"""
Model
Author: DaeHyeon Gi <spliter2157@gmail.com>
"""
from typing import Optional, Callable
from rich import print

import torch
from torch import nn, optim
from pytorch_lightning import LightningModule
from torchmetrics import (
    SpearmanCorrCoef,
    PearsonCorrCoef,
    MeanSquaredError,
    R2Score,
)
from transformers import AutoTokenizer
from rank_bm25 import BM25Plus

from .layers import (
    ContrastiveLoss,
    MLPLayer,
    Encoder,
)


class HardnessAnealingSampler:
    mode_dict = {"min": torch.lt, "max": torch.gt}
    order_dict = {"min": "<", "max": ">"}
    def __init__(
        self,
        mode: str,
        patience: int,
        min_delta: float,
        verbose: bool = True
    ) -> None:
        super().__init__()
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.wait_count = 0
        self.hardness = 0
        self.best_score = -torch.inf if mode == "max" else torch.inf

    def __call__(
        self,
        query: list[str],
        corpus: list[str],
        vectors: torch.Tensor
    ) -> torch.Tensor:
        if len(corpus) != vectors.size()[0]:
            raise ValueError()
        vectors = vectors.cpu()
        tokenized_corpus = [doc.lower().split(" ") for doc in set(corpus)]
        bm25 = BM25Plus(tokenized_corpus)

        scores = [bm25.get_scores(q.split()) for q in query]

        negative_samples = torch.Tensor()
        for score in scores:
            sorted_indices = torch.argsort(torch.Tensor(score), descending=True)
            negative_idx = sorted_indices[-self.hardness]
            negative_samples = torch.cat([negative_samples, vectors[negative_idx].unsqueeze(0)])
        return negative_samples.cuda()

    @property
    def monitor_op(self) -> Callable:
        return self.mode_dict[self.mode]

    def run_hardness_check(self, metrics: float) -> None:
        harder = self._evaluate_hardness_criteria(metrics)
        if harder:
            self.hardness += 1

    def _evaluate_hardness_criteria(self, current: torch.Tensor) -> tuple[bool, Optional[str]]:
        harder = False
        if self.monitor_op(current + self.min_delta, self.best_score):
            harder = False
            print(f"best score is updated!")
            print(f"best score {self.best_score} -> {current}")
            self.best_score = current
            self.wait_count = 0
        else:
            print(f"current score is {current}, best score is {self.best_score}")
            print(f"wait count {self.wait_count} -> {self.wait_count + 1}")
            self.wait_count += 1
            if self.wait_count >= self.patience:
                harder = True
                self.wait_count = 0
                print(f"patience {self.patience} 에 도달하였습니다.\n hardness {self.hardness} -> {self.hardness + 1}")
                return harder
        return harder


class Model(LightningModule):
    """DiffCSE Model"""
    def __init__(self, hparams: dict) -> None:
        super().__init__()
        self.hparams.update(hparams)
        self.save_hyperparameters()

        self.encoder = Encoder(hparams)
        self.tokenizer = AutoTokenizer.from_pretrained(
            hparams["pretrained_model"]
        )
        self.mlp = MLPLayer()
        self.loss_fn = ContrastiveLoss()
        self.spearman = SpearmanCorrCoef()
        self.pearson = PearsonCorrCoef()
        self.mse_p = MeanSquaredError()
        self.mse_n = MeanSquaredError()
        self.r_square = R2Score()

        hardness_config = self.hparams["hardness_annealing"]
        self.apply_hardness = hardness_config.pop("apply")
        if self.apply_hardness:
            self.hardness_annealing = HardnessAnealingSampler(**hardness_config)

    def configure_optimizers(self) -> dict:
        """Configure optimizer and lr scheduler"""
        params = [param for param in self.encoder.parameters()] \
            + [param for param in self.mlp.parameters()]
        optimizer = optim.AdamW(params, lr=self.hparams.learning_rate)
        return {
            "optimizer": optimizer,
        }

    def tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
        tokens = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.hparams["max_length"],
        )
        return tokens

    def forward(self, texts: list[str]) -> torch.Tensor:
        """Forward pass"""
        tokens = self.tokenize(texts)
        logit = self.encoder(tokens)
        logit = self.mlp(logit)
        return logit

    def training_step(self, batch: dict, batch_idx: int) -> dict:
        """Training Step"""
        if len(batch) == 2:
            anc_txt, pos_txt = batch
            anc, pos = self(anc_txt), self(pos_txt)
            if self.apply_hardness:
                neg = self.hardness_annealing(
                    query=anc_txt,
                    corpus=anc_txt + pos_txt,
                    vectors=torch.concat((anc, pos))
                )
                loss = self.loss_fn(anc=anc, pos=pos, neg=neg)
            else:
                loss = self.loss_fn(anc=anc, pos=pos)
        elif len(batch) == 3:
            anc, pos, neg = batch
            anc, pos, neg = self(anc), self(pos), self(neg)
            loss = self.loss_fn(anc=anc, pos=pos, neg=neg)
        batch_size = len(anc)
        self.log("loss/train_loss", loss, batch_size=batch_size, prog_bar=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        """Test during Training"""
        sent, hypo, label = batch
        sent, hypo = self(sent), self(hypo)
        sim = nn.functional.cosine_similarity(sent, hypo)
        self.pearson.update(preds=sim, target=label.to(self.hparams["device"]))
        self.spearman.update(preds=sim, target=label.to(self.hparams["device"]))
        self.r_square.update(preds=sim, target=label.to(self.hparams["device"]))
        for si, lab in zip(sim, label):
            if lab == 1:
                self.mse_p.update(preds=si, target=lab.to(self.hparams["device"]))
            else:
                self.mse_n.update(preds=si, target=lab.to(self.hparams["device"]))

    def on_validation_epoch_end(self) -> None:
        self.log("validation/spearman", self.spearman.compute(),  prog_bar=True, sync_dist=True)
        self.log("validation/pearson", self.pearson.compute(), prog_bar=True, sync_dist=True)
        self.log("validation/RMSE_p", self.mse_p.compute() ** 0.5, prog_bar=False, sync_dist=True)
        self.log("validation/RMSE_n", self.mse_n.compute() ** 0.5, prog_bar=False, sync_dist=True)
        self.log(
            "validation/RMSE",
            (self.mse_p.compute() + self.mse_n.compute()) ** 0.5,
            prog_bar=True,
            sync_dist=True
        )
        self.log("validation/R2", self.r_square.compute(), prog_bar=False, sync_dist=True)

        if self.apply_hardness:
            self.hardness_annealing.run_hardness_check(
                metrics=(self.mse_p.compute() + self.mse_n.compute()) ** 0.5,
            )
            self.log(
                "validation/hardness",
                self.hardness_annealing.hardness,
                prog_bar=True,
                sync_dist=False
            )

        self.spearman.reset()
        self.pearson.reset()
        self.mse_p.reset()
        self.mse_n.reset()
        self.r_square.reset()
