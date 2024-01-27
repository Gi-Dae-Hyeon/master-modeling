"""
Model
Author: DaeHyeon Gi <spliter2157@gmail.com>
"""

import torch
from torch import nn, optim
from pytorch_lightning import LightningModule
from torchmetrics import SpearmanCorrCoef, PearsonCorrCoef, MeanSquaredError
from transformers import AutoTokenizer

from .layers import (
    ContrastiveLoss,
    MLPLayer,
    Encoder,
)


class Model(LightningModule):
    """DiffCSE Model"""
    def __init__(self, hparams: dict) -> None:
        super().__init__()
        self.hparams.update(hparams)
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
            anc, pos = batch
            anc, pos = self(anc), self(pos)
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
            self.mse_p.compute() ** 0.5 + self.mse_n.compute() ** 0.5,
            prog_bar=True,
            sync_dist=True
        )

        self.spearman.reset()
        self.pearson.reset()
        self.mse_p.reset()
        self.mse_n.reset()
