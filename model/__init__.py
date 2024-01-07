"""
Model
Author: DaeHyeon Gi <spliter2157@gmail.com>
"""

import torch
from torch import nn, optim
from pytorch_lightning import LightningModule
from torchmetrics import SpearmanCorrCoef, PearsonCorrCoef
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
            max_length=50,
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
        anc, pos, neg = batch
        batch_size = len(anc)
        anc, pos, neg = self(anc), self(pos), self(neg)
        loss = self.loss_fn(anc=anc, pos=pos, neg=neg)
        self.log("loss/train_loss", loss, batch_size=batch_size, prog_bar=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        """Test during Training"""
        sent, hypo, label = batch
        sent, hypo = self(sent), self(hypo)
        sim = nn.functional.cosine_similarity(sent, hypo)
        self.pearson.update(preds=sim, target=label.cuda())
        self.spearman.update(preds=sim, target=label.cuda())

    def on_validation_epoch_end(self) -> None:
        self.log("corr/spearman", self.spearman.compute(),  prog_bar=True, sync_dist=True)
        self.log("corr/pearson", self.pearson.compute(), prog_bar=True, sync_dist=True)
        self.spearman.reset()
        self.pearson.reset()
