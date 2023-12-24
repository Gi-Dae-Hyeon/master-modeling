"""
Model
Author: DaeHyeon Gi <spliter2157@gmail.com>
"""

from typing import List, Dict

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule
from torchmetrics import SpearmanCorrCoef, PearsonCorrCoef
from transformers import AutoTokenizer

from .layers import (
    SupContrastiveLoss,
    MLPLayer,
    Encoder
)


class Model(LightningModule):
    """SimCSE Model"""
    def __init__(self, hparams: Dict) -> None:
        super().__init__()
        self.hparams.update(hparams)
        self.encoder = Encoder(hparams)
        self.tokenizer = AutoTokenizer.from_pretrained(
            hparams["pretrained_model"]
        )
        self.mlp = MLPLayer()
        self.loss_fn = SupContrastiveLoss()
        self.spearman = SpearmanCorrCoef()
        self.pearson = PearsonCorrCoef()

    def configure_optimizers(self) -> Dict:
        """Configure optimizer and lr scheduler"""
        params = [param for param in self.encoder.parameters()] \
            + [param for param in self.mlp.parameters()]
        optimizer = optim.AdamW(params, lr=self.hparams.learning_rate)
        sch_config = {
            "scheduler": optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=10
            ),
            "interval": "step"
        }
        return {
            "optimizer": optimizer,
            "lr_scheduler": sch_config
        }

    def forward(self, texts: List[str], do_mlm: bool = False) -> torch.Tensor:
        """Forward pass"""
        tokens = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=50,
        )
        logit = self.encoder(tokens)
        return self.mlp(logit)

    def training_step(self, batch: Dict, batch_idx: int) -> Dict:
        """Training Step"""
        anc, pos, neg = batch
        batch_size = len(anc)
        anc, pos, neg = self(anc), self(pos), self(neg)
        loss = self.loss_fn(anc=anc, pos=pos, neg=neg)
        self.log("loss/train_loss", loss, batch_size=batch_size, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        """Test during Training"""
        sent, hypo, label = batch
        sent, hypo = self(sent), self(hypo)
        sim = nn.functional.cosine_similarity(sent, hypo)
        self.pearson.update(preds=sim, target=label.cuda())
        self.spearman.update(preds=sim, target=label.cuda())
        self.log("corr/spearman", self.spearman.compute(), on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        self.log("corr/pearson", self.pearson.compute(), on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
