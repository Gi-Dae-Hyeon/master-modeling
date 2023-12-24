"""
Layers for the model
Author: Daehyeon Gi <spliter2157@gmail.com>
"""

from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor
import transformers


class MLPLayer(nn.Module):
    """Head for getting sent representation"""
    def __init__(self) -> None:
        super().__init__()
        self.dense = nn.Linear(1024, 1024)
        self.activation = nn.Tanh()

    def forward(self, feature_vector: Tensor) -> Tensor:
        """Forward pass"""
        return self.activation(self.dense(feature_vector))


class Encoder(nn.Module):
    """Encoder for getting sent representation"""
    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.encoder = getattr(transformers, config["model"]) \
            .from_pretrained(config["pretrained_model"], add_pooling_layer=False)

    def forward(self, tokens: Dict[str, Tensor]) -> Tensor:
        """Forward pass"""
        tokens = {key: value.cuda() for key, value in tokens.items()}
        return self.encoder(**tokens).last_hidden_state[:, 0]  # CLS Pooling


class SupContrastiveLoss(nn.Module):
    """Contrastive Loss"""
    def __init__(self, temp: float = .05) -> None:
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)
        self.criterion = nn.CrossEntropyLoss()

    def forward(
        self, anc: Tensor, pos: Tensor, neg: Tensor
        ) -> Tensor:
        """Forward pass"""
        pos_sim = self.cos(anc.unsqueeze(1), pos.unsqueeze(0)) / self.temp
        neg_sim = self.cos(anc.unsqueeze(1), neg.unsqueeze(0)) / self.temp
        similarity = torch.cat([pos_sim, neg_sim], dim=1)
        labels = torch.arange(similarity.size(0)).long().cuda()
        return self.criterion(similarity, labels)
