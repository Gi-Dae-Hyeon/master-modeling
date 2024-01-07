"""
Layers for the model
Author: Daehyeon Gi <spliter2157@gmail.com>
"""

import torch
import torch.nn as nn
from torch import Tensor
import transformers


class MLPLayer(nn.Module):
    """Head for getting sent representation"""
    def __init__(self, dim: int = 1024) -> None:
        super().__init__()
        self.dense = nn.Linear(dim, dim)
        self.activation = nn.Tanh()

    def forward(self, feature_vector: Tensor) -> Tensor:
        """Forward pass"""
        return self.activation(self.dense(feature_vector))


class ProjectionMLP(nn.Module):
    def __init__(self, dim: int = 1024) -> None:
        super().__init__()
        self.hidden_dim = dim * 2
        affine=False
        list_layers = [nn.Linear(dim, dim * 2, bias=False),
                       nn.BatchNorm1d(dim * 2),
                       nn.ReLU(inplace=True)]
        list_layers += [nn.Linear(dim * 2, dim, bias=False),
                        nn.BatchNorm1d(dim, affine=affine)]
        self.net = nn.Sequential(*list_layers)

    def forward(self, x):
        return self.net(x)


class Encoder(nn.Module):
    """Encoder for getting sent representation"""
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.encoder = getattr(transformers, config["model"]) \
            .from_pretrained(config["pretrained_model"], add_pooling_layer=False)

    def forward(self, tokens: dict[str, Tensor]) -> Tensor:
        """Forward pass"""
        tokens = {key: value.cuda() for key, value in tokens.items()}
        return self.encoder(**tokens).last_hidden_state[:, 0]


class ContrastiveLoss(nn.Module):
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
        similarity = pos_sim
        if neg is not None:
            neg_sim = self.cos(anc.unsqueeze(1), neg.unsqueeze(0)) / self.temp
            similarity = torch.cat([pos_sim, neg_sim], dim=1)
        labels = torch.arange(similarity.size(0)).long().cuda()
        return self.criterion(similarity, labels)
