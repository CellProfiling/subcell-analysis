import os
import random
from typing import Tuple

import numpy as np
import torch

from torch import nn
from torch import Tensor
from torch.nn import functional as F


def set_random_seed(seed):
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SigmoidFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super(SigmoidFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha > 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "none":
            pass
        elif self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        else:
            raise ValueError(
                f"Invalid Value for arg 'reduction': '{self.reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
            )
        return loss


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512, n_layers=1, dropout=0.5):
        super().__init__()

        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)

        if n_layers > 1:
            for i in range(n_layers):
                setattr(self, f"fc{i}", nn.Linear(input_dim, hidden_dim))
                setattr(self, f"dropout{i}", nn.Dropout(dropout))
                setattr(self, f"nonlin{i}", nn.ReLU())
                setattr(self, f"norm{i}", nn.BatchNorm1d(hidden_dim))
                input_dim = hidden_dim

        self.fc_out = nn.Linear(input_dim, output_dim)

    def forward(self, X):
        X = self.dropout(X)
        if self.n_layers > 1:
            for i in range(self.n_layers):
                X = getattr(self, f"fc{i}")(X)
                X = getattr(self, f"dropout{i}")(X)
                X = getattr(self, f"nonlin{i}")(X)
                X = getattr(self, f"norm{i}")(X)
        X = self.fc_out(X)
        return X


class GatedAttentionClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = None,
        pool_hidden_dim: int = 512,
        pool_dropout: float = 0.25,
        n_layers: int = 1,
        hidden_dim: int = 512,
        dropout: float = 0.25,
        num_heads: int = 1,
    ):
        super().__init__()

        self.num_heads = num_heads

        self.attention_v = nn.Sequential(
            nn.Dropout(pool_dropout), nn.Linear(input_dim, pool_hidden_dim), nn.Tanh()
        )
        self.attention_u = nn.Sequential(
            nn.Dropout(pool_dropout), nn.Linear(input_dim, pool_hidden_dim), nn.GELU()
        )
        self.attention = nn.Linear(pool_hidden_dim, num_heads)

        self.softmax = nn.Softmax(dim=-1)

        self.classifier = MLPClassifier(
            input_dim=input_dim * num_heads,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
        )

    def get_attention(self, x: torch.Tensor) -> torch.Tensor:
        v = self.attention_v(x)
        u = self.attention_u(x)

        attn = self.attention(v * u).permute(0, 2, 1)
        attn = self.softmax(attn)
        return attn

    def forward(self, x: torch.Tensor) -> Tuple[Tensor, Tensor]:
        attn = self.get_attention(x)

        x = torch.bmm(attn, x)
        x = x.view(x.shape[0], -1)

        x = self.classifier(x)
        return x
