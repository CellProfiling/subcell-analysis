from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from transformers import PretrainedConfig, ViTConfig, ViTMAEConfig, ViTModel
from transformers.utils import ModelOutput


@dataclass
class ViTPoolModelOutput(ModelOutput):
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Tuple[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None

    feature_vector: torch.FloatTensor = None
    pool_attn: torch.FloatTensor = None

    probabilties: torch.FloatTensor = None


class GatedAttentionPooler(nn.Module):
    def __init__(
        self, dim: int, int_dim: int = 512, num_heads: int = 1, out_dim: int = None
    ):
        super().__init__()

        self.num_heads = num_heads

        self.attention_v = nn.Sequential(
            nn.Dropout(0.1), nn.Linear(dim, int_dim), nn.Tanh()
        )
        self.attention_u = nn.Sequential(
            nn.Dropout(0.1), nn.Linear(dim, int_dim), nn.GELU()
        )
        self.attention = nn.Linear(int_dim, num_heads)

        self.softmax = nn.Softmax(dim=-1)

        if out_dim is None:
            self.out_dim = dim * num_heads
            self.out_proj = nn.Identity()
        else:
            self.out_dim = out_dim
            self.out_proj = nn.Linear(dim * num_heads, out_dim)

    def forward(self, x: torch.Tensor) -> Tuple[Tensor, Tensor]:
        v = self.attention_v(x)
        u = self.attention_u(x)

        attn = self.attention(v * u).permute(0, 2, 1)
        attn = self.softmax(attn)

        x = torch.bmm(attn, x)
        x = x.view(x.shape[0], -1)

        x = self.out_proj(x)
        return x, attn


class ViTPoolModel(nn.Module):
    def __init__(
        self, vit_config: Dict, pool_config: Dict, classifier_config: Dict = None
    ):
        super(ViTPoolModel, self).__init__()
        self.vit_config = ViTConfig(**vit_config)
        self.encoder = ViTModel(self.vit_config, add_pooling_layer=False)

        self.pool_model = GatedAttentionPooler(**pool_config) if pool_config else None

        if classifier_config:
            self.classifier = nn.Sequential(
                nn.Dropout(0.25),
                nn.Linear(self.pool_model.out_dim, self.pool_model.out_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(self.pool_model.out_dim // 2, 19),
            )
            self.sigmoid = nn.Sigmoid()
        else:
            self.classifier = None

    def forward(self, x):
        outputs = self.encoder(x, output_attentions=True, interpolate_pos_encoding=True)

        if self.pool_model:
            pool_op, pool_attn = self.pool_model(outputs.last_hidden_state)
        else:
            pool_op = torch.mean(outputs.last_hidden_state, dim=1)
            pool_attn = None

        if self.classifier:
            logits = self.classifier(pool_op)
            probs = self.sigmoid(logits)
        else:
            probs = None

        return ViTPoolModelOutput(
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            last_hidden_state=outputs.last_hidden_state,
            feature_vector=pool_op,
            pool_attn=pool_attn,
            probabilties=probs,
        )
