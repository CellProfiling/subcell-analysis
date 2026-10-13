"""The SubCell encoder: a ViT followed by gated-attention pooling over its tokens.

``ViTPoolModel`` returns a ``ViTPoolModelOutput`` whose ``feature_vector`` is the
pooled embedding used throughout the analyses (and ``pool_attn`` the pooling
attention). Built from the ``model.args`` block of a config by
``subcell.models.load.get_subcell_model``.
"""
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch
from torch import nn, Tensor
from transformers import ViTConfig, ViTModel
from transformers.modeling_outputs import BaseModelOutput
from transformers.utils import ModelOutput


@dataclass
class ViTPoolModelOutput(ModelOutput):
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Tuple[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None

    feature_vector: torch.FloatTensor = None
    pool_attn: torch.FloatTensor = None


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
        self,
        vit_config: Dict,
        pool_config: Dict,
        output_attentions: bool | str = True,
    ):
        super().__init__()
        self.output_attentions = output_attentions
        self.vit_config = ViTConfig(**vit_config)
        self.encoder = ViTModel(self.vit_config, add_pooling_layer=False)

        self.pool_model = GatedAttentionPooler(**pool_config) if pool_config else None

    def encode_last_attention(self, x):
        """The encoder's forward pass, returning the last layer's attention only.

        ``output_attentions=True`` makes every layer keep its full attention matrix
        and run the eager kernel; here only the last layer does.
        """
        hidden = self.encoder.embeddings(x, interpolate_pos_encoding=True)
        *layers, last = self.encoder.encoder.layer
        for layer in layers:
            hidden = layer(hidden)[0]
        hidden, attention = last(hidden, output_attentions=True)
        return BaseModelOutput(
            last_hidden_state=self.encoder.layernorm(hidden), attentions=(attention,)
        )

    def forward(self, x):
        if self.output_attentions == "last":
            outputs = self.encode_last_attention(x)
        else:
            outputs = self.encoder(
                x, output_attentions=self.output_attentions, interpolate_pos_encoding=True
            )

        if self.pool_model:
            pool_op, pool_attn = self.pool_model(outputs.last_hidden_state)
        else:
            pool_op = torch.mean(outputs.last_hidden_state, dim=1)
            pool_attn = None

        return ViTPoolModelOutput(
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            last_hidden_state=outputs.last_hidden_state,
            feature_vector=pool_op,
            pool_attn=pool_attn,
        )
