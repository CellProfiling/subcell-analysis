"""Cell-DINO wrapped to return a ``feature_vector`` like the other encoders.

The backbone is built from the dinov2 repository by
``subcell.models.load.get_cell_dino_model``. The feature vector concatenates the
CLS token with the mean of the patch tokens of the last block.

With ``output_attentions``, the output also carries the attention weights of the
last block, ``attentions[-1]`` of shape [B, heads, tokens, tokens], as for the
other encoders. The dinov2 blocks attend with ``scaled_dot_product_attention``,
which does not return its weights, so they are formed from that block's own
``qkv`` projection and scale applied to the tokens it receives.
"""
import torch
import torch.nn as nn

from subcell.models.vision_transformer import DINOOutput


class CellDINO(nn.Module):
    """A Cell-DINO backbone returning ``DINOOutput(feature_vector[, attentions])``."""

    def __init__(self, backbone, output_attentions=False):
        super().__init__()
        self.backbone = backbone
        self.output_attentions = output_attentions
        self.patch_size = backbone.patch_size
        if output_attentions:
            self.last_attn.register_forward_pre_hook(self._keep_tokens)

    @property
    def last_attn(self):
        last = self.backbone.blocks[-1]
        return last[-1].attn if self.backbone.chunked_blocks else last.attn

    def _keep_tokens(self, module, args):
        self._tokens = args[0]

    def last_attention(self):
        attn, tokens = self.last_attn, self._tokens
        B, N, C = tokens.shape
        q, k, _ = attn.qkv(tokens).reshape(B, N, 3, attn.num_heads, C // attn.num_heads).permute(2, 0, 3, 1, 4)
        return (q @ k.transpose(-2, -1) * attn.scale).softmax(dim=-1)

    def forward(self, x):
        ((patch, cls),) = self.backbone.get_intermediate_layers(x, n=1, return_class_token=True)
        output = DINOOutput(feature_vector=torch.cat([cls, patch.mean(dim=1)], dim=-1))
        if self.output_attentions:
            output.attentions = (self.last_attention(),)
        return output
