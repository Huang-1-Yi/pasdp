"""Transformer-based action heads.

``act`` contains the ACT DETR/CVAE network and ``diffusion_transformer``
contains the sequence model used by Diffusion Transformer Policy.
"""

from diffusion_policy.model.transformer_heads.diffusion_transformer import (
    TransformerForDiffusion,
)

__all__ = ["TransformerForDiffusion"]
