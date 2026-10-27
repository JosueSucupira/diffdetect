"""Wrapper for multi-crop ViT inference with patch projection and heads."""

from __future__ import annotations

import math
from typing import Optional

try:
    import numpy as np
    import timm
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    np = None  # type: ignore[assignment]
    timm = None  # type: ignore[assignment]
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

from .layers import Attention


def replicate_wrap(x: torch.Tensor, new_shape: tuple[int, int]) -> torch.Tensor:
    """Pad tensor by repeating edges to match the desired patch spatial shape."""
    rep_height = max(int(math.ceil(new_shape[-2] / x.shape[-2])), 1)
    rep_width = max(int(math.ceil(new_shape[-1] / x.shape[-1])), 1)
    x = x.repeat(1, 1, rep_height, rep_width)
    return x[..., : new_shape[-2], : new_shape[-1]]


class Wrapper5crops(nn.Module if nn is not None else object):  # type: ignore[misc]
    """Multi-crop wrapper around a timm Vision Transformer model."""

    def __init__(
        self,
        model: nn.Module,
        patch_size: int = 504,
        return_features: bool = True,
    ) -> None:
        if nn is None:
            raise RuntimeError("PyTorch is required to initialize Wrapper5crops")
        super().__init__()
        self.model = model
        self.model.set_input_size(img_size=patch_size)
        self.patch_embed = self.model.patch_embed
        self.model.patch_embed = nn.Identity()
        self.return_features = return_features
        self.patch_size = patch_size

        head_dim = self.model.head.in_features
        self.cls_head = nn.Linear(head_dim, 3)
        self.cls_head_binary = nn.Linear(head_dim, 1)
        self.cls_head_binary.weight = nn.Parameter(self.model.head.weight.detach().clone())
        self.cls_head_binary.bias = nn.Parameter(self.model.head.bias.detach().clone())
        self.attention = Attention(768, 8)
        self.cls_head_mlp = nn.Sequential(
            nn.Linear(head_dim, 768),
            nn.ReLU(),
            nn.Linear(768, 3),
        )
        self.conv = nn.Conv2d(768, 128, kernel_size=3, stride=1, padding=1)
        self.block: Optional[nn.Module] = None

    def forward(self, x: torch.Tensor):
        grid_size = self.patch_embed.grid_size
        embeddings = self.patch_embed.proj(x)

        pad_h = max(grid_size[0] - embeddings.shape[-2], 0)
        pad_w = max(grid_size[1] - embeddings.shape[-1], 0)
        if pad_h > 0 or pad_w > 0:
            embeddings = replicate_wrap(embeddings, grid_size)

        if getattr(self.patch_embed, "flatten", True):
            embeddings = embeddings.flatten(2).transpose(1, 2)
        embeddings = self.patch_embed.norm(embeddings)
        features = self.model.lora_vit.forward_features(embeddings)

        if self.block is not None:
            features = self.block(features)

        y = self.model.lora_vit.forward_head(features)
        if self.return_features:
            return y, features
        return y

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = False):
        patch_embed_dict = {
            k[12:]: v for k, v in state_dict.items() if k.startswith("patch_embed.")
        }
        if patch_embed_dict:
            self.patch_embed.load_state_dict(patch_embed_dict, strict=False)

        model_dict = {
            k: v
            for k, v in state_dict.items()
            if not (
                k.startswith("patch_embed.")
                or k.startswith("attention.")
                or k.startswith("cls_head.")
                or k.startswith("cls_head_binary.")
                or k.startswith("cls_head_mlp.")
                or k.startswith("block.")
                or k.startswith("conv.")
            )
        }
        if model_dict:
            self.model.load_state_dict(model_dict, strict=False)

        if any(k.startswith("attention.") for k in state_dict):
            self.attention.load_state_dict(
                {k[10:]: v for k, v in state_dict.items() if k.startswith("attention.")},
                strict=False,
            )
        if any(k.startswith("cls_head.") for k in state_dict):
            self.cls_head.load_state_dict(
                {k[9:]: v for k, v in state_dict.items() if k.startswith("cls_head.")},
                strict=False,
            )
        if any(k.startswith("cls_head_binary.") for k in state_dict):
            self.cls_head_binary.load_state_dict(
                {k[16:]: v for k, v in state_dict.items() if k.startswith("cls_head_binary.")},
                strict=False,
            )
        if any(k.startswith("cls_head_mlp.") for k in state_dict):
            self.cls_head_mlp.load_state_dict(
                {k[13:]: v for k, v in state_dict.items() if k.startswith("cls_head_mlp.")},
                strict=False,
            )
        if any(k.startswith("block.") for k in state_dict) and self.block is not None:
            self.block.load_state_dict(
                {k[6:]: v for k, v in state_dict.items() if k.startswith("block.")},
                strict=False,
            )
