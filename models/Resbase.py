# -*- coding: utf-8 -*-
"""Minimal Restormer backbone primitives used by GDF-Restormer.

This release file intentionally contains only the Restormer building blocks
required by ``models/Res_Strict.py``. The numerical structure follows the
Restormer-style MDTA/GDFN implementation used by the validated project code.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


DEBUG_RESTORMER = os.environ.get("DEBUG_RESTORMER", "0").lower() in {
    "1", "true", "yes", "on"
}
try:
    DEBUG_PROB = float(os.environ.get("DEBUG_RESTORMER_PROB", "0.05"))
except Exception:
    DEBUG_PROB = 0.05


def maybe_debug_tensor(name, tensor):
    """Optional tensor diagnostics; disabled by default in the public release."""
    if not DEBUG_RESTORMER:
        return
    if torch.rand(1).item() >= DEBUG_PROB:
        return
    try:
        cpu_tensor = tensor.detach().float().cpu()
        if torch.isnan(cpu_tensor).any() or torch.isinf(cpu_tensor).any():
            print(
                f"[DEBUG] {name} has NaN or Inf: "
                f"min={cpu_tensor.min().item():.4f}, "
                f"max={cpu_tensor.max().item():.4f}, "
                f"shape={tuple(tensor.shape)}"
            )
        else:
            print(
                f"[DEBUG] {name}: "
                f"min={cpu_tensor.min().item():.4f}, "
                f"max={cpu_tensor.max().item():.4f}, "
                f"mean={cpu_tensor.mean().item():.4f}, "
                f"shape={tuple(tensor.shape)}"
            )
    except Exception as exc:
        print(f"[DEBUG] Failed to debug {name}: {exc}")


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        sigma = x.float().var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mu = x.float().mean(-1, keepdim=True)
        sigma = x.float().var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super().__init__()
        self.body = (
            BiasFree_LayerNorm(dim)
            if LayerNorm_type == "BiasFree"
            else WithBias_LayerNorm(dim)
        )

    def forward(self, x):
        h, w = x.shape[-2:]
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.body(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, expansion_factor, bias):
        super().__init__()
        hidden = int(dim * expansion_factor)
        self.project_in = nn.Conv2d(
            dim, hidden * 2, kernel_size=1, bias=bias
        )
        self.dwconv = nn.Conv2d(
            hidden * 2,
            hidden * 2,
            3,
            1,
            1,
            groups=hidden * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(
            hidden, dim, kernel_size=1, bias=bias
        )

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            3,
            1,
            1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x):
        _, _, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(
            q, "b (head c) h1 w1 -> b head c (h1 w1)",
            head=self.num_heads
        )
        k = rearrange(
            k, "b (head c) h1 w1 -> b head c (h1 w1)",
            head=self.num_heads
        )
        v = rearrange(
            v, "b (head c) h1 w1 -> b head c (h1 w1)",
            head=self.num_heads
        )

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # Numerically stable form of the same softmax.
        attn = attn - attn.amax(dim=-1, keepdim=True)
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = rearrange(
            out,
            "b head c (h1 w1) -> b (head c) h1 w1",
            h1=h,
            w1=w,
        )
        return self.project_out(out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        expansion_factor,
        bias,
        LayerNorm_type,
    ):
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super().__init__()
        self.proj = nn.Conv2d(
            in_c, embed_dim, 3, 1, 1, bias=bias
        )

    def forward(self, x):
        x = self.proj(x)
        maybe_debug_tensor("patch_embed", x)
        return x


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat, n_feat // 2, 3, 1, 1, bias=False
            ),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat, n_feat * 2, 3, 1, 1, bias=False
            ),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)
