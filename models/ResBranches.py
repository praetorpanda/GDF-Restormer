# -*- coding: utf-8 -*-
"""
Restormer_ResBasisBranches
==========================

建议保存路径：
    models/Restormer_ResBasisBranches.py

本文件整合两个结构，方便在同一个文件中切换实验：

1. Restormer_LowRankBasis_ResBranch
   - prior_feat-driven aggressive residual branch
   - gate = PriorAwareResidualGate
   - aggressive branch 输入：x + prior_feat + coord
   - 适合作为较安全的三分支版本

2. Restormer_LowRankBasis_ExplicitBasisBranch
   - explicit basis-driven aggressive residual branch
   - gate = BasisAwareResidualGate
   - aggressive branch 输入：x + prior_feat + basis_prior + coeff_map + coord
   - 适合作为把 low-rank basis 显式加强为独立分支的激进版本

两个结构共同特点：
    - Restormer 主分支默认压缩为 2-level / num_blocks=(2,2)
    - 主分支拓扑按官方 Restormer 逻辑实现：
        encoder levels = level 1 到 level N-1
        deepest level = latent only
        decoder level1 concat 后不做 1x1 降维
        decoder level1 / refinement / output 使用 2*dim 通道
    - 保留 low-rank basis modulation path
    - 支持 return_monitor / return_aux
    - 支持 coord: [B,2,H,W] 或 [B,4,H,W]
    - coord=None 时自动生成 local normalized coordinate

推荐导入：
    from models.Restormer_ResBasisBranches import (
        Restormer_LowRankBasis_ResBranch,
        Restormer_LowRankBasis_ExplicitBasisBranch,
    )
"""

import math
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .Restormer import (
        OverlapPatchEmbed,
        TransformerBlock,
        Downsample,
        Upsample,
        maybe_debug_tensor,
    )
except Exception:
    from Restormer import (
        OverlapPatchEmbed,
        TransformerBlock,
        Downsample,
        Upsample,
        maybe_debug_tensor,
    )


# ============================================================
# Coordinate Utils
# ============================================================

def make_local_coord(
    x: torch.Tensor,
    coord_channels: int = 4,
    align_corners: bool = True,
) -> torch.Tensor:
    b, _, h, w = x.shape
    device, dtype = x.device, x.dtype

    if align_corners:
        yy = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        xx = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
    else:
        yy = (torch.arange(h, device=device, dtype=dtype) + 0.5) / h * 2.0 - 1.0
        xx = (torch.arange(w, device=device, dtype=dtype) + 0.5) / w * 2.0 - 1.0

    y, x_coord = torch.meshgrid(yy, xx, indexing="ij")
    x_coord = x_coord.unsqueeze(0).unsqueeze(0).expand(b, -1, -1, -1)
    y = y.unsqueeze(0).unsqueeze(0).expand(b, -1, -1, -1)

    if coord_channels == 2:
        return torch.cat([x_coord, y], dim=1)

    r = torch.sqrt(torch.clamp(x_coord ** 2 + y ** 2, min=1e-12))
    r2 = r ** 2
    return torch.cat([x_coord, y, r, r2], dim=1)


def normalize_coord_channels(
    coord: torch.Tensor,
    target_channels: int = 4,
) -> torch.Tensor:
    if coord.dim() != 4:
        raise ValueError(f"coord must be [B,C,H,W], got {tuple(coord.shape)}")

    if target_channels == 2:
        if coord.shape[1] < 2:
            raise ValueError(f"coord needs at least 2 channels, got {coord.shape[1]}")
        return coord[:, :2]

    if coord.shape[1] >= 4:
        return coord[:, :4]

    if coord.shape[1] == 2:
        x_coord = coord[:, 0:1]
        y = coord[:, 1:2]
        r = torch.sqrt(torch.clamp(x_coord ** 2 + y ** 2, min=1e-12))
        r2 = r ** 2
        return torch.cat([x_coord, y, r, r2], dim=1)

    raise ValueError(f"coord must have 2 or >=4 channels, got {coord.shape[1]}")


# ============================================================
# Monitor Utils
# ============================================================

def _safe_mean(x: torch.Tensor) -> torch.Tensor:
    return x.detach().float().mean()


def _safe_std(x: torch.Tensor) -> torch.Tensor:
    return x.detach().float().std(unbiased=False)


def _safe_min(x: torch.Tensor) -> torch.Tensor:
    return x.detach().float().min()


def _safe_max(x: torch.Tensor) -> torch.Tensor:
    return x.detach().float().max()


def radial_correlation_map(
    score: torch.Tensor,
    coord: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    if score.shape[1] > 1:
        score = score.mean(dim=1, keepdim=True)

    if coord.shape[1] >= 3:
        r = coord[:, 2:3]
    else:
        x_coord = coord[:, 0:1]
        y = coord[:, 1:2]
        r = torch.sqrt(torch.clamp(x_coord ** 2 + y ** 2, min=1e-12))

    s = score.detach().float().flatten(1)
    rv = r.detach().float().flatten(1)
    s = s - s.mean(dim=1, keepdim=True)
    rv = rv - rv.mean(dim=1, keepdim=True)
    corr = (s * rv).mean(dim=1) / (
        s.std(dim=1, unbiased=False) * rv.std(dim=1, unbiased=False) + eps
    )
    return corr.mean()


def edge_center_gap(
    score: torch.Tensor,
    coord: torch.Tensor,
) -> torch.Tensor:
    if score.shape[1] > 1:
        score = score.mean(dim=1, keepdim=True)

    score = score.detach().float()

    if coord.shape[1] >= 3:
        r = coord[:, 2:3].detach().float()
    else:
        x_coord = coord[:, 0:1]
        y = coord[:, 1:2]
        r = torch.sqrt(torch.clamp(x_coord.float() ** 2 + y.float() ** 2, min=1e-12))

    center_mask = (r <= 0.35).float()
    edge_mask = (r >= 0.75).float()

    center = (score * center_mask).sum() / (center_mask.sum() + 1e-6)
    edge = (score * edge_mask).sum() / (edge_mask.sum() + 1e-6)
    return edge - center


def basis_diversity_metric(basis: torch.Tensor) -> torch.Tensor:
    b = F.normalize(basis.detach().float(), dim=1)
    sim = b @ b.t()
    k = sim.shape[0]
    mask = ~torch.eye(k, device=sim.device, dtype=torch.bool)
    return sim[mask].abs().mean()


def coeff_stats(
    coeff: torch.Tensor,
    num_basis: int,
    prefix: str = "",
) -> Dict[str, torch.Tensor]:
    eps = 1e-8
    c = coeff.detach().float()
    entropy = -(c * torch.log(c + eps)).sum(dim=1).mean()
    entropy_norm = entropy / math.log(float(num_basis))
    top_prob = c.amax(dim=1).mean()
    usage = c.mean(dim=(0, 2, 3))
    return {
        f"{prefix}coeff_entropy_norm": entropy_norm,
        f"{prefix}coeff_top_prob_mean": top_prob,
        f"{prefix}coeff_spatial_std": c.std(unbiased=False),
        f"{prefix}basis_usage_std": usage.std(unbiased=False),
        f"{prefix}basis_usage_min": usage.min(),
        f"{prefix}basis_usage_max": usage.max(),
    }


def _logit_from_prob(p: float) -> float:
    p = float(max(min(p, 1.0 - 1e-6), 1e-6))
    return math.log(p / (1.0 - p))


# ============================================================
# Prior / Basis Modules
# ============================================================

class PriorEncoder(nn.Module):
    """
    输入 image 或 image + coord，输出 prior_feat。
    """

    def __init__(
        self,
        in_ch: int = 7,
        prior_ch: int = 32,
        mid_ch: int = 32,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(mid_ch, mid_ch, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(mid_ch, prior_ch, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LowRankBasisModulation(nn.Module):
    """
    原始 low-rank basis modulation 模块。

    注意：这个模块仍然用于调制 Restormer 主分支特征。
    新增的 explicit basis branch 使用 ExplicitBasisBranchPrior，
    不再只依赖本模块内部的 prior。
    """

    def __init__(
        self,
        prior_ch: int,
        feat_ch: int,
        num_basis: int = 8,
        hidden_ch: int = 32,
        zero_init_mod: bool = True,
        softmax_coeff: bool = True,
    ):
        super().__init__()
        self.num_basis = num_basis
        self.feat_ch = feat_ch
        self.softmax_coeff = softmax_coeff

        self.coeff_predictor = nn.Sequential(
            nn.Conv2d(prior_ch, hidden_ch, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden_ch, num_basis, 3, 1, 1),
        )
        self.basis = nn.Parameter(torch.randn(num_basis, feat_ch) * 0.02)
        self.to_scale_shift = nn.Sequential(
            nn.Conv2d(feat_ch, feat_ch * 2, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(feat_ch * 2, feat_ch * 2, 1, 1, 0),
        )

        if zero_init_mod:
            nn.init.zeros_(self.to_scale_shift[-1].weight)
            nn.init.zeros_(self.to_scale_shift[-1].bias)

    def forward(
        self,
        feat: torch.Tensor,
        prior_input: torch.Tensor,
        return_stats: bool = False,
        name: str = "",
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:

        if prior_input.shape[-2:] != feat.shape[-2:]:
            prior_input = F.interpolate(
                prior_input,
                size=feat.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        logits = self.coeff_predictor(prior_input)
        if self.softmax_coeff:
            coeff = torch.softmax(logits, dim=1)
        else:
            coeff = torch.sigmoid(logits)
            coeff = coeff / (coeff.sum(dim=1, keepdim=True) + 1e-6)

        prior = torch.einsum("bkhw,kc->bchw", coeff, self.basis)
        scale, shift = self.to_scale_shift(prior).chunk(2, dim=1)
        out = feat * (1.0 + scale) + shift

        if not return_stats:
            return out

        delta = out - feat
        stats = coeff_stats(coeff, self.num_basis, prefix=name)
        stats.update({
            f"{name}basis_cos_abs_offdiag": basis_diversity_metric(self.basis),
            f"{name}prior_abs_mean": _safe_mean(prior.abs()),
            f"{name}scale_abs_mean": _safe_mean(scale.abs()),
            f"{name}shift_abs_mean": _safe_mean(shift.abs()),
            f"{name}delta_ratio": (
                delta.detach().float().pow(2).mean().sqrt()
                / (feat.detach().float().pow(2).mean().sqrt() + 1e-6)
            ),
        })
        return out, stats

    def diversity_loss(self) -> torch.Tensor:
        b = F.normalize(self.basis.float(), dim=1)
        sim = b @ b.t()
        eye = torch.eye(self.num_basis, device=sim.device, dtype=sim.dtype)
        return ((sim - eye) ** 2).mean()


class ExplicitBasisBranchPrior(nn.Module):
    """
    显式 basis 分支 prior 生成器。

    这是本文件相对 ResBranch 版本的核心增强：
        prior_feat -> coeff_map + learnable branch basis -> basis_prior

    输出：
        basis_prior: [B, basis_ch, H, W]
        coeff:       [B, K, H, W]
        logits:      [B, K, H, W]

    basis_prior 会被直接送入：
        1. BasisDrivenAggressiveResidualBranch
        2. BasisAwareResidualGate
    """

    def __init__(
        self,
        prior_ch: int = 32,
        basis_ch: int = 32,
        num_basis: int = 8,
        hidden_ch: int = 32,
        softmax_coeff: bool = True,
        post_refine: bool = True,
    ):
        super().__init__()
        self.prior_ch = prior_ch
        self.basis_ch = basis_ch
        self.num_basis = num_basis
        self.softmax_coeff = softmax_coeff

        self.coeff_predictor = nn.Sequential(
            nn.Conv2d(prior_ch, hidden_ch, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden_ch, hidden_ch, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden_ch, num_basis, 3, 1, 1),
        )

        self.basis = nn.Parameter(torch.randn(num_basis, basis_ch) * 0.02)

        if post_refine:
            self.post = nn.Sequential(
                nn.Conv2d(basis_ch, basis_ch, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(basis_ch, basis_ch, 3, 1, 1),
            )
        else:
            self.post = nn.Identity()

    def forward(
        self,
        prior_feat: torch.Tensor,
        return_stats: bool = False,
        name: str = "branch_basis_",
    ):
        logits = self.coeff_predictor(prior_feat)
        if self.softmax_coeff:
            coeff = torch.softmax(logits, dim=1)
        else:
            coeff = torch.sigmoid(logits)
            coeff = coeff / (coeff.sum(dim=1, keepdim=True) + 1e-6)

        raw_basis_prior = torch.einsum("bkhw,kc->bchw", coeff, self.basis)
        basis_prior = self.post(raw_basis_prior)

        if not return_stats:
            return basis_prior, coeff, logits

        stats = coeff_stats(coeff, self.num_basis, prefix=name)
        stats.update({
            f"{name}basis_cos_abs_offdiag": basis_diversity_metric(self.basis),
            f"{name}prior_abs_mean": _safe_mean(basis_prior.abs()),
            f"{name}prior_std": _safe_std(basis_prior),
            f"{name}raw_prior_abs_mean": _safe_mean(raw_basis_prior.abs()),
        })
        return basis_prior, coeff, logits, stats

    def diversity_loss(self) -> torch.Tensor:
        b = F.normalize(self.basis.float(), dim=1)
        sim = b @ b.t()
        eye = torch.eye(self.num_basis, device=sim.device, dtype=sim.dtype)
        return ((sim - eye) ** 2).mean()


# ============================================================
# Basis-driven Aggressive Residual Branch
# ============================================================

class ConvResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        dilation: int = 1,
        use_depthwise: bool = False,
    ):
        super().__init__()
        if use_depthwise:
            self.body = nn.Sequential(
                nn.Conv2d(
                    channels,
                    channels,
                    3,
                    1,
                    dilation,
                    dilation=dilation,
                    groups=channels,
                    bias=True,
                ),
                nn.GELU(),
                nn.Conv2d(channels, channels, 1, 1, 0, bias=True),
                nn.GELU(),
                nn.Conv2d(
                    channels,
                    channels,
                    3,
                    1,
                    dilation,
                    dilation=dilation,
                    groups=channels,
                    bias=True,
                ),
                nn.GELU(),
                nn.Conv2d(channels, channels, 1, 1, 0, bias=True),
            )
        else:
            self.body = nn.Sequential(
                nn.Conv2d(channels, channels, 3, 1, dilation, dilation=dilation),
                nn.GELU(),
                nn.Conv2d(channels, channels, 3, 1, dilation, dilation=dilation),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class BasisDrivenAggressiveResidualBranch(nn.Module):
    """
    显式 basis-driven aggressive residual branch。

    输入：
        x            [B, 3, H, W]
        prior_feat   [B, prior_ch, H, W]
        basis_prior  [B, basis_ch, H, W]
        coeff        [B, K, H, W]
        coord        [B, coord_ch, H, W] or None

    输出：
        branch_residual [B, out_ch, H, W]

    这里 basis_prior 和 coeff_map 被显式送入分支，
    因此该 branch 不再只是 prior_feat-driven，而是 explicit basis-driven。
    """

    def __init__(
        self,
        inp_channels: int = 3,
        out_channels: int = 3,
        prior_ch: int = 32,
        basis_ch: int = 32,
        num_basis: int = 8,
        coord_channels: int = 4,
        hidden_ch: int = 48,
        num_blocks: int = 3,
        use_coord: bool = True,
        use_coeff_map: bool = True,
        use_depthwise: bool = False,
        zero_init_output: bool = True,
    ):
        super().__init__()
        self.use_coord = use_coord
        self.use_coeff_map = use_coeff_map
        self.coord_channels = coord_channels

        in_ch = inp_channels + prior_ch + basis_ch
        if use_coeff_map:
            in_ch += num_basis
        if use_coord:
            in_ch += coord_channels

        self.in_proj = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 3, 1, 1),
            nn.GELU(),
        )

        blocks = []
        for i in range(num_blocks):
            dilation = 1 if i % 2 == 0 else 2
            blocks.append(
                ConvResidualBlock(
                    hidden_ch,
                    dilation=dilation,
                    use_depthwise=use_depthwise,
                )
            )
        self.blocks = nn.Sequential(*blocks)
        self.out_proj = nn.Conv2d(hidden_ch, out_channels, 3, 1, 1)

        if zero_init_output:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        prior_feat: torch.Tensor,
        basis_prior: torch.Tensor,
        coeff: Optional[torch.Tensor] = None,
        coord: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if prior_feat.shape[-2:] != x.shape[-2:]:
            prior_feat = F.interpolate(
                prior_feat,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        if basis_prior.shape[-2:] != x.shape[-2:]:
            basis_prior = F.interpolate(
                basis_prior,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        inputs = [x, prior_feat, basis_prior]

        if self.use_coeff_map:
            if coeff is None:
                raise ValueError("BasisDrivenAggressiveResidualBranch requires coeff when use_coeff_map=True.")
            if coeff.shape[-2:] != x.shape[-2:]:
                coeff = F.interpolate(
                    coeff,
                    size=x.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            inputs.append(coeff)

        if self.use_coord:
            if coord is None:
                raise ValueError("BasisDrivenAggressiveResidualBranch requires coord when use_coord=True.")
            if coord.shape[-2:] != x.shape[-2:]:
                coord = F.interpolate(
                    coord,
                    size=x.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            inputs.append(coord)

        h = torch.cat(inputs, dim=1)
        h = self.in_proj(h)
        h = self.blocks(h)
        return self.out_proj(h)


class BasisAwareResidualGate(nn.Module):
    """
    Basis-aware residual gate。

    相对 PriorAwareResidualGate，额外显式使用：
        basis_prior
        coeff_map

    gate 判断依据：
        x
        prior_feat
        basis_prior
        coeff_map
        coord
        |main_residual|
        |branch_residual|
        |branch_residual - main_residual|

    gate:
        gate_min + (1 - gate_min) * sigmoid(logits + radial_bias * r)
    """

    def __init__(
        self,
        inp_channels: int = 3,
        prior_ch: int = 32,
        basis_ch: int = 32,
        num_basis: int = 8,
        coord_channels: int = 4,
        hidden_ch: int = 32,
        gate_min: float = 0.1,
        use_coord: bool = True,
        use_coeff_map: bool = True,
        use_radial_bias: bool = True,
        gate_init: float = 0.3,
    ):
        super().__init__()
        if not (0.0 <= gate_min < 1.0):
            raise ValueError(f"gate_min must be in [0,1), got {gate_min}")

        self.gate_min = float(gate_min)
        self.use_coord = use_coord
        self.use_coeff_map = use_coeff_map
        self.use_radial_bias = use_radial_bias

        in_ch = inp_channels + prior_ch + basis_ch
        if use_coeff_map:
            in_ch += num_basis
        if use_coord:
            in_ch += coord_channels
        in_ch += inp_channels  # |main_residual|
        in_ch += inp_channels  # |branch_residual|
        in_ch += inp_channels  # |branch-main|

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden_ch, hidden_ch, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden_ch, 1, 3, 1, 1),
        )

        gate_init = float(max(min(gate_init, 1.0 - 1e-6), self.gate_min + 1e-6))
        sigmoid_target = (gate_init - self.gate_min) / (1.0 - self.gate_min)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, _logit_from_prob(sigmoid_target))

        if use_radial_bias:
            self.radial_bias = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_parameter("radial_bias", None)

    def forward(
        self,
        x: torch.Tensor,
        prior_feat: torch.Tensor,
        basis_prior: torch.Tensor,
        coeff: Optional[torch.Tensor],
        coord: Optional[torch.Tensor],
        main_residual: torch.Tensor,
        branch_residual: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if prior_feat.shape[-2:] != x.shape[-2:]:
            prior_feat = F.interpolate(
                prior_feat,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        if basis_prior.shape[-2:] != x.shape[-2:]:
            basis_prior = F.interpolate(
                basis_prior,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        inputs = [x, prior_feat, basis_prior]

        if self.use_coeff_map:
            if coeff is None:
                raise ValueError("BasisAwareResidualGate requires coeff when use_coeff_map=True.")
            if coeff.shape[-2:] != x.shape[-2:]:
                coeff = F.interpolate(
                    coeff,
                    size=x.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            inputs.append(coeff)

        if self.use_coord:
            if coord is None:
                raise ValueError("BasisAwareResidualGate requires coord when use_coord=True.")
            if coord.shape[-2:] != x.shape[-2:]:
                coord = F.interpolate(
                    coord,
                    size=x.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            inputs.append(coord)

        residual_gap = (branch_residual - main_residual).abs()
        inputs.extend([
            main_residual.abs(),
            branch_residual.abs(),
            residual_gap,
        ])

        gate_logits = self.net(torch.cat(inputs, dim=1))

        if self.use_radial_bias and coord is not None:
            if coord.shape[1] >= 3:
                r = coord[:, 2:3]
            else:
                x_coord = coord[:, 0:1]
                y = coord[:, 1:2]
                r = torch.sqrt(torch.clamp(x_coord ** 2 + y ** 2, min=1e-12))
            if r.shape[-2:] != gate_logits.shape[-2:]:
                r = F.interpolate(
                    r,
                    size=gate_logits.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            gate_logits = gate_logits + self.radial_bias * r

        gate = self.gate_min + (1.0 - self.gate_min) * torch.sigmoid(gate_logits)
        return gate, gate_logits


# ============================================================
# Main Model
# ============================================================

class Restormer_LowRankBasis_ExplicitBasisBranch(nn.Module):
    """
    Coordinate-conditioned Low-rank Basis + Explicit Basis-driven Residual Branch Restormer.

    默认主分支：
        num_blocks=(2,2)，即 2-level Restormer，每个 level 2 个 TransformerBlock。

    三路径/三分支：
        1. Official-topology shallow Restormer main branch；
        2. Low-rank basis modulation path；
        3. Explicit basis-driven aggressive residual branch；

    融合：
        if use_branch_gate=True:
            out = main_out + branch_scale * gate * branch_residual
        else:
            out = main_out + branch_scale * branch_residual

    branch_scale:
        branch_scale = branch_scale_min
                     + (branch_scale_max - branch_scale_min) * sigmoid(branch_scale_logit)
    """

    def __init__(
        self,
        inp_channels: int = 3,
        out_channels: int = 3,
        dim: int = 48,
        num_blocks: Sequence[int] = (2, 2),
        num_refinement_blocks: int = 1,
        heads: Sequence[int] = (1, 2),
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
        LayerNorm_type: str = "BiasFree",
        stable_softmax: bool = False,
        use_coord: bool = True,
        coord_channels: int = 4,
        prior_ch: int = 32,
        prior_mid_ch: int = 32,
        num_basis: int = 8,
        basis_hidden_ch: int = 32,
        modulate_levels: Sequence[str] = ("shallow", "latent"),

        # Explicit branch basis
        branch_basis_ch: int = 32,
        branch_basis_hidden_ch: int = 32,
        branch_basis_post_refine: bool = True,
        branch_use_coeff_map: bool = True,

        # Aggressive branch
        branch_hidden_ch: int = 48,
        branch_num_blocks: int = 3,
        branch_use_depthwise: bool = False,
        branch_zero_init: bool = True,
        branch_scale_init: float = 0.1,
        branch_scale_min: float = 0.0,
        branch_scale_max: float = 1.0,

        # Gate
        # If use_branch_gate=False:
        #     out = main_out + branch_scale * branch_residual
        # If use_branch_gate=True:
        #     out = main_out + branch_scale * gate * branch_residual
        use_branch_gate: bool = True,
        gate_hidden_ch: int = 32,
        gate_min: float = 0.1,
        gate_init: float = 0.3,
        gate_use_radial_bias: bool = True,
    ):
        super().__init__()
        assert len(num_blocks) == len(heads), "num_blocks and heads must have same length"
        assert len(num_blocks) >= 2, "Restormer requires at least 2 levels."

        self.num_levels = len(num_blocks)
        self.use_coord = use_coord
        self.coord_channels = coord_channels
        self.prior_ch = prior_ch
        self.num_basis = num_basis
        self.modulate_levels = set(modulate_levels)
        self.branch_use_coeff_map = branch_use_coeff_map
        self.use_branch_gate = bool(use_branch_gate)
        self.last_monitor: Dict[str, torch.Tensor] = {}

        # ----------------------------
        # Shared prior path
        # ----------------------------
        prior_in_ch = inp_channels + (coord_channels if use_coord else 0)
        self.prior_encoder = PriorEncoder(
            in_ch=prior_in_ch,
            prior_ch=prior_ch,
            mid_ch=prior_mid_ch,
        )

        # Explicit basis branch prior.
        self.branch_basis_prior = ExplicitBasisBranchPrior(
            prior_ch=prior_ch,
            basis_ch=branch_basis_ch,
            num_basis=num_basis,
            hidden_ch=branch_basis_hidden_ch,
            softmax_coeff=True,
            post_refine=branch_basis_post_refine,
        )

        # ----------------------------
        # Official-topology Restormer main branch
        # ----------------------------
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim, bias=bias)

        encoder_dims = [int(dim * 2 ** level) for level in range(self.num_levels - 1)]
        latent_dim = int(dim * 2 ** (self.num_levels - 1))

        self.mod_shallow = (
            LowRankBasisModulation(
                prior_ch,
                dim,
                num_basis=num_basis,
                hidden_ch=basis_hidden_ch,
            )
            if "shallow" in self.modulate_levels else None
        )

        self.mod_encoders = (
            nn.ModuleList([
                LowRankBasisModulation(
                    prior_ch,
                    c,
                    num_basis=num_basis,
                    hidden_ch=basis_hidden_ch,
                )
                for c in encoder_dims
            ])
            if "enc" in self.modulate_levels else None
        )

        self.mod_latent = (
            LowRankBasisModulation(
                prior_ch,
                latent_dim,
                num_basis=num_basis,
                hidden_ch=basis_hidden_ch,
            )
            if "latent" in self.modulate_levels else None
        )

        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for level in range(self.num_levels - 1):
            level_dim = int(dim * 2 ** level)
            self.encoders.append(
                nn.Sequential(
                    *[
                        TransformerBlock(
                            dim=level_dim,
                            num_heads=heads[level],
                            ffn_expansion_factor=ffn_expansion_factor,
                            bias=bias,
                            LayerNorm_type=LayerNorm_type,
                            stable_softmax=stable_softmax,
                        )
                        for _ in range(num_blocks[level])
                    ]
                )
            )
            self.downsamples.append(Downsample(level_dim))

        self.latent = nn.Sequential(
            *[
                TransformerBlock(
                    dim=latent_dim,
                    num_heads=heads[-1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                    stable_softmax=stable_softmax,
                )
                for _ in range(num_blocks[-1])
            ]
        )

        self.upsamples = nn.ModuleList()
        self.reduce_chans = nn.ModuleList()
        self.decoders = nn.ModuleList()
        decoder_dims = []

        for level in reversed(range(self.num_levels - 1)):
            high_dim = int(dim * 2 ** (level + 1))
            skip_dim = int(dim * 2 ** level)
            self.upsamples.append(Upsample(high_dim))

            concat_dim = skip_dim * 2
            if level > 0:
                self.reduce_chans.append(
                    nn.Conv2d(concat_dim, skip_dim, kernel_size=1, bias=bias)
                )
                decoder_dim = skip_dim
            else:
                # Official Restormer detail: decoder level1 does not reduce channels.
                self.reduce_chans.append(nn.Identity())
                decoder_dim = concat_dim

            decoder_dims.append(decoder_dim)
            self.decoders.append(
                nn.Sequential(
                    *[
                        TransformerBlock(
                            dim=decoder_dim,
                            num_heads=heads[level],
                            ffn_expansion_factor=ffn_expansion_factor,
                            bias=bias,
                            LayerNorm_type=LayerNorm_type,
                            stable_softmax=stable_softmax,
                        )
                        for _ in range(num_blocks[level])
                    ]
                )
            )

        self.mod_decoders = (
            nn.ModuleList([
                LowRankBasisModulation(
                    prior_ch,
                    c,
                    num_basis=num_basis,
                    hidden_ch=basis_hidden_ch,
                )
                for c in decoder_dims
            ])
            if "dec" in self.modulate_levels else None
        )

        refinement_dim = int(dim * 2)
        self.refinement = nn.Sequential(
            *[
                TransformerBlock(
                    dim=refinement_dim,
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                    stable_softmax=stable_softmax,
                )
                for _ in range(num_refinement_blocks)
            ]
        )

        self.output = nn.Conv2d(
            refinement_dim,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )

        # ----------------------------
        # Explicit basis-driven branch + optional basis-aware gate
        # ----------------------------
        self.res_branch = BasisDrivenAggressiveResidualBranch(
            inp_channels=inp_channels,
            out_channels=out_channels,
            prior_ch=prior_ch,
            basis_ch=branch_basis_ch,
            num_basis=num_basis,
            coord_channels=coord_channels,
            hidden_ch=branch_hidden_ch,
            num_blocks=branch_num_blocks,
            use_coord=use_coord,
            use_coeff_map=branch_use_coeff_map,
            use_depthwise=branch_use_depthwise,
            zero_init_output=branch_zero_init,
        )

        self.branch_gate = BasisAwareResidualGate(
            inp_channels=inp_channels,
            prior_ch=prior_ch,
            basis_ch=branch_basis_ch,
            num_basis=num_basis,
            coord_channels=coord_channels,
            hidden_ch=gate_hidden_ch,
            gate_min=gate_min,
            use_coord=use_coord,
            use_coeff_map=branch_use_coeff_map,
            use_radial_bias=gate_use_radial_bias,
            gate_init=gate_init,
        )

        # ----------------------------
        # Branch scale with min/max constraint
        # ----------------------------
        self.branch_scale_min = float(branch_scale_min)
        self.branch_scale_max = float(branch_scale_max)

        if not (0.0 <= self.branch_scale_min < self.branch_scale_max <= 1.0):
            raise ValueError(
                f"Require 0 <= branch_scale_min < branch_scale_max <= 1, "
                f"got min={self.branch_scale_min}, max={self.branch_scale_max}"
            )

        # Map the user-facing initial scale into the internal sigmoid range.
        # Actual branch_scale will be:
        #   branch_scale_min + (branch_scale_max - branch_scale_min) * sigmoid(logit)
        branch_scale_init = float(
            max(
                min(branch_scale_init, self.branch_scale_max - 1e-6),
                self.branch_scale_min + 1e-6,
            )
        )

        internal_p = (
            (branch_scale_init - self.branch_scale_min)
            / (self.branch_scale_max - self.branch_scale_min)
        )

        self.branch_scale_logit = nn.Parameter(
            torch.tensor(_logit_from_prob(internal_p), dtype=torch.float32)
        )

    def _prepare_coord(
        self,
        x: torch.Tensor,
        coord: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if not self.use_coord:
            return None

        if coord is None:
            return make_local_coord(x, coord_channels=self.coord_channels)

        coord = coord.to(device=x.device, dtype=x.dtype)
        coord = normalize_coord_channels(coord, target_channels=self.coord_channels)
        if coord.shape[-2:] != x.shape[-2:]:
            coord = F.interpolate(
                coord,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return coord

    def _collect_basis_diversity_loss(self) -> torch.Tensor:
        mods = []
        if self.mod_shallow is not None:
            mods.append(self.mod_shallow)
        if self.mod_encoders is not None:
            mods += list(self.mod_encoders)
        if self.mod_latent is not None:
            mods.append(self.mod_latent)
        if self.mod_decoders is not None:
            mods += list(self.mod_decoders)

        losses = []
        if len(mods) > 0:
            losses += [m.diversity_loss() for m in mods]
        losses.append(self.branch_basis_prior.diversity_loss())

        return torch.stack(losses).mean()

    def _run_main_branch(
        self,
        x: torch.Tensor,
        prior_feat: torch.Tensor,
        return_monitor: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        monitor: Dict[str, torch.Tensor] = {}
        feats = []

        x = self.patch_embed(x)

        if self.mod_shallow is not None:
            if return_monitor:
                x, st = self.mod_shallow(
                    x,
                    prior_feat,
                    return_stats=True,
                    name="shallow_",
                )
                monitor.update(st)
            else:
                x = self.mod_shallow(x, prior_feat)

        for level in range(self.num_levels - 1):
            if self.mod_encoders is not None:
                if return_monitor:
                    x, st = self.mod_encoders[level](
                        x,
                        prior_feat,
                        return_stats=True,
                        name=f"enc{level+1}_",
                    )
                    monitor.update(st)
                else:
                    x = self.mod_encoders[level](x, prior_feat)

            x = self.encoders[level](x)
            feats.append(x)
            x = self.downsamples[level](x)

        if self.mod_latent is not None:
            if return_monitor:
                x, st = self.mod_latent(
                    x,
                    prior_feat,
                    return_stats=True,
                    name="latent_",
                )
                monitor.update(st)
            else:
                x = self.mod_latent(x, prior_feat)

        x = self.latent(x)
        maybe_debug_tensor("explicit_basis_latent", x)

        for decoder_index, level in enumerate(reversed(range(self.num_levels - 1))):
            x = self.upsamples[decoder_index](x)
            x = torch.cat([x, feats[level]], dim=1)
            x = self.reduce_chans[decoder_index](x)

            if self.mod_decoders is not None:
                if return_monitor:
                    x, st = self.mod_decoders[decoder_index](
                        x,
                        prior_feat,
                        return_stats=True,
                        name=f"dec{decoder_index+1}_",
                    )
                    monitor.update(st)
                else:
                    x = self.mod_decoders[decoder_index](x, prior_feat)

            x = self.decoders[decoder_index](x)

        x = self.refinement(x)
        main_residual = self.output(x)
        maybe_debug_tensor("explicit_basis_main_residual", main_residual)
        return main_residual, monitor

    def forward(
        self,
        x: torch.Tensor,
        coord: Optional[torch.Tensor] = None,
        return_monitor: bool = False,
        return_aux: bool = False,
    ):
        x_input = x
        monitor: Dict[str, torch.Tensor] = {}

        coord_map = self._prepare_coord(x_input, coord)
        if self.use_coord:
            prior_in = torch.cat([x_input, coord_map], dim=1)
        else:
            prior_in = x_input

        prior_feat = self.prior_encoder(prior_in)
        maybe_debug_tensor("explicit_basis_prior_feat", prior_feat)

        if return_monitor:
            monitor["prior_feat_abs_mean"] = _safe_mean(prior_feat.abs())
            monitor["prior_feat_std"] = _safe_std(prior_feat)

        if return_monitor:
            basis_prior, coeff, coeff_logits, basis_stats = self.branch_basis_prior(
                prior_feat,
                return_stats=True,
                name="branch_basis_",
            )
            monitor.update(basis_stats)
        else:
            basis_prior, coeff, coeff_logits = self.branch_basis_prior(
                prior_feat,
                return_stats=False,
            )

        maybe_debug_tensor("explicit_basis_basis_prior", basis_prior)

        main_residual, main_monitor = self._run_main_branch(
            x_input,
            prior_feat,
            return_monitor=return_monitor,
        )
        if return_monitor:
            monitor.update(main_monitor)

        main_out = x_input + main_residual

        branch_residual = self.res_branch(
            x=x_input,
            prior_feat=prior_feat,
            basis_prior=basis_prior,
            coeff=coeff if self.branch_use_coeff_map else None,
            coord=coord_map,
        )

        raw_branch_scale = torch.sigmoid(self.branch_scale_logit).to(
            device=x_input.device,
            dtype=x_input.dtype,
        )

        branch_scale = (
            self.branch_scale_min
            + (self.branch_scale_max - self.branch_scale_min) * raw_branch_scale
        )

        if self.use_branch_gate:
            gate, gate_logits = self.branch_gate(
                x=x_input,
                prior_feat=prior_feat,
                basis_prior=basis_prior,
                coeff=coeff if self.branch_use_coeff_map else None,
                coord=coord_map,
                main_residual=main_residual,
                branch_residual=branch_residual,
            )

            gated_branch_residual = branch_scale * gate * branch_residual

        else:
            # NoGate mode:
            # The explicit basis branch is forced to participate through branch_scale.
            # In this mode, gate is only a dummy tensor for monitor / aux compatibility.
            gate = torch.ones_like(branch_residual[:, :1])
            gate_logits = torch.zeros_like(gate)

            gated_branch_residual = branch_scale * branch_residual

        out = main_out + gated_branch_residual
        branch_out = x_input + branch_residual

        maybe_debug_tensor("explicit_basis_branch_residual", branch_residual)
        maybe_debug_tensor("explicit_basis_gate", gate)
        maybe_debug_tensor("explicit_basis_output", out)

        if return_monitor:
            main_norm = main_residual.detach().float().pow(2).mean().sqrt()
            branch_norm = branch_residual.detach().float().pow(2).mean().sqrt()
            gated_branch_norm = gated_branch_residual.detach().float().pow(2).mean().sqrt()

            monitor["main_residual_abs_mean"] = _safe_mean(main_residual.abs())
            monitor["branch_residual_abs_mean"] = _safe_mean(branch_residual.abs())
            monitor["gated_branch_abs_mean"] = _safe_mean(gated_branch_residual.abs())
            monitor["branch_to_main_ratio"] = branch_norm / (main_norm + 1e-6)
            monitor["gated_branch_to_main_ratio"] = gated_branch_norm / (main_norm + 1e-6)

            monitor["basis_prior_abs_mean"] = _safe_mean(basis_prior.abs())
            monitor["basis_prior_std"] = _safe_std(basis_prior)
            monitor["basis_coeff_logit_mean"] = _safe_mean(coeff_logits)
            monitor["basis_coeff_logit_std"] = _safe_std(coeff_logits)

            monitor["use_branch_gate"] = torch.tensor(
                1.0 if self.use_branch_gate else 0.0,
                device=x_input.device,
            )
            monitor["gate_mean"] = _safe_mean(gate)
            monitor["gate_std"] = _safe_std(gate)
            monitor["gate_min_value"] = _safe_min(gate)
            monitor["gate_max_value"] = _safe_max(gate)
            monitor["gate_logit_mean"] = _safe_mean(gate_logits)
            monitor["gate_logit_std"] = _safe_std(gate_logits)
            monitor["branch_scale"] = branch_scale.detach().float()

            monitor["basis_diversity_loss"] = self._collect_basis_diversity_loss().detach()
            monitor["output_residual_abs_mean"] = _safe_mean((out - x_input).abs())

            if coord_map is not None:
                monitor["gate_edge_center_gap"] = edge_center_gap(gate, coord_map)
                monitor["gate_radial_corr"] = radial_correlation_map(gate, coord_map)
                monitor["basis_prior_edge_center_gap"] = edge_center_gap(basis_prior, coord_map)
                monitor["basis_prior_radial_corr"] = radial_correlation_map(basis_prior, coord_map)

            self.last_monitor = monitor

        aux = {
            "main_out": main_out,
            "branch_out": branch_out,
            "main_residual": main_residual,
            "branch_residual": branch_residual,
            "gated_branch_residual": gated_branch_residual,
            "gate": gate,
            "gate_logits": gate_logits,
            "branch_scale": branch_scale,
            "prior_feat": prior_feat,
            "basis_prior": basis_prior,
            "basis_coeff": coeff,
            "basis_coeff_logits": coeff_logits,
        }

        if return_monitor and return_aux:
            return out, monitor, aux
        if return_monitor:
            return out, monitor
        if return_aux:
            return out, aux

        self.last_monitor = monitor
        return out

    def basis_diversity_loss(self) -> torch.Tensor:
        return self._collect_basis_diversity_loss()

    def get_last_monitor(
        self,
        to_float: bool = True,
    ) -> Dict[str, Union[float, torch.Tensor]]:
        if not to_float:
            return self.last_monitor
        out = {}
        for k, v in self.last_monitor.items():
            if torch.is_tensor(v):
                out[k] = float(v.detach().cpu())
            else:
                out[k] = v
        return out


# Backward-friendly alias.
Restormer_LowRankBasis_BasisBranch = Restormer_LowRankBasis_ExplicitBasisBranch




# ============================================================
# Prior-feature-driven ResBranch Variant
# ============================================================

class AggressiveResidualBranch(nn.Module):
    """
    新增 aggressive residual branch。

    该分支直接预测额外图像残差 branch_residual。
    它不是 MoE / 多专家结构，只是一个独立的增强残差分支。

    输入：
        x          [B, 3, H, W]
        prior_feat [B, prior_ch, H, W]
        coord      [B, coord_ch, H, W] or None

    输出：
        branch_residual [B, out_ch, H, W]
    """

    def __init__(
        self,
        inp_channels: int = 3,
        out_channels: int = 3,
        prior_ch: int = 32,
        coord_channels: int = 4,
        hidden_ch: int = 48,
        num_blocks: int = 3,
        use_coord: bool = True,
        use_depthwise: bool = False,
        zero_init_output: bool = True,
    ):
        super().__init__()

        self.use_coord = use_coord
        self.coord_channels = coord_channels

        in_ch = inp_channels + prior_ch + (coord_channels if use_coord else 0)

        self.in_proj = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 3, 1, 1),
            nn.GELU(),
        )

        blocks = []
        for i in range(num_blocks):
            # 轻微使用 dilation，让该分支比普通局部分支更 aggressive。
            dilation = 1 if i % 2 == 0 else 2
            blocks.append(
                ConvResidualBlock(
                    hidden_ch,
                    dilation=dilation,
                    use_depthwise=use_depthwise,
                )
            )

        self.blocks = nn.Sequential(*blocks)

        self.out_proj = nn.Conv2d(hidden_ch, out_channels, 3, 1, 1)

        if zero_init_output:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        prior_feat: torch.Tensor,
        coord: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if prior_feat.shape[-2:] != x.shape[-2:]:
            prior_feat = F.interpolate(
                prior_feat,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        inputs = [x, prior_feat]

        if self.use_coord:
            if coord is None:
                raise ValueError("AggressiveResidualBranch requires coord when use_coord=True.")

            if coord.shape[-2:] != x.shape[-2:]:
                coord = F.interpolate(
                    coord,
                    size=x.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            inputs.append(coord)

        h = torch.cat(inputs, dim=1)
        h = self.in_proj(h)
        h = self.blocks(h)
        branch_residual = self.out_proj(h)
        return branch_residual


class PriorAwareResidualGate(nn.Module):
    """
    Prior-aware residual gate。

    gate 判断依据：
        x
        prior_feat
        coord
        |main_residual|
        |branch_residual|
        |branch_residual - main_residual|

    gate:
        gate_min + (1 - gate_min) * sigmoid(logits + radial_bias * r)

    注意：
        这不是 MoE，也不是多专家；
        只是用于优化 aggressive residual branch 的空间参与强度判断。
    """

    def __init__(
        self,
        inp_channels: int = 3,
        prior_ch: int = 32,
        coord_channels: int = 4,
        hidden_ch: int = 32,
        gate_min: float = 0.1,
        use_coord: bool = True,
        use_radial_bias: bool = True,
        gate_init: float = 0.3,
    ):
        super().__init__()

        if not (0.0 <= gate_min < 1.0):
            raise ValueError(f"gate_min must be in [0,1), got {gate_min}")

        self.gate_min = float(gate_min)
        self.use_coord = use_coord
        self.use_radial_bias = use_radial_bias

        in_ch = (
            inp_channels
            + prior_ch
            + (coord_channels if use_coord else 0)
            + inp_channels  # |main_residual|
            + inp_channels  # |branch_residual|
            + inp_channels  # |branch-main|
        )

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden_ch, hidden_ch, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden_ch, 1, 3, 1, 1),
        )

        # 初始化 gate 到指定均值附近，而不是随机过强或完全关闭。
        # gate = gate_min + (1-gate_min) * sigmoid(logit)
        gate_init = float(max(min(gate_init, 1.0 - 1e-6), self.gate_min + 1e-6))
        sigmoid_target = (gate_init - self.gate_min) / (1.0 - self.gate_min)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, _logit_from_prob(sigmoid_target))

        if use_radial_bias:
            self.radial_bias = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_parameter("radial_bias", None)

    def forward(
        self,
        x: torch.Tensor,
        prior_feat: torch.Tensor,
        coord: Optional[torch.Tensor],
        main_residual: torch.Tensor,
        branch_residual: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if prior_feat.shape[-2:] != x.shape[-2:]:
            prior_feat = F.interpolate(
                prior_feat,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        inputs = [
            x,
            prior_feat,
        ]

        if self.use_coord:
            if coord is None:
                raise ValueError("PriorAwareResidualGate requires coord when use_coord=True.")

            if coord.shape[-2:] != x.shape[-2:]:
                coord = F.interpolate(
                    coord,
                    size=x.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            inputs.append(coord)

        residual_gap = (branch_residual - main_residual).abs()

        inputs.extend([
            main_residual.abs(),
            branch_residual.abs(),
            residual_gap,
        ])

        gate_logits = self.net(torch.cat(inputs, dim=1))

        if self.use_radial_bias and coord is not None:
            if coord.shape[1] >= 3:
                r = coord[:, 2:3]
            else:
                x_coord = coord[:, 0:1]
                y = coord[:, 1:2]
                r = torch.sqrt(torch.clamp(x_coord ** 2 + y ** 2, min=1e-12))

            if r.shape[-2:] != gate_logits.shape[-2:]:
                r = F.interpolate(
                    r,
                    size=gate_logits.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            gate_logits = gate_logits + self.radial_bias * r

        gate = self.gate_min + (1.0 - self.gate_min) * torch.sigmoid(gate_logits)
        return gate, gate_logits


# ============================================================
# Main Model
# ============================================================

class Restormer_LowRankBasis_ResBranch(nn.Module):
    """
    Coordinate-conditioned Low-rank Basis + Aggressive Residual Branch Restormer.

    三路径结构：
        1. Official-topology shallow Restormer main branch；
        2. Low-rank basis prior modulation path；
        3. Aggressive residual branch；

    融合：
        main_out = x + main_residual
        out = main_out + branch_scale * gate * branch_residual

    默认配置中 num_blocks=(2,2)，即 2-level Restormer，且每层 2 个 TransformerBlock。
    """

    def __init__(
        self,
        inp_channels: int = 3,
        out_channels: int = 3,
        dim: int = 48,
        num_blocks: Sequence[int] = (2, 2),
        num_refinement_blocks: int = 1,
        heads: Sequence[int] = (1, 2),
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
        LayerNorm_type: str = "BiasFree",
        stable_softmax: bool = False,
        use_coord: bool = True,
        coord_channels: int = 4,
        prior_ch: int = 32,
        prior_mid_ch: int = 32,
        num_basis: int = 8,
        basis_hidden_ch: int = 32,
        modulate_levels: Sequence[str] = ("shallow", "latent"),
        branch_hidden_ch: int = 48,
        branch_num_blocks: int = 3,
        branch_use_depthwise: bool = False,
        branch_zero_init: bool = True,
        branch_scale_init: float = 0.1,
        branch_scale_min: float = 0.0,
        branch_scale_max: float = 1.0,
        gate_hidden_ch: int = 32,
        gate_min: float = 0.1,
        gate_init: float = 0.3,
        gate_use_radial_bias: bool = True,
    ):
        super().__init__()

        assert len(num_blocks) == len(heads), "num_blocks and heads must have same length"
        assert len(num_blocks) >= 2, "Restormer requires at least 2 levels."

        self.num_levels = len(num_blocks)
        self.use_coord = use_coord
        self.coord_channels = coord_channels
        self.prior_ch = prior_ch
        self.num_basis = num_basis
        self.modulate_levels = set(modulate_levels)

        self.last_monitor: Dict[str, torch.Tensor] = {}

        # ----------------------------
        # Prior path
        # ----------------------------
        prior_in_ch = inp_channels + (coord_channels if use_coord else 0)

        self.prior_encoder = PriorEncoder(
            in_ch=prior_in_ch,
            prior_ch=prior_ch,
            mid_ch=prior_mid_ch,
        )

        # ----------------------------
        # Official-topology Restormer main branch
        # ----------------------------
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim, bias=bias)

        # Encoder levels: level 1 to level N-1.
        # Deepest level N is latent only.
        encoder_dims = [int(dim * 2 ** level) for level in range(self.num_levels - 1)]
        latent_dim = int(dim * 2 ** (self.num_levels - 1))

        self.mod_shallow = (
            LowRankBasisModulation(
                prior_ch,
                dim,
                num_basis=num_basis,
                hidden_ch=basis_hidden_ch,
            )
            if "shallow" in self.modulate_levels else None
        )

        self.mod_encoders = (
            nn.ModuleList([
                LowRankBasisModulation(
                    prior_ch,
                    c,
                    num_basis=num_basis,
                    hidden_ch=basis_hidden_ch,
                )
                for c in encoder_dims
            ])
            if "enc" in self.modulate_levels else None
        )

        self.mod_latent = (
            LowRankBasisModulation(
                prior_ch,
                latent_dim,
                num_basis=num_basis,
                hidden_ch=basis_hidden_ch,
            )
            if "latent" in self.modulate_levels else None
        )

        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for level in range(self.num_levels - 1):
            level_dim = int(dim * 2 ** level)

            self.encoders.append(
                nn.Sequential(
                    *[
                        TransformerBlock(
                            dim=level_dim,
                            num_heads=heads[level],
                            ffn_expansion_factor=ffn_expansion_factor,
                            bias=bias,
                            LayerNorm_type=LayerNorm_type,
                            stable_softmax=stable_softmax,
                        )
                        for _ in range(num_blocks[level])
                    ]
                )
            )

            self.downsamples.append(Downsample(level_dim))

        self.latent = nn.Sequential(
            *[
                TransformerBlock(
                    dim=latent_dim,
                    num_heads=heads[-1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                    stable_softmax=stable_softmax,
                )
                for _ in range(num_blocks[-1])
            ]
        )

        # Decoder levels: from level N-1 back to level 1.
        self.upsamples = nn.ModuleList()
        self.reduce_chans = nn.ModuleList()
        self.decoders = nn.ModuleList()

        decoder_dims = []

        for level in reversed(range(self.num_levels - 1)):
            high_dim = int(dim * 2 ** (level + 1))
            skip_dim = int(dim * 2 ** level)

            self.upsamples.append(Upsample(high_dim))

            concat_dim = skip_dim * 2

            if level > 0:
                self.reduce_chans.append(
                    nn.Conv2d(concat_dim, skip_dim, kernel_size=1, bias=bias)
                )
                decoder_dim = skip_dim
            else:
                # Official Restormer detail:
                # decoder level1 does not reduce channels after concat.
                self.reduce_chans.append(nn.Identity())
                decoder_dim = concat_dim

            decoder_dims.append(decoder_dim)

            self.decoders.append(
                nn.Sequential(
                    *[
                        TransformerBlock(
                            dim=decoder_dim,
                            num_heads=heads[level],
                            ffn_expansion_factor=ffn_expansion_factor,
                            bias=bias,
                            LayerNorm_type=LayerNorm_type,
                            stable_softmax=stable_softmax,
                        )
                        for _ in range(num_blocks[level])
                    ]
                )
            )

        self.mod_decoders = (
            nn.ModuleList([
                LowRankBasisModulation(
                    prior_ch,
                    c,
                    num_basis=num_basis,
                    hidden_ch=basis_hidden_ch,
                )
                for c in decoder_dims
            ])
            if "dec" in self.modulate_levels else None
        )

        refinement_dim = int(dim * 2)

        self.refinement = nn.Sequential(
            *[
                TransformerBlock(
                    dim=refinement_dim,
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                    stable_softmax=stable_softmax,
                )
                for _ in range(num_refinement_blocks)
            ]
        )

        # 输出 main_residual，而不是直接 + input。
        self.output = nn.Conv2d(
            refinement_dim,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )

        # ----------------------------
        # Aggressive residual branch + prior-aware gate
        # ----------------------------
        self.res_branch = AggressiveResidualBranch(
            inp_channels=inp_channels,
            out_channels=out_channels,
            prior_ch=prior_ch,
            coord_channels=coord_channels,
            hidden_ch=branch_hidden_ch,
            num_blocks=branch_num_blocks,
            use_coord=use_coord,
            use_depthwise=branch_use_depthwise,
            zero_init_output=branch_zero_init,
        )

        self.branch_gate = PriorAwareResidualGate(
            inp_channels=inp_channels,
            prior_ch=prior_ch,
            coord_channels=coord_channels,
            hidden_ch=gate_hidden_ch,
            gate_min=gate_min,
            use_coord=use_coord,
            use_radial_bias=gate_use_radial_bias,
            gate_init=gate_init,
        )

        self.branch_scale_min = float(branch_scale_min)
        self.branch_scale_max = float(branch_scale_max)

        if not (0.0 <= self.branch_scale_min < self.branch_scale_max <= 1.0):
            raise ValueError(
                f"Require 0 <= branch_scale_min < branch_scale_max <= 1, "
                f"got min={self.branch_scale_min}, max={self.branch_scale_max}"
            )

        # Map the user-facing initial scale into the internal sigmoid range.
        # Actual branch_scale will be:
        #   branch_scale_min + (branch_scale_max - branch_scale_min) * sigmoid(logit)
        branch_scale_init = float(
            max(
                min(branch_scale_init, self.branch_scale_max - 1e-6),
                self.branch_scale_min + 1e-6,
            )
        )

        internal_p = (
            (branch_scale_init - self.branch_scale_min)
            / (self.branch_scale_max - self.branch_scale_min)
        )

        self.branch_scale_logit = nn.Parameter(
            torch.tensor(_logit_from_prob(internal_p), dtype=torch.float32)
        )

    def _prepare_coord(
        self,
        x: torch.Tensor,
        coord: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:

        if not self.use_coord:
            return None

        if coord is None:
            return make_local_coord(x, coord_channels=self.coord_channels)

        coord = coord.to(device=x.device, dtype=x.dtype)
        coord = normalize_coord_channels(coord, target_channels=self.coord_channels)

        if coord.shape[-2:] != x.shape[-2:]:
            coord = F.interpolate(
                coord,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        return coord

    def _collect_basis_diversity_loss(self) -> torch.Tensor:
        mods = []

        if self.mod_shallow is not None:
            mods.append(self.mod_shallow)

        if self.mod_encoders is not None:
            mods += list(self.mod_encoders)

        if self.mod_latent is not None:
            mods.append(self.mod_latent)

        if self.mod_decoders is not None:
            mods += list(self.mod_decoders)

        if len(mods) == 0:
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device)

        losses = [m.diversity_loss() for m in mods]
        return torch.stack(losses).mean()

    def _run_main_branch(
        self,
        x: torch.Tensor,
        prior_feat: torch.Tensor,
        return_monitor: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        官方拓扑 Restormer 主分支。
        返回 main_residual，而不是 main_out。
        """
        monitor: Dict[str, torch.Tensor] = {}
        feats = []

        x = self.patch_embed(x)

        if self.mod_shallow is not None:
            if return_monitor:
                x, st = self.mod_shallow(
                    x,
                    prior_feat,
                    return_stats=True,
                    name="shallow_",
                )
                monitor.update(st)
            else:
                x = self.mod_shallow(x, prior_feat)

        for level in range(self.num_levels - 1):
            if self.mod_encoders is not None:
                if return_monitor:
                    x, st = self.mod_encoders[level](
                        x,
                        prior_feat,
                        return_stats=True,
                        name=f"enc{level+1}_",
                    )
                    monitor.update(st)
                else:
                    x = self.mod_encoders[level](x, prior_feat)

            x = self.encoders[level](x)
            feats.append(x)
            x = self.downsamples[level](x)

        if self.mod_latent is not None:
            if return_monitor:
                x, st = self.mod_latent(
                    x,
                    prior_feat,
                    return_stats=True,
                    name="latent_",
                )
                monitor.update(st)
            else:
                x = self.mod_latent(x, prior_feat)

        x = self.latent(x)
        maybe_debug_tensor("resbranch_latent", x)

        for decoder_index, level in enumerate(reversed(range(self.num_levels - 1))):
            x = self.upsamples[decoder_index](x)
            x = torch.cat([x, feats[level]], dim=1)
            x = self.reduce_chans[decoder_index](x)

            if self.mod_decoders is not None:
                if return_monitor:
                    x, st = self.mod_decoders[decoder_index](
                        x,
                        prior_feat,
                        return_stats=True,
                        name=f"dec{decoder_index+1}_",
                    )
                    monitor.update(st)
                else:
                    x = self.mod_decoders[decoder_index](x, prior_feat)

            x = self.decoders[decoder_index](x)

        x = self.refinement(x)
        main_residual = self.output(x)
        maybe_debug_tensor("resbranch_main_residual", main_residual)

        return main_residual, monitor

    def forward(
        self,
        x: torch.Tensor,
        coord: Optional[torch.Tensor] = None,
        return_monitor: bool = False,
        return_aux: bool = False,
    ):
        x_input = x
        monitor: Dict[str, torch.Tensor] = {}

        coord_map = self._prepare_coord(x_input, coord)

        if self.use_coord:
            prior_in = torch.cat([x_input, coord_map], dim=1)
        else:
            prior_in = x_input

        prior_feat = self.prior_encoder(prior_in)
        maybe_debug_tensor("resbranch_prior_feat", prior_feat)

        if return_monitor:
            monitor["prior_feat_abs_mean"] = _safe_mean(prior_feat.abs())
            monitor["prior_feat_std"] = _safe_std(prior_feat)

        main_residual, main_monitor = self._run_main_branch(
            x_input,
            prior_feat,
            return_monitor=return_monitor,
        )

        if return_monitor:
            monitor.update(main_monitor)

        main_out = x_input + main_residual

        branch_residual = self.res_branch(
            x_input,
            prior_feat,
            coord_map,
        )

        gate, gate_logits = self.branch_gate(
            x=x_input,
            prior_feat=prior_feat,
            coord=coord_map,
            main_residual=main_residual,
            branch_residual=branch_residual,
        )

        raw_branch_scale = torch.sigmoid(self.branch_scale_logit).to(
            device=x_input.device,
            dtype=x_input.dtype,
        )

        branch_scale = (
            self.branch_scale_min
            + (self.branch_scale_max - self.branch_scale_min) * raw_branch_scale
        )

        gated_branch_residual = branch_scale * gate * branch_residual
        out = main_out + gated_branch_residual
        branch_out = x_input + branch_residual

        maybe_debug_tensor("resbranch_branch_residual", branch_residual)
        maybe_debug_tensor("resbranch_gate", gate)
        maybe_debug_tensor("resbranch_output", out)

        if return_monitor:
            main_norm = main_residual.detach().float().pow(2).mean().sqrt()
            branch_norm = branch_residual.detach().float().pow(2).mean().sqrt()
            gated_branch_norm = gated_branch_residual.detach().float().pow(2).mean().sqrt()

            monitor["main_residual_abs_mean"] = _safe_mean(main_residual.abs())
            monitor["branch_residual_abs_mean"] = _safe_mean(branch_residual.abs())
            monitor["gated_branch_abs_mean"] = _safe_mean(gated_branch_residual.abs())
            monitor["branch_to_main_ratio"] = branch_norm / (main_norm + 1e-6)
            monitor["gated_branch_to_main_ratio"] = gated_branch_norm / (main_norm + 1e-6)

            monitor["gate_mean"] = _safe_mean(gate)
            monitor["gate_std"] = _safe_std(gate)
            monitor["gate_min_value"] = gate.detach().float().min()
            monitor["gate_max_value"] = gate.detach().float().max()
            monitor["gate_logit_mean"] = _safe_mean(gate_logits)
            monitor["gate_logit_std"] = _safe_std(gate_logits)
            monitor["branch_scale"] = branch_scale.detach().float()

            monitor["basis_diversity_loss"] = self._collect_basis_diversity_loss().detach()
            monitor["output_residual_abs_mean"] = _safe_mean((out - x_input).abs())

            if coord_map is not None:
                monitor["gate_edge_center_gap"] = edge_center_gap(gate, coord_map)
                monitor["gate_radial_corr"] = radial_correlation_map(gate, coord_map)

            self.last_monitor = monitor

        aux = {
            "main_out": main_out,
            "branch_out": branch_out,
            "main_residual": main_residual,
            "branch_residual": branch_residual,
            "gated_branch_residual": gated_branch_residual,
            "gate": gate,
            "gate_logits": gate_logits,
            "branch_scale": branch_scale,
            "prior_feat": prior_feat,
        }

        if return_monitor and return_aux:
            return out, monitor, aux

        if return_monitor:
            return out, monitor

        if return_aux:
            return out, aux

        self.last_monitor = monitor
        return out

    def basis_diversity_loss(self) -> torch.Tensor:
        """
        可选 basis diversity 正则。
        """
        return self._collect_basis_diversity_loss()

    def gate_smoothness_loss(self) -> torch.Tensor:
        """
        可选 gate 平滑正则。

        用法示例：
            pred = model(inp)
            loss = main_loss + 1e-5 * model.gate_smoothness_loss()

        注意：
            该函数依赖 forward 时保存的 monitor 不足以回传梯度，
            因此如果需要有梯度的 gate smooth loss，建议使用 return_aux=True
            后对 aux["gate"] 直接计算。
        """
        device = next(self.parameters()).device
        return torch.tensor(0.0, device=device)

    def get_last_monitor(
        self,
        to_float: bool = True,
    ) -> Dict[str, Union[float, torch.Tensor]]:

        if not to_float:
            return self.last_monitor

        out = {}

        for k, v in self.last_monitor.items():
            if torch.is_tensor(v):
                out[k] = float(v.detach().cpu())
            else:
                out[k] = v

        return out




# Backward-friendly aliases.
Restormer_LowRankBasis_BasisBranch = Restormer_LowRankBasis_ExplicitBasisBranch
Restormer_LowRankBasis_PriorResBranch = Restormer_LowRankBasis_ResBranch


# ============================================================
# Quick Smoke Test
# ============================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(1, 3, 64, 64).to(device)

    common_args = dict(
        inp_channels=3,
        out_channels=3,
        dim=24,
        num_blocks=(2, 2),
        num_refinement_blocks=1,
        heads=(1, 2),
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="BiasFree",
        use_coord=True,
        coord_channels=4,
        prior_ch=16,
        prior_mid_ch=16,
        num_basis=4,
        basis_hidden_ch=16,
        modulate_levels=("shallow", "latent"),
        branch_hidden_ch=24,
        branch_num_blocks=2,
        gate_hidden_ch=16,
        gate_min=0.1,
        gate_init=0.3,
    )

    print("Testing Restormer_LowRankBasis_ResBranch...")
    model1 = Restormer_LowRankBasis_ResBranch(**common_args).to(device)
    with torch.no_grad():
        y1, mon1, aux1 = model1(x, return_monitor=True, return_aux=True)
    print("Output:", y1.shape)
    print("gate_mean:", float(mon1["gate_mean"].detach().cpu()))
    print("branch_scale:", float(mon1["branch_scale"].detach().cpu()))

    print("Testing Restormer_LowRankBasis_ExplicitBasisBranch...")
    explicit_args = dict(common_args)
    explicit_args.update(
        branch_basis_ch=16,
        branch_basis_hidden_ch=16,
        branch_use_coeff_map=True,
    )
    model2 = Restormer_LowRankBasis_ExplicitBasisBranch(**explicit_args).to(device)
    with torch.no_grad():
        y2, mon2, aux2 = model2(x, return_monitor=True, return_aux=True)
    print("Output:", y2.shape)
    print("gate_mean:", float(mon2["gate_mean"].detach().cpu()))
    print("branch_scale:", float(mon2["branch_scale"].detach().cpu()))
    print("basis_prior:", aux2["basis_prior"].shape)
    print("basis_coeff:", aux2["basis_coeff"].shape)
