# -*- coding: utf-8 -*-
"""
Restormer Prior Variants
========================

保存路径建议：
    models/Restormer_PriorVariants.py

包含两个结构：

1. Restormer_DegField
   方案3：显式空间退化场估计
   image + coord -> degradation field -> feature modulation -> Restormer

2. Restormer_LowRankBasis
   方案2：低维退化基
   image + coord -> coefficient map -> learnable basis bank -> implicit prior -> feature modulation -> Restormer

两个模型均兼容普通训练：
    pred = model(inp)

需要监控指标时：
    pred, mon = model(inp, return_monitor=True)

如果你有全图 global coordinate：
    pred = model(inp, coord=global_coord)

coord 支持：
    [B, 2, H, W] = x, y
    [B, 4, H, W] = x, y, r, r^2

如果不传 coord，则自动生成 local normalized coordinate。
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
    """
    生成局部归一化坐标。

    coord_channels = 2:
        [x, y]

    coord_channels = 4:
        [x, y, r, r^2]

    输出 shape:
        [B, coord_channels, H, W]
    """
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
    """
    规范外部传入的 coord。

    支持：
        [B, 2, H, W] -> 自动补充 r, r^2
        [B, 4, H, W] -> 直接取前4通道
    """
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


def radial_correlation_map(
    score: torch.Tensor,
    coord: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    计算 score map 与半径 r 的 batch 平均 Pearson correlation。

    对 metalens 来说，如果退化明显随离轴距离增强，
    deg_score_radial_corr 往往应该逐渐变成正相关。
    """
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
    """
    计算边缘区域均值 - 中心区域均值。

    如果退化场确实学习到 metalens 的离轴退化，
    edge_center_gap 通常不应长期接近 0。
    """
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
    """
    返回 basis 两两 cosine similarity 的非对角绝对均值。

    越低说明 basis 越不容易 collapse 成同一种模式。
    """
    b = F.normalize(basis.detach().float(), dim=1)
    sim = b @ b.t()
    k = sim.shape[0]

    mask = ~torch.eye(k, device=sim.device, dtype=torch.bool)
    return sim[mask].abs().mean()


# ============================================================
# Common Modulation Module
# ============================================================

class FeatureAffineModulation(nn.Module):
    """
    通用 affine feature modulation。

    feat:
        [B, C, H, W]

    prior:
        [B, prior_ch, H0, W0]

    输出：
        feat * (1 + scale) + shift
    """

    def __init__(
        self,
        prior_ch: int,
        feat_ch: int,
        hidden_mult: float = 1.0,
        zero_init: bool = True,
    ):
        super().__init__()

        hidden = max(feat_ch, int(feat_ch * hidden_mult))

        self.net = nn.Sequential(
            nn.Conv2d(prior_ch, hidden, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(hidden, feat_ch * 2, 1, 1, 0),
        )

        if zero_init:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        feat: torch.Tensor,
        prior: torch.Tensor,
        return_stats: bool = False,
        name: str = "",
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:

        if prior.shape[-2:] != feat.shape[-2:]:
            prior = F.interpolate(
                prior,
                size=feat.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        scale, shift = self.net(prior).chunk(2, dim=1)
        out = feat * (1.0 + scale) + shift

        if not return_stats:
            return out

        delta = out - feat

        stats = {
            f"{name}scale_abs_mean": _safe_mean(scale.abs()),
            f"{name}shift_abs_mean": _safe_mean(shift.abs()),
            f"{name}delta_ratio": (
                delta.detach().float().pow(2).mean().sqrt()
                / (feat.detach().float().pow(2).mean().sqrt() + 1e-6)
            ),
        }

        return out, stats


# ============================================================
# Scheme 3: DegField Modules
# ============================================================

class DegFieldEstimator(nn.Module):
    """
    方案3核心模块：
    显式估计空间变化退化场。

    输入：
        image 或 image + coord

    输出：
        deg_feat:
            [B, deg_ch, H, W]

        deg_score:
            [B, 1, H, W]
            用于监控和可视化，可理解为退化强度响应。
    """

    def __init__(
        self,
        in_ch: int = 7,
        mid_ch: int = 32,
        deg_ch: int = 16,
        use_residual: bool = True,
    ):
        super().__init__()

        self.use_residual = use_residual

        self.conv1 = nn.Conv2d(in_ch, mid_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(mid_ch, deg_ch, 3, 1, 1)
        self.score = nn.Conv2d(deg_ch, 1, 1, 1, 0)

        # score zero-init，避免一开始 deg_score 出现过强随机响应
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h1 = F.gelu(self.conv1(x))
        h2 = F.gelu(self.conv2(h1))

        if self.use_residual and h2.shape == h1.shape:
            h2 = h2 + h1

        deg_feat = self.conv3(h2)
        deg_score = torch.sigmoid(self.score(deg_feat))

        return deg_feat, deg_score


class Restormer_DegField(nn.Module):
    """
    方案3：
    Coordinate-conditioned Degradation Field Modulated Restormer.

    设计逻辑：
        image + coord -> degradation field -> feature modulation -> Restormer

    forward:
        pred = model(x)
        pred, mon = model(x, return_monitor=True)
        pred = model(x, coord=global_coord)
    """

    def __init__(
        self,
        inp_channels: int = 3,
        out_channels: int = 3,
        dim: int = 48,
        num_blocks: Sequence[int] = (4, 6),
        num_refinement_blocks: int = 2,
        heads: Sequence[int] = (1, 2),
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
        LayerNorm_type: str = "BiasFree",
        use_coord: bool = True,
        coord_channels: int = 4,
        deg_ch: int = 16,
        deg_mid_ch: int = 32,
        modulate_levels: Sequence[str] = ("shallow", "enc", "latent", "dec"),
    ):
        super().__init__()

        assert len(num_blocks) == len(heads), "num_blocks and heads must have same length"

        self.num_levels = len(num_blocks)
        self.use_coord = use_coord
        self.coord_channels = coord_channels
        self.deg_ch = deg_ch
        self.modulate_levels = set(modulate_levels)

        self.last_monitor: Dict[str, torch.Tensor] = {}
        self.last_deg_score: Optional[torch.Tensor] = None

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim, bias=bias)

        deg_in_ch = inp_channels + (coord_channels if use_coord else 0)

        self.deg_estimator = DegFieldEstimator(
            in_ch=deg_in_ch,
            mid_ch=deg_mid_ch,
            deg_ch=deg_ch,
        )

        level_dims = [dim * (2 ** i) for i in range(self.num_levels)]

        self.mod_shallow = (
            FeatureAffineModulation(deg_ch, dim)
            if "shallow" in self.modulate_levels else None
        )

        self.mod_encoders = (
            nn.ModuleList([
                FeatureAffineModulation(deg_ch, c)
                for c in level_dims
            ])
            if "enc" in self.modulate_levels else None
        )

        self.mod_latent = (
            FeatureAffineModulation(deg_ch, level_dims[-1])
            if "latent" in self.modulate_levels else None
        )

        self.mod_decoders = (
            nn.ModuleList([
                FeatureAffineModulation(deg_ch, c)
                for c in reversed(level_dims[:-1])
            ])
            if "dec" in self.modulate_levels else None
        )

        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for i in range(self.num_levels):
            level_dim = level_dims[i]
            blocks = [
                TransformerBlock(
                    level_dim,
                    heads[i],
                    ffn_expansion_factor,
                    bias,
                    LayerNorm_type,
                )
                for _ in range(num_blocks[i])
            ]
            self.encoders.append(nn.Sequential(*blocks))

            if i < self.num_levels - 1:
                self.downsamples.append(Downsample(level_dim))

        self.latent = nn.Sequential(*[
            TransformerBlock(
                level_dims[-1],
                heads[-1],
                ffn_expansion_factor,
                bias,
                LayerNorm_type,
            )
            for _ in range(num_blocks[-1])
        ])

        self.upsamples = nn.ModuleList()
        self.reduce_chans = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for i in reversed(range(self.num_levels - 1)):
            in_dim = dim * (2 ** (i + 1))
            out_dim = dim * (2 ** i)

            self.upsamples.append(Upsample(in_dim))
            self.reduce_chans.append(nn.Conv2d(in_dim, out_dim, 1, bias=bias))

            blocks = [
                TransformerBlock(
                    out_dim,
                    heads[i],
                    ffn_expansion_factor,
                    bias,
                    LayerNorm_type,
                )
                for _ in range(num_blocks[i])
            ]

            self.decoders.append(nn.Sequential(*blocks))

        self.refinement = nn.Sequential(*[
            TransformerBlock(
                dim,
                heads[0],
                ffn_expansion_factor,
                bias,
                LayerNorm_type,
            )
            for _ in range(num_refinement_blocks)
        ])

        self.output = nn.Conv2d(dim, out_channels, 3, 1, 1, bias=bias)

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

    def forward(
        self,
        x: torch.Tensor,
        coord: Optional[torch.Tensor] = None,
        return_monitor: bool = False,
    ):
        x_input = x
        monitor: Dict[str, torch.Tensor] = {}

        coord_map = self._prepare_coord(x, coord)

        if self.use_coord:
            deg_in = torch.cat([x, coord_map], dim=1)
        else:
            deg_in = x

        deg_feat, deg_score = self.deg_estimator(deg_in)
        self.last_deg_score = deg_score

        maybe_debug_tensor("deg_feat", deg_feat)
        maybe_debug_tensor("deg_score", deg_score)

        if return_monitor:
            monitor["deg_score_mean"] = _safe_mean(deg_score)
            monitor["deg_score_std"] = _safe_std(deg_score)
            monitor["deg_feat_abs_mean"] = _safe_mean(deg_feat.abs())

            if coord_map is not None:
                monitor["deg_score_edge_center_gap"] = edge_center_gap(deg_score, coord_map)
                monitor["deg_score_radial_corr"] = radial_correlation_map(deg_score, coord_map)

        feats = []

        x = self.patch_embed(x)

        if self.mod_shallow is not None:
            if return_monitor:
                x, st = self.mod_shallow(x, deg_feat, return_stats=True, name="shallow_")
                monitor.update(st)
            else:
                x = self.mod_shallow(x, deg_feat)

        for i in range(self.num_levels):
            if self.mod_encoders is not None:
                if return_monitor:
                    x, st = self.mod_encoders[i](x, deg_feat, return_stats=True, name=f"enc{i+1}_")
                    monitor.update(st)
                else:
                    x = self.mod_encoders[i](x, deg_feat)

            x = self.encoders[i](x)
            feats.append(x)

            if i < self.num_levels - 1:
                x = self.downsamples[i](x)

        if self.mod_latent is not None:
            if return_monitor:
                x, st = self.mod_latent(x, deg_feat, return_stats=True, name="latent_")
                monitor.update(st)
            else:
                x = self.mod_latent(x, deg_feat)

        x = self.latent(x)
        maybe_debug_tensor("latent", x)

        dec_idx = 0

        for i in range(self.num_levels - 1)[::-1]:
            x = self.upsamples[self.num_levels - 2 - i](x)
            x = torch.cat([x, feats[i]], dim=1)
            x = self.reduce_chans[self.num_levels - 2 - i](x)

            if self.mod_decoders is not None:
                if return_monitor:
                    x, st = self.mod_decoders[dec_idx](x, deg_feat, return_stats=True, name=f"dec{dec_idx+1}_")
                    monitor.update(st)
                else:
                    x = self.mod_decoders[dec_idx](x, deg_feat)

            x = self.decoders[self.num_levels - 2 - i](x)
            dec_idx += 1

        x = self.refinement(x)
        x_out = self.output(x)

        maybe_debug_tensor("output", x_out)

        out = x_out + x_input

        if return_monitor:
            monitor["output_residual_abs_mean"] = _safe_mean((out - x_input).abs())
            self.last_monitor = monitor
            return out, monitor

        self.last_monitor = monitor
        return out

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

    def deg_smoothness_loss(self) -> torch.Tensor:
        """
        可选辅助 loss：
            loss = main_loss + lambda_smooth * model.deg_smoothness_loss()

        建议：
            lambda_smooth 从 1e-4 或 1e-5 开始试。
        """
        if self.last_deg_score is None:
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device)

        score = self.last_deg_score

        dx = score[:, :, :, 1:] - score[:, :, :, :-1]
        dy = score[:, :, 1:, :] - score[:, :, :-1, :]

        return dx.abs().mean() + dy.abs().mean()


# ============================================================
# Scheme 2: Low-rank Basis Modules
# ============================================================

class PriorEncoder(nn.Module):
    """
    方案2前置 prior encoder。

    输入：
        image 或 image + coord

    输出：
        prior_feat [B, prior_ch, H, W]

    这个 prior_feat 用于预测 low-rank basis coefficient map。
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
    方案2核心模块。

    prior_input:
        [B, prior_ch, H, W]

    coefficient map:
        [B, K, H, W]

    learnable basis:
        [K, C]

    implicit prior:
        [B, C, H, W]

    modulation:
        feat * (1 + scale) + shift
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

        eps = 1e-8

        coeff_detached = coeff.detach().float()

        entropy = -(
            coeff_detached * torch.log(coeff_detached + eps)
        ).sum(dim=1).mean()

        entropy_norm = entropy / math.log(float(self.num_basis))
        top_prob = coeff_detached.amax(dim=1).mean()

        usage = coeff_detached.mean(dim=(0, 2, 3))

        delta = out - feat

        stats = {
            f"{name}coeff_entropy_norm": entropy_norm,
            f"{name}coeff_top_prob_mean": top_prob,
            f"{name}coeff_spatial_std": coeff_detached.std(unbiased=False),
            f"{name}basis_usage_std": usage.std(unbiased=False),
            f"{name}basis_usage_min": usage.min(),
            f"{name}basis_usage_max": usage.max(),
            f"{name}basis_cos_abs_offdiag": basis_diversity_metric(self.basis),
            f"{name}prior_abs_mean": _safe_mean(prior.abs()),
            f"{name}scale_abs_mean": _safe_mean(scale.abs()),
            f"{name}shift_abs_mean": _safe_mean(shift.abs()),
            f"{name}delta_ratio": (
                delta.detach().float().pow(2).mean().sqrt()
                / (feat.detach().float().pow(2).mean().sqrt() + 1e-6)
            ),
        }

        return out, stats

    def diversity_loss(self) -> torch.Tensor:
        """
        可选 basis diversity loss。

        用法：
            loss = main_loss + 1e-5 * model.basis_diversity_loss()

        建议：
            先不加，确认模型能训练后再试 1e-5 或 1e-4。
        """
        b = F.normalize(self.basis.float(), dim=1)
        sim = b @ b.t()
        eye = torch.eye(self.num_basis, device=sim.device, dtype=sim.dtype)
        return ((sim - eye) ** 2).mean()


class Restormer_LowRankBasis(nn.Module):
    """
    方案2：
    Coordinate-conditioned Low-rank Degradation Basis Modulated Restormer.

    设计逻辑：
        image + coord -> prior feature
        prior feature -> coefficient map
        coefficient map + learnable basis bank -> implicit prior
        implicit prior -> feature modulation
        modulation -> Restormer

    forward:
        pred = model(x)
        pred, mon = model(x, return_monitor=True)
        pred = model(x, coord=global_coord)
    """

    def __init__(
        self,
        inp_channels: int = 3,
        out_channels: int = 3,
        dim: int = 48,
        num_blocks: Sequence[int] = (4, 6),
        num_refinement_blocks: int = 2,
        heads: Sequence[int] = (1, 2),
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
        LayerNorm_type: str = "BiasFree",
        use_coord: bool = True,
        coord_channels: int = 4,
        prior_ch: int = 32,
        prior_mid_ch: int = 32,
        num_basis: int = 8,
        basis_hidden_ch: int = 32,
        modulate_levels: Sequence[str] = ("shallow", "enc", "latent", "dec"),
    ):
        super().__init__()

        assert len(num_blocks) == len(heads), "num_blocks and heads must have same length"

        self.num_levels = len(num_blocks)
        self.use_coord = use_coord
        self.coord_channels = coord_channels
        self.prior_ch = prior_ch
        self.num_basis = num_basis
        self.modulate_levels = set(modulate_levels)

        self.last_monitor: Dict[str, torch.Tensor] = {}

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim, bias=bias)

        prior_in_ch = inp_channels + (coord_channels if use_coord else 0)

        self.prior_encoder = PriorEncoder(
            in_ch=prior_in_ch,
            prior_ch=prior_ch,
            mid_ch=prior_mid_ch,
        )

        level_dims = [dim * (2 ** i) for i in range(self.num_levels)]

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
                for c in level_dims
            ])
            if "enc" in self.modulate_levels else None
        )

        self.mod_latent = (
            LowRankBasisModulation(
                prior_ch,
                level_dims[-1],
                num_basis=num_basis,
                hidden_ch=basis_hidden_ch,
            )
            if "latent" in self.modulate_levels else None
        )

        self.mod_decoders = (
            nn.ModuleList([
                LowRankBasisModulation(
                    prior_ch,
                    c,
                    num_basis=num_basis,
                    hidden_ch=basis_hidden_ch,
                )
                for c in reversed(level_dims[:-1])
            ])
            if "dec" in self.modulate_levels else None
        )

        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for i in range(self.num_levels):
            level_dim = level_dims[i]

            blocks = [
                TransformerBlock(
                    level_dim,
                    heads[i],
                    ffn_expansion_factor,
                    bias,
                    LayerNorm_type,
                )
                for _ in range(num_blocks[i])
            ]

            self.encoders.append(nn.Sequential(*blocks))

            if i < self.num_levels - 1:
                self.downsamples.append(Downsample(level_dim))

        self.latent = nn.Sequential(*[
            TransformerBlock(
                level_dims[-1],
                heads[-1],
                ffn_expansion_factor,
                bias,
                LayerNorm_type,
            )
            for _ in range(num_blocks[-1])
        ])

        self.upsamples = nn.ModuleList()
        self.reduce_chans = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for i in reversed(range(self.num_levels - 1)):
            in_dim = dim * (2 ** (i + 1))
            out_dim = dim * (2 ** i)

            self.upsamples.append(Upsample(in_dim))
            self.reduce_chans.append(nn.Conv2d(in_dim, out_dim, 1, bias=bias))

            blocks = [
                TransformerBlock(
                    out_dim,
                    heads[i],
                    ffn_expansion_factor,
                    bias,
                    LayerNorm_type,
                )
                for _ in range(num_blocks[i])
            ]

            self.decoders.append(nn.Sequential(*blocks))

        self.refinement = nn.Sequential(*[
            TransformerBlock(
                dim,
                heads[0],
                ffn_expansion_factor,
                bias,
                LayerNorm_type,
            )
            for _ in range(num_refinement_blocks)
        ])

        self.output = nn.Conv2d(dim, out_channels, 3, 1, 1, bias=bias)

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

    def forward(
        self,
        x: torch.Tensor,
        coord: Optional[torch.Tensor] = None,
        return_monitor: bool = False,
    ):
        x_input = x
        monitor: Dict[str, torch.Tensor] = {}

        coord_map = self._prepare_coord(x, coord)

        if self.use_coord:
            prior_in = torch.cat([x, coord_map], dim=1)
        else:
            prior_in = x

        prior_feat = self.prior_encoder(prior_in)

        maybe_debug_tensor("lowrank_prior_feat", prior_feat)

        if return_monitor:
            monitor["prior_feat_abs_mean"] = _safe_mean(prior_feat.abs())
            monitor["prior_feat_std"] = _safe_std(prior_feat)

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

        for i in range(self.num_levels):
            if self.mod_encoders is not None:
                if return_monitor:
                    x, st = self.mod_encoders[i](
                        x,
                        prior_feat,
                        return_stats=True,
                        name=f"enc{i+1}_",
                    )
                    monitor.update(st)
                else:
                    x = self.mod_encoders[i](x, prior_feat)

            x = self.encoders[i](x)
            feats.append(x)

            if i < self.num_levels - 1:
                x = self.downsamples[i](x)

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

        maybe_debug_tensor("latent", x)

        dec_idx = 0

        for i in range(self.num_levels - 1)[::-1]:
            x = self.upsamples[self.num_levels - 2 - i](x)
            x = torch.cat([x, feats[i]], dim=1)
            x = self.reduce_chans[self.num_levels - 2 - i](x)

            if self.mod_decoders is not None:
                if return_monitor:
                    x, st = self.mod_decoders[dec_idx](
                        x,
                        prior_feat,
                        return_stats=True,
                        name=f"dec{dec_idx+1}_",
                    )
                    monitor.update(st)
                else:
                    x = self.mod_decoders[dec_idx](x, prior_feat)

            x = self.decoders[self.num_levels - 2 - i](x)
            dec_idx += 1

        x = self.refinement(x)
        x_out = self.output(x)

        maybe_debug_tensor("output", x_out)

        out = x_out + x_input

        if return_monitor:
            monitor["basis_diversity_loss"] = self._collect_basis_diversity_loss().detach()
            monitor["output_residual_abs_mean"] = _safe_mean((out - x_input).abs())
            self.last_monitor = monitor
            return out, monitor

        self.last_monitor = monitor
        return out

    def basis_diversity_loss(self) -> torch.Tensor:
        """
        可选训练正则。

        用法：
            pred = model(inp)
            loss = main_loss + 1e-5 * model.basis_diversity_loss()
        """
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


# ============================================================
# Quick Smoke Test
# ============================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x = torch.randn(1, 3, 128, 128).to(device)

    common_args = dict(
        inp_channels=3,
        out_channels=3,
        dim=24,
        num_blocks=(1, 1),
        num_refinement_blocks=1,
        heads=(1, 2),
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="BiasFree",
    )

    print("Testing Restormer_DegField...")
    model1 = Restormer_DegField(
        **common_args,
        use_coord=True,
        coord_channels=4,
        deg_ch=8,
        modulate_levels=("shallow", "enc", "latent", "dec"),
    ).to(device)

    y1, mon1 = model1(x, return_monitor=True)
    print("Output:", y1.shape)
    print("Monitor keys:", list(mon1.keys())[:10])

    print("Testing Restormer_LowRankBasis...")
    model2 = Restormer_LowRankBasis(
        **common_args,
        use_coord=True,
        coord_channels=4,
        prior_ch=16,
        num_basis=4,
        modulate_levels=("shallow", "enc", "latent", "dec"),
    ).to(device)

    y2, mon2 = model2(x, return_monitor=True)
    print("Output:", y2.shape)
    print("Monitor keys:", list(mon2.keys())[:10])


