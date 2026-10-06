# -*- coding: utf-8 -*-
"""
Res_Strict.py
=============

Self-contained strict Restormer-aligned PSF-like DegField variants.

This file provides two new structures:

1) Restormer_StrictDegField
   - Non-global / local PSF-like DegField.
   - If coord is omitted, local normalized coordinate is generated internally.
   - Strictly follows the Restormer-style encoder-bottleneck-decoder hierarchy:
       num_blocks=[4,6,6,8] -> enc1=4, enc2=6, enc3=6, latent=8
     The last entry is used only for the bottleneck / latent stage, not for
     both an extra lowest encoder and latent.

2) Restormer_StrictGlobalDegField
   - Strict global-coordinate PSF-like DegField.
   - Requires external full-canvas coord with shape [B,2,H,W] or [B,4,H,W].
   - Uses the same strict Restormer-style hierarchy as Restormer_StrictDegField.

The Strict implementation is self-contained for release: the required coordinate,
DegField estimator, monitor, and affine-modulation utilities are defined in this file.

Expected save path:
    models/Res_Strict.py
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Robust imports
# ============================================================
try:
    from .Resbase import (
        OverlapPatchEmbed,
        TransformerBlock,
        Downsample,
        Upsample,
        maybe_debug_tensor,
    )
except Exception:
    try:
        from Resbase import (
            OverlapPatchEmbed,
            TransformerBlock,
            Downsample,
            Upsample,
            maybe_debug_tensor,
        )
    except Exception:
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
# Inlined Strict DegField dependencies
# ============================================================
# These definitions are copied verbatim from the original Res_Psfbasis.py
# so that the public Strict model is self-contained. Their numerical behavior,
# initialization, tensor shapes, and monitor semantics are intentionally unchanged.

# -----------------------------
# Coordinate utilities
# -----------------------------
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

    输出:
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

def get_radius_from_coord(coord: torch.Tensor, size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    """
    从 coord 得到 radius map。
    """
    if coord.shape[1] >= 3:
        r = coord[:, 2:3]
    else:
        x_coord = coord[:, 0:1]
        y = coord[:, 1:2]
        r = torch.sqrt(torch.clamp(x_coord ** 2 + y ** 2, min=1e-12))

    if size is not None and r.shape[-2:] != size:
        r = F.interpolate(r, size=size, mode="bilinear", align_corners=False)
    return r

# -----------------------------
# Monitor / field utilities
# -----------------------------
def _safe_mean(x: torch.Tensor) -> torch.Tensor:
    return x.detach().float().mean()

def _safe_std(x: torch.Tensor) -> torch.Tensor:
    return x.detach().float().std(unbiased=False)

def tv_loss_map(x: torch.Tensor) -> torch.Tensor:
    """
    Total variation smoothness loss for field-like maps.
    """
    if x is None:
        return torch.tensor(0.0)
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    return dx.abs().mean() + dy.abs().mean()

def radial_correlation_map(
    score: torch.Tensor,
    coord: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    计算 score map 与半径 r 的 batch 平均 Pearson correlation。
    """
    if score.shape[1] > 1:
        score = score.mean(dim=1, keepdim=True)

    r = get_radius_from_coord(coord, size=score.shape[-2:])

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
    """
    if score.shape[1] > 1:
        score = score.mean(dim=1, keepdim=True)

    score = score.detach().float()
    r = get_radius_from_coord(coord, size=score.shape[-2:]).detach().float()

    center_mask = (r <= 0.35).float()
    edge_mask = (r >= 0.75).float()

    center = (score * center_mask).sum() / (center_mask.sum() + 1e-6)
    edge = (score * edge_mask).sum() / (edge_mask.sum() + 1e-6)
    return edge - center

# -----------------------------
# DegField modulation components
# -----------------------------
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

class PSFLikeDegFieldEstimator(nn.Module):
    """
    低分辨率 PSF-like degradation field estimator。

    输入：
        image 或 image + coord

    输出：
        deg_feat:
            [B, deg_ch, H/s, W/s]

        deg_score:
            [B, 1, H/s, W/s]

    设计原因：
        PSF 退化场通常是空间连续变化的，不应逐像素强烈震荡。
        使用低分辨率 field 可以减少内容纹理泄漏，使其更接近 optical degradation field。
    """

    def __init__(
        self,
        in_ch: int = 7,
        mid_ch: int = 32,
        deg_ch: int = 16,
        downsample_factor: int = 4,
        use_residual: bool = True,
    ):
        super().__init__()

        if downsample_factor not in (1, 2, 4, 8):
            raise ValueError("downsample_factor should be one of {1,2,4,8}")

        self.downsample_factor = downsample_factor
        self.use_residual = use_residual

        layers = []
        cur_ch = in_ch
        cur_factor = 1

        while cur_factor < downsample_factor:
            layers += [
                nn.Conv2d(cur_ch, mid_ch, 3, 2, 1),
                nn.GELU(),
            ]
            cur_ch = mid_ch
            cur_factor *= 2

        if len(layers) == 0:
            layers += [
                nn.Conv2d(cur_ch, mid_ch, 3, 1, 1),
                nn.GELU(),
            ]
            cur_ch = mid_ch

        self.down = nn.Sequential(*layers)

        self.conv1 = nn.Conv2d(cur_ch, mid_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(mid_ch, deg_ch, 3, 1, 1)
        self.score = nn.Conv2d(deg_ch, 1, 1, 1, 0)

        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.down(x)
        h1 = F.gelu(self.conv1(h))
        h2 = F.gelu(self.conv2(h1))

        if self.use_residual and h2.shape == h1.shape:
            h2 = h2 + h1

        deg_feat = self.conv3(h2)
        deg_score = torch.sigmoid(self.score(deg_feat))
        return deg_feat, deg_score


# ============================================================
# Small compatibility helpers
# ============================================================
def _make_patch_embed(in_channels: int, dim: int, bias: bool):
    """Create OverlapPatchEmbed under both Resbase-style and Restormer-style signatures."""
    try:
        return OverlapPatchEmbed(in_c=in_channels, embed_dim=dim, bias=bias)
    except TypeError:
        return OverlapPatchEmbed(in_channels, dim, bias=bias)


def _as_float_dict(d: Dict[str, Union[torch.Tensor, float, int, str, bool]]):
    out = {}
    for k, v in d.items():
        if torch.is_tensor(v):
            if v.numel() == 1:
                out[k] = float(v.detach().cpu())
            else:
                out[k] = float(v.detach().float().mean().cpu())
        else:
            out[k] = v
    return out


# ============================================================
# Strict Restormer-aligned DegField backbone
# ============================================================
class _StrictRestormerDegFieldBase(nn.Module):
    """
    Restormer-style PSF-like DegField backbone with strict official hierarchy.

    Important hierarchy:
        If num_blocks=[4,6,6,8] and heads=[1,2,4,8], then:
            encoder levels: 0..2 -> 4,6,6 blocks
            latent level  : 3    -> 8 blocks
            decoder levels: 2..0 -> 6,6,4 blocks

    This is intentionally different from the older Restormer_PSFLikeDegField
    implementation that builds encoders for all num_levels and then builds
    an additional latent using num_blocks[-1].
    """

    coord_mode_name = "base"

    def __init__(
        self,
        inp_channels: int = 3,
        out_channels: int = 3,
        dim: int = 48,
        num_blocks: Sequence[int] = (4, 6, 6, 8),
        num_refinement_blocks: int = 4,
        heads: Sequence[int] = (1, 2, 4, 8),
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
        LayerNorm_type: str = "BiasFree",
        use_coord: bool = True,
        coord_channels: int = 4,
        deg_ch: int = 16,
        deg_mid_ch: int = 32,
        deg_downsample_factor: int = 4,
        modulate_levels: Sequence[str] = ("shallow", "enc", "latent", "dec"),
    ):
        super().__init__()

        if len(num_blocks) != len(heads):
            raise ValueError(
                f"num_blocks and heads must have same length, got "
                f"{len(num_blocks)} and {len(heads)}."
            )
        if len(num_blocks) < 2:
            raise ValueError("Strict Restormer DegField expects at least 2 levels.")
        if int(coord_channels) not in (2, 4):
            raise ValueError(f"coord_channels must be 2 or 4, got {coord_channels}.")

        self.inp_channels = int(inp_channels)
        self.out_channels = int(out_channels)
        self.num_levels = len(num_blocks)
        self.use_coord = bool(use_coord)
        self.coord_channels = int(coord_channels)
        self.deg_ch = int(deg_ch)
        self.modulate_levels = set(modulate_levels)

        self.last_monitor: Dict[str, torch.Tensor] = {}
        self.last_deg_feat: Optional[torch.Tensor] = None
        self.last_deg_score: Optional[torch.Tensor] = None
        self.last_coord_map: Optional[torch.Tensor] = None

        # RGB-only restoration backbone input.
        self.patch_embed = _make_patch_embed(inp_channels, dim, bias=bias)

        # Degradation-field branch input: RGB + coordinate if enabled.
        deg_in_ch = inp_channels + (coord_channels if self.use_coord else 0)
        self.deg_estimator = PSFLikeDegFieldEstimator(
            in_ch=deg_in_ch,
            mid_ch=deg_mid_ch,
            deg_ch=deg_ch,
            downsample_factor=deg_downsample_factor,
        )

        level_dims = [dim * (2 ** i) for i in range(self.num_levels)]

        # Modulation modules. For official hierarchy:
        #   encoder modulation only covers levels before latent, i.e. level_dims[:-1].
        self.mod_shallow = (
            FeatureAffineModulation(deg_ch, dim)
            if "shallow" in self.modulate_levels else None
        )

        self.mod_encoders = (
            nn.ModuleList([
                FeatureAffineModulation(deg_ch, c)
                for c in level_dims[:-1]
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

        # Encoder: all levels except deepest latent.
        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i in range(self.num_levels - 1):
            level_dim = level_dims[i]
            self.encoders.append(
                nn.Sequential(*[
                    TransformerBlock(
                        level_dim,
                        heads[i],
                        ffn_expansion_factor,
                        bias,
                        LayerNorm_type,
                    )
                    for _ in range(num_blocks[i])
                ])
            )
            self.downsamples.append(Downsample(level_dim))

        # Deepest bottleneck / latent: last num_blocks entry is used here only.
        latent_dim = level_dims[-1]
        self.latent = nn.Sequential(*[
            TransformerBlock(
                latent_dim,
                heads[-1],
                ffn_expansion_factor,
                bias,
                LayerNorm_type,
            )
            for _ in range(num_blocks[-1])
        ])

        # Decoder mirrors encoder levels.
        self.upsamples = nn.ModuleList()
        self.reduce_chans = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in reversed(range(self.num_levels - 1)):
            in_dim = dim * (2 ** (i + 1))
            out_dim = dim * (2 ** i)

            self.upsamples.append(Upsample(in_dim))
            self.reduce_chans.append(nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=bias))
            self.decoders.append(
                nn.Sequential(*[
                    TransformerBlock(
                        out_dim,
                        heads[i],
                        ffn_expansion_factor,
                        bias,
                        LayerNorm_type,
                    )
                    for _ in range(num_blocks[i])
                ])
            )

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

        self.output = nn.Conv2d(
            dim,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )

    # ------------------------------------------------------------
    # Coordinate behavior is specialized by subclasses.
    # ------------------------------------------------------------
    def _prepare_coord(self, x: torch.Tensor, coord: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        raise NotImplementedError

    def _coord_aux(self, coord_map: Optional[torch.Tensor]) -> Dict[str, Union[str, bool, torch.Tensor]]:
        if coord_map is None:
            return {
                "coord_mode": "disabled",
                "coord_used": False,
            }

        aux: Dict[str, Union[str, bool, torch.Tensor]] = {
            "coord_mode": self.coord_mode_name,
            "coord_used": True,
            "coord_mean": coord_map.detach().mean(),
            "coord_std": coord_map.detach().std(unbiased=False),
        }

        if coord_map.shape[1] >= 3:
            r = coord_map[:, 2:3]
        else:
            r = get_radius_from_coord(coord_map)

        if self.coord_mode_name == "global":
            prefix = "global_radius"
        else:
            prefix = "radius"

        aux.update({
            f"{prefix}_mean": r.detach().mean(),
            f"{prefix}_std": r.detach().std(unbiased=False),
            f"{prefix}_min": r.detach().min(),
            f"{prefix}_max": r.detach().max(),
        })
        return aux

    # ------------------------------------------------------------
    # Forward / monitors
    # ------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        coord: Optional[torch.Tensor] = None,
        return_monitor: bool = False,
        return_aux: bool = False,
    ):
        x_input = x
        monitor: Dict[str, Union[torch.Tensor, str, bool]] = {}

        coord_map = self._prepare_coord(x, coord) if self.use_coord else None
        self.last_coord_map = coord_map

        if self.use_coord:
            deg_in = torch.cat([x, coord_map], dim=1)
        else:
            deg_in = x

        deg_feat, deg_score = self.deg_estimator(deg_in)
        self.last_deg_feat = deg_feat
        self.last_deg_score = deg_score

        if return_monitor or return_aux:
            monitor["deg_score_mean"] = _safe_mean(deg_score)
            monitor["deg_score_std"] = _safe_std(deg_score)
            monitor["deg_feat_abs_mean"] = _safe_mean(deg_feat.abs())
            monitor["deg_field_tv"] = self.deg_smoothness_loss().detach()
            if coord_map is not None:
                monitor["deg_score_edge_center_gap"] = edge_center_gap(deg_score, coord_map)
                monitor["deg_score_radial_corr"] = radial_correlation_map(deg_score, coord_map)

        feats: List[torch.Tensor] = []

        # Shallow feature.
        x = self.patch_embed(x)
        if self.mod_shallow is not None:
            if return_monitor or return_aux:
                x, st = self.mod_shallow(x, deg_feat, return_stats=True, name="shallow_")
                monitor.update(st)
            else:
                x = self.mod_shallow(x, deg_feat)

        # Encoder levels except latent.
        for i in range(self.num_levels - 1):
            if self.mod_encoders is not None:
                if return_monitor or return_aux:
                    x, st = self.mod_encoders[i](x, deg_feat, return_stats=True, name=f"enc{i+1}_")
                    monitor.update(st)
                else:
                    x = self.mod_encoders[i](x, deg_feat)

            x = self.encoders[i](x)
            feats.append(x)
            x = self.downsamples[i](x)

        # Latent / bottleneck.
        if self.mod_latent is not None:
            if return_monitor or return_aux:
                x, st = self.mod_latent(x, deg_feat, return_stats=True, name="latent_")
                monitor.update(st)
            else:
                x = self.mod_latent(x, deg_feat)

        x = self.latent(x)
        maybe_debug_tensor("strict_degfield_latent", x)

        # Decoder.
        for dec_idx, i in enumerate(reversed(range(self.num_levels - 1))):
            x = self.upsamples[dec_idx](x)
            x = torch.cat([x, feats[i]], dim=1)
            x = self.reduce_chans[dec_idx](x)

            if self.mod_decoders is not None:
                if return_monitor or return_aux:
                    x, st = self.mod_decoders[dec_idx](x, deg_feat, return_stats=True, name=f"dec{dec_idx+1}_")
                    monitor.update(st)
                else:
                    x = self.mod_decoders[dec_idx](x, deg_feat)

            x = self.decoders[dec_idx](x)

        x = self.refinement(x)
        x_out = self.output(x)
        out = x_out + x_input

        if return_monitor or return_aux:
            monitor["output_residual_abs_mean"] = _safe_mean((out - x_input).abs())
            monitor.update(self._coord_aux(coord_map))
            self.last_monitor = monitor
            return out, monitor

        self.last_monitor = monitor
        return out

    def deg_smoothness_loss(self) -> torch.Tensor:
        if self.last_deg_score is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return tv_loss_map(self.last_deg_score)

    def deg_radial_prior_loss(
        self,
        min_corr: float = 0.10,
        require_edge_larger: bool = False,
        min_gap: float = 0.00,
    ) -> torch.Tensor:
        """
        Weak radial prior, compatible with the existing engine optional regularizer.
        """
        if self.last_deg_score is None or self.last_coord_map is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        corr = radial_correlation_map(self.last_deg_score, self.last_coord_map)
        loss = F.relu(torch.as_tensor(min_corr, device=corr.device, dtype=corr.dtype) - corr)

        if require_edge_larger:
            gap = edge_center_gap(self.last_deg_score, self.last_coord_map)
            loss = loss + F.relu(torch.as_tensor(min_gap, device=gap.device, dtype=gap.dtype) - gap)

        return loss

    def get_deg_grid_embedding(
        self,
        grid_size: Tuple[int, int] = (9, 9),
        normalize: bool = True,
        detach: bool = False,
    ) -> torch.Tensor:
        """
        Pool last_deg_feat into grid embeddings for optional visualization or contrastive usage.
        """
        if self.last_deg_feat is None:
            raise RuntimeError("last_deg_feat is None. Call forward() before get_deg_grid_embedding().")

        z = F.adaptive_avg_pool2d(self.last_deg_feat, grid_size)
        b, c, gh, gw = z.shape
        z = z.permute(0, 2, 3, 1).reshape(b * gh * gw, c)

        if normalize:
            z = F.normalize(z.float(), dim=1)
        if detach:
            z = z.detach()

        return z

    def get_last_monitor(self, to_float: bool = True) -> Dict[str, Union[float, torch.Tensor, str, bool]]:
        if not to_float:
            return self.last_monitor
        return _as_float_dict(self.last_monitor)


# ============================================================
# 1) Strict non-global / local DegField
# ============================================================
class Restormer_StrictDegField(_StrictRestormerDegFieldBase):
    """
    Strict Restormer-aligned non-global/local PSF-like DegField.

    Coordinate behavior:
        - If use_coord=True and coord is None, local normalized coordinate is generated.
        - If coord is provided, it is accepted and normalized, but normal non-global
          ablation should call this model without external coord.
    """
    coord_mode_name = "non_global_or_local"

    def _prepare_coord(self, x: torch.Tensor, coord: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if not self.use_coord:
            return None

        if coord is None:
            return make_local_coord(x, coord_channels=self.coord_channels)

        if not torch.is_tensor(coord):
            raise TypeError(f"coord must be torch.Tensor, got {type(coord)}")
        if coord.ndim != 4:
            raise ValueError(f"coord must be 4D [B,C,H,W], got shape {tuple(coord.shape)}")
        if coord.shape[0] != x.shape[0]:
            raise ValueError(f"coord batch mismatch: input batch={x.shape[0]}, coord batch={coord.shape[0]}")
        if coord.shape[1] not in (2, 4):
            raise ValueError(f"coord channel should be 2 or 4, got {coord.shape[1]}")

        coord = coord.to(device=x.device, dtype=x.dtype)
        coord = normalize_coord_channels(coord, target_channels=self.coord_channels)
        if coord.shape[-2:] != x.shape[-2:]:
            coord = F.interpolate(coord, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return coord


# Backward-friendly aliases.
Restormer_StrictLocalDegField = Restormer_StrictDegField
Restormer_StrictPSFLikeDegField = Restormer_StrictDegField
Restormer_StrictDegField_NonGlobal = Restormer_StrictDegField


# ============================================================
# 2) Strict global-coordinate DegField
# ============================================================
class Restormer_StrictGlobalDegField(_StrictRestormerDegFieldBase):
    """
    Strict Restormer-aligned global-coordinate PSF-like DegField.

    Coordinate behavior:
        - Requires external full-canvas coord when use_coord=True.
        - coord shape: [B,2,H,W] or [B,4,H,W].
        - Does not fall back to local coordinate, so a missing coord is treated
          as an experiment error.
    """
    coord_mode_name = "global"

    def _prepare_coord(self, x: torch.Tensor, coord: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if not self.use_coord:
            return None

        b, _, h, w = x.shape

        if coord is None:
            raise ValueError(
                f"{self.__class__.__name__} requires external global coord. "
                "Expected coord with shape [B,2,H,W] or [B,4,H,W]."
            )
        if not torch.is_tensor(coord):
            raise TypeError(f"coord must be torch.Tensor, got {type(coord)}")
        if coord.ndim != 4:
            raise ValueError(f"coord must be 4D [B,C,H,W], got shape {tuple(coord.shape)}")
        if coord.shape[0] != b:
            raise ValueError(f"coord batch mismatch: input batch={b}, coord batch={coord.shape[0]}")
        if coord.shape[1] not in (2, 4):
            raise ValueError(f"coord channel should be 2 or 4, got {coord.shape[1]}")

        coord = coord.to(device=x.device, dtype=x.dtype)
        coord = normalize_coord_channels(coord, target_channels=self.coord_channels)
        if coord.shape[-2:] != (h, w):
            coord = F.interpolate(coord, size=(h, w), mode="bilinear", align_corners=False)
        return coord


# Backward-friendly aliases.
Restormer_StrictGDFRestormer = Restormer_StrictGlobalDegField
Restormer_StrictGlobalPSFLikeDegField = Restormer_StrictGlobalDegField
Restormer_StrictGlobalDegField_4L = Restormer_StrictGlobalDegField


# ============================================================
# Default argument dictionaries and optional registry
# ============================================================
RESTORMER4_STRICT_ARGS = {
    "inp_channels": 3,
    "out_channels": 3,
    "dim": 48,
    "num_blocks": [4, 6, 6, 8],
    "num_refinement_blocks": 4,
    "heads": [1, 2, 4, 8],
    "ffn_expansion_factor": 2.66,
    "bias": False,
    "LayerNorm_type": "BiasFree",
}

DEGFIELD4_STRICT_ARGS = {
    **RESTORMER4_STRICT_ARGS,
    "use_coord": True,
    "coord_channels": 4,
    "deg_ch": 16,
    "deg_mid_ch": 32,
    "deg_downsample_factor": 4,
    "modulate_levels": ("shallow", "enc", "latent", "dec"),
}

STRICT_MODEL_REGISTRY = {
    "StrictDegField_4L_NonGlobal": {
        "model_class": Restormer_StrictDegField,
        "model_args": DEGFIELD4_STRICT_ARGS,
        "requires_external_coord": False,
        "description": "Strict Restormer-aligned non-global/local PSF-like DegField.",
    },
    "StrictGlobalDegField_4L": {
        "model_class": Restormer_StrictGlobalDegField,
        "model_args": DEGFIELD4_STRICT_ARGS,
        "requires_external_coord": True,
        "description": "Strict Restormer-aligned global-coordinate PSF-like DegField.",
    },
}


# ============================================================
# Quick smoke test
# ============================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(1, 3, 64, 64, device=device)

    yy = torch.linspace(-1.0, 1.0, 64, device=device).view(1, 1, 64, 1).expand(1, 1, 64, 64)
    xx = torch.linspace(-1.0, 1.0, 64, device=device).view(1, 1, 1, 64).expand(1, 1, 64, 64)
    coord2 = torch.cat([xx, yy], dim=1)

    small_args = dict(
        inp_channels=3,
        out_channels=3,
        dim=24,
        num_blocks=[1, 1, 1],
        num_refinement_blocks=1,
        heads=[1, 2, 4],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="BiasFree",
        use_coord=True,
        coord_channels=4,
        deg_ch=8,
        deg_mid_ch=16,
        deg_downsample_factor=4,
        modulate_levels=("shallow", "enc", "latent", "dec"),
    )

    print("Testing Restormer_StrictDegField...")
    m1 = Restormer_StrictDegField(**small_args).to(device)
    y1, aux1 = m1(x, return_aux=True)
    print("Output:", tuple(y1.shape), "coord_mode:", aux1.get("coord_mode"), "keys:", list(aux1.keys())[:8])

    print("Testing Restormer_StrictGlobalDegField...")
    m2 = Restormer_StrictGlobalDegField(**small_args).to(device)
    y2, aux2 = m2(x, coord=coord2, return_aux=True)
    print("Output:", tuple(y2.shape), "coord_mode:", aux2.get("coord_mode"), "keys:", list(aux2.keys())[:8])

    try:
        print("Testing missing coord for strict global model...")
        _ = m2(x)
    except Exception as exc:
        print("Expected error:", repr(exc))
