# -*- coding: utf-8 -*-
"""
Res_Global_RetinexMoE.py
========================

Strict global-coordinate PSF-prior variants, including:
    1) GlobalPSFLikeDegField          (existing strict global wrapper)
    2) GlobalPSFLikeLowRankBasis      (existing strict global wrapper)
    3) GlobalPSFScaleMoE              (3-expert scale-aware MoE)
    4) GlobalPSFScaleRetinexMoE       (scale-aware MoE + Retinex-style illumination expert)

This file is intended as a drop-in upgrade of Res_Global.py while preserving
all previously available structures.
"""

from typing import Dict, Optional, Union, Tuple, Sequence, List

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .Res_Psfbasis import (
        Restormer_PSFLikeDegField,
        Restormer_PSFLikeLowRankBasis,
        normalize_coord_channels,
        get_radius_from_coord,
        radial_correlation_map,
        edge_center_gap,
        _safe_mean,
        _safe_std,
        tv_loss_map,
    )
except Exception:
    from Res_Psfbasis import (
        Restormer_PSFLikeDegField,
        Restormer_PSFLikeLowRankBasis,
        normalize_coord_channels,
        get_radius_from_coord,
        radial_correlation_map,
        edge_center_gap,
        _safe_mean,
        _safe_std,
        tv_loss_map,
    )


# ============================================================
# Shared strict global-coordinate helper
# ============================================================
class _StrictGlobalCoordMixin:
    """
    Mixin that replaces local fallback coord with strict external global coord.

    This class assumes the child class defines:
        self.use_coord
        self.coord_channels
        self.last_coord_map
    """

    def _prepare_coord(self, x: torch.Tensor, coord: Optional[torch.Tensor]):
        if not self.use_coord:
            return None

        b, _, h, w = x.shape

        if coord is None:
            raise ValueError(
                f"{self.__class__.__name__} requires external global coord. "
                "Expected coord with shape [B,2,H,W] or [B,4,H,W]. "
                "Do not omit coord; otherwise the experiment is no longer a "
                "true global-coordinate PSF-prior run."
            )

        if not torch.is_tensor(coord):
            raise TypeError(f"coord must be torch.Tensor, got {type(coord)}")

        if coord.ndim != 4:
            raise ValueError(f"coord must be 4D [B,C,H,W], got shape {tuple(coord.shape)}")

        if coord.shape[0] != b:
            raise ValueError(
                f"coord batch mismatch: input batch={b}, coord batch={coord.shape[0]}"
            )

        if coord.shape[1] not in (2, 4):
            raise ValueError(
                f"coord channel should be 2 or 4, got {coord.shape[1]}. "
                "Use [x,y] or [x,y,r,r^2]."
            )

        coord = coord.to(device=x.device, dtype=x.dtype)
        coord = normalize_coord_channels(coord, target_channels=self.coord_channels)

        if coord.shape[-2:] != (h, w):
            coord = F.interpolate(
                coord,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )

        return coord

    @staticmethod
    def _coord_aux(coord_map: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        if coord_map is None:
            return {
                "coord_mode": "global_disabled",
                "coord_used": False,
            }

        aux = {
            "coord_mode": "global",
            "coord_used": True,
            "coord_mean": coord_map.detach().mean(),
            "coord_std": coord_map.detach().std(unbiased=False),
        }

        if coord_map.shape[1] >= 3:
            r = coord_map[:, 2:3]
        else:
            r = get_radius_from_coord(coord_map)

        aux.update({
            "global_radius_mean": r.detach().mean(),
            "global_radius_std": r.detach().std(unbiased=False),
            "global_radius_min": r.detach().min(),
            "global_radius_max": r.detach().max(),
        })
        return aux


# ============================================================
# Existing strict global wrappers (preserved unchanged)
# ============================================================
class Restormer_GlobalPSFLikeDegField(_StrictGlobalCoordMixin, Restormer_PSFLikeDegField):
    def forward(
        self,
        x: torch.Tensor,
        coord: Optional[torch.Tensor] = None,
        return_monitor: bool = False,
        return_aux: bool = False,
    ):
        if return_aux:
            pred, monitor = super().forward(x, coord=coord, return_monitor=True)
            aux = dict(monitor)
            aux.update(self._coord_aux(self.last_coord_map))
            return pred, aux

        return super().forward(x, coord=coord, return_monitor=return_monitor)

    def get_last_monitor(self, to_float: bool = True) -> Dict[str, Union[float, torch.Tensor]]:
        base = super().get_last_monitor(to_float=False)
        if self.last_coord_map is not None:
            coord = self.last_coord_map.detach()
            base = dict(base)
            base["global_coord_mean"] = coord.mean()
            base["global_coord_std"] = coord.std(unbiased=False)
            r = get_radius_from_coord(coord)
            base["global_radius_mean"] = r.mean()
            base["global_radius_std"] = r.std(unbiased=False)
            base["global_radius_min"] = r.min()
            base["global_radius_max"] = r.max()

        if not to_float:
            return base

        out = {}
        for k, v in base.items():
            if torch.is_tensor(v):
                out[k] = float(v.detach().cpu())
            else:
                out[k] = v
        return out


class Restormer_GlobalPSFLikeLowRankBasis(_StrictGlobalCoordMixin, Restormer_PSFLikeLowRankBasis):
    def forward(
        self,
        x: torch.Tensor,
        coord: Optional[torch.Tensor] = None,
        return_monitor: bool = False,
        return_aux: bool = False,
    ):
        if return_aux:
            pred, monitor = super().forward(x, coord=coord, return_monitor=True)
            aux = dict(monitor)
            aux.update(self._coord_aux(self.last_coord_map))
            return pred, aux

        return super().forward(x, coord=coord, return_monitor=return_monitor)

    def get_last_monitor(self, to_float: bool = True) -> Dict[str, Union[float, torch.Tensor]]:
        base = super().get_last_monitor(to_float=False)
        if self.last_coord_map is not None:
            coord = self.last_coord_map.detach()
            base = dict(base)
            base["global_coord_mean"] = coord.mean()
            base["global_coord_std"] = coord.std(unbiased=False)
            r = get_radius_from_coord(coord)
            base["global_radius_mean"] = r.mean()
            base["global_radius_std"] = r.std(unbiased=False)
            base["global_radius_min"] = r.min()
            base["global_radius_max"] = r.max()

        if not to_float:
            return base

        out = {}
        for k, v in base.items():
            if torch.is_tensor(v):
                out[k] = float(v.detach().cpu())
            else:
                out[k] = v
        return out


# Backward-friendly aliases.
Restormer_GlobalPSFDegField = Restormer_GlobalPSFLikeDegField
Restormer_GlobalPSFLowRankBasis = Restormer_GlobalPSFLikeLowRankBasis


# ============================================================
# MoE helper modules
# ============================================================
class _DepthwiseResidualExpert(nn.Module):
    def __init__(self, ch: int, dilation: int = 1, zero_init: bool = True):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, 3, 1, padding=dilation, dilation=dilation, groups=ch)
        self.pw1 = nn.Conv2d(ch, ch, 1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(ch, ch, 1)
        if zero_init:
            nn.init.zeros_(self.pw2.weight)
            if self.pw2.bias is not None:
                nn.init.zeros_(self.pw2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dw(x)
        h = self.act(self.pw1(h))
        h = self.pw2(h)
        return h


class _LargeScaleResidualExpert(nn.Module):
    def __init__(self, ch: int, dilation: int = 4, zero_init: bool = True):
        super().__init__()
        self.local = _DepthwiseResidualExpert(ch, dilation=dilation, zero_init=False)
        self.gc_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch, 1),
            nn.GELU(),
            nn.Conv2d(ch, ch, 1),
        )
        self.out = nn.Conv2d(ch, ch, 1)
        if zero_init:
            nn.init.zeros_(self.out.weight)
            if self.out.bias is not None:
                nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.local(x)
        gc = self.gc_proj(x)
        gc = gc.expand(-1, -1, x.shape[-2], x.shape[-1])
        return self.out(local + gc)


class _RetinexIlluminationExpert(nn.Module):
    """
    Retinex-style illumination expert.

    It estimates a low-frequency illumination map from luminance and combines it
    with the degradation feature to produce an illumination-aware prior.
    """
    def __init__(self, deg_ch: int, hidden_ch: int = 16, zero_init: bool = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(deg_ch + 2, hidden_ch, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden_ch, hidden_ch, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden_ch, deg_ch, 3, 1, 1),
        )
        if zero_init:
            last = self.net[-1]
            nn.init.zeros_(last.weight)
            if last.bias is not None:
                nn.init.zeros_(last.bias)

    @staticmethod
    def _lowpass(y: torch.Tensor) -> torch.Tensor:
        # A simple, stable low-frequency approximation.
        y = F.avg_pool2d(y, kernel_size=7, stride=1, padding=3)
        y = F.avg_pool2d(y, kernel_size=5, stride=1, padding=2)
        return y

    def forward(self, x_input: torch.Tensor, deg_feat: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # luminance in [roughly] RGB range; stay differentiable.
        y = 0.299 * x_input[:, 0:1] + 0.587 * x_input[:, 1:2] + 0.114 * x_input[:, 2:3]
        y_low = self._lowpass(y)
        y_low = F.interpolate(y_low, size=deg_feat.shape[-2:], mode="bilinear", align_corners=False)
        reflect = y / (self._lowpass(y) + 1e-4)
        reflect = F.interpolate(reflect, size=deg_feat.shape[-2:], mode="bilinear", align_corners=False)
        feat = torch.cat([deg_feat, y_low, reflect], dim=1)
        out = self.net(feat)
        monitor = {
            "illum_low_mean": _safe_mean(y_low),
            "illum_low_std": _safe_std(y_low),
            "reflect_proxy_mean": _safe_mean(reflect),
            "reflect_proxy_std": _safe_std(reflect),
        }
        return out, monitor


class _RouterHead(nn.Module):
    def __init__(self, in_ch: int, num_experts: int):
        super().__init__()
        hid = max(16, in_ch)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hid, 1),
            nn.GELU(),
            nn.Conv2d(hid, num_experts, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# MoE base class
# ============================================================
class _GlobalPSFMoEBase(_StrictGlobalCoordMixin, Restormer_PSFLikeDegField):
    def _prepare_router_coord(self, coord_map: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        return F.interpolate(coord_map, size=size, mode="bilinear", align_corners=False)

    @staticmethod
    def _region_masks_from_coord(coord: torch.Tensor, size: Tuple[int, int]):
        r = get_radius_from_coord(coord, size=size)
        center = (r < 0.30).float()
        middle = ((r >= 0.30) & (r < 0.55)).float()
        outer = (r >= 0.55).float()
        return center, middle, outer, r

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (x * mask).sum() / (mask.sum() + 1e-6)

    def _log_common_router_stats(
        self,
        monitor: Dict[str, torch.Tensor],
        router_prob: torch.Tensor,
        coord_map: torch.Tensor,
        prefix: str,
        expert_names: Sequence[str],
    ):
        usage = router_prob.mean(dim=(0, 2, 3))
        monitor[f"{prefix}_usage_std"] = usage.std(unbiased=False)
        entropy = -(router_prob.clamp_min(1e-8) * router_prob.clamp_min(1e-8).log()).sum(dim=1, keepdim=True)
        entropy = entropy / float(torch.log(torch.tensor(router_prob.shape[1], device=router_prob.device, dtype=router_prob.dtype)))
        monitor[f"{prefix}_router_entropy_norm"] = _safe_mean(entropy)
        monitor[f"{prefix}_top_prob_mean"] = _safe_mean(router_prob.max(dim=1, keepdim=True).values)
        monitor[f"{prefix}_router_tv"] = tv_loss_map(router_prob)

        center_mask, middle_mask, outer_mask, _ = self._region_masks_from_coord(coord_map, size=router_prob.shape[-2:])

        for i, name in enumerate(expert_names):
            p = router_prob[:, i:i+1]
            center_usage = self._masked_mean(p, center_mask)
            middle_usage = self._masked_mean(p, middle_mask)
            outer_usage = self._masked_mean(p, outer_mask)

            monitor[f"{prefix}_usage_{name}"] = _safe_mean(p)
            monitor[f"{prefix}_usage_center_{name}"] = center_usage
            monitor[f"{prefix}_usage_middle_region_{name}"] = middle_usage
            monitor[f"{prefix}_usage_outer_{name}"] = outer_usage
            monitor[f"{prefix}_{name}_outer_minus_center"] = outer_usage - center_usage
            monitor[f"{prefix}_{name}_radial_corr"] = radial_correlation_map(p, coord_map)

        if "large" in expert_names and "center" not in expert_names:
            idx = expert_names.index("large")
            p = router_prob[:, idx:idx+1]
            monitor[f"{prefix}_large_outer_minus_center"] = self._masked_mean(p, outer_mask) - self._masked_mean(p, center_mask)

        if "small" in expert_names:
            idx = expert_names.index("small")
            p = router_prob[:, idx:idx+1]
            monitor[f"{prefix}_small_center_minus_outer"] = self._masked_mean(p, center_mask) - self._masked_mean(p, outer_mask)

    def scale_moe_balance_loss(self) -> torch.Tensor:
        prob = getattr(self, "last_scale_router_prob", None)
        if prob is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        usage = prob.mean(dim=(0, 2, 3))
        target = torch.full_like(usage, 1.0 / usage.numel())
        return F.mse_loss(usage, target)

    def scale_moe_router_tv_loss(self) -> torch.Tensor:
        prob = getattr(self, "last_scale_router_prob", None)
        if prob is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return tv_loss_map(prob)

    def scale_moe_outer_large_prior_loss(self, margin: float = 0.02) -> torch.Tensor:
        prob = getattr(self, "last_scale_router_prob", None)
        coord = getattr(self, "last_coord_map", None)
        names = getattr(self, "scale_expert_names", None)
        if prob is None or coord is None or names is None or "large" not in names:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        idx = names.index("large")
        large = prob[:, idx:idx+1]
        center_mask, _, outer_mask, _ = self._region_masks_from_coord(coord, size=large.shape[-2:])
        outer_mean = self._masked_mean(large, outer_mask)
        center_mean = self._masked_mean(large, center_mask)
        return F.relu(torch.as_tensor(margin, device=large.device, dtype=large.dtype) - (outer_mean - center_mean))

    def get_last_monitor(self, to_float: bool = True) -> Dict[str, Union[float, torch.Tensor]]:
        base = getattr(self, "last_monitor", {})
        if self.last_coord_map is not None:
            coord = self.last_coord_map.detach()
            base = dict(base)
            base["global_coord_mean"] = coord.mean()
            base["global_coord_std"] = coord.std(unbiased=False)
            r = get_radius_from_coord(coord)
            base["global_radius_mean"] = r.mean()
            base["global_radius_std"] = r.std(unbiased=False)
            base["global_radius_min"] = r.min()
            base["global_radius_max"] = r.max()

        if not to_float:
            return base
        out = {}
        for k, v in base.items():
            out[k] = float(v.detach().cpu()) if torch.is_tensor(v) else v
        return out


# ============================================================
# 3-expert ScaleMoE
# ============================================================
class Restormer_GlobalPSFScaleMoE(_GlobalPSFMoEBase):
    def __init__(
        self,
        *args,
        scale_router_temperature: float = 1.0,
        scale_expert_zero_init: bool = True,
        scale_expert_residual_scale: float = 0.25,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.scale_router_temperature = float(scale_router_temperature)
        self.scale_expert_residual_scale = float(scale_expert_residual_scale)
        self.scale_expert_names = ["small", "middle", "large"]

        self.scale_small_expert = _DepthwiseResidualExpert(self.deg_ch, dilation=1, zero_init=scale_expert_zero_init)
        self.scale_middle_expert = _DepthwiseResidualExpert(self.deg_ch, dilation=2, zero_init=scale_expert_zero_init)
        self.scale_large_expert = _LargeScaleResidualExpert(self.deg_ch, dilation=4, zero_init=scale_expert_zero_init)
        router_in_ch = self.deg_ch + (self.coord_channels if self.use_coord else 0)
        self.scale_router = _RouterHead(router_in_ch, num_experts=3)

        self.last_scale_router_prob: Optional[torch.Tensor] = None
        self.last_scale_prior: Optional[torch.Tensor] = None

    def _mix_prior(self, x_input: torch.Tensor, deg_feat: torch.Tensor, coord_map: torch.Tensor, monitor: Dict[str, torch.Tensor]) -> torch.Tensor:
        coord_low = self._prepare_router_coord(coord_map, deg_feat.shape[-2:]) if coord_map is not None else None
        router_in = torch.cat([deg_feat, coord_low], dim=1) if coord_low is not None else deg_feat
        logits = self.scale_router(router_in) / max(self.scale_router_temperature, 1e-6)
        prob = torch.softmax(logits, dim=1)

        small = self.scale_small_expert(deg_feat)
        middle = self.scale_middle_expert(deg_feat)
        large = self.scale_large_expert(deg_feat)

        mix = (
            prob[:, 0:1] * small +
            prob[:, 1:2] * middle +
            prob[:, 2:3] * large
        )
        prior = deg_feat + self.scale_expert_residual_scale * mix

        self.last_scale_router_prob = prob
        self.last_scale_prior = prior

        monitor["scale_prior_abs_mean"] = _safe_mean(prior.abs())
        monitor["scale_delta_from_deg_abs_mean"] = _safe_mean((prior - deg_feat).abs())
        self._log_common_router_stats(monitor, prob, coord_map, prefix="scale", expert_names=self.scale_expert_names)
        return prior

    def forward(
        self,
        x: torch.Tensor,
        coord: Optional[torch.Tensor] = None,
        return_monitor: bool = False,
        return_aux: bool = False,
    ):
        x_input = x
        monitor: Dict[str, torch.Tensor] = {}

        coord_map = self._prepare_coord(x, coord)
        self.last_coord_map = coord_map

        deg_in = torch.cat([x, coord_map], dim=1) if self.use_coord else x
        deg_feat, deg_score = self.deg_estimator(deg_in)
        self.last_deg_feat = deg_feat
        self.last_deg_score = deg_score

        prior = self._mix_prior(x_input, deg_feat, coord_map, monitor)

        monitor["deg_score_mean"] = _safe_mean(deg_score)
        monitor["deg_score_std"] = _safe_std(deg_score)
        monitor["deg_feat_abs_mean"] = _safe_mean(deg_feat.abs())
        monitor["deg_field_tv"] = self.deg_smoothness_loss().detach()
        if coord_map is not None:
            monitor["deg_score_edge_center_gap"] = edge_center_gap(deg_score, coord_map)
            monitor["deg_score_radial_corr"] = radial_correlation_map(deg_score, coord_map)

        feats: List[torch.Tensor] = []
        x = self.patch_embed(x)

        if self.mod_shallow is not None:
            if return_monitor:
                x, st = self.mod_shallow(x, prior, return_stats=True, name="shallow_")
                monitor.update(st)
            else:
                x = self.mod_shallow(x, prior)

        for i in range(self.num_levels):
            if self.mod_encoders is not None:
                if return_monitor:
                    x, st = self.mod_encoders[i](x, prior, return_stats=True, name=f"enc{i+1}_")
                    monitor.update(st)
                else:
                    x = self.mod_encoders[i](x, prior)

            x = self.encoders[i](x)
            feats.append(x)
            if i < self.num_levels - 1:
                x = self.downsamples[i](x)

        if self.mod_latent is not None:
            if return_monitor:
                x, st = self.mod_latent(x, prior, return_stats=True, name="latent_")
                monitor.update(st)
            else:
                x = self.mod_latent(x, prior)

        x = self.latent(x)
        dec_idx = 0
        for i in range(self.num_levels - 1)[::-1]:
            x = self.upsamples[self.num_levels - 2 - i](x)
            x = torch.cat([x, feats[i]], dim=1)
            x = self.reduce_chans[self.num_levels - 2 - i](x)

            if self.mod_decoders is not None:
                if return_monitor:
                    x, st = self.mod_decoders[dec_idx](x, prior, return_stats=True, name=f"dec{dec_idx+1}_")
                    monitor.update(st)
                else:
                    x = self.mod_decoders[dec_idx](x, prior)

            x = self.decoders[self.num_levels - 2 - i](x)
            dec_idx += 1

        x = self.refinement(x)
        x_out = self.output(x)
        out = x_out + x_input

        monitor["output_residual_abs_mean"] = _safe_mean((out - x_input).abs())
        self.last_monitor = monitor

        if return_aux:
            aux = dict(monitor)
            aux.update(self._coord_aux(self.last_coord_map))
            return out, aux
        if return_monitor:
            return out, monitor
        return out


Restormer_GlobalPSFScaleMoEDegField = Restormer_GlobalPSFScaleMoE
Restormer_GlobalPSFScaleMoE_DegField = Restormer_GlobalPSFScaleMoE


# ============================================================
# 4-expert ScaleMoE + Retinex-style illumination expert
# ============================================================
class Restormer_GlobalPSFScaleRetinexMoE(_GlobalPSFMoEBase):
    def __init__(
        self,
        *args,
        scale_router_temperature: float = 1.0,
        scale_expert_zero_init: bool = True,
        scale_expert_residual_scale: float = 0.25,
        illum_hidden_ch: int = 16,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.scale_router_temperature = float(scale_router_temperature)
        self.scale_expert_residual_scale = float(scale_expert_residual_scale)
        self.scale_expert_names = ["small", "middle", "large", "illumination"]

        self.scale_small_expert = _DepthwiseResidualExpert(self.deg_ch, dilation=1, zero_init=scale_expert_zero_init)
        self.scale_middle_expert = _DepthwiseResidualExpert(self.deg_ch, dilation=2, zero_init=scale_expert_zero_init)
        self.scale_large_expert = _LargeScaleResidualExpert(self.deg_ch, dilation=4, zero_init=scale_expert_zero_init)
        self.scale_illum_expert = _RetinexIlluminationExpert(self.deg_ch, hidden_ch=illum_hidden_ch, zero_init=scale_expert_zero_init)
        router_in_ch = self.deg_ch + (self.coord_channels if self.use_coord else 0)
        self.scale_router = _RouterHead(router_in_ch, num_experts=4)

        self.last_scale_router_prob: Optional[torch.Tensor] = None
        self.last_scale_prior: Optional[torch.Tensor] = None

    def _mix_prior(self, x_input: torch.Tensor, deg_feat: torch.Tensor, coord_map: torch.Tensor, monitor: Dict[str, torch.Tensor]) -> torch.Tensor:
        coord_low = self._prepare_router_coord(coord_map, deg_feat.shape[-2:]) if coord_map is not None else None
        router_in = torch.cat([deg_feat, coord_low], dim=1) if coord_low is not None else deg_feat
        logits = self.scale_router(router_in) / max(self.scale_router_temperature, 1e-6)
        prob = torch.softmax(logits, dim=1)

        small = self.scale_small_expert(deg_feat)
        middle = self.scale_middle_expert(deg_feat)
        large = self.scale_large_expert(deg_feat)
        illum, illum_mon = self.scale_illum_expert(x_input, deg_feat)
        monitor.update(illum_mon)

        mix = (
            prob[:, 0:1] * small +
            prob[:, 1:2] * middle +
            prob[:, 2:3] * large +
            prob[:, 3:4] * illum
        )
        prior = deg_feat + self.scale_expert_residual_scale * mix

        self.last_scale_router_prob = prob
        self.last_scale_prior = prior

        monitor["scale_prior_abs_mean"] = _safe_mean(prior.abs())
        monitor["scale_delta_from_deg_abs_mean"] = _safe_mean((prior - deg_feat).abs())
        monitor["illum_prior_abs_mean"] = _safe_mean(illum.abs())
        monitor["illum_prior_tv"] = tv_loss_map(illum).detach()
        self._log_common_router_stats(monitor, prob, coord_map, prefix="scale", expert_names=self.scale_expert_names)
        return prior

    def forward(
        self,
        x: torch.Tensor,
        coord: Optional[torch.Tensor] = None,
        return_monitor: bool = False,
        return_aux: bool = False,
    ):
        x_input = x
        monitor: Dict[str, torch.Tensor] = {}

        coord_map = self._prepare_coord(x, coord)
        self.last_coord_map = coord_map

        deg_in = torch.cat([x, coord_map], dim=1) if self.use_coord else x
        deg_feat, deg_score = self.deg_estimator(deg_in)
        self.last_deg_feat = deg_feat
        self.last_deg_score = deg_score

        prior = self._mix_prior(x_input, deg_feat, coord_map, monitor)

        monitor["deg_score_mean"] = _safe_mean(deg_score)
        monitor["deg_score_std"] = _safe_std(deg_score)
        monitor["deg_feat_abs_mean"] = _safe_mean(deg_feat.abs())
        monitor["deg_field_tv"] = self.deg_smoothness_loss().detach()
        if coord_map is not None:
            monitor["deg_score_edge_center_gap"] = edge_center_gap(deg_score, coord_map)
            monitor["deg_score_radial_corr"] = radial_correlation_map(deg_score, coord_map)

        feats: List[torch.Tensor] = []
        x = self.patch_embed(x)

        if self.mod_shallow is not None:
            if return_monitor:
                x, st = self.mod_shallow(x, prior, return_stats=True, name="shallow_")
                monitor.update(st)
            else:
                x = self.mod_shallow(x, prior)

        for i in range(self.num_levels):
            if self.mod_encoders is not None:
                if return_monitor:
                    x, st = self.mod_encoders[i](x, prior, return_stats=True, name=f"enc{i+1}_")
                    monitor.update(st)
                else:
                    x = self.mod_encoders[i](x, prior)

            x = self.encoders[i](x)
            feats.append(x)
            if i < self.num_levels - 1:
                x = self.downsamples[i](x)

        if self.mod_latent is not None:
            if return_monitor:
                x, st = self.mod_latent(x, prior, return_stats=True, name="latent_")
                monitor.update(st)
            else:
                x = self.mod_latent(x, prior)

        x = self.latent(x)
        dec_idx = 0
        for i in range(self.num_levels - 1)[::-1]:
            x = self.upsamples[self.num_levels - 2 - i](x)
            x = torch.cat([x, feats[i]], dim=1)
            x = self.reduce_chans[self.num_levels - 2 - i](x)

            if self.mod_decoders is not None:
                if return_monitor:
                    x, st = self.mod_decoders[dec_idx](x, prior, return_stats=True, name=f"dec{dec_idx+1}_")
                    monitor.update(st)
                else:
                    x = self.mod_decoders[dec_idx](x, prior)

            x = self.decoders[self.num_levels - 2 - i](x)
            dec_idx += 1

        x = self.refinement(x)
        x_out = self.output(x)
        out = x_out + x_input

        monitor["output_residual_abs_mean"] = _safe_mean((out - x_input).abs())
        self.last_monitor = monitor

        if return_aux:
            aux = dict(monitor)
            aux.update(self._coord_aux(self.last_coord_map))
            return out, aux
        if return_monitor:
            return out, monitor
        return out


# Helpful aliases.
Restormer_GlobalPSFRetinexMoE = Restormer_GlobalPSFScaleRetinexMoE
Restormer_GlobalPSFScaleIllumMoE = Restormer_GlobalPSFScaleRetinexMoE
Restormer_GlobalPSFScaleRetinexMoEDegField = Restormer_GlobalPSFScaleRetinexMoE


# ============================================================
# 5-expert ScaleMoE + Retinex illumination + Wavelet-LL expert
# ============================================================

class _HaarWaveletLLExpert(nn.Module):
    """
    Lightweight Haar wavelet low-frequency expert.

    Motivation:
        For endoscopic images, illumination and smooth degradation cues are often
        concentrated in low-frequency components. This expert uses a Haar-like
        LL component of deg_feat as an additional candidate prior. It is
        router-controlled and therefore does not force frequency information into
        all regions.

    Input:
        deg_feat [B, C, H, W]

    Output:
        wavelet_prior [B, C, H, W]
        monitor dict
    """

    def __init__(self, deg_ch: int, hidden_ch: int = 16, zero_init: bool = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(deg_ch + 1, hidden_ch, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden_ch, hidden_ch, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden_ch, deg_ch, 3, 1, 1),
        )
        if zero_init:
            last = self.net[-1]
            nn.init.zeros_(last.weight)
            if last.bias is not None:
                nn.init.zeros_(last.bias)

    @staticmethod
    def _pad_even(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        h, w = x.shape[-2:]
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x, (pad_h, pad_w)

    @staticmethod
    def _crop_back(x: torch.Tensor, pad_hw: Tuple[int, int]) -> torch.Tensor:
        pad_h, pad_w = pad_hw
        if pad_h:
            x = x[..., :-pad_h, :]
        if pad_w:
            x = x[..., :, :-pad_w]
        return x

    def _haar_decompose(self, x: torch.Tensor):
        x_pad, pad_hw = self._pad_even(x)
        x00 = x_pad[:, :, 0::2, 0::2]
        x01 = x_pad[:, :, 0::2, 1::2]
        x10 = x_pad[:, :, 1::2, 0::2]
        x11 = x_pad[:, :, 1::2, 1::2]

        # Orthonormal Haar-like subbands.
        ll = (x00 + x01 + x10 + x11) * 0.5
        lh = (x00 + x01 - x10 - x11) * 0.5
        hl = (x00 - x01 + x10 - x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5
        return ll, lh, hl, hh, pad_hw

    def forward(self, deg_feat: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        ll, lh, hl, hh, pad_hw = self._haar_decompose(deg_feat)

        ll_up = F.interpolate(ll, scale_factor=2, mode="bilinear", align_corners=False)
        ll_up = self._crop_back(ll_up, pad_hw)

        # High-frequency energy is only used as a monitor hint.
        hf_energy = (lh.abs().mean(dim=1, keepdim=True) +
                     hl.abs().mean(dim=1, keepdim=True) +
                     hh.abs().mean(dim=1, keepdim=True)) / 3.0
        hf_energy_up = F.interpolate(hf_energy, size=deg_feat.shape[-2:], mode="bilinear", align_corners=False)

        wavelet_in = torch.cat([ll_up, hf_energy_up], dim=1)
        out = self.net(wavelet_in)

        monitor = {
            "wavelet_ll_abs_mean": _safe_mean(ll_up.abs()),
            "wavelet_ll_std": _safe_std(ll_up),
            "wavelet_hf_energy_mean": _safe_mean(hf_energy_up),
            "wavelet_hf_energy_std": _safe_std(hf_energy_up),
        }
        return out, monitor


class Restormer_GlobalPSFScaleRetinexWaveletMoE(Restormer_GlobalPSFScaleRetinexMoE):
    """
    Strict global-coordinate PSF-like DegField with five router-controlled experts:

        1. small-scale expert
        2. middle-scale expert
        3. large-scale expert
        4. Retinex-style illumination expert
        5. Wavelet-LL low-frequency expert

    This preserves all previous modules and only adds a wavelet candidate expert.
    The router decides whether/where the wavelet prior is used.
    """

    def __init__(
        self,
        *args,
        scale_router_temperature: float = 1.0,
        scale_expert_zero_init: bool = True,
        scale_expert_residual_scale: float = 0.25,
        illum_hidden_ch: int = 16,
        wavelet_hidden_ch: int = 16,
        **kwargs,
    ):
        # Initialize base Restormer_PSFLikeDegField directly through the grandparent
        # to avoid constructing a 4-expert router first.
        _GlobalPSFMoEBase.__init__(self, *args, **kwargs)

        self.scale_router_temperature = float(scale_router_temperature)
        self.scale_expert_residual_scale = float(scale_expert_residual_scale)
        self.scale_expert_names = ["small", "middle", "large", "illumination", "wavelet"]

        self.scale_small_expert = _DepthwiseResidualExpert(self.deg_ch, dilation=1, zero_init=scale_expert_zero_init)
        self.scale_middle_expert = _DepthwiseResidualExpert(self.deg_ch, dilation=2, zero_init=scale_expert_zero_init)
        self.scale_large_expert = _LargeScaleResidualExpert(self.deg_ch, dilation=4, zero_init=scale_expert_zero_init)
        self.scale_illum_expert = _RetinexIlluminationExpert(
            self.deg_ch,
            hidden_ch=illum_hidden_ch,
            zero_init=scale_expert_zero_init,
        )
        self.scale_wavelet_expert = _HaarWaveletLLExpert(
            self.deg_ch,
            hidden_ch=wavelet_hidden_ch,
            zero_init=scale_expert_zero_init,
        )

        router_in_ch = self.deg_ch + (self.coord_channels if self.use_coord else 0)
        self.scale_router = _RouterHead(router_in_ch, num_experts=5)

        self.last_scale_router_prob: Optional[torch.Tensor] = None
        self.last_scale_prior: Optional[torch.Tensor] = None

    def _mix_prior(self, x_input: torch.Tensor, deg_feat: torch.Tensor, coord_map: torch.Tensor, monitor: Dict[str, torch.Tensor]) -> torch.Tensor:
        coord_low = self._prepare_router_coord(coord_map, deg_feat.shape[-2:]) if coord_map is not None else None
        router_in = torch.cat([deg_feat, coord_low], dim=1) if coord_low is not None else deg_feat
        logits = self.scale_router(router_in) / max(self.scale_router_temperature, 1e-6)
        prob = torch.softmax(logits, dim=1)

        small = self.scale_small_expert(deg_feat)
        middle = self.scale_middle_expert(deg_feat)
        large = self.scale_large_expert(deg_feat)

        illum, illum_mon = self.scale_illum_expert(x_input, deg_feat)
        wavelet, wavelet_mon = self.scale_wavelet_expert(deg_feat)
        monitor.update(illum_mon)
        monitor.update(wavelet_mon)

        mix = (
            prob[:, 0:1] * small +
            prob[:, 1:2] * middle +
            prob[:, 2:3] * large +
            prob[:, 3:4] * illum +
            prob[:, 4:5] * wavelet
        )
        prior = deg_feat + self.scale_expert_residual_scale * mix

        self.last_scale_router_prob = prob
        self.last_scale_prior = prior

        monitor["scale_prior_abs_mean"] = _safe_mean(prior.abs())
        monitor["scale_delta_from_deg_abs_mean"] = _safe_mean((prior - deg_feat).abs())

        monitor["illum_prior_abs_mean"] = _safe_mean(illum.abs())
        monitor["illum_prior_tv"] = tv_loss_map(illum).detach()

        monitor["wavelet_prior_abs_mean"] = _safe_mean(wavelet.abs())
        monitor["wavelet_prior_tv"] = tv_loss_map(wavelet).detach()

        self._log_common_router_stats(monitor, prob, coord_map, prefix="scale", expert_names=self.scale_expert_names)
        return prior


# Helpful aliases for the 5-expert Retinex + Wavelet variant.
Restormer_GlobalPSFRetinexWaveletMoE = Restormer_GlobalPSFScaleRetinexWaveletMoE
Restormer_GlobalPSFScaleIllumWaveletMoE = Restormer_GlobalPSFScaleRetinexWaveletMoE
Restormer_GlobalPSFScaleRetinexWaveletMoEDegField = Restormer_GlobalPSFScaleRetinexWaveletMoE


# ============================================================
# 6-expert ScaleMoE + Retinex + Wavelet-LL + Wavelet-HF-light
# ============================================================

class _HaarWaveletHFLightExpert(nn.Module):
    """
    Lightweight Haar wavelet high-frequency expert.

    This expert is intentionally light. It does not reconstruct full HF details.
    It only uses LH/HL/HH directional energy as a compact high-frequency hint,
    then projects it into a prior candidate controlled by the MoE router.
    """

    def __init__(self, deg_ch: int, hidden_ch: int = 8, zero_init: bool = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, hidden_ch, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden_ch, hidden_ch, 3, 1, 1, groups=hidden_ch),
            nn.GELU(),
            nn.Conv2d(hidden_ch, deg_ch, 1),
        )
        if zero_init:
            last = self.net[-1]
            nn.init.zeros_(last.weight)
            if last.bias is not None:
                nn.init.zeros_(last.bias)

    @staticmethod
    def _pad_even(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        h, w = x.shape[-2:]
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x, (pad_h, pad_w)

    @staticmethod
    def _crop_back(x: torch.Tensor, pad_hw: Tuple[int, int]) -> torch.Tensor:
        pad_h, pad_w = pad_hw
        if pad_h:
            x = x[..., :-pad_h, :]
        if pad_w:
            x = x[..., :, :-pad_w]
        return x

    def _haar_hf_energy(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x_pad, pad_hw = self._pad_even(x)
        x00 = x_pad[:, :, 0::2, 0::2]
        x01 = x_pad[:, :, 0::2, 1::2]
        x10 = x_pad[:, :, 1::2, 0::2]
        x11 = x_pad[:, :, 1::2, 1::2]

        lh = (x00 + x01 - x10 - x11) * 0.5
        hl = (x00 - x01 + x10 - x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5

        lh_e = lh.abs().mean(dim=1, keepdim=True)
        hl_e = hl.abs().mean(dim=1, keepdim=True)
        hh_e = hh.abs().mean(dim=1, keepdim=True)

        hf = torch.cat([lh_e, hl_e, hh_e], dim=1)
        hf = F.interpolate(hf, scale_factor=2, mode="bilinear", align_corners=False)
        hf = self._crop_back(hf, pad_hw)

        monitor = {
            "wavelet_hf_lh_energy_mean": _safe_mean(hf[:, 0:1]),
            "wavelet_hf_hl_energy_mean": _safe_mean(hf[:, 1:2]),
            "wavelet_hf_hh_energy_mean": _safe_mean(hf[:, 2:3]),
            "wavelet_hf_light_energy_mean": _safe_mean(hf),
            "wavelet_hf_light_energy_std": _safe_std(hf),
        }
        return hf, monitor

    def forward(self, deg_feat: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        hf, monitor = self._haar_hf_energy(deg_feat)
        out = self.net(hf)
        return out, monitor


class Restormer_GlobalPSFScaleRetinexWaveletLLHFMoE(Restormer_GlobalPSFScaleRetinexWaveletMoE):
    """
    Strict global-coordinate PSF-like DegField with six router-controlled experts:

        1. small-scale expert
        2. middle-scale expert
        3. large-scale expert
        4. Retinex-style illumination expert
        5. Wavelet-LL low-frequency expert
        6. Wavelet-HF-light directional energy expert
    """

    def __init__(
        self,
        *args,
        scale_router_temperature: float = 1.0,
        scale_expert_zero_init: bool = True,
        scale_expert_residual_scale: float = 0.25,
        illum_hidden_ch: int = 16,
        wavelet_hidden_ch: int = 16,
        wavelet_hf_hidden_ch: int = 8,
        **kwargs,
    ):
        _GlobalPSFMoEBase.__init__(self, *args, **kwargs)

        self.scale_router_temperature = float(scale_router_temperature)
        self.scale_expert_residual_scale = float(scale_expert_residual_scale)
        self.scale_expert_names = ["small", "middle", "large", "illumination", "wavelet_ll", "wavelet_hf"]

        self.scale_small_expert = _DepthwiseResidualExpert(self.deg_ch, dilation=1, zero_init=scale_expert_zero_init)
        self.scale_middle_expert = _DepthwiseResidualExpert(self.deg_ch, dilation=2, zero_init=scale_expert_zero_init)
        self.scale_large_expert = _LargeScaleResidualExpert(self.deg_ch, dilation=4, zero_init=scale_expert_zero_init)
        self.scale_illum_expert = _RetinexIlluminationExpert(
            self.deg_ch,
            hidden_ch=illum_hidden_ch,
            zero_init=scale_expert_zero_init,
        )
        self.scale_wavelet_ll_expert = _HaarWaveletLLExpert(
            self.deg_ch,
            hidden_ch=wavelet_hidden_ch,
            zero_init=scale_expert_zero_init,
        )
        self.scale_wavelet_hf_expert = _HaarWaveletHFLightExpert(
            self.deg_ch,
            hidden_ch=wavelet_hf_hidden_ch,
            zero_init=scale_expert_zero_init,
        )

        router_in_ch = self.deg_ch + (self.coord_channels if self.use_coord else 0)
        self.scale_router = _RouterHead(router_in_ch, num_experts=6)

        self.last_scale_router_prob: Optional[torch.Tensor] = None
        self.last_scale_prior: Optional[torch.Tensor] = None

    def _mix_prior(self, x_input: torch.Tensor, deg_feat: torch.Tensor, coord_map: torch.Tensor, monitor: Dict[str, torch.Tensor]) -> torch.Tensor:
        coord_low = self._prepare_router_coord(coord_map, deg_feat.shape[-2:]) if coord_map is not None else None
        router_in = torch.cat([deg_feat, coord_low], dim=1) if coord_low is not None else deg_feat
        logits = self.scale_router(router_in) / max(self.scale_router_temperature, 1e-6)
        prob = torch.softmax(logits, dim=1)

        small = self.scale_small_expert(deg_feat)
        middle = self.scale_middle_expert(deg_feat)
        large = self.scale_large_expert(deg_feat)

        illum, illum_mon = self.scale_illum_expert(x_input, deg_feat)
        wavelet_ll, wavelet_ll_mon = self.scale_wavelet_ll_expert(deg_feat)
        wavelet_hf, wavelet_hf_mon = self.scale_wavelet_hf_expert(deg_feat)

        monitor.update(illum_mon)
        monitor.update(wavelet_ll_mon)
        monitor.update(wavelet_hf_mon)

        mix = (
            prob[:, 0:1] * small +
            prob[:, 1:2] * middle +
            prob[:, 2:3] * large +
            prob[:, 3:4] * illum +
            prob[:, 4:5] * wavelet_ll +
            prob[:, 5:6] * wavelet_hf
        )
        prior = deg_feat + self.scale_expert_residual_scale * mix

        self.last_scale_router_prob = prob
        self.last_scale_prior = prior

        monitor["scale_prior_abs_mean"] = _safe_mean(prior.abs())
        monitor["scale_delta_from_deg_abs_mean"] = _safe_mean((prior - deg_feat).abs())

        monitor["illum_prior_abs_mean"] = _safe_mean(illum.abs())
        monitor["illum_prior_tv"] = tv_loss_map(illum).detach()

        # Backward-compatible G5 names refer to the LL expert.
        monitor["wavelet_prior_abs_mean"] = _safe_mean(wavelet_ll.abs())
        monitor["wavelet_prior_tv"] = tv_loss_map(wavelet_ll).detach()
        monitor["wavelet_ll_prior_abs_mean"] = _safe_mean(wavelet_ll.abs())
        monitor["wavelet_ll_prior_tv"] = tv_loss_map(wavelet_ll).detach()

        monitor["wavelet_hf_prior_abs_mean"] = _safe_mean(wavelet_hf.abs())
        monitor["wavelet_hf_prior_tv"] = tv_loss_map(wavelet_hf).detach()

        self._log_common_router_stats(monitor, prob, coord_map, prefix="scale", expert_names=self.scale_expert_names)
        return prior


# Helpful aliases for the 6-expert Retinex + Wavelet-LL/HF-light variant.
Restormer_GlobalPSFRetinexWaveletLLHFMoE = Restormer_GlobalPSFScaleRetinexWaveletLLHFMoE
Restormer_GlobalPSFScaleIllumWaveletLLHFMoE = Restormer_GlobalPSFScaleRetinexWaveletLLHFMoE
Restormer_GlobalPSFScaleRetinexWaveletHFMoE = Restormer_GlobalPSFScaleRetinexWaveletLLHFMoE
Restormer_GlobalPSFScaleRetinexWaveletLLHFMoEDegField = Restormer_GlobalPSFScaleRetinexWaveletLLHFMoE


# ============================================================
# Quick smoke test
# ============================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x = torch.randn(1, 3, 128, 128, device=device)
    yy = torch.linspace(-1.0, 1.0, 128, device=device).view(1, 1, 128, 1).expand(1, 1, 128, 128)
    xx = torch.linspace(-1.0, 1.0, 128, device=device).view(1, 1, 1, 128).expand(1, 1, 128, 128)
    coord = torch.cat([xx, yy], dim=1)

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
        use_coord=True,
        coord_channels=4,
        deg_ch=8,
        deg_downsample_factor=4,
        modulate_levels=("shallow", "enc", "latent", "dec"),
    )

    print("Testing Restormer_GlobalPSFLikeDegField...")
    m1 = Restormer_GlobalPSFLikeDegField(**common_args).to(device)
    y1, aux1 = m1(x, coord=coord, return_aux=True)
    print("Output:", tuple(y1.shape), "aux keys:", list(aux1.keys())[:8])

    print("Testing Restormer_GlobalPSFScaleMoE...")
    m2 = Restormer_GlobalPSFScaleMoE(**common_args).to(device)
    y2, aux2 = m2(x, coord=coord, return_aux=True)
    print("Output:", tuple(y2.shape), "aux keys:", list(aux2.keys())[:12])

    print("Testing Restormer_GlobalPSFScaleRetinexMoE...")
    m3 = Restormer_GlobalPSFScaleRetinexMoE(**common_args).to(device)
    y3, aux3 = m3(x, coord=coord, return_aux=True)
    print("Output:", tuple(y3.shape), "aux keys:", list(aux3.keys())[:14])

    print("Testing Restormer_GlobalPSFScaleRetinexWaveletMoE...")
    m4 = Restormer_GlobalPSFScaleRetinexWaveletMoE(**common_args).to(device)
    y4, aux4 = m4(x, coord=coord, return_aux=True)
    print("Output:", tuple(y4.shape), "aux keys:", list(aux4.keys())[:16])

    print("Testing Restormer_GlobalPSFScaleRetinexWaveletLLHFMoE...")
    m5 = Restormer_GlobalPSFScaleRetinexWaveletLLHFMoE(**common_args).to(device)
    y5, aux5 = m5(x, coord=coord, return_aux=True)
    print("Output:", tuple(y5.shape), "aux keys:", list(aux5.keys())[:18])
