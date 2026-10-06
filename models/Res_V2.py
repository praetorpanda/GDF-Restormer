# -*- coding: utf-8 -*-
"""
Res_V2
======

保存路径建议：
    models/Res_V2.py

本文件在当前最稳定的 Restormer_PSFLikeDegField V1 基础上，集中保存后续更值得尝试的
frequency-aware / residual-aware 变体。设计目标是：

1. 不修改 Res_Psfbasis.py；
2. 尽量继承并复用 Restormer_PSFLikeDegField 的主干、PSF-like deg field、monitor 接口；
3. 每个新结构都兼容：
       pred = model(inp)
       pred, mon = model(inp, return_monitor=True)
4. 后续 engine/run 只需替换 model_class 和 model_args 即可测试。

包含结构：
    1) Restormer_PSFResidualFreqCorrection
       V1 coarse output + residual-frequency correction branch

    2) Restormer_PSFGuidedDynamicFreqSelect
       PSF-guided dynamic frequency selection, 不再固定 LH/HL/HH 权重

    3) Restormer_PSFLatentFFTGate
       latent feature 上的 lightweight FFT spectral gate

    4) Restormer_PSFBlockDCTLocalGate
       block-DCT local frequency energy gate

    5) Restormer_PSFGaborOrientationGate
       Gabor orientation-energy gate, 用于方向性边缘/像差建模

注意：
    这些结构都属于“值得跑的候选结构”。建议优先顺序：
        ResidualFreqCorrection -> DynamicFreqSelect -> LatentFFTGate -> BlockDCT -> Gabor
"""

from typing import Dict, Optional, Sequence, Tuple, Union, List
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .Res_Psfbasis import (
        Restormer_PSFLikeDegField,
        FeatureAffineModulation,
        tv_loss_map,
        radial_correlation_map,
        edge_center_gap,
        _safe_mean,
        _safe_std,
        maybe_debug_tensor,
    )
except Exception:
    from Res_Psfbasis import (
        Restormer_PSFLikeDegField,
        FeatureAffineModulation,
        tv_loss_map,
        radial_correlation_map,
        edge_center_gap,
        _safe_mean,
        _safe_std,
        maybe_debug_tensor,
    )


# ============================================================
# Basic frequency utilities
# ============================================================

class HaarDWT2D(nn.Module):
    """Parameter-free single-level Haar DWT."""

    def __init__(self, pad_mode: str = "reflect"):
        super().__init__()
        if pad_mode not in ("reflect", "replicate"):
            raise ValueError("pad_mode should be 'reflect' or 'replicate'")
        self.pad_mode = pad_mode

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.dim() != 4:
            raise ValueError(f"x must be [B,C,H,W], got {tuple(x.shape)}")
        pad_h = x.shape[-2] % 2
        pad_w = x.shape[-1] % 2
        if pad_h != 0 or pad_w != 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode=self.pad_mode)

        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]

        ll = (x00 + x01 + x10 + x11) * 0.5
        lh = (-x00 - x01 + x10 + x11) * 0.5
        hl = (-x00 + x01 - x10 + x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5
        return ll, lh, hl, hh


def _zero_init_last_conv(module: nn.Module) -> None:
    for m in reversed(list(module.modules())):
        if isinstance(m, nn.Conv2d):
            nn.init.zeros_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
            return


def _bounded_scalar(param: nn.Parameter, scale: float) -> torch.Tensor:
    return float(scale) * torch.tanh(param)


class SmallConvBranch(nn.Module):
    """Small CNN used by several side branches."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        mid_ch: int = 32,
        zero_init: bool = True,
        depth: int = 2,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        ch = in_ch
        for _ in range(max(1, int(depth))):
            layers += [nn.Conv2d(ch, mid_ch, 3, 1, 1), nn.GELU()]
            ch = mid_ch
        layers += [nn.Conv2d(ch, out_ch, 3, 1, 1)]
        self.net = nn.Sequential(*layers)
        if zero_init:
            _zero_init_last_conv(self.net)

    def forward(self, x: torch.Tensor, target_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        y = self.net(x)
        if target_size is not None and y.shape[-2:] != target_size:
            y = F.interpolate(y, size=target_size, mode="bilinear", align_corners=False)
        return y


class LaplacianHighPass(nn.Module):
    """Lightweight high-pass extractor based on Gaussian-like local averaging."""

    def __init__(self, kernel_size: int = 5):
        super().__init__()
        if kernel_size not in (3, 5, 7):
            raise ValueError("kernel_size should be 3, 5, or 7")
        self.kernel_size = int(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low = F.avg_pool2d(x, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2)
        return x - low


# ============================================================
# V1 backbone reuse mixin
# ============================================================

class _V1BackboneForwardMixin:
    """
    Helper methods for subclasses that need to reuse V1 backbone but replace stage priors.
    """

    def _compute_deg_feat(
        self,
        x: torch.Tensor,
        coord: Optional[torch.Tensor],
        return_monitor: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        monitor: Dict[str, torch.Tensor] = {}
        coord_map = self._prepare_coord(x, coord)
        self.last_coord_map = coord_map

        if self.use_coord:
            deg_in = torch.cat([x, coord_map], dim=1)
        else:
            deg_in = x

        deg_feat, deg_score = self.deg_estimator(deg_in)
        self.last_deg_feat = deg_feat
        self.last_deg_score = deg_score

        if return_monitor:
            monitor["deg_score_mean"] = _safe_mean(deg_score)
            monitor["deg_score_std"] = _safe_std(deg_score)
            monitor["deg_feat_abs_mean"] = _safe_mean(deg_feat.abs())
            monitor["deg_field_tv"] = self.deg_smoothness_loss().detach()
            if coord_map is not None:
                monitor["deg_score_edge_center_gap"] = edge_center_gap(deg_score, coord_map)
                monitor["deg_score_radial_corr"] = radial_correlation_map(deg_score, coord_map)

        return deg_feat, deg_score, coord_map, monitor

    def _forward_backbone_with_stage_priors(
        self,
        x_input: torch.Tensor,
        prior_default: torch.Tensor,
        *,
        return_monitor: bool = False,
        monitor: Optional[Dict[str, torch.Tensor]] = None,
        prior_shallow: Optional[torch.Tensor] = None,
        prior_enc: Optional[torch.Tensor] = None,
        prior_latent: Optional[torch.Tensor] = None,
        prior_dec: Optional[torch.Tensor] = None,
        latent_hook=None,
    ):
        if monitor is None:
            monitor = {}

        prior_shallow = prior_default if prior_shallow is None else prior_shallow
        prior_enc = prior_default if prior_enc is None else prior_enc
        prior_latent = prior_default if prior_latent is None else prior_latent
        prior_dec = prior_default if prior_dec is None else prior_dec

        feats = []
        x = self.patch_embed(x_input)

        if self.mod_shallow is not None:
            if return_monitor:
                x, st = self.mod_shallow(x, prior_shallow, return_stats=True, name="shallow_")
                monitor.update(st)
            else:
                x = self.mod_shallow(x, prior_shallow)

        for i in range(self.num_levels):
            if self.mod_encoders is not None:
                if return_monitor:
                    x, st = self.mod_encoders[i](x, prior_enc, return_stats=True, name=f"enc{i+1}_")
                    monitor.update(st)
                else:
                    x = self.mod_encoders[i](x, prior_enc)

            x = self.encoders[i](x)
            feats.append(x)

            if i < self.num_levels - 1:
                x = self.downsamples[i](x)

        if self.mod_latent is not None:
            if return_monitor:
                x, st = self.mod_latent(x, prior_latent, return_stats=True, name="latent_")
                monitor.update(st)
            else:
                x = self.mod_latent(x, prior_latent)

        if latent_hook is not None:
            x, hook_stats = latent_hook(x)
            if return_monitor and hook_stats:
                monitor.update(hook_stats)

        x = self.latent(x)
        maybe_debug_tensor("latent", x)

        dec_idx = 0
        for i in range(self.num_levels - 1)[::-1]:
            x = self.upsamples[self.num_levels - 2 - i](x)
            x = torch.cat([x, feats[i]], dim=1)
            x = self.reduce_chans[self.num_levels - 2 - i](x)

            if self.mod_decoders is not None:
                if return_monitor:
                    x, st = self.mod_decoders[dec_idx](x, prior_dec, return_stats=True, name=f"dec{dec_idx+1}_")
                    monitor.update(st)
                else:
                    x = self.mod_decoders[dec_idx](x, prior_dec)

            x = self.decoders[self.num_levels - 2 - i](x)
            dec_idx += 1

        x = self.refinement(x)
        x_out = self.output(x)
        out = x_out + x_input

        if return_monitor:
            monitor["output_residual_abs_mean"] = _safe_mean((out - x_input).abs())
            self.last_monitor = monitor
            return out, monitor

        self.last_monitor = monitor
        return out


# ============================================================
# 1) Residual-frequency correction branch
# ============================================================

class ResidualFrequencyCorrectionBranch(nn.Module):
    """
    Correction branch applied after a V1 coarse output.

    Input contains:
        x, coarse, residual_hint=(x-coarse), highpass(residual_hint)
    Output:
        RGB correction map.
    """

    def __init__(
        self,
        inp_channels: int = 3,
        mid_ch: int = 32,
        zero_init: bool = True,
        highpass_kernel: int = 5,
    ):
        super().__init__()
        self.highpass = LaplacianHighPass(kernel_size=highpass_kernel)
        in_ch = inp_channels * 4
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(mid_ch, mid_ch, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(mid_ch, inp_channels, 3, 1, 1),
        )
        if zero_init:
            _zero_init_last_conv(self.net)

    def forward(self, x: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
        residual_hint = x - coarse
        hp = self.highpass(residual_hint)
        return self.net(torch.cat([x, coarse, residual_hint, hp], dim=1))


class Restormer_PSFResidualFreqCorrection(Restormer_PSFLikeDegField):
    """
    V1 + residual-frequency correction branch.

    流程：
        coarse = V1(input)
        correction = FreqCorrection(input, coarse, input-coarse)
        out = coarse + beta * correction

    这是当前最建议优先尝试的结构，因为它建模的是“V1 尚未恢复的频率残差”，
    而不是直接从 input 高频中抽取纹理。
    """

    def __init__(
        self,
        *args,
        corr_mid_ch: int = 32,
        corr_beta_init: float = 0.30,
        corr_beta_scale: float = 0.10,
        corr_zero_init: bool = True,
        corr_highpass_kernel: int = 5,
        **kwargs,
    ):
        inp_channels = int(kwargs.get("inp_channels", 3))
        super().__init__(*args, **kwargs)
        self.corr_beta = nn.Parameter(torch.tensor(float(corr_beta_init)))
        self.corr_beta_scale = float(corr_beta_scale)
        self.correction_branch = ResidualFrequencyCorrectionBranch(
            inp_channels=inp_channels,
            mid_ch=corr_mid_ch,
            zero_init=corr_zero_init,
            highpass_kernel=corr_highpass_kernel,
        )
        self.last_correction: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, coord: Optional[torch.Tensor] = None, return_monitor: bool = False):
        if return_monitor:
            coarse, monitor = super().forward(x, coord=coord, return_monitor=True)
        else:
            coarse = super().forward(x, coord=coord, return_monitor=False)
            monitor = {}

        correction = self.correction_branch(x, coarse)
        beta = _bounded_scalar(self.corr_beta, self.corr_beta_scale).to(device=x.device, dtype=x.dtype)
        out = coarse + beta * correction
        self.last_correction = correction

        if return_monitor:
            monitor["corr_beta_raw"] = self.corr_beta.detach().float()
            monitor["corr_beta_effective"] = beta.detach().float()
            monitor["corr_abs_mean"] = _safe_mean(correction.abs())
            monitor["corr_std"] = _safe_std(correction)
            monitor["corr_tv"] = tv_loss_map(correction).detach()
            monitor["coarse_residual_abs_mean"] = _safe_mean((coarse - x).abs())
            monitor["output_residual_abs_mean"] = _safe_mean((out - x).abs())
            self.last_monitor = monitor
            return out, monitor

        self.last_monitor = monitor
        return out

    def correction_smoothness_loss(self) -> torch.Tensor:
        if self.last_correction is None:
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device)
        return tv_loss_map(self.last_correction)


# ============================================================
# 2) PSF-guided dynamic frequency selection
# ============================================================

class PSFGuidedDynamicDWTSelect(nn.Module):
    """
    DWT high-frequency branch with dynamic LH/HL/HH weights guided by deg_feat.
    """

    def __init__(
        self,
        inp_channels: int = 3,
        deg_ch: int = 16,
        out_ch: int = 16,
        mid_ch: int = 32,
        pad_mode: str = "reflect",
        zero_init: bool = True,
    ):
        super().__init__()
        self.dwt = HaarDWT2D(pad_mode=pad_mode)
        self.band_proj = nn.Sequential(
            nn.Conv2d(inp_channels * 3, mid_ch, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(mid_ch, out_ch, 3, 1, 1),
        )
        self.weight_net = nn.Sequential(
            nn.Conv2d(deg_ch, mid_ch, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(mid_ch, 3, 1, 1, 0),
        )
        if zero_init:
            _zero_init_last_conv(self.band_proj)

        self.last_weights: Optional[torch.Tensor] = None
        self.last_bands: Dict[str, torch.Tensor] = {}

    def forward(self, x: torch.Tensor, deg_feat: torch.Tensor) -> torch.Tensor:
        ll, lh, hl, hh = self.dwt(x)
        if deg_feat.shape[-2:] != lh.shape[-2:]:
            deg_small = F.interpolate(deg_feat, size=lh.shape[-2:], mode="bilinear", align_corners=False)
        else:
            deg_small = deg_feat

        logits = self.weight_net(deg_small)
        weights = torch.softmax(logits, dim=1)
        selected = torch.cat([
            weights[:, 0:1] * lh,
            weights[:, 1:2] * hl,
            weights[:, 2:3] * hh,
        ], dim=1)
        freq_feat = self.band_proj(selected)
        if freq_feat.shape[-2:] != deg_feat.shape[-2:]:
            freq_feat = F.interpolate(freq_feat, size=deg_feat.shape[-2:], mode="bilinear", align_corners=False)

        self.last_weights = weights
        self.last_bands = {"ll": ll.detach(), "lh": lh.detach(), "hl": hl.detach(), "hh": hh.detach()}
        return freq_feat

    def band_energy_dict(self) -> Dict[str, torch.Tensor]:
        if not self.last_bands:
            device = next(self.parameters()).device
            zero = torch.tensor(0.0, device=device)
            return {"ll": zero, "lh": zero, "hl": zero, "hh": zero}
        return {k: v.float().abs().mean() for k, v in self.last_bands.items()}


class Restormer_PSFGuidedDynamicFreqSelect(_V1BackboneForwardMixin, Restormer_PSFLikeDegField):
    """V1 + PSF-guided dynamic LH/HL/HH frequency selection."""

    def __init__(
        self,
        *args,
        dyn_freq_mid_ch: int = 32,
        dyn_freq_alpha_init: float = 0.30,
        dyn_freq_alpha_scale: float = 0.10,
        dyn_freq_zero_init: bool = True,
        dyn_freq_apply_shallow: bool = True,
        dyn_freq_apply_encoder: bool = False,
        dyn_freq_apply_latent: bool = False,
        dyn_freq_apply_decoder: bool = True,
        dyn_freq_pad_mode: str = "reflect",
        **kwargs,
    ):
        inp_channels = int(kwargs.get("inp_channels", 3))
        super().__init__(*args, **kwargs)
        self.dyn_freq_alpha = nn.Parameter(torch.tensor(float(dyn_freq_alpha_init)))
        self.dyn_freq_alpha_scale = float(dyn_freq_alpha_scale)
        self.dyn_freq_apply_shallow = bool(dyn_freq_apply_shallow)
        self.dyn_freq_apply_encoder = bool(dyn_freq_apply_encoder)
        self.dyn_freq_apply_latent = bool(dyn_freq_apply_latent)
        self.dyn_freq_apply_decoder = bool(dyn_freq_apply_decoder)
        self.dynamic_freq = PSFGuidedDynamicDWTSelect(
            inp_channels=inp_channels,
            deg_ch=self.deg_ch,
            out_ch=self.deg_ch,
            mid_ch=dyn_freq_mid_ch,
            pad_mode=dyn_freq_pad_mode,
            zero_init=dyn_freq_zero_init,
        )
        self.last_freq_feat: Optional[torch.Tensor] = None
        self.last_deg_prior: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, coord: Optional[torch.Tensor] = None, return_monitor: bool = False):
        deg_feat, deg_score, coord_map, monitor = self._compute_deg_feat(x, coord, return_monitor)
        freq_feat = self.dynamic_freq(x, deg_feat)
        alpha = _bounded_scalar(self.dyn_freq_alpha, self.dyn_freq_alpha_scale).to(device=x.device, dtype=x.dtype)
        deg_freq_prior = deg_feat + alpha * freq_feat

        self.last_freq_feat = freq_feat
        self.last_deg_prior = deg_freq_prior

        if return_monitor:
            monitor["deg_prior_abs_mean"] = _safe_mean(deg_freq_prior.abs())
            monitor["freq_feat_abs_mean"] = _safe_mean(freq_feat.abs())
            monitor["freq_feat_std"] = _safe_std(freq_feat)
            monitor["freq_alpha_raw"] = self.dyn_freq_alpha.detach().float()
            monitor["freq_alpha_effective"] = alpha.detach().float()
            monitor["freq_prior_tv"] = tv_loss_map(freq_feat).detach()
            if self.dynamic_freq.last_weights is not None:
                w = self.dynamic_freq.last_weights.detach().float()
                monitor["dyn_w_lh_mean"] = w[:, 0:1].mean()
                monitor["dyn_w_hl_mean"] = w[:, 1:2].mean()
                monitor["dyn_w_hh_mean"] = w[:, 2:3].mean()
                monitor["dyn_w_std"] = w.std(unbiased=False)
            bands = self.dynamic_freq.band_energy_dict()
            monitor["dwt_lh_abs_mean"] = bands["lh"].detach()
            monitor["dwt_hl_abs_mean"] = bands["hl"].detach()
            monitor["dwt_hh_abs_mean"] = bands["hh"].detach()
            monitor["dwt_hf_abs_mean"] = (bands["lh"] + bands["hl"] + bands["hh"]).detach()

        prior_shallow = deg_freq_prior if self.dyn_freq_apply_shallow else deg_feat
        prior_enc = deg_freq_prior if self.dyn_freq_apply_encoder else deg_feat
        prior_latent = deg_freq_prior if self.dyn_freq_apply_latent else deg_feat
        prior_dec = deg_freq_prior if self.dyn_freq_apply_decoder else deg_feat

        return self._forward_backbone_with_stage_priors(
            x,
            deg_feat,
            return_monitor=return_monitor,
            monitor=monitor,
            prior_shallow=prior_shallow,
            prior_enc=prior_enc,
            prior_latent=prior_latent,
            prior_dec=prior_dec,
        )

    def freq_smoothness_loss(self) -> torch.Tensor:
        if self.last_freq_feat is None:
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device)
        return tv_loss_map(self.last_freq_feat)


# ============================================================
# 3) latent FFT gate
# ============================================================

class LatentFFTGate(nn.Module):
    """Lightweight spectral gate on latent feature."""

    def __init__(self, channels: int, hidden_ch: int = 32, zero_init: bool = True):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels, hidden_ch, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(hidden_ch, channels, 1, 1, 0),
        )
        if zero_init:
            _zero_init_last_conv(self.gate)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # Use log magnitude as stable spectral descriptor.
        fft = torch.fft.rfft2(feat.float(), norm="ortho")
        mag = torch.log1p(torch.abs(fft))
        # Convert a compact global spectral descriptor to channel gates.
        desc = mag.mean(dim=(-2, -1), keepdim=True).to(dtype=feat.dtype)
        gate = torch.tanh(self.gate(desc))
        return feat * (1.0 + gate), gate


class Restormer_PSFLatentFFTGate(_V1BackboneForwardMixin, Restormer_PSFLikeDegField):
    """V1 + latent FFT spectral gate."""

    def __init__(
        self,
        *args,
        fft_hidden_ch: int = 32,
        fft_beta_init: float = 0.30,
        fft_beta_scale: float = 0.10,
        fft_zero_init: bool = True,
        **kwargs,
    ):
        dim = int(kwargs.get("dim", 48))
        num_blocks = kwargs.get("num_blocks", (4, 6))
        latent_ch = dim * (2 ** (len(num_blocks) - 1))
        super().__init__(*args, **kwargs)
        self.fft_beta = nn.Parameter(torch.tensor(float(fft_beta_init)))
        self.fft_beta_scale = float(fft_beta_scale)
        self.latent_fft_gate = LatentFFTGate(latent_ch, hidden_ch=fft_hidden_ch, zero_init=fft_zero_init)
        self.last_fft_gate: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, coord: Optional[torch.Tensor] = None, return_monitor: bool = False):
        deg_feat, deg_score, coord_map, monitor = self._compute_deg_feat(x, coord, return_monitor)

        def hook(latent_feat: torch.Tensor):
            gated, gate = self.latent_fft_gate(latent_feat)
            beta = _bounded_scalar(self.fft_beta, self.fft_beta_scale).to(device=latent_feat.device, dtype=latent_feat.dtype)
            out = latent_feat + beta * (gated - latent_feat)
            self.last_fft_gate = gate
            stats = {
                "fft_beta_raw": self.fft_beta.detach().float(),
                "fft_beta_effective": beta.detach().float(),
                "fft_gate_abs_mean": _safe_mean(gate.abs()),
                "fft_gate_std": _safe_std(gate),
            }
            return out, stats

        return self._forward_backbone_with_stage_priors(
            x,
            deg_feat,
            return_monitor=return_monitor,
            monitor=monitor,
            latent_hook=hook,
        )


# ============================================================
# 4) block-DCT local frequency gate
# ============================================================

class BlockDCTEnergyPrior(nn.Module):
    """
    Fixed block-DCT filters -> local frequency energy -> deg_ch prior.

    This keeps locality better than global FFT and is more flexible than DWT LH/HL/HH.
    """

    def __init__(
        self,
        inp_channels: int = 3,
        out_ch: int = 16,
        block_size: int = 8,
        selected_freqs: Optional[Sequence[Tuple[int, int]]] = None,
        mid_ch: int = 32,
        zero_init: bool = True,
    ):
        super().__init__()
        self.inp_channels = int(inp_channels)
        self.block_size = int(block_size)
        if selected_freqs is None:
            # mid/high DCT frequencies, avoid pure DC.
            selected_freqs = [(0, 2), (2, 0), (1, 2), (2, 1), (2, 2), (3, 1), (1, 3), (3, 3)]
        self.selected_freqs = list(selected_freqs)
        weight = self._make_dct_kernels(self.block_size, self.selected_freqs)
        # [K,1,B,B], repeated at runtime for grouped conv.
        self.register_buffer("dct_weight", weight, persistent=False)
        self.energy_to_prior = SmallConvBranch(
            in_ch=len(self.selected_freqs),
            out_ch=out_ch,
            mid_ch=mid_ch,
            zero_init=zero_init,
            depth=2,
        )
        self.last_energy: Optional[torch.Tensor] = None

    @staticmethod
    def _make_dct_kernels(block_size: int, freqs: Sequence[Tuple[int, int]]) -> torch.Tensor:
        n = int(block_size)
        yy = torch.arange(n).float().view(n, 1)
        xx = torch.arange(n).float().view(1, n)
        kernels = []
        for u, v in freqs:
            au = math.sqrt(1.0 / n) if u == 0 else math.sqrt(2.0 / n)
            av = math.sqrt(1.0 / n) if v == 0 else math.sqrt(2.0 / n)
            ky = torch.cos(math.pi * (2 * yy + 1) * u / (2 * n))
            kx = torch.cos(math.pi * (2 * xx + 1) * v / (2 * n))
            k = au * av * ky * kx
            kernels.append(k.view(1, 1, n, n))
        return torch.cat(kernels, dim=0)

    def forward(self, x: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        b, c, h, w = x.shape
        bs = self.block_size
        pad_h = (bs - h % bs) % bs
        pad_w = (bs - w % bs) % bs
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        k = self.dct_weight.to(device=x.device, dtype=x.dtype)
        num_k = k.shape[0]
        weight = k.repeat(c, 1, 1, 1)
        # [B, C*K, H/bs, W/bs]
        coeff = F.conv2d(x, weight, stride=bs, padding=0, groups=c)
        coeff = coeff.view(b, c, num_k, coeff.shape[-2], coeff.shape[-1])
        energy = coeff.abs().mean(dim=1)  # [B,K,hb,wb]
        self.last_energy = energy
        prior = self.energy_to_prior(energy, target_size=target_size)
        return prior


class Restormer_PSFBlockDCTLocalGate(_V1BackboneForwardMixin, Restormer_PSFLikeDegField):
    """V1 + block-DCT local frequency prior/gate."""

    def __init__(
        self,
        *args,
        dct_block_size: int = 8,
        dct_mid_ch: int = 32,
        dct_alpha_init: float = 0.30,
        dct_alpha_scale: float = 0.10,
        dct_zero_init: bool = True,
        dct_apply_shallow: bool = True,
        dct_apply_encoder: bool = False,
        dct_apply_latent: bool = False,
        dct_apply_decoder: bool = True,
        **kwargs,
    ):
        inp_channels = int(kwargs.get("inp_channels", 3))
        super().__init__(*args, **kwargs)
        self.dct_alpha = nn.Parameter(torch.tensor(float(dct_alpha_init)))
        self.dct_alpha_scale = float(dct_alpha_scale)
        self.dct_apply_shallow = bool(dct_apply_shallow)
        self.dct_apply_encoder = bool(dct_apply_encoder)
        self.dct_apply_latent = bool(dct_apply_latent)
        self.dct_apply_decoder = bool(dct_apply_decoder)
        self.dct_prior = BlockDCTEnergyPrior(
            inp_channels=inp_channels,
            out_ch=self.deg_ch,
            block_size=dct_block_size,
            mid_ch=dct_mid_ch,
            zero_init=dct_zero_init,
        )
        self.last_dct_prior: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, coord: Optional[torch.Tensor] = None, return_monitor: bool = False):
        deg_feat, deg_score, coord_map, monitor = self._compute_deg_feat(x, coord, return_monitor)
        dct_feat = self.dct_prior(x, target_size=deg_feat.shape[-2:])
        alpha = _bounded_scalar(self.dct_alpha, self.dct_alpha_scale).to(device=x.device, dtype=x.dtype)
        deg_dct_prior = deg_feat + alpha * dct_feat
        self.last_dct_prior = dct_feat

        if return_monitor:
            monitor["dct_alpha_effective"] = alpha.detach().float()
            monitor["dct_feat_abs_mean"] = _safe_mean(dct_feat.abs())
            monitor["dct_feat_std"] = _safe_std(dct_feat)
            monitor["dct_prior_abs_mean"] = _safe_mean(deg_dct_prior.abs())
            if self.dct_prior.last_energy is not None:
                monitor["dct_energy_abs_mean"] = _safe_mean(self.dct_prior.last_energy.abs())
                monitor["dct_energy_std"] = _safe_std(self.dct_prior.last_energy)

        return self._forward_backbone_with_stage_priors(
            x,
            deg_feat,
            return_monitor=return_monitor,
            monitor=monitor,
            prior_shallow=deg_dct_prior if self.dct_apply_shallow else deg_feat,
            prior_enc=deg_dct_prior if self.dct_apply_encoder else deg_feat,
            prior_latent=deg_dct_prior if self.dct_apply_latent else deg_feat,
            prior_dec=deg_dct_prior if self.dct_apply_decoder else deg_feat,
        )

    def freq_smoothness_loss(self) -> torch.Tensor:
        if self.last_dct_prior is None:
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device)
        return tv_loss_map(self.last_dct_prior)


# ============================================================
# 5) Gabor orientation-energy gate
# ============================================================

class GaborOrientationEnergyPrior(nn.Module):
    """Fixed Gabor filter bank -> orientation energy -> deg_ch prior."""

    def __init__(
        self,
        inp_channels: int = 3,
        out_ch: int = 16,
        kernel_size: int = 15,
        sigmas: Sequence[float] = (2.0, 4.0),
        orientations: int = 8,
        mid_ch: int = 32,
        zero_init: bool = True,
    ):
        super().__init__()
        self.inp_channels = int(inp_channels)
        kernels = self._make_gabor_bank(kernel_size, sigmas, orientations)
        self.register_buffer("gabor_weight", kernels, persistent=False)  # [K,1,ks,ks]
        self.energy_to_prior = SmallConvBranch(
            in_ch=kernels.shape[0],
            out_ch=out_ch,
            mid_ch=mid_ch,
            zero_init=zero_init,
            depth=2,
        )
        self.last_energy: Optional[torch.Tensor] = None

    @staticmethod
    def _make_gabor_bank(kernel_size: int, sigmas: Sequence[float], orientations: int) -> torch.Tensor:
        ks = int(kernel_size)
        half = ks // 2
        y, x = torch.meshgrid(
            torch.arange(-half, half + 1).float(),
            torch.arange(-half, half + 1).float(),
            indexing="ij",
        )
        kernels = []
        for sigma in sigmas:
            lambd = max(2.0, float(sigma) * 2.5)
            for i in range(int(orientations)):
                theta = math.pi * i / int(orientations)
                xr = x * math.cos(theta) + y * math.sin(theta)
                yr = -x * math.sin(theta) + y * math.cos(theta)
                g = torch.exp(-(xr ** 2 + yr ** 2) / (2 * float(sigma) ** 2)) * torch.cos(2 * math.pi * xr / lambd)
                g = g - g.mean()
                g = g / (g.abs().sum() + 1e-6)
                kernels.append(g.view(1, 1, ks, ks))
        return torch.cat(kernels, dim=0)

    def forward(self, x: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        b, c, h, w = x.shape
        k = self.gabor_weight.to(device=x.device, dtype=x.dtype)
        num_k = k.shape[0]
        weight = k.repeat(c, 1, 1, 1)
        resp = F.conv2d(x, weight, padding=k.shape[-1] // 2, groups=c)
        resp = resp.view(b, c, num_k, h, w)
        energy = resp.abs().mean(dim=1)  # [B,K,H,W]
        # Downsample before prior projection to reduce content leakage.
        if energy.shape[-2] > target_size[0] or energy.shape[-1] > target_size[1]:
            energy_small = F.interpolate(energy, size=target_size, mode="bilinear", align_corners=False)
        else:
            energy_small = energy
        self.last_energy = energy_small
        return self.energy_to_prior(energy_small, target_size=target_size)


class Restormer_PSFGaborOrientationGate(_V1BackboneForwardMixin, Restormer_PSFLikeDegField):
    """V1 + Gabor orientation-energy prior/gate."""

    def __init__(
        self,
        *args,
        gabor_kernel_size: int = 15,
        gabor_sigmas: Sequence[float] = (2.0, 4.0),
        gabor_orientations: int = 8,
        gabor_mid_ch: int = 32,
        gabor_alpha_init: float = 0.30,
        gabor_alpha_scale: float = 0.10,
        gabor_zero_init: bool = True,
        gabor_apply_shallow: bool = True,
        gabor_apply_encoder: bool = False,
        gabor_apply_latent: bool = False,
        gabor_apply_decoder: bool = True,
        **kwargs,
    ):
        inp_channels = int(kwargs.get("inp_channels", 3))
        super().__init__(*args, **kwargs)
        self.gabor_alpha = nn.Parameter(torch.tensor(float(gabor_alpha_init)))
        self.gabor_alpha_scale = float(gabor_alpha_scale)
        self.gabor_apply_shallow = bool(gabor_apply_shallow)
        self.gabor_apply_encoder = bool(gabor_apply_encoder)
        self.gabor_apply_latent = bool(gabor_apply_latent)
        self.gabor_apply_decoder = bool(gabor_apply_decoder)
        self.gabor_prior = GaborOrientationEnergyPrior(
            inp_channels=inp_channels,
            out_ch=self.deg_ch,
            kernel_size=gabor_kernel_size,
            sigmas=gabor_sigmas,
            orientations=gabor_orientations,
            mid_ch=gabor_mid_ch,
            zero_init=gabor_zero_init,
        )
        self.last_gabor_prior: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, coord: Optional[torch.Tensor] = None, return_monitor: bool = False):
        deg_feat, deg_score, coord_map, monitor = self._compute_deg_feat(x, coord, return_monitor)
        gabor_feat = self.gabor_prior(x, target_size=deg_feat.shape[-2:])
        alpha = _bounded_scalar(self.gabor_alpha, self.gabor_alpha_scale).to(device=x.device, dtype=x.dtype)
        deg_gabor_prior = deg_feat + alpha * gabor_feat
        self.last_gabor_prior = gabor_feat

        if return_monitor:
            monitor["gabor_alpha_effective"] = alpha.detach().float()
            monitor["gabor_feat_abs_mean"] = _safe_mean(gabor_feat.abs())
            monitor["gabor_feat_std"] = _safe_std(gabor_feat)
            monitor["gabor_prior_abs_mean"] = _safe_mean(deg_gabor_prior.abs())
            if self.gabor_prior.last_energy is not None:
                monitor["gabor_energy_abs_mean"] = _safe_mean(self.gabor_prior.last_energy.abs())
                monitor["gabor_energy_std"] = _safe_std(self.gabor_prior.last_energy)

        return self._forward_backbone_with_stage_priors(
            x,
            deg_feat,
            return_monitor=return_monitor,
            monitor=monitor,
            prior_shallow=deg_gabor_prior if self.gabor_apply_shallow else deg_feat,
            prior_enc=deg_gabor_prior if self.gabor_apply_encoder else deg_feat,
            prior_latent=deg_gabor_prior if self.gabor_apply_latent else deg_feat,
            prior_dec=deg_gabor_prior if self.gabor_apply_decoder else deg_feat,
        )

    def freq_smoothness_loss(self) -> torch.Tensor:
        if self.last_gabor_prior is None:
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device)
        return tv_loss_map(self.last_gabor_prior)


# ============================================================
# Backward-friendly aliases
# ============================================================

Restormer_PSFResidualFreq = Restormer_PSFResidualFreqCorrection
Restormer_PSFDynamicFreqSelect = Restormer_PSFGuidedDynamicFreqSelect
Restormer_PSFFFTGate = Restormer_PSFLatentFFTGate
Restormer_PSFBlockDCTGate = Restormer_PSFBlockDCTLocalGate
Restormer_PSFGaborGate = Restormer_PSFGaborOrientationGate


# ============================================================
# Quick smoke test
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
        use_coord=True,
        coord_channels=4,
        deg_ch=8,
        deg_mid_ch=16,
        deg_downsample_factor=4,
        modulate_levels=("shallow", "enc", "latent", "dec"),
    )

    classes = [
        Restormer_PSFResidualFreqCorrection,
        Restormer_PSFGuidedDynamicFreqSelect,
        Restormer_PSFLatentFFTGate,
        Restormer_PSFBlockDCTLocalGate,
        Restormer_PSFGaborOrientationGate,
    ]

    for cls in classes:
        print(f"Testing {cls.__name__}...")
        model = cls(**common_args).to(device)
        y, mon = model(x, return_monitor=True)
        print("  output:", tuple(y.shape), "monitor keys:", list(mon.keys())[:8])
