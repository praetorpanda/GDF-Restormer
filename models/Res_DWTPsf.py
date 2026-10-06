# -*- coding: utf-8 -*-
"""
Res_DWTPsf
==========

保存路径建议：
    models/Res_DWTPsf.py

本文件在现有 PSFLikeDegField V1 的基础上，新增一个轻量 DWT 高频补充分支：

    image + coord -> PSF-like low-resolution deg_feat
    image         -> DWT high/mid-frequency branch -> freq_feat
    deg_prior = deg_feat + alpha * freq_feat
    deg_prior -> 原 V1 的 affine modulation -> Restormer

设计原则：
    1. 不修改旧文件 Res_Psfbasis.py；
    2. 默认 alpha=0，初始行为几乎等价于 V1；
    3. DWT 分支默认只使用 LH/HL，优先补充结构边缘，而不是强行增强 HH 噪声；
    4. 分支输出对齐到 deg_feat 的低分辨率尺寸，避免过强逐像素纹理泄漏；
    5. 兼容原 engine 的 pred = model(inp) 和 pred, mon = model(inp, return_monitor=True)。

推荐第一组消融：
    Restormer_DWTPSFLikeDegField(freq_band_mode="lh_hl")
    Restormer_DWTPSFLikeDegField(freq_band_mode="hf")
    Restormer_DWTPSFLikeDegField(freq_band_mode="ll")
    Restormer_DWTPSFLikeDegField(freq_band_mode="all")

注意：
    本文件需要和 Res_Psfbasis.py 放在同一个 models 目录下。
"""

from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .Res_Psfbasis import (
        Restormer_PSFLikeDegField,
        FeatureAffineModulation,
        make_local_coord,
        normalize_coord_channels,
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
        make_local_coord,
        normalize_coord_channels,
        tv_loss_map,
        radial_correlation_map,
        edge_center_gap,
        _safe_mean,
        _safe_std,
        maybe_debug_tensor,
    )


# ============================================================
# DWT Utilities
# ============================================================

class HaarDWT2D(nn.Module):
    """
    Parameter-free single-level Haar DWT.

    输入:
        x: [B, C, H, W]

    输出:
        ll, lh, hl, hh: each [B, C, ceil(H/2), ceil(W/2)]

    说明：
        - 当 H/W 为奇数时，使用 reflect/replicate padding 到偶数尺寸；
        - 系数使用 0.5 缩放，保持数值范围相对稳定；
        - 命名不强依赖严格物理方向，只用于区分三个高频子带。
    """

    def __init__(self, pad_mode: str = "reflect"):
        super().__init__()
        if pad_mode not in ("reflect", "replicate"):
            raise ValueError("pad_mode should be 'reflect' or 'replicate'")
        self.pad_mode = pad_mode

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.dim() != 4:
            raise ValueError(f"x must be [B,C,H,W], got {tuple(x.shape)}")

        _, _, h, w = x.shape
        pad_h = h % 2
        pad_w = w % 2
        if pad_h != 0 or pad_w != 0:
            # pad format: left, right, top, bottom
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


class TinyDWTFrequencyPrior(nn.Module):
    """
    轻量 DWT frequency prior branch。

    默认使用 LH + HL 两个结构边缘子带，避免 HH 噪声/合成伪影过强。

    freq_band_mode:
        "lh_hl"             -> concat(LH, HL)，推荐第一版
        "hf"                -> concat(LH, HL, HH)
        "ll"                -> LL only
        "all"               -> concat(LL, LH, HL, HH)
        "lh_hl_hh_weighted" -> concat(LH, HL, hh_weight * HH)

    输出:
        freq_feat: [B, out_ch, H0, W0]
    """

    def __init__(
        self,
        inp_channels: int = 3,
        out_ch: int = 16,
        mid_ch: int = 24,
        freq_band_mode: str = "lh_hl",
        hh_weight: float = 0.30,
        pad_mode: str = "reflect",
        zero_init: bool = True,
    ):
        super().__init__()

        valid_modes = {"lh_hl", "hf", "ll", "all", "lh_hl_hh_weighted"}
        if freq_band_mode not in valid_modes:
            raise ValueError(f"freq_band_mode must be one of {valid_modes}, got {freq_band_mode}")

        self.inp_channels = inp_channels
        self.out_ch = out_ch
        self.mid_ch = mid_ch
        self.freq_band_mode = freq_band_mode
        self.hh_weight = float(hh_weight)
        self.dwt = HaarDWT2D(pad_mode=pad_mode)

        if freq_band_mode == "lh_hl":
            band_ch = inp_channels * 2
        elif freq_band_mode == "hf":
            band_ch = inp_channels * 3
        elif freq_band_mode == "ll":
            band_ch = inp_channels
        elif freq_band_mode == "all":
            band_ch = inp_channels * 4
        elif freq_band_mode == "lh_hl_hh_weighted":
            band_ch = inp_channels * 3
        else:
            raise RuntimeError("unreachable")

        self.net = nn.Sequential(
            nn.Conv2d(band_ch, mid_ch, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(mid_ch, mid_ch, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(mid_ch, out_ch, 3, 1, 1),
        )

        # 关键：默认 zero init，使 alpha 非零时也不会一开始破坏 V1；
        # 同时 alpha 默认 0，因此初始严格接近 V1。
        if zero_init:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

        self.last_bands: Dict[str, torch.Tensor] = {}
        self.last_freq_feat: Optional[torch.Tensor] = None
        self.last_hf_energy: Optional[torch.Tensor] = None

    def _select_bands(
        self,
        ll: torch.Tensor,
        lh: torch.Tensor,
        hl: torch.Tensor,
        hh: torch.Tensor,
    ) -> torch.Tensor:
        if self.freq_band_mode == "lh_hl":
            return torch.cat([lh, hl], dim=1)
        if self.freq_band_mode == "hf":
            return torch.cat([lh, hl, hh], dim=1)
        if self.freq_band_mode == "ll":
            return ll
        if self.freq_band_mode == "all":
            return torch.cat([ll, lh, hl, hh], dim=1)
        if self.freq_band_mode == "lh_hl_hh_weighted":
            return torch.cat([lh, hl, self.hh_weight * hh], dim=1)
        raise RuntimeError("unreachable")

    def forward(
        self,
        x: torch.Tensor,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        ll, lh, hl, hh = self.dwt(x)
        bands = self._select_bands(ll, lh, hl, hh)
        freq_feat = self.net(bands)

        if target_size is not None and freq_feat.shape[-2:] != target_size:
            freq_feat = F.interpolate(
                freq_feat,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )

        # monitor cache
        with torch.no_grad():
            self.last_bands = {
                "ll": ll.detach(),
                "lh": lh.detach(),
                "hl": hl.detach(),
                "hh": hh.detach(),
            }
            self.last_hf_energy = (lh.abs().mean() + hl.abs().mean() + hh.abs().mean()).detach()
        self.last_freq_feat = freq_feat
        return freq_feat

    def band_energy_dict(self) -> Dict[str, torch.Tensor]:
        if not self.last_bands:
            device = next(self.parameters()).device
            zero = torch.tensor(0.0, device=device)
            return {"ll": zero, "lh": zero, "hl": zero, "hh": zero}
        return {k: v.float().abs().mean() for k, v in self.last_bands.items()}


# ============================================================
# DWT + PSF-like DegField V1
# ============================================================

class Restormer_DWTPSFLikeDegField(Restormer_PSFLikeDegField):
    """
    PSFLikeDegField V1 + weak DWT frequency prior branch.

    相比 Restormer_PSFLikeDegField：
        - 保留原始 deg_estimator 与所有 modulation 逻辑；
        - 新增 TinyDWTFrequencyPrior；
        - 使用 deg_prior = deg_feat + alpha * freq_feat；
        - alpha 默认 0，初始几乎等价于 V1；
        - 默认使用 LH/HL 子带作为中高频结构先验。

    推荐默认配置：
        freq_band_mode="lh_hl"
        freq_alpha_init=0.0
        freq_alpha_scale=0.10
        freq_zero_init=True
    """

    def __init__(
        self,
        *args,
        freq_branch: bool = True,
        freq_band_mode: str = "lh_hl",
        freq_mid_ch: int = 24,
        freq_alpha_init: float = 0.0,
        freq_alpha_scale: float = 0.10,
        freq_hh_weight: float = 0.30,
        freq_zero_init: bool = True,
        freq_pad_mode: str = "reflect",
        freq_detach_input: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.freq_branch = bool(freq_branch)
        self.freq_band_mode = freq_band_mode
        self.freq_alpha_scale = float(freq_alpha_scale)
        self.freq_detach_input = bool(freq_detach_input)
        self.freq_alpha = nn.Parameter(torch.tensor(float(freq_alpha_init)))

        if self.freq_branch:
            self.freq_prior = TinyDWTFrequencyPrior(
                inp_channels=kwargs.get("inp_channels", 3),
                out_ch=self.deg_ch,
                mid_ch=freq_mid_ch,
                freq_band_mode=freq_band_mode,
                hh_weight=freq_hh_weight,
                pad_mode=freq_pad_mode,
                zero_init=freq_zero_init,
            )
        else:
            self.freq_prior = None

        self.last_freq_feat: Optional[torch.Tensor] = None
        self.last_deg_prior: Optional[torch.Tensor] = None

    def _get_alpha(self) -> torch.Tensor:
        # bounded scalar, default 0.0
        return self.freq_alpha_scale * torch.tanh(self.freq_alpha)

    def forward(
        self,
        x: torch.Tensor,
        coord: Optional[torch.Tensor] = None,
        return_monitor: bool = False,
    ):
        x_input = x
        monitor: Dict[str, torch.Tensor] = {}

        coord_map = self._prepare_coord(x, coord)
        self.last_coord_map = coord_map

        if self.use_coord:
            deg_in = torch.cat([x, coord_map], dim=1)
        else:
            deg_in = x

        deg_feat, deg_score = self.deg_estimator(deg_in)

        # DWT frequency branch: output aligned to deg_feat resolution.
        if self.freq_branch and self.freq_prior is not None:
            freq_input = x.detach() if self.freq_detach_input else x
            freq_feat = self.freq_prior(freq_input, target_size=deg_feat.shape[-2:])
            alpha = self._get_alpha().to(device=x.device, dtype=x.dtype)
            deg_prior = deg_feat + alpha * freq_feat
        else:
            freq_feat = torch.zeros_like(deg_feat)
            alpha = torch.zeros((), device=x.device, dtype=x.dtype)
            deg_prior = deg_feat

        self.last_deg_feat = deg_feat
        self.last_deg_score = deg_score
        self.last_freq_feat = freq_feat
        self.last_deg_prior = deg_prior

        maybe_debug_tensor("dwtpsf_deg_feat", deg_feat)
        maybe_debug_tensor("dwtpsf_deg_score", deg_score)
        maybe_debug_tensor("dwtpsf_freq_feat", freq_feat)
        maybe_debug_tensor("dwtpsf_deg_prior", deg_prior)

        if return_monitor:
            monitor["deg_score_mean"] = _safe_mean(deg_score)
            monitor["deg_score_std"] = _safe_std(deg_score)
            monitor["deg_feat_abs_mean"] = _safe_mean(deg_feat.abs())
            monitor["deg_prior_abs_mean"] = _safe_mean(deg_prior.abs())
            monitor["freq_feat_abs_mean"] = _safe_mean(freq_feat.abs())
            monitor["freq_feat_std"] = _safe_std(freq_feat)
            monitor["freq_alpha_raw"] = self.freq_alpha.detach().float()
            monitor["freq_alpha_tanh"] = torch.tanh(self.freq_alpha.detach().float())
            monitor["freq_alpha_effective"] = alpha.detach().float()
            monitor["deg_field_tv"] = self.deg_smoothness_loss().detach()
            monitor["freq_prior_tv"] = tv_loss_map(freq_feat).detach()

            # branch band energy monitor
            if self.freq_prior is not None:
                band_energy = self.freq_prior.band_energy_dict()
                monitor["dwt_ll_abs_mean"] = band_energy["ll"].detach()
                monitor["dwt_lh_abs_mean"] = band_energy["lh"].detach()
                monitor["dwt_hl_abs_mean"] = band_energy["hl"].detach()
                monitor["dwt_hh_abs_mean"] = band_energy["hh"].detach()
                monitor["dwt_hf_abs_mean"] = (
                    band_energy["lh"] + band_energy["hl"] + band_energy["hh"]
                ).detach()

            # cosine similarity between spatial PSF prior and frequency prior
            with torch.no_grad():
                d = deg_feat.detach().float().flatten(1)
                f = freq_feat.detach().float().flatten(1)
                if d.shape == f.shape:
                    d = F.normalize(d, dim=1)
                    f = F.normalize(f, dim=1)
                    monitor["deg_freq_cos_sim"] = (d * f).sum(dim=1).mean().detach()
                else:
                    monitor["deg_freq_cos_sim"] = torch.tensor(0.0, device=x.device)

            if coord_map is not None:
                monitor["deg_score_edge_center_gap"] = edge_center_gap(deg_score, coord_map)
                monitor["deg_score_radial_corr"] = radial_correlation_map(deg_score, coord_map)

        # The following is copied from V1 forward, but uses deg_prior instead of deg_feat.
        feats = []

        x = self.patch_embed(x)

        if self.mod_shallow is not None:
            if return_monitor:
                x, st = self.mod_shallow(x, deg_prior, return_stats=True, name="shallow_")
                monitor.update(st)
            else:
                x = self.mod_shallow(x, deg_prior)

        for i in range(self.num_levels):
            if self.mod_encoders is not None:
                if return_monitor:
                    x, st = self.mod_encoders[i](x, deg_prior, return_stats=True, name=f"enc{i+1}_")
                    monitor.update(st)
                else:
                    x = self.mod_encoders[i](x, deg_prior)

            x = self.encoders[i](x)
            feats.append(x)

            if i < self.num_levels - 1:
                x = self.downsamples[i](x)

        if self.mod_latent is not None:
            if return_monitor:
                x, st = self.mod_latent(x, deg_prior, return_stats=True, name="latent_")
                monitor.update(st)
            else:
                x = self.mod_latent(x, deg_prior)

        x = self.latent(x)
        maybe_debug_tensor("latent", x)

        dec_idx = 0

        for i in range(self.num_levels - 1)[::-1]:
            x = self.upsamples[self.num_levels - 2 - i](x)
            x = torch.cat([x, feats[i]], dim=1)
            x = self.reduce_chans[self.num_levels - 2 - i](x)

            if self.mod_decoders is not None:
                if return_monitor:
                    x, st = self.mod_decoders[dec_idx](x, deg_prior, return_stats=True, name=f"dec{dec_idx+1}_")
                    monitor.update(st)
                else:
                    x = self.mod_decoders[dec_idx](x, deg_prior)

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

    def freq_smoothness_loss(self) -> torch.Tensor:
        if self.last_freq_feat is None:
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device)
        return tv_loss_map(self.last_freq_feat)

    def get_freq_grid_embedding(
        self,
        grid_size: Tuple[int, int] = (9, 9),
        normalize: bool = True,
        detach: bool = False,
    ) -> torch.Tensor:
        """
        Pool last_freq_feat into grid embeddings.
        返回: [B * Gh * Gw, C]
        """
        if self.last_freq_feat is None:
            raise RuntimeError("last_freq_feat is None. Call forward() before get_freq_grid_embedding().")

        z = F.adaptive_avg_pool2d(self.last_freq_feat, grid_size)
        b, c, gh, gw = z.shape
        z = z.permute(0, 2, 3, 1).reshape(b * gh * gw, c)

        if normalize:
            z = F.normalize(z.float(), dim=1)
        if detach:
            z = z.detach()
        return z


# Backward-friendly aliases.
Restormer_DWTPSF = Restormer_DWTPSFLikeDegField
Restormer_DWTPSFLikeDegFieldV1 = Restormer_DWTPSFLikeDegField


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
        use_coord=True,
        coord_channels=4,
        deg_ch=8,
        deg_downsample_factor=4,
        modulate_levels=("shallow", "enc", "latent", "dec"),
        freq_branch=True,
        freq_band_mode="lh_hl",
        freq_mid_ch=12,
        freq_alpha_init=0.0,
        freq_alpha_scale=0.10,
        freq_zero_init=True,
    )

    model = Restormer_DWTPSFLikeDegField(**common_args).to(device)
    y, mon = model(x, return_monitor=True)
    print("Output:", tuple(y.shape))
    print("Monitor keys:", list(mon.keys())[:20])
    print("freq_alpha_effective:", float(mon["freq_alpha_effective"].detach().cpu()))


# ============================================================
# Metric-ready frequency utilities
#   These helpers are intentionally lightweight and dependency-free.
#   They can be imported by a future evaluation engine for:
#       FFT magnitude error
#       DWT high-frequency error
#       Gradient error
#       Center / Edge region masks
# ============================================================

def dwt_high_frequency_bands(
    x: torch.Tensor,
    pad_mode: str = "reflect",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return single-level Haar DWT high-frequency bands: LH, HL, HH."""
    dwt = HaarDWT2D(pad_mode=pad_mode).to(device=x.device)
    _, lh, hl, hh = dwt(x)
    return lh, hl, hh


def dwt_high_frequency_concat(
    x: torch.Tensor,
    pad_mode: str = "reflect",
    hh_weight: float = 1.0,
) -> torch.Tensor:
    """Return concat([LH, HL, hh_weight * HH]) for DWT-HF error metrics."""
    lh, hl, hh = dwt_high_frequency_bands(x, pad_mode=pad_mode)
    return torch.cat([lh, hl, float(hh_weight) * hh], dim=1)


def fft_log_magnitude(
    x: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Return log(1 + |FFT2(x)|), useful for FFT magnitude error.

    x: [B,C,H,W], value range does not need to be fixed.
    """
    x_float = x.float()
    fft = torch.fft.fft2(x_float, dim=(-2, -1), norm="ortho")
    mag = torch.log1p(torch.abs(fft) + float(eps))
    return mag


def gradient_magnitude(
    x: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Sobel-like gradient magnitude for gradient error metrics.
    Output shape: [B,C,H,W].
    """
    b, c, h, w = x.shape
    device = x.device
    dtype = x.dtype

    kx = torch.tensor(
        [[-1.0, 0.0, 1.0],
         [-2.0, 0.0, 2.0],
         [-1.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3) / 8.0

    ky = torch.tensor(
        [[-1.0, -2.0, -1.0],
         [0.0, 0.0, 0.0],
         [1.0, 2.0, 1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3) / 8.0

    kx = kx.repeat(c, 1, 1, 1)
    ky = ky.repeat(c, 1, 1, 1)

    gx = F.conv2d(x, kx, padding=1, groups=c)
    gy = F.conv2d(x, ky, padding=1, groups=c)
    return torch.sqrt(gx * gx + gy * gy + float(eps))


def make_center_edge_masks(
    h: int,
    w: int,
    center_ratio: float = 0.50,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build center/edge masks for Center-PSNR and Edge-PSNR.

    center_ratio=0.50 means the central 50% width/height is center,
    the remaining ring is edge.
    Returns:
        center_mask, edge_mask: [1,1,H,W]
    """
    center_ratio = float(center_ratio)
    if not (0.0 < center_ratio < 1.0):
        raise ValueError("center_ratio should be in (0, 1).")

    yy = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype).view(h, 1).expand(h, w)
    xx = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype).view(1, w).expand(h, w)
    half = center_ratio
    center = ((xx.abs() <= half) & (yy.abs() <= half)).float().view(1, 1, h, w)
    edge = 1.0 - center
    return center, edge


# ============================================================
# Structure 1:
# Decoder-only / stage-selective frequency prompt
# ============================================================

class Restormer_DWTPSFDecoderPrompt(Restormer_DWTPSFLikeDegField):
    """
    Structure-1: PSF field keeps global all-level modulation, while the
    DWT frequency prior is only injected into selected stages.

    Motivation:
        Previous all-level addition:
            deg_prior = deg_feat + alpha * freq_feat
        did activate the DWT branch, but did not outperform V1.
        This class separates the spatial PSF prior and frequency compensation:

            encoder / latent: deg_feat
            shallow / decoder: deg_feat + alpha * freq_feat

    Default:
        frequency prior is used in shallow and decoder stages only.
    """

    def __init__(
        self,
        *args,
        freq_apply_shallow: bool = True,
        freq_apply_encoder: bool = False,
        freq_apply_latent: bool = False,
        freq_apply_decoder: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.freq_apply_shallow = bool(freq_apply_shallow)
        self.freq_apply_encoder = bool(freq_apply_encoder)
        self.freq_apply_latent = bool(freq_apply_latent)
        self.freq_apply_decoder = bool(freq_apply_decoder)

    def _compute_deg_and_freq_priors(
        self,
        x: torch.Tensor,
        coord: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        coord_map = self._prepare_coord(x, coord)
        self.last_coord_map = coord_map

        deg_in = torch.cat([x, coord_map], dim=1) if self.use_coord else x
        deg_feat, deg_score = self.deg_estimator(deg_in)

        if self.freq_branch and self.freq_prior is not None:
            freq_input = x.detach() if self.freq_detach_input else x
            try:
                freq_feat = self.freq_prior(
                    freq_input,
                    target_size=deg_feat.shape[-2:],
                    cond=deg_feat,
                )
            except TypeError:
                freq_feat = self.freq_prior(
                    freq_input,
                    target_size=deg_feat.shape[-2:],
                )
            alpha = self._get_alpha().to(device=x.device, dtype=x.dtype)
            freq_prior = alpha * freq_feat
            deg_freq_prior = deg_feat + freq_prior
        else:
            freq_feat = torch.zeros_like(deg_feat)
            alpha = torch.zeros((), device=x.device, dtype=x.dtype)
            deg_freq_prior = deg_feat

        self.last_deg_feat = deg_feat
        self.last_deg_score = deg_score
        self.last_freq_feat = freq_feat
        self.last_deg_prior = deg_freq_prior

        return deg_feat, deg_score, freq_feat, deg_freq_prior, alpha, coord_map

    def _stage_prior(self, deg_feat: torch.Tensor, deg_freq_prior: torch.Tensor, stage: str) -> torch.Tensor:
        if stage == "shallow":
            return deg_freq_prior if self.freq_apply_shallow else deg_feat
        if stage == "encoder":
            return deg_freq_prior if self.freq_apply_encoder else deg_feat
        if stage == "latent":
            return deg_freq_prior if self.freq_apply_latent else deg_feat
        if stage == "decoder":
            return deg_freq_prior if self.freq_apply_decoder else deg_feat
        return deg_feat

    def _add_common_monitor(
        self,
        monitor: Dict[str, torch.Tensor],
        x: torch.Tensor,
        deg_feat: torch.Tensor,
        deg_score: torch.Tensor,
        freq_feat: torch.Tensor,
        deg_freq_prior: torch.Tensor,
        alpha: torch.Tensor,
        coord_map: Optional[torch.Tensor],
    ) -> None:
        monitor["deg_score_mean"] = _safe_mean(deg_score)
        monitor["deg_score_std"] = _safe_std(deg_score)
        monitor["deg_feat_abs_mean"] = _safe_mean(deg_feat.abs())
        monitor["deg_prior_abs_mean"] = _safe_mean(deg_freq_prior.abs())
        monitor["freq_feat_abs_mean"] = _safe_mean(freq_feat.abs())
        monitor["freq_feat_std"] = _safe_std(freq_feat)
        monitor["freq_alpha_raw"] = self.freq_alpha.detach().float()
        monitor["freq_alpha_tanh"] = torch.tanh(self.freq_alpha.detach().float())
        monitor["freq_alpha_effective"] = alpha.detach().float()
        monitor["deg_field_tv"] = self.deg_smoothness_loss().detach()
        monitor["freq_prior_tv"] = tv_loss_map(freq_feat).detach()

        # Stage-usage flags are helpful when comparing all-level vs decoder-only designs.
        monitor["freq_apply_shallow"] = torch.tensor(float(self.freq_apply_shallow), device=x.device)
        monitor["freq_apply_encoder"] = torch.tensor(float(self.freq_apply_encoder), device=x.device)
        monitor["freq_apply_latent"] = torch.tensor(float(self.freq_apply_latent), device=x.device)
        monitor["freq_apply_decoder"] = torch.tensor(float(self.freq_apply_decoder), device=x.device)

        if self.freq_prior is not None and hasattr(self.freq_prior, "band_energy_dict"):
            band_energy = self.freq_prior.band_energy_dict()
            monitor["dwt_ll_abs_mean"] = band_energy["ll"].detach()
            monitor["dwt_lh_abs_mean"] = band_energy["lh"].detach()
            monitor["dwt_hl_abs_mean"] = band_energy["hl"].detach()
            monitor["dwt_hh_abs_mean"] = band_energy["hh"].detach()
            monitor["dwt_hf_abs_mean"] = (
                band_energy["lh"] + band_energy["hl"] + band_energy["hh"]
            ).detach()

        with torch.no_grad():
            d = deg_feat.detach().float().flatten(1)
            f = freq_feat.detach().float().flatten(1)
            if d.shape == f.shape:
                d = F.normalize(d, dim=1)
                f = F.normalize(f, dim=1)
                monitor["deg_freq_cos_sim"] = (d * f).sum(dim=1).mean().detach()
            else:
                monitor["deg_freq_cos_sim"] = torch.tensor(0.0, device=x.device)

        if coord_map is not None:
            monitor["deg_score_edge_center_gap"] = edge_center_gap(deg_score, coord_map)
            monitor["deg_score_radial_corr"] = radial_correlation_map(deg_score, coord_map)

    def forward(
        self,
        x: torch.Tensor,
        coord: Optional[torch.Tensor] = None,
        return_monitor: bool = False,
    ):
        x_input = x
        monitor: Dict[str, torch.Tensor] = {}

        deg_feat, deg_score, freq_feat, deg_freq_prior, alpha, coord_map = self._compute_deg_and_freq_priors(
            x,
            coord=coord,
        )

        maybe_debug_tensor("dwtpsf_dec_prompt_deg_feat", deg_feat)
        maybe_debug_tensor("dwtpsf_dec_prompt_freq_feat", freq_feat)
        maybe_debug_tensor("dwtpsf_dec_prompt_deg_freq_prior", deg_freq_prior)

        if return_monitor:
            self._add_common_monitor(
                monitor,
                x=x,
                deg_feat=deg_feat,
                deg_score=deg_score,
                freq_feat=freq_feat,
                deg_freq_prior=deg_freq_prior,
                alpha=alpha,
                coord_map=coord_map,
            )

        # Restormer forward path with stage-specific prior.
        feats = []
        x = self.patch_embed(x)

        if self.mod_shallow is not None:
            prior = self._stage_prior(deg_feat, deg_freq_prior, "shallow")
            if return_monitor:
                x, st = self.mod_shallow(x, prior, return_stats=True, name="shallow_")
                monitor.update(st)
            else:
                x = self.mod_shallow(x, prior)

        for i in range(self.num_levels):
            if self.mod_encoders is not None:
                prior = self._stage_prior(deg_feat, deg_freq_prior, "encoder")
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
            prior = self._stage_prior(deg_feat, deg_freq_prior, "latent")
            if return_monitor:
                x, st = self.mod_latent(x, prior, return_stats=True, name="latent_")
                monitor.update(st)
            else:
                x = self.mod_latent(x, prior)

        x = self.latent(x)
        maybe_debug_tensor("latent", x)

        dec_idx = 0
        for i in range(self.num_levels - 1)[::-1]:
            x = self.upsamples[self.num_levels - 2 - i](x)
            x = torch.cat([x, feats[i]], dim=1)
            x = self.reduce_chans[self.num_levels - 2 - i](x)

            if self.mod_decoders is not None:
                prior = self._stage_prior(deg_feat, deg_freq_prior, "decoder")
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

        if return_monitor:
            monitor["output_residual_abs_mean"] = _safe_mean((out - x_input).abs())
            self.last_monitor = monitor
            return out, monitor

        self.last_monitor = monitor
        return out


# ============================================================
# Structure 2:
# PSF-guided adaptive DWT frequency selection
# ============================================================

class AdaptiveDWTFrequencySelectionPrior(nn.Module):
    """
    PSF-guided / degradation-guided adaptive DWT band selection.

    Instead of fixed:
        freq_feat = f(LH, HL, 0.3*HH)

    It predicts local band weights:
        w_lh, w_hl, w_hh = softmax(WeightNet(deg_feat) / T)

    Then builds a weighted high-frequency representation:
        concat(w_lh*LH, w_hl*HL, w_hh*HH)

    This is suitable for spatially-varying PSF / metalens degradation.
    """

    def __init__(
        self,
        inp_channels: int = 3,
        cond_ch: int = 16,
        out_ch: int = 16,
        mid_ch: int = 32,
        pad_mode: str = "reflect",
        temperature: float = 1.0,
        min_hh_weight: float = 0.0,
        zero_init: bool = False,
    ):
        super().__init__()
        self.inp_channels = int(inp_channels)
        self.cond_ch = int(cond_ch)
        self.out_ch = int(out_ch)
        self.mid_ch = int(mid_ch)
        self.temperature = float(temperature)
        self.min_hh_weight = float(min_hh_weight)

        self.dwt = HaarDWT2D(pad_mode=pad_mode)

        self.weight_net = nn.Sequential(
            nn.Conv2d(cond_ch, mid_ch, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(mid_ch, 3, 1, 1, 0),
        )

        self.net = nn.Sequential(
            nn.Conv2d(inp_channels * 3, mid_ch, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(mid_ch, mid_ch, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(mid_ch, out_ch, 3, 1, 1),
        )

        if zero_init:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

        self.last_bands: Dict[str, torch.Tensor] = {}
        self.last_weights: Optional[torch.Tensor] = None
        self.last_freq_feat: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        target_size: Optional[Tuple[int, int]] = None,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if cond is None:
            raise ValueError("AdaptiveDWTFrequencySelectionPrior requires cond=deg_feat.")

        ll, lh, hl, hh = self.dwt(x)

        if target_size is None:
            target_size = cond.shape[-2:]

        if lh.shape[-2:] != target_size:
            lh_r = F.interpolate(lh, size=target_size, mode="bilinear", align_corners=False)
            hl_r = F.interpolate(hl, size=target_size, mode="bilinear", align_corners=False)
            hh_r = F.interpolate(hh, size=target_size, mode="bilinear", align_corners=False)
            ll_r = F.interpolate(ll, size=target_size, mode="bilinear", align_corners=False)
        else:
            lh_r, hl_r, hh_r, ll_r = lh, hl, hh, ll

        logits = self.weight_net(cond)
        temperature = max(float(self.temperature), 1e-6)
        weights = torch.softmax(logits / temperature, dim=1)

        if self.min_hh_weight > 0:
            # Keep a small non-zero HH path but renormalize all weights.
            weights = weights.clone()
            weights[:, 2:3] = torch.clamp(weights[:, 2:3], min=self.min_hh_weight)
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)

        w_lh = weights[:, 0:1]
        w_hl = weights[:, 1:2]
        w_hh = weights[:, 2:3]

        selected = torch.cat(
            [
                w_lh * lh_r,
                w_hl * hl_r,
                w_hh * hh_r,
            ],
            dim=1,
        )

        freq_feat = self.net(selected)

        with torch.no_grad():
            self.last_bands = {
                "ll": ll_r.detach(),
                "lh": lh_r.detach(),
                "hl": hl_r.detach(),
                "hh": hh_r.detach(),
            }
            self.last_weights = weights.detach()

        self.last_freq_feat = freq_feat
        return freq_feat

    def band_energy_dict(self) -> Dict[str, torch.Tensor]:
        if not self.last_bands:
            device = next(self.parameters()).device
            zero = torch.tensor(0.0, device=device)
            return {"ll": zero, "lh": zero, "hl": zero, "hh": zero}
        return {k: v.float().abs().mean() for k, v in self.last_bands.items()}

    def band_weight_dict(self) -> Dict[str, torch.Tensor]:
        if self.last_weights is None:
            device = next(self.parameters()).device
            zero = torch.tensor(0.0, device=device)
            return {
                "w_lh_mean": zero,
                "w_hl_mean": zero,
                "w_hh_mean": zero,
                "w_lh_std": zero,
                "w_hl_std": zero,
                "w_hh_std": zero,
            }

        w = self.last_weights.float()
        return {
            "w_lh_mean": w[:, 0:1].mean(),
            "w_hl_mean": w[:, 1:2].mean(),
            "w_hh_mean": w[:, 2:3].mean(),
            "w_lh_std": w[:, 0:1].std(unbiased=False),
            "w_hl_std": w[:, 1:2].std(unbiased=False),
            "w_hh_std": w[:, 2:3].std(unbiased=False),
        }


class Restormer_DWTPSFAdaptiveFreqSelect(Restormer_DWTPSFDecoderPrompt):
    """
    Structure-2: PSF-guided adaptive frequency selection.

    Default stage use:
        shallow + decoder only.

    Compared with manually fixed LH/HL/HH weights, this class lets deg_feat
    predict local band weights, which is better aligned with spatially-varying
    PSF degradation.
    """

    def __init__(
        self,
        *args,
        freq_mid_ch: int = 32,
        freq_selection_temperature: float = 1.0,
        freq_min_hh_weight: float = 0.0,
        freq_zero_init: bool = False,
        freq_pad_mode: str = "reflect",
        **kwargs,
    ):
        super().__init__(
            *args,
            freq_mid_ch=freq_mid_ch,
            freq_zero_init=freq_zero_init,
            freq_pad_mode=freq_pad_mode,
            **kwargs,
        )

        if self.freq_branch:
            self.freq_prior = AdaptiveDWTFrequencySelectionPrior(
                inp_channels=kwargs.get("inp_channels", 3),
                cond_ch=self.deg_ch,
                out_ch=self.deg_ch,
                mid_ch=freq_mid_ch,
                pad_mode=freq_pad_mode,
                temperature=freq_selection_temperature,
                min_hh_weight=freq_min_hh_weight,
                zero_init=freq_zero_init,
            )

    def _add_common_monitor(
        self,
        monitor: Dict[str, torch.Tensor],
        x: torch.Tensor,
        deg_feat: torch.Tensor,
        deg_score: torch.Tensor,
        freq_feat: torch.Tensor,
        deg_freq_prior: torch.Tensor,
        alpha: torch.Tensor,
        coord_map: Optional[torch.Tensor],
    ) -> None:
        super()._add_common_monitor(
            monitor,
            x=x,
            deg_feat=deg_feat,
            deg_score=deg_score,
            freq_feat=freq_feat,
            deg_freq_prior=deg_freq_prior,
            alpha=alpha,
            coord_map=coord_map,
        )

        if self.freq_prior is not None and hasattr(self.freq_prior, "band_weight_dict"):
            wd = self.freq_prior.band_weight_dict()
            for k, v in wd.items():
                monitor[f"freq_select_{k}"] = v.detach()


# ============================================================
# Structure 3:
# Metric/loss-ready frequency-supervised PSF model
# ============================================================

class Restormer_DWTPSFFrequencyLossReady(Restormer_DWTPSFDecoderPrompt):
    """
    Structure-3: Decoder-prompt DWT-PSF model with built-in metric/loss helpers.

    This class does not require the training engine to use the helpers, but it
    makes the model file ready for the planned comparison metrics:

        Stage-1:
            PSNR / SSIM / LPIPS / DISTS / inference time / params / FLOPs
            -> computed in engine/eval script.

        Stage-2:
            Center / Edge PSNR
            FFT magnitude error
            DWT high-frequency error
            Gradient error
            -> helper functions below can be called by engine/eval script.

        Stage-3:
            NIQE / MUSIQ / MANIQA
            -> no-reference metrics should be computed outside the model.

    The actual network is the safer decoder-only prompt design.
    """

    @staticmethod
    def fft_magnitude_error(pred: torch.Tensor, target: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        loss = (fft_log_magnitude(pred) - fft_log_magnitude(target)).abs()
        return loss.mean() if reduction == "mean" else loss

    @staticmethod
    def dwt_hf_error(
        pred: torch.Tensor,
        target: torch.Tensor,
        hh_weight: float = 1.0,
        reduction: str = "mean",
    ) -> torch.Tensor:
        pred_hf = dwt_high_frequency_concat(pred, hh_weight=hh_weight)
        tar_hf = dwt_high_frequency_concat(target, hh_weight=hh_weight)
        loss = (pred_hf - tar_hf).abs()
        return loss.mean() if reduction == "mean" else loss

    @staticmethod
    def gradient_error(pred: torch.Tensor, target: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        loss = (gradient_magnitude(pred) - gradient_magnitude(target)).abs()
        return loss.mean() if reduction == "mean" else loss

    @staticmethod
    def center_edge_l1_error(
        pred: torch.Tensor,
        target: torch.Tensor,
        center_ratio: float = 0.50,
    ) -> Dict[str, torch.Tensor]:
        _, _, h, w = pred.shape
        center, edge = make_center_edge_masks(
            h,
            w,
            center_ratio=center_ratio,
            device=pred.device,
            dtype=pred.dtype,
        )
        abs_err = (pred - target).abs()
        center_l1 = (abs_err * center).sum() / (center.sum() * pred.shape[1] + 1e-6)
        edge_l1 = (abs_err * edge).sum() / (edge.sum() * pred.shape[1] + 1e-6)
        return {
            "center_l1": center_l1,
            "edge_l1": edge_l1,
        }

    def frequency_aux_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        lambda_fft: float = 0.0,
        lambda_dwt_hf: float = 0.0,
        lambda_grad: float = 0.0,
        dwt_hh_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Optional auxiliary loss for a future engine.

        It returns:
            total_aux_loss, stats_dict

        Existing engines can ignore this method safely.
        """
        total = pred.sum() * 0.0
        stats: Dict[str, torch.Tensor] = {}

        if lambda_fft > 0:
            l_fft = self.fft_magnitude_error(pred, target)
            total = total + float(lambda_fft) * l_fft
            stats["loss_fft_mag"] = l_fft.detach()

        if lambda_dwt_hf > 0:
            l_dwt = self.dwt_hf_error(pred, target, hh_weight=dwt_hh_weight)
            total = total + float(lambda_dwt_hf) * l_dwt
            stats["loss_dwt_hf"] = l_dwt.detach()

        if lambda_grad > 0:
            l_grad = self.gradient_error(pred, target)
            total = total + float(lambda_grad) * l_grad
            stats["loss_gradient"] = l_grad.detach()

        return total, stats


# Backward/new aliases for run files.
Restormer_DWTPSF_DecoderPrompt = Restormer_DWTPSFDecoderPrompt
Restormer_DWTPSF_FreqSelect = Restormer_DWTPSFAdaptiveFreqSelect
Restormer_DWTPSF_FrequencyLossReady = Restormer_DWTPSFFrequencyLossReady
