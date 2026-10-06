# -*- coding: utf-8 -*-
"""
Res_Psfbasis
============

保存路径建议：
    models/Res_Psfbasis.py

本文件基于 Restormer_PriorVariants.py 的两个最好结构继续改造，提供两个更接近
unknown-PSF / PSF-like degradation learning 的版本：

1. Restormer_PSFLikeDegField
   image + coord -> low-resolution PSF-like degradation field
   -> feature affine modulation -> Restormer

   主要新增：
   - 低分辨率 deg field，减少内容纹理泄漏
   - deg_score / deg_feat 保存，用于可视化与对比学习
   - deg_smoothness_loss()
   - deg_radial_prior_loss()
   - get_deg_grid_embedding()

2. Restormer_PSFLikeLowRankBasis
   image + coord -> prior feature -> shared low-resolution coefficient map
   -> PSF-like low-rank basis bank -> layer-wise modulation -> Restormer

   主要新增：
   - shared coefficient map，所有层共享同一个 PSF-like mixture field
   - low-resolution coefficient map
   - softmax floor，防止 basis collapse
   - coefficient smoothness loss
   - basis diversity loss
   - get_coeff_grid_embedding()

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

可选对比学习：
    z = model.get_deg_grid_embedding(grid_size=(9, 9))
    # 或
    z = model.get_coeff_grid_embedding(grid_size=(9, 9))
    labels = make_field_position_labels(batch_size=B, grid_size=(9, 9), device=z.device)
    loss_ctr = supervised_nt_xent(z, labels)

注意：
    对比学习建议在训练脚本里调用，不强制写进 forward，避免破坏原有 engine。
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
    try:
        from Restormer import (
            OverlapPatchEmbed,
            TransformerBlock,
            Downsample,
            Upsample,
            maybe_debug_tensor,
        )
    except Exception:
        from .Restormer import (
            OverlapPatchEmbed,
            TransformerBlock,
            Downsample,
            Upsample,
        )

        def maybe_debug_tensor(name, tensor):
            return None


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


def resize_like(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if src.shape[-2:] == ref.shape[-2:]:
        return src
    return F.interpolate(src, size=ref.shape[-2:], mode="bilinear", align_corners=False)


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


# ============================================================
# Monitor / Loss Utils
# ============================================================

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


def basis_diversity_metric(basis: torch.Tensor) -> torch.Tensor:
    """
    返回 basis 两两 cosine similarity 的非对角绝对均值。
    """
    b = F.normalize(basis.detach().float(), dim=1)
    sim = b @ b.t()
    k = sim.shape[0]
    mask = ~torch.eye(k, device=sim.device, dtype=torch.bool)
    return sim[mask].abs().mean()


def supervised_nt_xent(
    z: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Supervised NT-Xent loss。

    用途：
        让相同/相近 PSF field position 的 degradation embedding 靠近，
        让远距离 position 或不同退化强度的 embedding 分离。

    z:
        [N, D]

    labels:
        [N]
        相同 label 视作 positive。
        labels < 0 的样本会被忽略。

    返回：
        scalar loss

    注意：
        如果 batch 内没有任何 positive pair，返回 0。
    """
    if z.dim() != 2:
        raise ValueError(f"z must be [N,D], got {tuple(z.shape)}")

    labels = labels.view(-1).to(device=z.device)
    valid = labels >= 0
    z = z[valid]
    labels = labels[valid]

    n = z.shape[0]
    if n <= 1:
        return z.sum() * 0.0

    z = F.normalize(z.float(), dim=1)

    logits = (z @ z.t()) / temperature
    logits = logits - logits.detach().max(dim=1, keepdim=True).values

    eye = torch.eye(n, device=z.device, dtype=torch.bool)
    pos_mask = (labels[:, None] == labels[None, :]) & (~eye)

    # 没有 positive pair 时，直接返回 0，避免 NaN。
    if pos_mask.sum() == 0:
        return z.sum() * 0.0

    logits_mask = ~eye
    logits_for_denom = logits.masked_fill(~logits_mask, -1e9)
    log_prob = logits - torch.logsumexp(logits_for_denom, dim=1, keepdim=True)

    pos_count = pos_mask.sum(dim=1)
    valid_anchor = pos_count > 0

    mean_log_prob_pos = (log_prob * pos_mask.float()).sum(dim=1) / (pos_count.float() + eps)
    loss = -mean_log_prob_pos[valid_anchor].mean()
    return loss


def make_field_position_labels(
    batch_size: int,
    grid_size: Tuple[int, int] = (9, 9),
    device: Optional[torch.device] = None,
    neighbor_as_positive: bool = False,
) -> torch.Tensor:
    """
    生成 field-position labels，用于 grid embedding 的 supervised contrastive loss。

    默认：
        同一个 grid cell 跨不同图像为同一类。
        labels shape = [B * Gh * Gw]

    neighbor_as_positive=False:
        每个 cell 一个 label。

    neighbor_as_positive=True:
        将 2x2 邻域粗分组为同一 label，更宽松。
    """
    gh, gw = grid_size
    yy = torch.arange(gh, device=device).view(gh, 1).expand(gh, gw)
    xx = torch.arange(gw, device=device).view(1, gw).expand(gh, gw)

    if neighbor_as_positive:
        labels = (yy // 2) * math.ceil(gw / 2) + (xx // 2)
    else:
        labels = yy * gw + xx

    labels = labels.reshape(1, gh * gw).repeat(batch_size, 1).reshape(-1)
    return labels


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
# PSF-like DegField
# ============================================================

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


class Restormer_PSFLikeDegField(nn.Module):
    """
    PSF-like Degradation Field Modulated Restormer.

    相比原 Restormer_DegField：
        1. deg field 默认为低分辨率；
        2. 保存 deg_feat / deg_score / coord_map；
        3. 提供 smoothness、radial weak prior、grid embedding 接口；
        4. 更适合配合 field-aware contrastive loss。
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
        deg_downsample_factor: int = 4,
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
        self.last_deg_feat: Optional[torch.Tensor] = None
        self.last_deg_score: Optional[torch.Tensor] = None
        self.last_coord_map: Optional[torch.Tensor] = None

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim, bias=bias)

        deg_in_ch = inp_channels + (coord_channels if use_coord else 0)

        self.deg_estimator = PSFLikeDegFieldEstimator(
            in_ch=deg_in_ch,
            mid_ch=deg_mid_ch,
            deg_ch=deg_ch,
            downsample_factor=deg_downsample_factor,
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
        self.last_coord_map = coord_map

        if self.use_coord:
            deg_in = torch.cat([x, coord_map], dim=1)
        else:
            deg_in = x

        deg_feat, deg_score = self.deg_estimator(deg_in)

        self.last_deg_feat = deg_feat
        self.last_deg_score = deg_score

        maybe_debug_tensor("psf_deg_feat", deg_feat)
        maybe_debug_tensor("psf_deg_score", deg_score)

        if return_monitor:
            monitor["deg_score_mean"] = _safe_mean(deg_score)
            monitor["deg_score_std"] = _safe_std(deg_score)
            monitor["deg_feat_abs_mean"] = _safe_mean(deg_feat.abs())
            monitor["deg_field_tv"] = self.deg_smoothness_loss().detach()

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

    def deg_smoothness_loss(self) -> torch.Tensor:
        if self.last_deg_score is None:
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device)
        return tv_loss_map(self.last_deg_score)

    def deg_radial_prior_loss(
        self,
        min_corr: float = 0.10,
        require_edge_larger: bool = False,
        min_gap: float = 0.00,
    ) -> torch.Tensor:
        """
        弱 radial prior。

        min_corr:
            希望 deg_score 与 radius 至少有一定正相关。
            只是 weak prior，不建议设置太大。

        require_edge_larger:
            是否要求边缘 deg_score 大于中心 deg_score。

        min_gap:
            edge_center_gap 的最低值。
        """
        if self.last_deg_score is None or self.last_coord_map is None:
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device)

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
        将 last_deg_feat 池化成 grid embedding。

        返回:
            [B * Gh * Gw, C]

        可用于：
            labels = make_field_position_labels(B, grid_size, device=z.device)
            loss_ctr = supervised_nt_xent(z, labels)
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

    def get_last_monitor(self, to_float: bool = True) -> Dict[str, Union[float, torch.Tensor]]:
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
# PSF-like Low-rank Basis
# ============================================================

class PriorEncoder(nn.Module):
    """
    输入 image 或 image + coord，输出 prior feature。
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


class SharedPSFCoeffPredictor(nn.Module):
    """
    从 prior feature 中预测 shared low-resolution coefficient map。

    输出:
        coeff [B, K, H/s, W/s]

    这里的 coeff 被所有层共享，表示同一个 PSF-like degradation mixture field。
    """

    def __init__(
        self,
        prior_ch: int,
        num_basis: int = 8,
        hidden_ch: int = 32,
        downsample_factor: int = 4,
        coeff_floor: float = 0.05,
        softmax_coeff: bool = True,
    ):
        super().__init__()

        if downsample_factor not in (1, 2, 4, 8):
            raise ValueError("downsample_factor should be one of {1,2,4,8}")

        self.num_basis = num_basis
        self.downsample_factor = downsample_factor
        self.coeff_floor = float(coeff_floor)
        self.softmax_coeff = softmax_coeff

        # Debug caches. These are intentionally not detached here so the
        # caller can inspect the exact tensors produced in the last forward.
        # Monitor code will detach before logging.
        self.last_logits: Optional[torch.Tensor] = None
        self.last_soft: Optional[torch.Tensor] = None
        self.last_coeff: Optional[torch.Tensor] = None

        layers = []
        cur_ch = prior_ch
        cur_factor = 1

        while cur_factor < downsample_factor:
            layers += [
                nn.Conv2d(cur_ch, hidden_ch, 3, 2, 1),
                nn.GELU(),
            ]
            cur_ch = hidden_ch
            cur_factor *= 2

        if len(layers) == 0:
            layers += [
                nn.Conv2d(cur_ch, hidden_ch, 3, 1, 1),
                nn.GELU(),
            ]
            cur_ch = hidden_ch

        layers += [
            nn.Conv2d(cur_ch, hidden_ch, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden_ch, num_basis, 3, 1, 1),
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, prior_feat: torch.Tensor) -> torch.Tensor:
        logits = self.net(prior_feat)

        if self.softmax_coeff:
            soft = torch.softmax(logits, dim=1)

            # coeff_floor 表示总保底质量，均匀分配给 K 个 basis。
            # 例如 coeff_floor=0.10, K=8，则每个 basis 至少 0.0125。
            if self.coeff_floor > 0:
                coeff = self.coeff_floor / self.num_basis + (1.0 - self.coeff_floor) * soft
            else:
                coeff = soft
        else:
            soft = torch.sigmoid(logits)
            coeff = soft / (soft.sum(dim=1, keepdim=True) + 1e-6)

            if self.coeff_floor > 0:
                coeff = self.coeff_floor / self.num_basis + (1.0 - self.coeff_floor) * coeff
                coeff = coeff / (coeff.sum(dim=1, keepdim=True) + 1e-6)

        # Save exact last tensors for coefficient-map sanity checks.
        self.last_logits = logits
        self.last_soft = soft
        self.last_coeff = coeff

        return coeff


class PSFLikeBasisModulation(nn.Module):
    """
    使用 shared coeff map + layer-specific basis bank 生成 implicit prior，再做 affine modulation。

    feat:
        [B, C, H, W]

    coeff:
        [B, K, H0, W0]

    basis:
        [K, C]

    prior:
        [B, C, H, W]
    """

    def __init__(
        self,
        feat_ch: int,
        num_basis: int = 8,
        zero_init_mod: bool = True,
    ):
        super().__init__()

        self.num_basis = num_basis
        self.feat_ch = feat_ch

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
        coeff: torch.Tensor,
        return_stats: bool = False,
        name: str = "",
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:

        if coeff.shape[-2:] != feat.shape[-2:]:
            coeff_resized = F.interpolate(
                coeff,
                size=feat.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            coeff_resized = coeff_resized / (coeff_resized.sum(dim=1, keepdim=True) + 1e-6)
        else:
            coeff_resized = coeff

        prior = torch.einsum("bkhw,kc->bchw", coeff_resized, self.basis)

        scale, shift = self.to_scale_shift(prior).chunk(2, dim=1)
        out = feat * (1.0 + scale) + shift

        if not return_stats:
            return out

        delta = out - feat
        stats = {
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
        b = F.normalize(self.basis.float(), dim=1)
        sim = b @ b.t()
        eye = torch.eye(self.num_basis, device=sim.device, dtype=sim.dtype)
        return ((sim - eye) ** 2).mean()


class Restormer_PSFLikeLowRankBasis(nn.Module):
    """
    PSF-like Low-rank Degradation Basis Modulated Restormer。

    相比原 Restormer_LowRankBasis：
        1. coefficient map 由 shared predictor 统一预测；
        2. 所有层共享同一个 coefficient map，但各层有自己的 basis bank；
        3. coefficient map 默认低分辨率；
        4. softmax floor 防止 basis collapse；
        5. 提供 coefficient smoothness loss 和 grid embedding 接口。
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
        coeff_hidden_ch: int = 32,
        coeff_downsample_factor: int = 4,
        coeff_floor: float = 0.05,
        softmax_coeff: bool = True,
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
        self.last_prior_feat: Optional[torch.Tensor] = None
        self.last_coeff_map: Optional[torch.Tensor] = None
        self.last_coord_map: Optional[torch.Tensor] = None

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim, bias=bias)

        prior_in_ch = inp_channels + (coord_channels if use_coord else 0)

        self.prior_encoder = PriorEncoder(
            in_ch=prior_in_ch,
            prior_ch=prior_ch,
            mid_ch=prior_mid_ch,
        )

        self.coeff_predictor = SharedPSFCoeffPredictor(
            prior_ch=prior_ch,
            num_basis=num_basis,
            hidden_ch=coeff_hidden_ch,
            downsample_factor=coeff_downsample_factor,
            coeff_floor=coeff_floor,
            softmax_coeff=softmax_coeff,
        )

        level_dims = [dim * (2 ** i) for i in range(self.num_levels)]

        self.mod_shallow = (
            PSFLikeBasisModulation(dim, num_basis=num_basis)
            if "shallow" in self.modulate_levels else None
        )

        self.mod_encoders = (
            nn.ModuleList([
                PSFLikeBasisModulation(c, num_basis=num_basis)
                for c in level_dims
            ])
            if "enc" in self.modulate_levels else None
        )

        self.mod_latent = (
            PSFLikeBasisModulation(level_dims[-1], num_basis=num_basis)
            if "latent" in self.modulate_levels else None
        )

        self.mod_decoders = (
            nn.ModuleList([
                PSFLikeBasisModulation(c, num_basis=num_basis)
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

    def _collect_mods(self):
        mods = []

        if self.mod_shallow is not None:
            mods.append(self.mod_shallow)

        if self.mod_encoders is not None:
            mods += list(self.mod_encoders)

        if self.mod_latent is not None:
            mods.append(self.mod_latent)

        if self.mod_decoders is not None:
            mods += list(self.mod_decoders)

        return mods

    def _collect_basis_diversity_loss(self) -> torch.Tensor:
        mods = self._collect_mods()
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
        self.last_coord_map = coord_map

        if self.use_coord:
            prior_in = torch.cat([x, coord_map], dim=1)
        else:
            prior_in = x

        prior_feat = self.prior_encoder(prior_in)
        coeff_map = self.coeff_predictor(prior_feat)

        self.last_prior_feat = prior_feat
        self.last_coeff_map = coeff_map

        maybe_debug_tensor("psf_lowrank_prior_feat", prior_feat)
        maybe_debug_tensor("psf_lowrank_coeff_map", coeff_map)

        if return_monitor:
            monitor["prior_feat_abs_mean"] = _safe_mean(prior_feat.abs())
            monitor["prior_feat_std"] = _safe_std(prior_feat)
            monitor["coeff_tv"] = self.coeff_smoothness_loss().detach()

            coeff_detached = coeff_map.detach().float()
            coeff_sum = coeff_detached.sum(dim=1, keepdim=True)
            coeff_max_map = coeff_detached.amax(dim=1)
            coeff_min_map = coeff_detached.amin(dim=1)

            entropy = -(
                coeff_detached * torch.log(coeff_detached + 1e-8)
            ).sum(dim=1).mean()
            entropy_norm = entropy / math.log(float(self.num_basis))
            top_prob = coeff_max_map.mean()
            usage = coeff_detached.mean(dim=(0, 2, 3))

            # Shape / probability sanity checks.
            monitor["coeff_shape_h"] = torch.as_tensor(
                coeff_detached.shape[-2], device=coeff_detached.device
            ).float()
            monitor["coeff_shape_w"] = torch.as_tensor(
                coeff_detached.shape[-1], device=coeff_detached.device
            ).float()
            monitor["coeff_sum_mean"] = _safe_mean(coeff_sum)
            monitor["coeff_sum_std"] = _safe_std(coeff_sum)
            monitor["coeff_sum_min"] = coeff_sum.min().detach().float()
            monitor["coeff_sum_max"] = coeff_sum.max().detach().float()
            monitor["coeff_mean"] = _safe_mean(coeff_detached)
            monitor["coeff_min"] = coeff_min_map.min().detach().float()
            monitor["coeff_max"] = coeff_max_map.max().detach().float()

            # Original coefficient statistics.
            monitor["coeff_entropy_norm"] = entropy_norm
            monitor["coeff_top_prob_mean"] = top_prob
            monitor["coeff_spatial_std"] = coeff_detached.std(unbiased=False)
            monitor["basis_usage_sum"] = usage.sum().detach().float()
            monitor["basis_usage_mean"] = usage.mean().detach().float()
            monitor["basis_usage_std"] = usage.std(unbiased=False)
            monitor["basis_usage_min"] = usage.min()
            monitor["basis_usage_max"] = usage.max()

            # Pre-floor / logits sanity checks.
            if self.coeff_predictor.last_logits is not None:
                logits_detached = self.coeff_predictor.last_logits.detach().float()
                monitor["coeff_logits_mean"] = _safe_mean(logits_detached)
                monitor["coeff_logits_std"] = _safe_std(logits_detached)
                monitor["coeff_logits_min"] = logits_detached.min().detach().float()
                monitor["coeff_logits_max"] = logits_detached.max().detach().float()

            if self.coeff_predictor.last_soft is not None:
                soft_detached = self.coeff_predictor.last_soft.detach().float()
                soft_sum = soft_detached.sum(dim=1, keepdim=True)
                monitor["soft_sum_mean"] = _safe_mean(soft_sum)
                monitor["soft_sum_std"] = _safe_std(soft_sum)
                monitor["soft_top_prob_mean"] = soft_detached.amax(dim=1).mean().detach().float()
                monitor["soft_entropy_norm"] = (
                    -(
                        soft_detached * torch.log(soft_detached + 1e-8)
                    ).sum(dim=1).mean() / math.log(float(self.num_basis))
                ).detach().float()

            if coord_map is not None:
                # 用 max coefficient response 作为空间退化响应的弱可视化指标。
                coeff_score = coeff_detached.amax(dim=1, keepdim=True)
                monitor["coeff_score_edge_center_gap"] = edge_center_gap(coeff_score, coord_map)
                monitor["coeff_score_radial_corr"] = radial_correlation_map(coeff_score, coord_map)

        feats = []

        x = self.patch_embed(x)

        if self.mod_shallow is not None:
            if return_monitor:
                x, st = self.mod_shallow(
                    x,
                    coeff_map,
                    return_stats=True,
                    name="shallow_",
                )
                monitor.update(st)
            else:
                x = self.mod_shallow(x, coeff_map)

        for i in range(self.num_levels):
            if self.mod_encoders is not None:
                if return_monitor:
                    x, st = self.mod_encoders[i](
                        x,
                        coeff_map,
                        return_stats=True,
                        name=f"enc{i+1}_",
                    )
                    monitor.update(st)
                else:
                    x = self.mod_encoders[i](x, coeff_map)

            x = self.encoders[i](x)
            feats.append(x)

            if i < self.num_levels - 1:
                x = self.downsamples[i](x)

        if self.mod_latent is not None:
            if return_monitor:
                x, st = self.mod_latent(
                    x,
                    coeff_map,
                    return_stats=True,
                    name="latent_",
                )
                monitor.update(st)
            else:
                x = self.mod_latent(x, coeff_map)

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
                        coeff_map,
                        return_stats=True,
                        name=f"dec{dec_idx+1}_",
                    )
                    monitor.update(st)
                else:
                    x = self.mod_decoders[dec_idx](x, coeff_map)

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
        return self._collect_basis_diversity_loss()

    def coeff_smoothness_loss(self) -> torch.Tensor:
        if self.last_coeff_map is None:
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device)
        return tv_loss_map(self.last_coeff_map)

    def coeff_entropy_loss(
        self,
        target_entropy: float = 0.65,
        mode: str = "min",
    ) -> torch.Tensor:
        """
        可选 entropy regularization。

        mode="min":
            希望 entropy 不低于 target_entropy，防止过早 collapse。
            loss = relu(target - entropy_norm)

        mode="max":
            希望 entropy 不高于 target_entropy，鼓励更明确的 basis assignment。
            loss = relu(entropy_norm - target)

        一般建议：
            前期可以 mode="min"，后期可关闭。
        """
        if self.last_coeff_map is None:
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device)

        coeff = self.last_coeff_map.float()
        entropy = -(coeff * torch.log(coeff + 1e-8)).sum(dim=1).mean()
        entropy_norm = entropy / math.log(float(self.num_basis))

        target = torch.as_tensor(target_entropy, device=coeff.device, dtype=coeff.dtype)

        if mode == "min":
            return F.relu(target - entropy_norm)
        if mode == "max":
            return F.relu(entropy_norm - target)

        raise ValueError("mode should be 'min' or 'max'")

    def get_coeff_grid_embedding(
        self,
        grid_size: Tuple[int, int] = (9, 9),
        normalize: bool = True,
        detach: bool = False,
    ) -> torch.Tensor:
        """
        将 last_coeff_map 池化成 grid embedding。

        返回:
            [B * Gh * Gw, K]

        可用于：
            labels = make_field_position_labels(B, grid_size, device=z.device)
            loss_ctr = supervised_nt_xent(z, labels)
        """
        if self.last_coeff_map is None:
            raise RuntimeError("last_coeff_map is None. Call forward() before get_coeff_grid_embedding().")

        z = F.adaptive_avg_pool2d(self.last_coeff_map, grid_size)
        b, k, gh, gw = z.shape
        z = z.permute(0, 2, 3, 1).reshape(b * gh * gw, k)

        if normalize:
            z = F.normalize(z.float(), dim=1)

        if detach:
            z = z.detach()

        return z

    def coeff_debug_dict(self) -> Dict[str, Union[float, Tuple[int, ...]]]:
        """
        Return coefficient-map sanity-check values after forward().

        This is intended for debugging PSFLikeLowRankBasis only. The key checks are:
            coeff_sum_mean ~= 1
            basis_usage_sum ~= 1
            coeff_top_prob_mean >= 1 / num_basis
            soft_sum_mean ~= 1 when softmax_coeff=True
        """
        if self.last_coeff_map is None:
            return {}

        coeff = self.last_coeff_map.detach().float()
        coeff_sum = coeff.sum(dim=1)
        entropy_norm = -(
            coeff * torch.log(coeff + 1e-8)
        ).sum(dim=1).mean() / math.log(float(self.num_basis))

        out: Dict[str, Union[float, Tuple[int, ...]]] = {
            "coeff_shape": tuple(coeff.shape),
            "coeff_sum_mean": float(coeff_sum.mean().cpu()),
            "coeff_sum_std": float(coeff_sum.std(unbiased=False).cpu()),
            "coeff_sum_min": float(coeff_sum.min().cpu()),
            "coeff_sum_max": float(coeff_sum.max().cpu()),
            "coeff_min": float(coeff.min().cpu()),
            "coeff_max": float(coeff.max().cpu()),
            "coeff_mean": float(coeff.mean().cpu()),
            "coeff_top_prob_mean": float(coeff.amax(dim=1).mean().cpu()),
            "coeff_entropy_norm": float(entropy_norm.cpu()),
            "basis_usage_sum": float(coeff.mean(dim=(0, 2, 3)).sum().cpu()),
        }

        if self.coeff_predictor.last_logits is not None:
            logits = self.coeff_predictor.last_logits.detach().float()
            out.update({
                "logits_mean": float(logits.mean().cpu()),
                "logits_std": float(logits.std(unbiased=False).cpu()),
                "logits_min": float(logits.min().cpu()),
                "logits_max": float(logits.max().cpu()),
            })

        if self.coeff_predictor.last_soft is not None:
            soft = self.coeff_predictor.last_soft.detach().float()
            soft_sum = soft.sum(dim=1)
            soft_entropy_norm = -(
                soft * torch.log(soft + 1e-8)
            ).sum(dim=1).mean() / math.log(float(self.num_basis))
            out.update({
                "soft_sum_mean": float(soft_sum.mean().cpu()),
                "soft_sum_std": float(soft_sum.std(unbiased=False).cpu()),
                "soft_top_prob_mean": float(soft.amax(dim=1).mean().cpu()),
                "soft_entropy_norm": float(soft_entropy_norm.cpu()),
            })

        return out

    def get_last_monitor(self, to_float: bool = True) -> Dict[str, Union[float, torch.Tensor]]:
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
# PSF-like DegField V2-Light: weak residual score-gated prior
# ============================================================

def radial_correlation_map_train(
    score: torch.Tensor,
    coord: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Differentiable radial correlation for training loss.
    Do NOT detach score here.
    """
    if score.shape[1] > 1:
        score = score.mean(dim=1, keepdim=True)

    r = get_radius_from_coord(coord, size=score.shape[-2:])

    s = score.float().flatten(1)
    rv = r.float().flatten(1)

    s = s - s.mean(dim=1, keepdim=True)
    rv = rv - rv.mean(dim=1, keepdim=True)

    corr = (s * rv).mean(dim=1) / (
        s.std(dim=1, unbiased=False) * rv.std(dim=1, unbiased=False) + eps
    )
    return corr.mean()


def edge_center_gap_train(
    score: torch.Tensor,
    coord: torch.Tensor,
) -> torch.Tensor:
    """
    Differentiable edge-center gap for training loss.
    Do NOT detach score here.
    """
    if score.shape[1] > 1:
        score = score.mean(dim=1, keepdim=True)

    r = get_radius_from_coord(coord, size=score.shape[-2:])

    center_mask = (r <= 0.35).float()
    edge_mask = (r >= 0.75).float()

    center = (score * center_mask).sum() / (center_mask.sum() + 1e-6)
    edge = (score * edge_mask).sum() / (edge_mask.sum() + 1e-6)
    return edge - center


class Restormer_PSFLikeDegFieldV2Light(Restormer_PSFLikeDegField):
    """
    PSF-like DegField V2-Light.

    This class directly inherits Restormer_PSFLikeDegField, so it does NOT
    depend on a separate Restormer_PSFLikeDegFieldV2 definition.

    Weak residual gate:
        score_centered = deg_score - mean(deg_score)
        score_gate = 1 + gate_strength * tanh(gain) * score_centered
        deg_prior = deg_feat * score_gate

    If deg_score_gain_init = 0.0, tanh(gain)=0 at initialization, so the
    structure starts almost exactly from V1 behavior.
    """

    def __init__(
        self,
        *args,
        deg_score_gain_init: float = 0.0,
        deg_score_gate_strength: float = 0.10,
        score_init_std: float = 1e-3,
        clamp_gate: bool = True,
        gate_min: float = 0.80,
        gate_max: float = 1.20,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.deg_score_gain = nn.Parameter(torch.tensor(float(deg_score_gain_init)))
        self.deg_score_gate_strength = float(deg_score_gate_strength)
        self.clamp_gate = bool(clamp_gate)
        self.gate_min = float(gate_min)
        self.gate_max = float(gate_max)
        self.last_deg_prior: Optional[torch.Tensor] = None

        # Keep a tiny non-zero score head init to avoid exact constant symmetry.
        if hasattr(self.deg_estimator, "score"):
            nn.init.normal_(self.deg_estimator.score.weight, mean=0.0, std=float(score_init_std))
            nn.init.zeros_(self.deg_estimator.score.bias)

    def _make_score_gate(self, deg_score: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build weak residual gate from deg_score.

        Returns:
            score_gate, score_centered
        """
        score_centered = deg_score - deg_score.mean(dim=(2, 3), keepdim=True)
        gain = torch.tanh(self.deg_score_gain)
        score_gate = 1.0 + self.deg_score_gate_strength * gain * score_centered

        if self.clamp_gate:
            score_gate = torch.clamp(score_gate, min=self.gate_min, max=self.gate_max)

        return score_gate, score_centered

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

        score_gate, score_centered = self._make_score_gate(deg_score)
        deg_prior = deg_feat * score_gate

        self.last_deg_feat = deg_feat
        self.last_deg_score = deg_score
        self.last_deg_prior = deg_prior

        maybe_debug_tensor("psf_v2light_deg_feat", deg_feat)
        maybe_debug_tensor("psf_v2light_deg_score", deg_score)
        maybe_debug_tensor("psf_v2light_score_gate", score_gate)
        maybe_debug_tensor("psf_v2light_deg_prior", deg_prior)

        if return_monitor:
            monitor["deg_score_mean"] = _safe_mean(deg_score)
            monitor["deg_score_std"] = _safe_std(deg_score)
            monitor["deg_score_centered_std"] = _safe_std(score_centered)
            monitor["deg_feat_abs_mean"] = _safe_mean(deg_feat.abs())
            monitor["deg_prior_abs_mean"] = _safe_mean(deg_prior.abs())
            monitor["score_gate_mean"] = _safe_mean(score_gate)
            monitor["score_gate_std"] = _safe_std(score_gate)
            monitor["score_gate_min"] = score_gate.detach().float().min()
            monitor["score_gate_max"] = score_gate.detach().float().max()
            monitor["deg_score_gain_raw"] = self.deg_score_gain.detach().float()
            monitor["deg_score_gain_tanh"] = torch.tanh(self.deg_score_gain.detach().float())
            monitor["deg_score_gate_strength"] = torch.as_tensor(
                self.deg_score_gate_strength,
                device=deg_score.device,
                dtype=deg_score.dtype,
            )
            monitor["deg_field_tv"] = self.deg_smoothness_loss().detach()

            if coord_map is not None:
                monitor["deg_score_edge_center_gap"] = edge_center_gap(deg_score, coord_map)
                monitor["deg_score_radial_corr"] = radial_correlation_map(deg_score, coord_map)

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
                    x, st = self.mod_encoders[i](
                        x,
                        deg_prior,
                        return_stats=True,
                        name=f"enc{i+1}_",
                    )
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
                    x, st = self.mod_decoders[dec_idx](
                        x,
                        deg_prior,
                        return_stats=True,
                        name=f"dec{dec_idx+1}_",
                    )
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

    def deg_radial_prior_loss(
        self,
        min_corr: float = 0.10,
        require_edge_larger: bool = False,
        min_gap: float = 0.00,
    ) -> torch.Tensor:
        """
        Differentiable weak radial prior for V2-Light.
        """
        if self.last_deg_score is None or self.last_coord_map is None:
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device)

        corr = radial_correlation_map_train(self.last_deg_score, self.last_coord_map)
        loss = F.relu(
            torch.as_tensor(min_corr, device=corr.device, dtype=corr.dtype) - corr
        )

        if require_edge_larger:
            gap = edge_center_gap_train(self.last_deg_score, self.last_coord_map)
            loss = loss + F.relu(
                torch.as_tensor(min_gap, device=gap.device, dtype=gap.dtype) - gap
            )

        return loss

    def get_deg_grid_embedding(
        self,
        grid_size: Tuple[int, int] = (9, 9),
        normalize: bool = True,
        detach: bool = False,
    ) -> torch.Tensor:
        """
        Pool the score-gated degradation prior into grid embeddings.
        """
        if self.last_deg_prior is not None:
            source = self.last_deg_prior
        elif self.last_deg_feat is not None:
            source = self.last_deg_feat
        else:
            raise RuntimeError("last_deg_prior / last_deg_feat is None. Call forward() first.")

        z = F.adaptive_avg_pool2d(source, grid_size)
        b, c, gh, gw = z.shape
        z = z.permute(0, 2, 3, 1).reshape(b * gh * gw, c)

        if normalize:
            z = F.normalize(z.float(), dim=1)

        if detach:
            z = z.detach()

        return z


# Backward-compatible alias.
# If an old run file imports Restormer_PSFLikeDegFieldV2, it will not crash.
Restormer_PSFLikeDegFieldV2 = Restormer_PSFLikeDegFieldV2Light


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

    print("Testing Restormer_PSFLikeDegField...")
    model1 = Restormer_PSFLikeDegField(
        **common_args,
        use_coord=True,
        coord_channels=4,
        deg_ch=8,
        deg_downsample_factor=4,
        modulate_levels=("shallow", "enc", "latent", "dec"),
    ).to(device)

    y1, mon1 = model1(x, return_monitor=True)
    z1 = model1.get_deg_grid_embedding(grid_size=(9, 9))
    labels1 = make_field_position_labels(
        batch_size=x.shape[0],
        grid_size=(9, 9),
        device=z1.device,
    )
    ctr1 = supervised_nt_xent(z1, labels1)
    print("Output:", y1.shape)
    print("Embedding:", z1.shape, "Contrastive loss:", float(ctr1.detach().cpu()))
    print("Monitor keys:", list(mon1.keys())[:10])

    print("Testing Restormer_PSFLikeLowRankBasis...")
    model2 = Restormer_PSFLikeLowRankBasis(
        **common_args,
        use_coord=True,
        coord_channels=4,
        prior_ch=16,
        num_basis=4,
        coeff_downsample_factor=4,
        coeff_floor=0.05,
        modulate_levels=("shallow", "enc", "latent", "dec"),
    ).to(device)

    y2, mon2 = model2(x, return_monitor=True)
    z2 = model2.get_coeff_grid_embedding(grid_size=(9, 9))
    labels2 = make_field_position_labels(
        batch_size=x.shape[0],
        grid_size=(9, 9),
        device=z2.device,
    )
    ctr2 = supervised_nt_xent(z2, labels2)
    print("Output:", y2.shape)
    print("Embedding:", z2.shape, "Contrastive loss:", float(ctr2.detach().cpu()))
    print("Monitor keys:", list(mon2.keys())[:10])
