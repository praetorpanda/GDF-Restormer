"""Fixed-split full-canvas training, fast validation and tiled evaluation."""
from __future__ import annotations

import csv
import gc
import math
import os
import re
import socket
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.utils.data import DataLoader
from torchmetrics.functional import peak_signal_noise_ratio
from torchmetrics.functional import structural_similarity_index_measure as ssim
from tqdm import tqdm

try:
    import cv2
except Exception as exc:  # pragma: no cover
    cv2 = None
    _CV2_IMPORT_ERROR = exc
else:
    _CV2_IMPORT_ERROR = None

from config import Config
from gdf_loss import SSIMLoss
from utils import seed_everything
from data.dataset_GlobalMeta_FullCanvas import build_globalmeta_fullcanvas_datasets as build_globalmeta_rectroi_datasets

try:
    from muon import MuonWithAuxAdam
except Exception:
    MuonWithAuxAdam = None

_CREATED_PG = False
PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# Unified comparison settings.
PATCH_SIZE = 256
ROI_W = 768
ROI_H = 640
VAL_CROP_MODE = "grid9"

# Spatial error bins for global-radius monitoring.
GLOBAL_CENTER_RADIUS_THR = 0.30
GLOBAL_EDGE_RADIUS_THR = 0.55


# ============================================================
# Distributed helpers
# ============================================================
def _find_free_port() -> int:
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return int(port)


def _ensure_dist_initialized(use_muon: bool) -> None:
    global _CREATED_PG
    if use_muon and dist.is_available() and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        url = f"tcp://127.0.0.1:{_find_free_port()}"
        dist.init_process_group(
            backend=backend,
            init_method=url,
            rank=0,
            world_size=1,
        )
        _CREATED_PG = True


def cleanup_dist_if_needed() -> None:
    global _CREATED_PG
    if _CREATED_PG and dist.is_available() and dist.is_initialized():
        print("[INFO] Destroying process group...")
        dist.destroy_process_group()
        _CREATED_PG = False


# ============================================================
# Dataset builder: no temporary split creation here
# ============================================================
def build_all_datasets_for_globalcoord_6500(
    train_gt_dir: str,
    train_meta_dir: str,
    val_gt_dir: str,
    val_meta_dir: str,
    train_repeat: int = 1,
    patch_size: int = PATCH_SIZE,
    roi_w: int = ROI_W,
    roi_h: int = ROI_H,
    val_crop_mode: str = VAL_CROP_MODE,
):
    """Build online-coordinate FullCanvas datasets from existing fixed split folders."""
    return build_globalmeta_rectroi_datasets(
        train_gt_dir=train_gt_dir,
        train_meta_dir=train_meta_dir,
        test_gt_dir=val_gt_dir,
        test_meta_dir=val_meta_dir,
        patch_size=patch_size,
        roi_w=roi_w,
        roi_h=roi_h,
        train_repeat=train_repeat,
        val_crop_mode=val_crop_mode,
    )


# ============================================================
# Model/stat helpers
# ============================================================
def compute_model_params(model: torch.nn.Module) -> Dict[str, int]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": total_params, "trainable_params": trainable_params}


def _model_forward_for_stats(
    model: torch.nn.Module,
    dummy: torch.Tensor,
    coord: Optional[torch.Tensor],
    pass_external_coord: bool = True,
) -> torch.Tensor:
    if pass_external_coord and coord is not None:
        try:
            return model(dummy, coord=coord)
        except TypeError:
            return model(dummy)
    return model(dummy)


def compute_inference_time(
    model: torch.nn.Module,
    device: torch.device,
    patch_size: int = PATCH_SIZE,
    runs: int = 50,
    pass_external_coord: bool = True,
) -> Dict[str, float]:
    dummy = torch.randn(1, 3, patch_size, patch_size, device=device)
    coord = torch.zeros(1, 2, patch_size, patch_size, device=device) if pass_external_coord else None
    model.eval()

    if runs <= 0:
        return {"infer_time_avg": None, "infer_time_std": None, "infer_runs": 0}

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(10):
            _ = _model_forward_for_stats(model, dummy, coord, pass_external_coord=pass_external_coord)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    timings = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.time()
            _ = _model_forward_for_stats(model, dummy, coord, pass_external_coord=pass_external_coord)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            timings.append((time.time() - t0) * 1000.0)

    arr = np.array(timings, dtype=np.float64)
    return {
        "infer_time_avg": float(arr.mean()),
        "infer_time_std": float(arr.std()),
        "infer_runs": int(len(timings)),
    }


def analyze_model_statistics(
    model: torch.nn.Module,
    device: torch.device,
    patch_size: int = PATCH_SIZE,
    runs: int = 50,
    pass_external_coord: bool = True,
) -> Dict[str, object]:
    stats_params = compute_model_params(model)
    latency = compute_inference_time(
        model,
        device=device,
        patch_size=patch_size,
        runs=runs,
        pass_external_coord=pass_external_coord,
    )
    stats = {
        **stats_params,
        "flops": -1,
        **latency,
        "resolution": (patch_size, patch_size),
    }

    msg = (
        f"[MODEL] params={stats['total_params']:,} | "
        f"trainable={stats['trainable_params']:,} | "
        f"patch={stats['resolution']}"
    )
    if stats["infer_time_avg"] is not None:
        msg += f" | infer={stats['infer_time_avg']:.3f} ms"
    print(msg)
    return stats


def build_optimizer(
    model: torch.nn.Module,
    base_lr: float,
    weight_decay: float,
    use_muon: bool,
    muon_lr: Optional[float] = None,
    muon_lr_mult: float = 5,
):
    if not use_muon:
        return optim.AdamW(model.parameters(), lr=base_lr, betas=(0.9, 0.999), weight_decay=weight_decay)

    if MuonWithAuxAdam is None:
        raise RuntimeError("USE_MUON=True, but MuonWithAuxAdam is not installed.")

    conv_kernels, others = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 4:
            conv_kernels.append(p)
        else:
            others.append(p)

    muon_lr = muon_lr if muon_lr is not None else base_lr * muon_lr_mult
    param_groups = []
    if conv_kernels:
        param_groups.append(dict(params=conv_kernels, use_muon=True, lr=muon_lr, weight_decay=weight_decay))
    if others:
        param_groups.append(dict(params=others, use_muon=False, lr=base_lr, betas=(0.9, 0.999), weight_decay=weight_decay))
    return MuonWithAuxAdam(param_groups)


def _unwrap_model_if_needed(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _forward_with_aux(
    model: torch.nn.Module,
    inp: torch.Tensor,
    coord: Optional[torch.Tensor] = None,
    pass_external_coord: bool = True,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Unified forward adapter for models.

    pass_external_coord=True:
        use full-image global coord, required by GlobalCoordRestormer and Global-DegField.
    pass_external_coord=False:
        call the model without coord so non-global/local DegField variants keep their
        original internal/local coordinate behavior.
    """
    if pass_external_coord and coord is not None:
        try:
            out = model(inp, coord=coord, return_aux=True)
            if isinstance(out, tuple) and len(out) == 2:
                return out[0], out[1] or {}
            return out, {}
        except TypeError:
            pass

        try:
            out = model(inp, coord=coord, return_monitor=True)
            if isinstance(out, tuple) and len(out) == 2:
                return out[0], out[1] or {}
            return out, {}
        except TypeError:
            pred = model(inp, coord=coord)
            return pred, {}

    try:
        out = model(inp, return_aux=True)
        if isinstance(out, tuple) and len(out) == 2:
            return out[0], out[1] or {}
        return out, {}
    except TypeError:
        pass

    try:
        out = model(inp, return_monitor=True)
        if isinstance(out, tuple) and len(out) == 2:
            return out[0], out[1] or {}
        return out, {}
    except TypeError:
        return model(inp), {}


def _optional_prior_regularization(
    model: torch.nn.Module,
    lambda_basis: float = 0.0,
    lambda_deg_smooth: float = 0.0,
    lambda_coeff_smooth: float = 0.0,
    lambda_coeff_entropy: float = 0.0,
    coeff_entropy_target: float = 0.65,
    coeff_entropy_mode: str = "min",
    lambda_deg_radial: float = 0.0,
    deg_radial_min_corr: float = 0.10,
    deg_radial_min_gap: float = 0.0,
    deg_radial_require_edge_larger: bool = False,
):
    raw_model = _unwrap_model_if_needed(model)
    reg = None
    reg_terms: Dict[str, float] = {}

    def _add(name: str, value: torch.Tensor, weight: float):
        nonlocal reg
        if weight <= 0:
            return
        term = value * weight
        reg = term if reg is None else reg + term
        try:
            reg_terms[name] = float(value.detach().float().mean().cpu().item())
        except Exception:
            pass

    if lambda_basis > 0 and hasattr(raw_model, "basis_diversity_loss"):
        _add("basis_diversity", raw_model.basis_diversity_loss(), lambda_basis)

    if lambda_deg_smooth > 0 and hasattr(raw_model, "deg_smoothness_loss"):
        _add("deg_smooth", raw_model.deg_smoothness_loss(), lambda_deg_smooth)

    if lambda_coeff_smooth > 0 and hasattr(raw_model, "coeff_smoothness_loss"):
        _add("coeff_smooth", raw_model.coeff_smoothness_loss(), lambda_coeff_smooth)

    if lambda_coeff_entropy > 0 and hasattr(raw_model, "coeff_entropy_loss"):
        try:
            ent = raw_model.coeff_entropy_loss(target_entropy=coeff_entropy_target, mode=coeff_entropy_mode)
        except TypeError:
            ent = raw_model.coeff_entropy_loss()
        _add("coeff_entropy", ent, lambda_coeff_entropy)

    if lambda_deg_radial > 0 and hasattr(raw_model, "deg_radial_prior_loss"):
        try:
            radial = raw_model.deg_radial_prior_loss(
                min_corr=deg_radial_min_corr,
                min_gap=deg_radial_min_gap,
                require_edge_larger=deg_radial_require_edge_larger,
            )
        except TypeError:
            radial = raw_model.deg_radial_prior_loss(min_corr=deg_radial_min_corr)
        _add("deg_radial", radial, lambda_deg_radial)

    if reg is None:
        device = next(raw_model.parameters()).device
        reg = torch.tensor(0.0, device=device)
    return reg, reg_terms


# ============================================================
# Tensor metrics / spatial stats for crop validation
# ============================================================
def compute_pair_metrics_tensor(pred: torch.Tensor, tar: torch.Tensor) -> Dict[str, float]:
    pred_metric = torch.clamp(pred, 0.0, 1.0)
    tar_metric = torch.clamp(tar, 0.0, 1.0)
    return {
        "psnr": float(peak_signal_noise_ratio(pred_metric, tar_metric, data_range=1.0).detach().item()),
        "ssim": float(ssim(pred_metric, tar_metric, data_range=1.0).detach().item()),
        "l1": float(torch.mean(torch.abs(pred_metric - tar_metric)).detach().item()),
    }


def _safe_masked_mean(x: torch.Tensor, mask: torch.Tensor):
    if mask is None or mask.sum().item() <= 0:
        return None
    return x[mask].mean()


def compute_global_radius_error_stats(pred: torch.Tensor, tar: torch.Tensor, global_radius: torch.Tensor) -> Dict[str, object]:
    with torch.no_grad():
        err = (pred.detach() - tar.detach()).abs().mean(dim=1, keepdim=True)
        r = global_radius.to(device=err.device, dtype=err.dtype)

        center_mask = r < GLOBAL_CENTER_RADIUS_THR
        middle_mask = (r >= GLOBAL_CENTER_RADIUS_THR) & (r < GLOBAL_EDGE_RADIUS_THR)
        edge_mask = r >= GLOBAL_EDGE_RADIUS_THR

        center = _safe_masked_mean(err, center_mask)
        middle = _safe_masked_mean(err, middle_mask)
        edge = _safe_masked_mean(err, edge_mask)

        return {
            "l1_center": float(center.item()) if center is not None else 0.0,
            "l1_middle": float(middle.item()) if middle is not None else 0.0,
            "l1_edge": float(edge.item()) if edge is not None else 0.0,
            "has_center": center is not None,
            "has_middle": middle is not None,
            "has_edge": edge is not None,
        }


def _prepare_meta_clean_global_batch(clean_crop, meta_crop, global_coord, device: torch.device):
    clean_crop = clean_crop.to(device)
    meta_crop = meta_crop.to(device)
    global_coord = global_coord.to(device)
    return meta_crop, clean_crop, global_coord


# ============================================================
# Crop-level training meters
# ============================================================
def _safe_scalar(v):
    if v is None:
        return None
    if isinstance(v, (float, int)):
        return float(v)
    if torch.is_tensor(v):
        if v.numel() == 1:
            return float(v.detach().item())
        return float(v.detach().float().mean().item())
    return None


def _init_meter():
    return {
        "loss_total": 0.0,
        "loss_l1": 0.0,
        "loss_ssim": 0.0,
        "pred_mean": 0.0,
        "pred_std": 0.0,
        "tar_mean": 0.0,
        "tar_std": 0.0,
        "pred_res_mean": 0.0,
        "pred_res_std": 0.0,
        "tar_res_mean": 0.0,
        "tar_res_std": 0.0,
        "global_l1_center": 0.0,
        "global_l1_middle": 0.0,
        "global_l1_edge": 0.0,
        "global_center_count": 0,
        "global_middle_count": 0,
        "global_edge_count": 0,
        "aux_sum": {},
        "aux_batches": 0,
        "count": 0,
    }


def _update_meter(meter, pred, tar, inp, global_radius, loss_total, loss_l1, loss_ssim, aux=None):
    with torch.no_grad():
        pred_det = pred.detach()
        tar_det = tar.detach()
        inp_det = inp.detach()
        pred_res = pred_det - inp_det
        tar_res = tar_det - inp_det

        meter["loss_total"] += float(loss_total.detach().item())
        meter["loss_l1"] += float(loss_l1.detach().item())
        meter["loss_ssim"] += float(loss_ssim.detach().item())
        meter["pred_mean"] += float(pred_det.mean().item())
        meter["pred_std"] += float(pred_det.std(unbiased=False).item())
        meter["tar_mean"] += float(tar_det.mean().item())
        meter["tar_std"] += float(tar_det.std(unbiased=False).item())
        meter["pred_res_mean"] += float(pred_res.mean().item())
        meter["pred_res_std"] += float(pred_res.std(unbiased=False).item())
        meter["tar_res_mean"] += float(tar_res.mean().item())
        meter["tar_res_std"] += float(tar_res.std(unbiased=False).item())

        gstats = compute_global_radius_error_stats(pred_det, tar_det, global_radius)
        if gstats["has_center"]:
            meter["global_l1_center"] += gstats["l1_center"]
            meter["global_center_count"] += 1
        if gstats["has_middle"]:
            meter["global_l1_middle"] += gstats["l1_middle"]
            meter["global_middle_count"] += 1
        if gstats["has_edge"]:
            meter["global_l1_edge"] += gstats["l1_edge"]
            meter["global_edge_count"] += 1

        if aux is not None and len(aux) > 0:
            meter["aux_batches"] += 1
            for k, v in aux.items():
                val = _safe_scalar(v)
                if val is None:
                    continue
                meter["aux_sum"][k] = meter["aux_sum"].get(k, 0.0) + val

        meter["count"] += 1


def _format_meter(meter):
    cnt = max(meter["count"], 1)
    center_cnt = max(meter["global_center_count"], 1)
    middle_cnt = max(meter["global_middle_count"], 1)
    edge_cnt = max(meter["global_edge_count"], 1)
    aux_cnt = max(meter["aux_batches"], 1)

    stats = {
        "TrainLoss": meter["loss_total"] / cnt,
        "L1": meter["loss_l1"] / cnt,
        "SSIMLoss": meter["loss_ssim"] / cnt,
        "PredMean": meter["pred_mean"] / cnt,
        "PredStd": meter["pred_std"] / cnt,
        "TarMean": meter["tar_mean"] / cnt,
        "TarStd": meter["tar_std"] / cnt,
        "PredResMean": meter["pred_res_mean"] / cnt,
        "PredResStd": meter["pred_res_std"] / cnt,
        "TarResMean": meter["tar_res_mean"] / cnt,
        "TarResStd": meter["tar_res_std"] / cnt,
        "GlobalL1Center": meter["global_l1_center"] / center_cnt,
        "GlobalL1Middle": meter["global_l1_middle"] / middle_cnt,
        "GlobalL1Edge": meter["global_l1_edge"] / edge_cnt,
        "GlobalCenterCount": meter["global_center_count"],
        "GlobalMiddleCount": meter["global_middle_count"],
        "GlobalEdgeCount": meter["global_edge_count"],
        "AuxAvailRate": meter["aux_batches"] / cnt,
    }
    for k, v in meter["aux_sum"].items():
        stats[k] = v / aux_cnt
    return stats


def _short_aux_string(stats: Dict[str, object], max_items: int = 18) -> str:
    preferred = [
        "deg_score_mean", "deg_score_std", "deg_score_edge_center_gap", "deg_score_radial_corr",
        "deg_feat_abs_mean", "scale_usage_std", "scale_router_entropy_norm", "scale_top_prob_mean",
        "scale_usage_small", "scale_usage_middle", "scale_usage_large", "scale_usage_illumination",
        "scale_usage_wavelet", "scale_wavelet_radial_corr", "wavelet_ll_abs_mean", "wavelet_hf_energy_mean",
        "illum_low_mean", "reflect_proxy_mean", "global_coord_std", "global_radius_mean",
        "reg_deg_smooth", "reg_coeff_smooth", "reg_basis_diversity", "reg_coeff_entropy", "reg_deg_radial",
    ]
    keys = [k for k in preferred if k in stats]
    if not keys:
        keys = [k for k in stats.keys() if isinstance(stats[k], (float, int))]
    keys = keys[:max_items]
    if not keys:
        return ""
    return " | " + " | ".join([f"{k}={float(stats[k]):.6f}" for k in keys if isinstance(stats[k], (float, int))])


def _append_aux_to_file(f, stats: Dict[str, object]) -> None:
    if not stats:
        return
    f.write("\nLast Epoch Global/Prior Monitor:\n")
    for k in sorted(stats.keys()):
        v = stats[k]
        if isinstance(v, (float, int)):
            f.write(f"  {k:<34}: {v:.8f}\n")
        else:
            f.write(f"  {k:<34}: {v}\n")
    f.write("\n")


# ============================================================
# Numpy/CV2 unified-eval helpers copied in spirit from official runners
# ============================================================
def require_cv2() -> None:
    if cv2 is None:
        raise RuntimeError(f"OpenCV/cv2 is required by unified eval, but import failed: {_CV2_IMPORT_ERROR}")


def list_images(folder: Path) -> List[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Image folder does not exist: {folder}")
    files = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    files.sort(key=lambda x: str(x.relative_to(folder)))
    return files


def normalize_key(stem: str) -> str:
    key = stem
    common_suffixes = [
        "_smoke_low", "_smoke_mid", "_smoke_high", "_low", "_mid", "_high",
        "_meta", "_lq", "_input", "_degraded", "_blur", "_masked",
    ]
    for suffix in common_suffixes:
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return key


def build_pairs(gt_dir: Path, lq_dir: Path, allow_fuzzy_pairing: bool = False) -> List[Tuple[Path, Path]]:
    gt_files = list_images(gt_dir)
    lq_files = list_images(lq_dir)
    if len(gt_files) == 0:
        raise RuntimeError(f"No GT images found in {gt_dir}")
    if len(lq_files) == 0:
        raise RuntimeError(f"No LQ/META images found in {lq_dir}")

    lq_by_rel_noext: Dict[str, Path] = {}
    lq_by_stem: Dict[str, List[Path]] = {}
    lq_by_norm: Dict[str, List[Path]] = {}
    for p in lq_files:
        rel_noext = str(p.relative_to(lq_dir).with_suffix(""))
        lq_by_rel_noext[rel_noext] = p
        lq_by_stem.setdefault(p.stem, []).append(p)
        lq_by_norm.setdefault(normalize_key(p.stem), []).append(p)

    pairs: List[Tuple[Path, Path]] = []
    missing: List[str] = []
    used_lq = set()
    for gt in gt_files:
        rel_noext = str(gt.relative_to(gt_dir).with_suffix(""))
        lq = lq_by_rel_noext.get(rel_noext)
        if lq is None:
            candidates = lq_by_stem.get(gt.stem, [])
            if len(candidates) == 1:
                lq = candidates[0]
        if lq is None and allow_fuzzy_pairing:
            candidates = lq_by_norm.get(normalize_key(gt.stem), [])
            if len(candidates) == 1:
                lq = candidates[0]
            elif len(candidates) > 1:
                unused = [c for c in candidates if c not in used_lq]
                lq = (unused or candidates)[0]
        if lq is None:
            missing.append(str(gt.relative_to(gt_dir)))
        else:
            used_lq.add(lq)
            pairs.append((gt, lq))

    pairs.sort(key=lambda x: str(x[0].relative_to(gt_dir)))
    if not pairs:
        raise RuntimeError(f"No matching pairs found between GT={gt_dir} and LQ/META={lq_dir}.")
    if missing:
        print(f"[WARN] {len(missing)} GT files had no LQ/META match. First few: {missing[:5]}")
    return pairs


def read_rgb_float(path: Path) -> np.ndarray:
    require_cv2()
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def save_rgb_float(path: Path, img: np.ndarray) -> None:
    require_cv2()
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.clip(img, 0.0, 1.0)
    bgr = cv2.cvtColor((img * 255.0 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def calc_l1(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def calc_psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * math.log10(1.0 / mse))


def _ssim_single_channel(x: np.ndarray, y: np.ndarray) -> float:
    require_cv2()
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    kernel = (11, 11)
    sigma = 1.5
    mux = cv2.GaussianBlur(x, kernel, sigma)
    muy = cv2.GaussianBlur(y, kernel, sigma)
    mux2 = mux * mux
    muy2 = muy * muy
    muxy = mux * muy
    sigx2 = cv2.GaussianBlur(x * x, kernel, sigma) - mux2
    sigy2 = cv2.GaussianBlur(y * y, kernel, sigma) - muy2
    sigxy = cv2.GaussianBlur(x * y, kernel, sigma) - muxy
    ssim_map = ((2 * muxy + c1) * (2 * sigxy + c2)) / ((mux2 + muy2 + c1) * (sigx2 + sigy2 + c2) + 1e-12)
    return float(ssim_map.mean())


def calc_ssim(a: np.ndarray, b: np.ndarray) -> float:
    if a.ndim == 2:
        return _ssim_single_channel(a, b)
    return float(np.mean([_ssim_single_channel(a[..., c], b[..., c]) for c in range(a.shape[2])]))


def ensure_same_size(a: np.ndarray, b: np.ndarray, pa: Path, pb: Path) -> None:
    if a.shape != b.shape:
        raise RuntimeError(f"Image size mismatch: {pa} {a.shape} vs {pb} {b.shape}")


def center_rect_roi_slice(h: int, w: int, roi_w: int, roi_h: int) -> Tuple[int, int, int, int]:
    rw = min(int(roi_w), w)
    rh = min(int(roi_h), h)
    x0 = max(0, (w - rw) // 2)
    y0 = max(0, (h - rh) // 2)
    return y0, y0 + rh, x0, x0 + rw


def center_rect_roi(img: np.ndarray, roi_w: int, roi_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    y0, y1, x0, x1 = center_rect_roi_slice(h, w, roi_w, roi_h)
    return img[y0:y1, x0:x1]


def grid9_slices(h: int, w: int, crop_size: int) -> List[Tuple[int, int, int, int]]:
    ch = min(int(crop_size), h)
    cw = min(int(crop_size), w)
    ys = [0, max(0, (h - ch) // 2), max(0, h - ch)]
    xs = [0, max(0, (w - cw) // 2), max(0, w - cw)]
    out: List[Tuple[int, int, int, int]] = []
    seen = set()
    for y in ys:
        for x in xs:
            key = (y, x)
            if key in seen:
                continue
            seen.add(key)
            out.append((y, y + ch, x, x + cw))
    return out


def grid9_slices_in_full_image(full_h: int, full_w: int, roi_w: int, roi_h: int, crop_size: int) -> List[Tuple[int, int, int, int]]:
    ry0, ry1, rx0, rx1 = center_rect_roi_slice(full_h, full_w, roi_w, roi_h)
    rh, rw = ry1 - ry0, rx1 - rx0
    out = []
    for y0, y1, x0, x1 in grid9_slices(rh, rw, crop_size):
        out.append((ry0 + y0, ry0 + y1, rx0 + x0, rx0 + x1))
    return out


def build_global_coord_for_region(
    full_h: int,
    full_w: int,
    y0: int,
    x0: int,
    h: int,
    w: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ys = np.arange(h, dtype=np.float32) + float(y0)
    xs = np.arange(w, dtype=np.float32) + float(x0)
    ys = np.clip(ys, 0, max(full_h - 1, 0))
    xs = np.clip(xs, 0, max(full_w - 1, 0))

    if full_h <= 1:
        y_norm = np.zeros_like(ys, dtype=np.float32)
    else:
        y_norm = 2.0 * ys / float(full_h - 1) - 1.0
    if full_w <= 1:
        x_norm = np.zeros_like(xs, dtype=np.float32)
    else:
        x_norm = 2.0 * xs / float(full_w - 1) - 1.0

    yy, xx = np.meshgrid(y_norm, x_norm, indexing="ij")
    coord = np.stack([xx, yy], axis=0).astype(np.float32)
    radius = np.sqrt(xx ** 2 + yy ** 2) / np.sqrt(2.0)
    radius = np.clip(radius, 0.0, 1.0).astype(np.float32)[None, ...]
    return torch.from_numpy(coord), torch.from_numpy(radius)


def _pad_tensor_to_multiple(x: torch.Tensor, multiple: int, mode: str = "reflect") -> Tuple[torch.Tensor, Tuple[int, int]]:
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h > 0 or pad_w > 0:
        actual_mode = mode
        if h <= 1 or w <= 1:
            actual_mode = "replicate"
        x = F.pad(x, (0, pad_w, 0, pad_h), mode=actual_mode)
    return x, (pad_h, pad_w)


def _crop_tensor_back(x: torch.Tensor, pad_hw: Tuple[int, int]) -> torch.Tensor:
    pad_h, pad_w = pad_hw
    if pad_h:
        x = x[..., :-pad_h, :]
    if pad_w:
        x = x[..., :, :-pad_w]
    return x


def run_model_on_region_global(
    model: torch.nn.Module,
    lq_rgb: np.ndarray,
    full_h: int,
    full_w: int,
    y0: int,
    x0: int,
    device: Union[str, torch.device],
    window_size: int = 8,
    pass_external_coord: bool = True,
) -> np.ndarray:
    """Run model on one image region with its corresponding global coordinate map."""
    h, w = lq_rgb.shape[:2]
    x = torch.from_numpy(lq_rgb.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    coord = None
    if pass_external_coord:
        coord, _ = build_global_coord_for_region(full_h=full_h, full_w=full_w, y0=y0, x0=x0, h=h, w=w)
        coord = coord.unsqueeze(0).to(device=device, dtype=x.dtype)

    x, pad_hw = _pad_tensor_to_multiple(x, window_size, mode="reflect")
    if coord is not None:
        coord, _ = _pad_tensor_to_multiple(coord, window_size, mode="replicate")

    with torch.no_grad():
        pred, _ = _forward_with_aux(model, x, coord=coord, pass_external_coord=pass_external_coord)
    pred = _crop_tensor_back(pred, pad_hw)
    pred = pred[:, :, :h, :w].clamp(0, 1)
    out = pred.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    return out.astype(np.float32)


def _positions(length: int, tile: int, overlap: int) -> List[int]:
    tile = int(tile)
    overlap = int(overlap)
    if length <= tile:
        return [0]
    stride = max(1, tile - overlap)
    pos = list(range(0, max(1, length - tile + 1), stride))
    last = length - tile
    if pos[-1] != last:
        pos.append(last)
    return pos


def run_model_on_image_global_tiled(
    model: torch.nn.Module,
    lq_rgb: np.ndarray,
    device: Union[str, torch.device],
    window_size: int = 8,
    tile: int = 256,
    overlap: int = 32,
    pass_external_coord: bool = True,
) -> np.ndarray:
    """Full-image tiled inference with correct global coordinates per tile."""
    full_h, full_w = lq_rgb.shape[:2]
    if full_h <= tile and full_w <= tile:
        return run_model_on_region_global(
            model, lq_rgb, full_h, full_w, 0, 0, device, window_size,
            pass_external_coord=pass_external_coord,
        )

    ys = _positions(full_h, tile, overlap)
    xs = _positions(full_w, tile, overlap)
    acc = np.zeros((full_h, full_w, 3), dtype=np.float32)
    weight = np.zeros((full_h, full_w, 1), dtype=np.float32)

    for y0 in ys:
        for x0 in xs:
            y1 = min(y0 + tile, full_h)
            x1 = min(x0 + tile, full_w)
            patch = lq_rgb[y0:y1, x0:x1, :]
            pred_patch = run_model_on_region_global(
                model,
                patch,
                full_h=full_h,
                full_w=full_w,
                y0=y0,
                x0=x0,
                device=device,
                window_size=window_size,
                pass_external_coord=pass_external_coord,
            )
            acc[y0:y1, x0:x1, :] += pred_patch[: y1 - y0, : x1 - x0, :]
            weight[y0:y1, x0:x1, :] += 1.0

    return acc / np.maximum(weight, 1e-6)


def evaluate_raw_pairs(gt_dir: Path, lq_dir: Path, max_images: Optional[int] = None) -> Dict[str, object]:
    pairs = build_pairs(gt_dir, lq_dir, allow_fuzzy_pairing=False)
    if max_images is not None and max_images > 0:
        pairs = pairs[:max_images]
    psnr_vals, ssim_vals, l1_vals = [], [], []
    for gt_path, lq_path in tqdm(pairs, desc="Raw full eval", leave=False):
        gt = read_rgb_float(gt_path)
        lq = read_rgb_float(lq_path)
        ensure_same_size(gt, lq, gt_path, lq_path)
        psnr_vals.append(calc_psnr(lq, gt))
        ssim_vals.append(calc_ssim(lq, gt))
        l1_vals.append(calc_l1(lq, gt))
    return {
        "count": len(pairs),
        "psnr": float(np.mean(psnr_vals)) if psnr_vals else None,
        "ssim": float(np.mean(ssim_vals)) if ssim_vals else None,
        "l1": float(np.mean(l1_vals)) if l1_vals else None,
    }


def evaluate_model_pairs_global(
    model: torch.nn.Module,
    gt_dir: Path,
    lq_dir: Path,
    device: Union[str, torch.device],
    window_size: int = 8,
    tile: int = 256,
    overlap: int = 32,
    max_images: Optional[int] = None,
    pass_external_coord: bool = True,
) -> Dict[str, object]:
    pairs = build_pairs(gt_dir, lq_dir, allow_fuzzy_pairing=False)
    if max_images is not None and max_images > 0:
        pairs = pairs[:max_images]
    psnr_vals, ssim_vals, l1_vals = [], [], []
    start = time.perf_counter()
    for gt_path, lq_path in tqdm(pairs, desc="Global full eval", leave=False):
        gt = read_rgb_float(gt_path)
        lq = read_rgb_float(lq_path)
        ensure_same_size(gt, lq, gt_path, lq_path)
        pred = run_model_on_image_global_tiled(
            model,
            lq,
            device=device,
            window_size=window_size,
            tile=tile,
            overlap=overlap,
            pass_external_coord=pass_external_coord,
        )
        psnr_vals.append(calc_psnr(pred, gt))
        ssim_vals.append(calc_ssim(pred, gt))
        l1_vals.append(calc_l1(pred, gt))
    elapsed = time.perf_counter() - start
    return {
        "count": len(pairs),
        "psnr": float(np.mean(psnr_vals)) if psnr_vals else None,
        "ssim": float(np.mean(ssim_vals)) if ssim_vals else None,
        "l1": float(np.mean(l1_vals)) if l1_vals else None,
        "elapsed_sec": float(elapsed),
    }


def evaluate_raw_pairs_grid9(gt_dir: Path, lq_dir: Path, roi_w: int, roi_h: int, crop_size: int, max_images: Optional[int] = None) -> Dict[str, object]:
    pairs = build_pairs(gt_dir, lq_dir, allow_fuzzy_pairing=False)
    if max_images is not None and max_images > 0:
        pairs = pairs[:max_images]
    psnr_vals, ssim_vals, l1_vals = [], [], []
    crop_count = 0
    for gt_path, lq_path in tqdm(pairs, desc="Raw grid9 eval", leave=False):
        gt = read_rgb_float(gt_path)
        lq = read_rgb_float(lq_path)
        ensure_same_size(gt, lq, gt_path, lq_path)
        gt_roi = center_rect_roi(gt, roi_w, roi_h)
        lq_roi = center_rect_roi(lq, roi_w, roi_h)
        h, w = gt_roi.shape[:2]
        for y0, y1, x0, x1 in grid9_slices(h, w, crop_size):
            gt_crop = gt_roi[y0:y1, x0:x1]
            lq_crop = lq_roi[y0:y1, x0:x1]
            psnr_vals.append(calc_psnr(lq_crop, gt_crop))
            ssim_vals.append(calc_ssim(lq_crop, gt_crop))
            l1_vals.append(calc_l1(lq_crop, gt_crop))
            crop_count += 1
    return {
        "count": len(pairs),
        "crop_count": crop_count,
        "psnr": float(np.mean(psnr_vals)) if psnr_vals else None,
        "ssim": float(np.mean(ssim_vals)) if ssim_vals else None,
        "l1": float(np.mean(l1_vals)) if l1_vals else None,
    }


def evaluate_model_pairs_grid9_global(
    model: torch.nn.Module,
    gt_dir: Path,
    lq_dir: Path,
    device: Union[str, torch.device],
    window_size: int,
    roi_w: int,
    roi_h: int,
    crop_size: int,
    max_images: Optional[int] = None,
    pass_external_coord: bool = True,
) -> Dict[str, object]:
    pairs = build_pairs(gt_dir, lq_dir, allow_fuzzy_pairing=False)
    if max_images is not None and max_images > 0:
        pairs = pairs[:max_images]
    psnr_vals, ssim_vals, l1_vals = [], [], []
    crop_count = 0
    start = time.perf_counter()
    for gt_path, lq_path in tqdm(pairs, desc="Global grid9 eval", leave=False):
        gt = read_rgb_float(gt_path)
        lq = read_rgb_float(lq_path)
        ensure_same_size(gt, lq, gt_path, lq_path)
        full_h, full_w = gt.shape[:2]
        for y0, y1, x0, x1 in grid9_slices_in_full_image(full_h, full_w, roi_w, roi_h, crop_size):
            gt_crop = gt[y0:y1, x0:x1]
            lq_crop = lq[y0:y1, x0:x1]
            pred_crop = run_model_on_region_global(
                model,
                lq_crop,
                full_h=full_h,
                full_w=full_w,
                y0=y0,
                x0=x0,
                device=device,
                window_size=window_size,
                pass_external_coord=pass_external_coord,
            )
            psnr_vals.append(calc_psnr(pred_crop, gt_crop))
            ssim_vals.append(calc_ssim(pred_crop, gt_crop))
            l1_vals.append(calc_l1(pred_crop, gt_crop))
            crop_count += 1
    elapsed = time.perf_counter() - start
    return {
        "count": len(pairs),
        "crop_count": crop_count,
        "psnr": float(np.mean(psnr_vals)) if psnr_vals else None,
        "ssim": float(np.mean(ssim_vals)) if ssim_vals else None,
        "l1": float(np.mean(l1_vals)) if l1_vals else None,
        "elapsed_sec": float(elapsed),
    }


def make_labeled_panel(label: str, img_rgb: np.ndarray, label_h: int = 32) -> np.ndarray:
    require_cv2()
    img_u8 = np.clip(img_rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
    h, w = img_u8.shape[:2]
    panel = np.ones((h + label_h, w, 3), dtype=np.uint8) * 255
    panel[label_h:, :, :] = img_u8
    cv2.putText(panel, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
    return panel


def save_triplet(meta_rgb: np.ndarray, pred_rgb: np.ndarray, gt_rgb: np.ndarray, out_path: Path) -> None:
    require_cv2()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = gt_rgb.shape[:2]
    if meta_rgb.shape[:2] != (h, w):
        meta_rgb = cv2.resize(meta_rgb, (w, h), interpolation=cv2.INTER_AREA)
    if pred_rgb.shape[:2] != (h, w):
        pred_rgb = cv2.resize(pred_rgb, (w, h), interpolation=cv2.INTER_AREA)
    triplet = np.concatenate(
        [make_labeled_panel("META / Input", meta_rgb), make_labeled_panel("Global DegField", pred_rgb), make_labeled_panel("GT", gt_rgb)],
        axis=1,
    )
    bgr = cv2.cvtColor(triplet, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path), bgr)


def run_visualization_global(
    model: torch.nn.Module,
    visual_gt_dir: Path,
    visual_lq_dir: Path,
    out_root: Path,
    device: Union[str, torch.device],
    window_size: int = 8,
    tile: int = 256,
    overlap: int = 32,
    max_images: Optional[int] = None,
    pass_external_coord: bool = True,
) -> Dict[str, object]:
    pairs = build_pairs(visual_gt_dir, visual_lq_dir, allow_fuzzy_pairing=False)
    if max_images is not None and max_images > 0:
        pairs = pairs[:max_images]

    pred_dir = out_root / "pred"
    input_dir = out_root / "meta"
    gt_copy_dir = out_root / "gt"
    triplet_dir = out_root / "triplet"
    for d in [pred_dir, input_dir, gt_copy_dir, triplet_dir]:
        d.mkdir(parents=True, exist_ok=True)

    csv_path = out_root / "visual_metrics.csv"
    psnr_vals, ssim_vals, l1_vals = [], [], []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "name", "psnr", "ssim", "l1", "gt", "meta", "pred", "triplet"])
        for i, (gt_path, lq_path) in enumerate(tqdm(pairs, desc="Global visual", leave=False)):
            gt = read_rgb_float(gt_path)
            lq = read_rgb_float(lq_path)
            ensure_same_size(gt, lq, gt_path, lq_path)
            pred = run_model_on_image_global_tiled(
            model,
            lq,
            device=device,
            window_size=window_size,
            tile=tile,
            overlap=overlap,
            pass_external_coord=pass_external_coord,
        )
            psnr_val = calc_psnr(pred, gt)
            ssim_val = calc_ssim(pred, gt)
            l1_val = calc_l1(pred, gt)
            psnr_vals.append(psnr_val)
            ssim_vals.append(ssim_val)
            l1_vals.append(l1_val)

            rel = gt_path.relative_to(visual_gt_dir)
            pred_path = pred_dir / rel
            meta_copy_path = input_dir / rel
            gt_copy_path = gt_copy_dir / rel
            triplet_path = triplet_dir / rel.with_name(rel.stem + "_triplet.png")
            save_rgb_float(pred_path, pred)
            save_rgb_float(meta_copy_path, lq)
            save_rgb_float(gt_copy_path, gt)
            save_triplet(lq, pred, gt, triplet_path)
            writer.writerow([i, str(rel), f"{psnr_val:.6f}", f"{ssim_val:.6f}", f"{l1_val:.8f}", gt_path, lq_path, pred_path, triplet_path])

    return {
        "count": len(pairs),
        "psnr": float(np.mean(psnr_vals)) if psnr_vals else None,
        "ssim": float(np.mean(ssim_vals)) if ssim_vals else None,
        "l1": float(np.mean(l1_vals)) if l1_vals else None,
        "out_root": out_root,
        "pred_dir": pred_dir,
        "triplet_dir": triplet_dir,
        "csv": csv_path,
    }


# ============================================================
# Checkpoint and result formatting
# ============================================================
def _save_compact_checkpoint(path: Path, model: torch.nn.Module, optimizer, epoch: int, best_psnr: float, best_ssim: float, extra: Optional[Dict[str, object]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "best_psnr": float(best_psnr),
        "best_ssim": float(best_ssim),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, str(path))


def _load_state_dict_from_checkpoint(model: torch.nn.Module, ckpt_path: Path, device: Union[str, torch.device]) -> None:
    ckpt = torch.load(str(ckpt_path), map_location=device)
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state = ckpt["state_dict"]
        elif "params_ema" in ckpt and isinstance(ckpt["params_ema"], dict):
            state = ckpt["params_ema"]
        elif "params" in ckpt and isinstance(ckpt["params"], dict):
            state = ckpt["params"]
        else:
            state = ckpt
    else:
        state = ckpt
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] Missing keys when loading checkpoint: {len(missing)}")
    if unexpected:
        print(f"[WARN] Unexpected keys when loading checkpoint: {len(unexpected)}")


def fmt_num(x: object, nd: int = 4, none: str = "N/A") -> str:
    if x is None:
        return none
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def fmt_int_commas(x: object, none: str = "N/A") -> str:
    if x is None:
        return none
    try:
        return f"{int(x):,}"
    except Exception:
        return str(x) if str(x) != "" else none


def fmt_list(xs: Iterable[object], nd: int = 4) -> str:
    out = []
    for x in xs:
        if x is None:
            out.append("N/A")
            continue
        try:
            out.append(f"{float(x):.{nd}f}")
        except Exception:
            out.append(str(x))
    return str(out)


# ============================================================
# Main training entry
# ============================================================

def _warmstart_model_from_checkpoint(model: torch.nn.Module, ckpt_path: str, device: Union[str, torch.device]) -> Dict[str, object]:
    """Load model weights only. Optimizer/scheduler are intentionally reset."""
    ckpt_path = str(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)

    meta = {}
    if isinstance(ckpt, dict):
        meta["epoch"] = ckpt.get("epoch", None)
        meta["best_psnr"] = ckpt.get("best_psnr", None)
        meta["best_ssim"] = ckpt.get("best_ssim", None)
        meta["checkpoint_type"] = ckpt.get("checkpoint_type", None)

        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state = ckpt["state_dict"]
        elif "params_ema" in ckpt and isinstance(ckpt["params_ema"], dict):
            state = ckpt["params_ema"]
        elif "params" in ckpt and isinstance(ckpt["params"], dict):
            state = ckpt["params"]
        else:
            state = ckpt
    else:
        state = ckpt

    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    meta["missing_keys"] = len(missing)
    meta["unexpected_keys"] = len(unexpected)

    print("=" * 100)
    print(f"[WARMSTART] Loaded model weights only from: {ckpt_path}")
    print(f"[WARMSTART] checkpoint meta: {meta}")
    print("[WARMSTART] Optimizer and scheduler are reset for the new formal run.")
    print("=" * 100)
    return meta


def train_variant_6500(
    model_name: str,
    model_class,
    batch_size: int,
    model_args: Dict[str, object],
    *,
    config_path: str = "config.yml",
    checkpoint_root: Optional[str] = None,
    result_file: Optional[str] = None,
    train_repeat: int = 1,
    train_gt_dir: str,
    train_meta_dir: str,
    val_gt_dir: str,
    val_meta_dir: str,
    patch_size: int = PATCH_SIZE,
    roi_w: int = ROI_W,
    roi_h: int = ROI_H,
    val_crop_mode: str = VAL_CROP_MODE,
    train_num_workers: int = 4,
    val_batch_size: int = 1,
    val_num_workers: int = 2,
    grid9_val_every: int = 0,
    grid9_val_batch_size: int = 1,
    grid9_val_num_workers: int = 2,
    eval_ckpt_mode: str = "center",
    prior_monitor_every: int = 10,
    lambda_basis_diversity: float = 0.0,
    lambda_deg_smoothness: float = 0.0,
    lambda_coeff_smoothness: float = 0.0,
    lambda_coeff_entropy: float = 0.0,
    coeff_entropy_target: float = 0.65,
    coeff_entropy_mode: str = "min",
    lambda_deg_radial: float = 0.0,
    deg_radial_min_corr: float = 0.10,
    deg_radial_min_gap: float = 0.0,
    deg_radial_require_edge_larger: bool = False,
    # Unified post-eval
    run_unified_eval: bool = True,
    eval_full: bool = True,
    eval_grid9: bool = True,
    eval_raw: bool = True,
    max_eval_val_images: int = -1,
    full_eval_tile: int = 256,
    full_eval_overlap: int = 32,
    window_size: int = 8,
    visual_gt_dir: Optional[str] = None,
    visual_meta_dir: Optional[str] = None,
    visual_out_root: Optional[str] = None,
    max_visual_images: int = -1,
    skip_visual: bool = False,
    benchmark_runs: int = 50,
    save_latest: bool = True,
    pretrained_ckpt: Optional[str] = None,
    pass_external_coord: bool = True,
):
    """Train one model on fixed6500/1729 folders and append unified metrics."""
    config_path = str((PROJECT_ROOT / config_path).resolve()) if not os.path.isabs(config_path) else config_path
    opt = Config(config_path)
    seed_everything(opt.OPTIM.SEED)

    use_muon = getattr(opt.OPTIM, "USE_MUON", False)
    base_lr = opt.OPTIM.LR_INITIAL
    lr_min = opt.OPTIM.LR_MIN
    weight_decay = getattr(opt.OPTIM, "WEIGHT_DECAY", 0.0)
    muon_lr = getattr(opt.OPTIM, "MUON_LR", None)
    muon_lr_mult = getattr(opt.OPTIM, "MUON_LR_MULT", 5)

    if checkpoint_root is None:
        checkpoint_root = str(PROJECT_ROOT / "checkpoints" / "global_psf_6500")
    checkpoint_root = str((PROJECT_ROOT / checkpoint_root).resolve()) if not os.path.isabs(checkpoint_root) else checkpoint_root
    model_save_dir = Path(checkpoint_root) / model_name
    model_save_dir.mkdir(parents=True, exist_ok=True)

    if result_file is None:
        result_file = str(model_save_dir / "result.txt")
    result_path = Path(result_file)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[TRAIN] {model_name} | meta -> clean | patch={patch_size} | val={val_crop_mode}")
    print(f"[TRAIN] workers={train_num_workers} | grid9_every={grid9_val_every} | external_coord={pass_external_coord}")
    print(f"[TRAIN] checkpoint={model_save_dir}")

    start_time = time.time()
    accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    device = accelerator.device

    if use_muon:
        _ensure_dist_initialized(True)

    datasets = build_all_datasets_for_globalcoord_6500(
        train_gt_dir=train_gt_dir,
        train_meta_dir=train_meta_dir,
        val_gt_dir=val_gt_dir,
        val_meta_dir=val_meta_dir,
        train_repeat=train_repeat,
        patch_size=patch_size,
        roi_w=roi_w,
        roi_h=roi_h,
        val_crop_mode=val_crop_mode,
    )
    train_dataset = datasets["train"]
    val_dataset = datasets["val"]

    grid9_val_loader = None
    grid9_val_dataset = None
    grid9_val_every = int(grid9_val_every) if grid9_val_every is not None else 0
    if grid9_val_every > 0 and str(val_crop_mode).lower() != "grid9":
        print(f"[FASTVAL] Building extra periodic RectROI-grid9 validation loader every {grid9_val_every} epoch(s).")
        grid9_datasets = build_all_datasets_for_globalcoord_6500(
            train_gt_dir=train_gt_dir,
            train_meta_dir=train_meta_dir,
            val_gt_dir=val_gt_dir,
            val_meta_dir=val_meta_dir,
            train_repeat=1,
            patch_size=patch_size,
            roi_w=roi_w,
            roi_h=roi_h,
            val_crop_mode="grid9",
        )
        grid9_val_dataset = grid9_datasets["val"]
    elif grid9_val_every > 0 and str(val_crop_mode).lower() == "grid9":
        print("[FASTVAL] Main validation is already grid9; extra periodic grid9 loader is disabled to avoid duplicate work.")
        grid9_val_every = 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=max(0, int(train_num_workers)),
        pin_memory=True,
    )
    val_loader = DataLoader(val_dataset, batch_size=max(1, int(val_batch_size)), shuffle=False, num_workers=max(0, int(val_num_workers)), pin_memory=True)
    if grid9_val_dataset is not None:
        grid9_val_loader = DataLoader(
            grid9_val_dataset,
            batch_size=max(1, int(grid9_val_batch_size)),
            shuffle=False,
            num_workers=max(0, int(grid9_val_num_workers)),
            pin_memory=True,
        )

    model = model_class(**model_args).to(device)
    warmstart_meta = {}
    if pretrained_ckpt:
        warmstart_meta = _warmstart_model_from_checkpoint(model, pretrained_ckpt, device)
    model_stats = analyze_model_statistics(
        model,
        device=device,
        patch_size=patch_size,
        runs=benchmark_runs,
        pass_external_coord=pass_external_coord,
    )

    optimizer = build_optimizer(model, base_lr, weight_decay, use_muon, muon_lr=muon_lr, muon_lr_mult=muon_lr_mult)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.OPTIM.NUM_EPOCHS, eta_min=lr_min)

    if grid9_val_loader is not None:
        train_loader, val_loader, grid9_val_loader, model, optimizer, scheduler = accelerator.prepare(
            train_loader, val_loader, grid9_val_loader, model, optimizer, scheduler
        )
    else:
        train_loader, val_loader, model, optimizer, scheduler = accelerator.prepare(train_loader, val_loader, model, optimizer, scheduler)

    l1_criterion = torch.nn.L1Loss()
    ssim_criterion = SSIMLoss()
    lambda_l1 = 1.0
    lambda_ssim = 0.2

    best_psnr, best_ssim, best_epoch = 0.0, 0.0, -1
    best_grid9_psnr, best_grid9_ssim, best_grid9_epoch = 0.0, 0.0, -1
    psnr_curve, ssim_curve = [], []
    periodic_grid9_psnr_curve, periodic_grid9_ssim_curve, periodic_grid9_epoch_curve = [], [], []
    raw_input_psnr_curve, raw_input_ssim_curve, raw_input_l1_curve = [], [], []
    val_global_l1_center_curve, val_global_l1_middle_curve, val_global_l1_edge_curve = [], [], []
    train_stats: Dict[str, object] = {}

    total_epochs = int(opt.OPTIM.NUM_EPOCHS)
    val_after_every = int(getattr(opt.TRAINING, "VAL_AFTER_EVERY", 1))

    best_ckpt_path = model_save_dir / "net_g_best.pth"
    best_grid9_ckpt_path = model_save_dir / "net_g_best_grid9.pth"
    latest_ckpt_path = model_save_dir / "net_g_latest.pth"

    for epoch in range(1, total_epochs + 1):
        model.train()
        meter = _init_meter()

        for clean_crop, meta_crop, global_coord, global_radius, _ in tqdm(train_loader, desc=f"{model_name} Epoch {epoch} [global-6500]"):
            global_radius = global_radius.to(device)
            inp, tar, coord = _prepare_meta_clean_global_batch(clean_crop, meta_crop, global_coord, device)

            optimizer.zero_grad(set_to_none=True)
            pred, aux = _forward_with_aux(model, inp, coord=coord, pass_external_coord=pass_external_coord)
            if pred.shape != tar.shape:
                pred = F.interpolate(pred, size=tar.shape[2:], mode="bilinear", align_corners=False)

            pred_for_loss = torch.clamp(pred, 0.0, 1.0)
            loss_l1 = l1_criterion(pred_for_loss, tar)
            loss_ssim = ssim_criterion(pred_for_loss, tar)
            loss = lambda_l1 * loss_l1 + lambda_ssim * loss_ssim

            prior_reg, reg_terms = _optional_prior_regularization(
                model,
                lambda_basis=lambda_basis_diversity,
                lambda_deg_smooth=lambda_deg_smoothness,
                lambda_coeff_smooth=lambda_coeff_smoothness,
                lambda_coeff_entropy=lambda_coeff_entropy,
                coeff_entropy_target=coeff_entropy_target,
                coeff_entropy_mode=coeff_entropy_mode,
                lambda_deg_radial=lambda_deg_radial,
                deg_radial_min_corr=deg_radial_min_corr,
                deg_radial_min_gap=deg_radial_min_gap,
                deg_radial_require_edge_larger=deg_radial_require_edge_larger,
            )
            if torch.is_tensor(prior_reg) and prior_reg.detach().abs().item() > 0:
                loss = loss + prior_reg

            if aux is None:
                aux = {}
            for k, v in reg_terms.items():
                aux[f"reg_{k}"] = torch.as_tensor(v, device=inp.device)

            accelerator.backward(loss)
            optimizer.step()
            _update_meter(meter, pred, tar, inp, global_radius, loss, loss_l1, loss_ssim, aux=aux)

        scheduler.step()
        train_stats = _format_meter(meter)
        print(
            f"[Epoch {epoch:03d}/{total_epochs:03d}] "
            f"loss={train_stats['TrainLoss']:.6f} | "
            f"l1={train_stats['L1']:.6f} | "
            f"ssim_loss={train_stats['SSIMLoss']:.6f}"
        )

        if epoch % val_after_every == 0:
            model.eval()
            psnr_total, ssim_total = 0.0, 0.0
            raw_psnr_total, raw_ssim_total, raw_l1_total = 0.0, 0.0, 0.0
            val_l1_center_sum, val_l1_middle_sum, val_l1_edge_sum = 0.0, 0.0, 0.0
            val_center_count, val_middle_count, val_edge_count = 0, 0, 0

            with torch.no_grad():
                for clean_crop, meta_crop, global_coord, global_radius, _ in tqdm(val_loader, desc=f"{model_name} Val {epoch}", leave=False):
                    global_radius = global_radius.to(device)
                    inp, tar, coord = _prepare_meta_clean_global_batch(clean_crop, meta_crop, global_coord, device)

                    raw_metrics = compute_pair_metrics_tensor(inp, tar)
                    raw_psnr_total += raw_metrics["psnr"]
                    raw_ssim_total += raw_metrics["ssim"]
                    raw_l1_total += raw_metrics["l1"]

                    pred, _ = _forward_with_aux(model, inp, coord=coord, pass_external_coord=pass_external_coord)
                    if pred.shape != tar.shape:
                        pred = F.interpolate(pred, size=tar.shape[2:], mode="bilinear", align_corners=False)
                    pred_metric = torch.clamp(pred, 0.0, 1.0)

                    psnr_total += peak_signal_noise_ratio(pred_metric, tar, data_range=1.0)
                    ssim_total += ssim(pred_metric, tar, data_range=1.0)

                    gstats = compute_global_radius_error_stats(pred_metric, tar, global_radius)
                    if gstats["has_center"]:
                        val_l1_center_sum += gstats["l1_center"]
                        val_center_count += 1
                    if gstats["has_middle"]:
                        val_l1_middle_sum += gstats["l1_middle"]
                        val_middle_count += 1
                    if gstats["has_edge"]:
                        val_l1_edge_sum += gstats["l1_edge"]
                        val_edge_count += 1

            avg_psnr = psnr_total / len(val_loader)
            avg_ssim = ssim_total / len(val_loader)
            avg_raw_psnr = raw_psnr_total / len(val_loader)
            avg_raw_ssim = raw_ssim_total / len(val_loader)
            avg_raw_l1 = raw_l1_total / len(val_loader)
            avg_val_l1_center = (val_l1_center_sum / val_center_count) if val_center_count > 0 else None
            avg_val_l1_middle = (val_l1_middle_sum / val_middle_count) if val_middle_count > 0 else None
            avg_val_l1_edge = (val_l1_edge_sum / val_edge_count) if val_edge_count > 0 else None

            avg_psnr_f = float(avg_psnr.detach().item() if torch.is_tensor(avg_psnr) else avg_psnr)
            avg_ssim_f = float(avg_ssim.detach().item() if torch.is_tensor(avg_ssim) else avg_ssim)

            psnr_curve.append(avg_psnr_f)
            ssim_curve.append(avg_ssim_f)
            raw_input_psnr_curve.append(float(avg_raw_psnr))
            raw_input_ssim_curve.append(float(avg_raw_ssim))
            raw_input_l1_curve.append(float(avg_raw_l1))
            val_global_l1_center_curve.append(avg_val_l1_center)
            val_global_l1_middle_curve.append(avg_val_l1_middle)
            val_global_l1_edge_curve.append(avg_val_l1_edge)

            raw_model = accelerator.unwrap_model(model)
            if avg_psnr_f > best_psnr:
                best_psnr = avg_psnr_f
                best_ssim = max(best_ssim, avg_ssim_f)
                best_epoch = epoch
                if accelerator.is_local_main_process:
                    _save_compact_checkpoint(
                        best_ckpt_path,
                        raw_model,
                        optimizer,
                        epoch=epoch,
                        best_psnr=best_psnr,
                        best_ssim=best_ssim,
                        extra={"checkpoint_type": "best_psnr"},
                    )
                    print(f"[CKPT] Saved best checkpoint: {best_ckpt_path}")
            if avg_ssim_f > best_ssim:
                best_ssim = avg_ssim_f

            if save_latest and accelerator.is_local_main_process:
                _save_compact_checkpoint(
                    latest_ckpt_path,
                    raw_model,
                    optimizer,
                    epoch=epoch,
                    best_psnr=best_psnr,
                    best_ssim=best_ssim,
                    extra={"checkpoint_type": "latest_val_epoch"},
                )

            print(
                f"[Val {epoch:03d}] "
                f"PSNR={avg_psnr_f:.4f} | SSIM={avg_ssim_f:.4f} | "
                f"best={best_psnr:.4f}@{best_epoch}"
            )

            # Optional periodic RectROI-grid9 validation. This is intentionally
            # decoupled from the per-epoch fast validation so that training does
            # not spend every epoch on 1729*9 crops.
            if grid9_val_loader is not None and grid9_val_every > 0 and epoch % grid9_val_every == 0:
                model.eval()
                g9_psnr_total, g9_ssim_total = 0.0, 0.0
                g9_count = 0
                with torch.no_grad():
                    for clean_crop, meta_crop, global_coord, global_radius, _ in tqdm(
                        grid9_val_loader,
                        desc=f"{model_name} PeriodicGrid9 {epoch}",
                        leave=False,
                    ):
                        inp, tar, coord = _prepare_meta_clean_global_batch(clean_crop, meta_crop, global_coord, device)
                        pred, _ = _forward_with_aux(model, inp, coord=coord, pass_external_coord=pass_external_coord)
                        if pred.shape != tar.shape:
                            pred = F.interpolate(pred, size=tar.shape[2:], mode="bilinear", align_corners=False)
                        pred_metric = torch.clamp(pred, 0.0, 1.0)
                        # Keep batch_size=1 by default for exactly matching previous crop-average metrics.
                        g9_psnr_total += peak_signal_noise_ratio(pred_metric, tar, data_range=1.0)
                        g9_ssim_total += ssim(pred_metric, tar, data_range=1.0)
                        g9_count += 1

                avg_g9_psnr = g9_psnr_total / max(g9_count, 1)
                avg_g9_ssim = g9_ssim_total / max(g9_count, 1)
                avg_g9_psnr_f = float(avg_g9_psnr.detach().item() if torch.is_tensor(avg_g9_psnr) else avg_g9_psnr)
                avg_g9_ssim_f = float(avg_g9_ssim.detach().item() if torch.is_tensor(avg_g9_ssim) else avg_g9_ssim)
                periodic_grid9_epoch_curve.append(int(epoch))
                periodic_grid9_psnr_curve.append(avg_g9_psnr_f)
                periodic_grid9_ssim_curve.append(avg_g9_ssim_f)

                raw_model = accelerator.unwrap_model(model)
                if avg_g9_psnr_f > best_grid9_psnr:
                    best_grid9_psnr = avg_g9_psnr_f
                    best_grid9_ssim = avg_g9_ssim_f
                    best_grid9_epoch = epoch
                    if accelerator.is_local_main_process:
                        _save_compact_checkpoint(
                            best_grid9_ckpt_path,
                            raw_model,
                            optimizer,
                            epoch=epoch,
                            best_psnr=best_grid9_psnr,
                            best_ssim=best_grid9_ssim,
                            extra={"checkpoint_type": "best_periodic_grid9"},
                        )
                        print(f"[CKPT] Saved best periodic grid9 checkpoint: {best_grid9_ckpt_path}")

                print(
                    f"[Grid9 {epoch:03d}] "
                    f"PSNR={avg_g9_psnr_f:.4f} | SSIM={avg_g9_ssim_f:.4f} | "
                    f"best={best_grid9_psnr:.4f}@{best_grid9_epoch}"
                )

    raw_model = accelerator.unwrap_model(model)
    if save_latest and accelerator.is_local_main_process:
        _save_compact_checkpoint(
            latest_ckpt_path,
            raw_model,
            optimizer,
            epoch=total_epochs,
            best_psnr=best_psnr,
            best_ssim=best_ssim,
            extra={"checkpoint_type": "latest_final"},
        )

    # Unified post-eval uses the best checkpoint if available.
    raw_full = {"count": 0, "psnr": None, "ssim": None, "l1": None}
    eval_full_res = {"count": 0, "psnr": None, "ssim": None, "l1": None, "elapsed_sec": None}
    raw_grid9 = {"count": 0, "crop_count": 0, "psnr": None, "ssim": None, "l1": None}
    eval_grid9_res = {"count": 0, "crop_count": 0, "psnr": None, "ssim": None, "l1": None, "elapsed_sec": None}
    visual_stats = {"count": 0}
    eval_ckpt_mode = str(eval_ckpt_mode or "center").lower()
    if eval_ckpt_mode == "grid9":
        ckpt_for_eval = best_grid9_ckpt_path if best_grid9_ckpt_path.is_file() else best_ckpt_path if best_ckpt_path.is_file() else latest_ckpt_path if latest_ckpt_path.is_file() else None
        ckpt_reason = "best_periodic_grid9 checkpoint" if best_grid9_ckpt_path.is_file() else "best fast-val checkpoint" if best_ckpt_path.is_file() else "latest checkpoint" if ckpt_for_eval else "no checkpoint"
    elif eval_ckpt_mode == "latest":
        ckpt_for_eval = latest_ckpt_path if latest_ckpt_path.is_file() else best_ckpt_path if best_ckpt_path.is_file() else None
        ckpt_reason = "latest checkpoint" if latest_ckpt_path.is_file() else "best fast-val checkpoint" if ckpt_for_eval else "no checkpoint"
    elif eval_ckpt_mode == "auto":
        ckpt_for_eval = best_grid9_ckpt_path if best_grid9_ckpt_path.is_file() else best_ckpt_path if best_ckpt_path.is_file() else latest_ckpt_path if latest_ckpt_path.is_file() else None
        ckpt_reason = "best_periodic_grid9 checkpoint" if best_grid9_ckpt_path.is_file() else "best fast-val checkpoint" if best_ckpt_path.is_file() else "latest checkpoint" if ckpt_for_eval else "no checkpoint"
    else:
        ckpt_for_eval = best_ckpt_path if best_ckpt_path.is_file() else latest_ckpt_path if latest_ckpt_path.is_file() else None
        ckpt_reason = "best fast-val checkpoint" if best_ckpt_path.is_file() else "latest checkpoint" if ckpt_for_eval else "no checkpoint"

    # Let accelerator finish before heavy single-process image evaluation.
    accelerator.wait_for_everyone()

    if accelerator.is_local_main_process and run_unified_eval and ckpt_for_eval is not None:
        eval_device = device
        raw_model.eval()
        _load_state_dict_from_checkpoint(raw_model, ckpt_for_eval, device=eval_device)
        raw_model.to(eval_device)
        raw_model.eval()

        max_eval = None if max_eval_val_images is None or max_eval_val_images <= 0 else int(max_eval_val_images)
        print(f"[EVAL] checkpoint={ckpt_for_eval} ({ckpt_reason})")

        if eval_raw and eval_full:
            print("[EVAL] raw full-image baseline")
            raw_full = evaluate_raw_pairs(Path(val_gt_dir), Path(val_meta_dir), max_images=max_eval)
            print(f"[INFO] Raw full PSNR/SSIM/L1: {raw_full.get('psnr')}, {raw_full.get('ssim')}, {raw_full.get('l1')}")

        if eval_full:
            print("[EVAL] model full-image tiled")
            eval_full_res = evaluate_model_pairs_global(
                raw_model,
                Path(val_gt_dir),
                Path(val_meta_dir),
                device=eval_device,
                window_size=window_size,
                tile=full_eval_tile,
                overlap=full_eval_overlap,
                max_images=max_eval,
                pass_external_coord=pass_external_coord,
            )
            print(f"[INFO] Eval full PSNR/SSIM/L1: {eval_full_res.get('psnr')}, {eval_full_res.get('ssim')}, {eval_full_res.get('l1')}")

        if eval_grid9:
            if eval_raw:
                print("[INFO] Evaluating raw META input on val split: RectROI-grid9...")
                raw_grid9 = evaluate_raw_pairs_grid9(
                    Path(val_gt_dir),
                    Path(val_meta_dir),
                    roi_w=roi_w,
                    roi_h=roi_h,
                    crop_size=patch_size,
                    max_images=max_eval,
                )
                print(f"[INFO] Raw grid9 PSNR/SSIM/L1: {raw_grid9.get('psnr')}, {raw_grid9.get('ssim')}, {raw_grid9.get('l1')}")

            print("[INFO] Evaluating selected checkpoint on val split: RectROI-grid9 global...")
            eval_grid9_res = evaluate_model_pairs_grid9_global(
                raw_model,
                Path(val_gt_dir),
                Path(val_meta_dir),
                device=eval_device,
                window_size=window_size,
                roi_w=roi_w,
                roi_h=roi_h,
                crop_size=patch_size,
                max_images=max_eval,
                pass_external_coord=pass_external_coord,
            )
            print(f"[INFO] Eval grid9 PSNR/SSIM/L1: {eval_grid9_res.get('psnr')}, {eval_grid9_res.get('ssim')}, {eval_grid9_res.get('l1')}")

        if not skip_visual and visual_gt_dir and visual_meta_dir:
            vg = Path(visual_gt_dir)
            vm = Path(visual_meta_dir)
            if vg.is_dir() and vm.is_dir():
                out_root = Path(visual_out_root) if visual_out_root else (PROJECT_ROOT / "visual_results" / model_name)
                max_vis = None if max_visual_images is None or max_visual_images <= 0 else int(max_visual_images)
                print(f"[INFO] Running visualization on: {vm}")
                visual_stats = run_visualization_global(
                    raw_model,
                    vg,
                    vm,
                    out_root=out_root,
                    device=eval_device,
                    window_size=window_size,
                    tile=full_eval_tile,
                    overlap=full_eval_overlap,
                    max_images=max_vis,
                    pass_external_coord=pass_external_coord,
                )
                print(f"[INFO] Visual outputs saved to: {visual_stats.get('out_root')}")
            else:
                print("[WARN] visual_gt_dir/visual_meta_dir not found; skip visualization.")

    if accelerator.is_local_main_process:
        with result_path.open("a", encoding="utf-8") as f:
            f.write("\n" + "=" * 100 + "\n")
            f.write(f"[STRICT GLOBAL DEGFIELD RUN] {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"TRAIN_GT_DIR={train_gt_dir}\n")
            f.write(f"TRAIN_META_DIR={train_meta_dir}\n")
            f.write(f"VAL_GT_DIR={val_gt_dir}\n")
            f.write(f"VAL_META_DIR={val_meta_dir}\n")
            f.write(f"PATCH_SIZE={patch_size}\n")
            f.write(f"ROI_W={roi_w}\n")
            f.write(f"ROI_H={roi_h}\n")
            f.write(f"VAL_CROP_MODE={val_crop_mode}\n")
            f.write(f"CHECKPOINT_ROOT={model_save_dir}\n")
            f.write(f"RESULT_FILE={result_path}\n")
            f.write("=" * 100 + "\n")

            f.write(f"======================== Summary: {model_name} ========================\n")
            f.write("Task                 : meta -> clean restoration\n")
            f.write("Engine Type          : StrictGlobalDegField FullCanvas Training + Unified Evaluation\n")
            f.write(f"Train GT Dir         : {train_gt_dir}\n")
            f.write(f"Train Meta Dir       : {train_meta_dir}\n")
            f.write(f"Test GT Dir          : {val_gt_dir}\n")
            f.write(f"Test Meta Dir        : {val_meta_dir}\n")
            f.write("Dataset Mode         : Fixed6500/1729 existing split + online full-canvas global coord\n")
            f.write(f"Patch Size           : {patch_size}\n")
            f.write(f"ROI W                : {roi_w}\n")
            f.write(f"ROI H                : {roi_h}\n")
            f.write(f"Val Crop Mode        : unified_full_and_rectroi_grid9 + fullcanvas_train_crop_val_{val_crop_mode}\n")
            f.write(f"Global Center Thr    : {GLOBAL_CENTER_RADIUS_THR}\n")
            f.write(f"Global Edge/Outer Thr: {GLOBAL_EDGE_RADIUS_THR}\n")
            f.write(f"Model Name           : {model_name}\n")
            f.write(f"Model Class          : {model_class.__name__}\n")
            f.write(f"Pass External Coord  : {pass_external_coord}\n")
            f.write("DegField Modulation  : deg_feat -> FeatureAffineModulation\n")
            f.write("Auxiliary Score Map  : deg_score (monitor / optional prior utility)\n")
            f.write(f"Train Num Workers    : {train_num_workers}\n")
            f.write(f"WarmStart Checkpoint : {pretrained_ckpt if pretrained_ckpt else 'N/A'}\n")
            f.write(f"WarmStart Meta       : {warmstart_meta if pretrained_ckpt else 'N/A'}\n")
            f.write(f"Total Params         : {fmt_int_commas(model_stats.get('total_params'))}\n")
            f.write(f"Trainable Params     : {fmt_int_commas(model_stats.get('trainable_params'))}\n")
            f.write("FLOPs (MACs*2)       : N/A\n")
            f.write(f"Resolution(FLOPs)    : {model_stats.get('resolution')}\n")
            f.write(f"InferTime Avg (ms)   : {fmt_num(model_stats.get('infer_time_avg'), 3)}\n")
            f.write(f"InferTime Std (ms)   : {fmt_num(model_stats.get('infer_time_std'), 3)}\n")
            f.write(f"Num Epochs           : {total_epochs}\n")
            f.write(f"Loss                 : L1 + {lambda_ssim} * SSIMLoss\n")
            f.write(f"Lambda Basis Div     : {lambda_basis_diversity}\n")
            f.write(f"Lambda Deg Smooth    : {lambda_deg_smoothness}\n")
            f.write(f"Lambda Coeff Smooth  : {lambda_coeff_smoothness}\n")
            f.write(f"Lambda Coeff Entropy : {lambda_coeff_entropy}\n")
            f.write(f"Lambda Deg Radial    : {lambda_deg_radial}\n")
            f.write(f"Best Checkpoint      : {ckpt_for_eval if ckpt_for_eval else 'N/A'}\n")
            f.write(f"Checkpoint Selection : {ckpt_reason}\n")
            f.write(f"Best Epoch           : {best_epoch}\n")
            f.write(f"Best PSNR            : {fmt_num(best_psnr, 4)}\n")
            f.write(f"Final PSNR           : {fmt_num(psnr_curve[-1] if psnr_curve else None, 4)}\n")
            f.write(f"Best SSIM            : {fmt_num(best_ssim, 4)}\n")
            f.write(f"Final SSIM           : {fmt_num(ssim_curve[-1] if ssim_curve else None, 4)}\n")

            f.write(f"\nOfficial Full Val PSNR    : {fmt_num(eval_full_res.get('psnr'), 4)}\n")
            f.write(f"Official Full Val SSIM    : {fmt_num(eval_full_res.get('ssim'), 4)}\n")
            f.write(f"Official Full Val L1      : {fmt_num(eval_full_res.get('l1'), 6)}\n")
            f.write(f"Official Full Val Count   : {eval_full_res.get('count', 0)}\n")
            f.write(f"RawInput Full PSNR        : {fmt_num(raw_full.get('psnr'), 4)}\n")
            f.write(f"RawInput Full SSIM        : {fmt_num(raw_full.get('ssim'), 4)}\n")
            f.write(f"RawInput Full L1          : {fmt_num(raw_full.get('l1'), 6)}\n")

            f.write(f"\nRectROI Grid9 Val PSNR    : {fmt_num(eval_grid9_res.get('psnr'), 4)}\n")
            f.write(f"RectROI Grid9 Val SSIM    : {fmt_num(eval_grid9_res.get('ssim'), 4)}\n")
            f.write(f"RectROI Grid9 Val L1      : {fmt_num(eval_grid9_res.get('l1'), 6)}\n")
            f.write(f"RectROI Grid9 Image Count : {eval_grid9_res.get('count', 0)}\n")
            f.write(f"RectROI Grid9 Crop Count  : {eval_grid9_res.get('crop_count', 0)}\n")
            f.write(f"RawInput Grid9 PSNR       : {fmt_num(raw_grid9.get('psnr'), 4)}\n")
            f.write(f"RawInput Grid9 SSIM       : {fmt_num(raw_grid9.get('ssim'), 4)}\n")
            f.write(f"RawInput Grid9 L1         : {fmt_num(raw_grid9.get('l1'), 6)}\n")

            f.write(f"\nFinal RawInput PSNR  : {fmt_num(raw_full.get('psnr'), 4)}\n")
            f.write(f"Final RawInput SSIM  : {fmt_num(raw_full.get('ssim'), 4)}\n")
            f.write(f"Final RawInput L1    : {fmt_num(raw_full.get('l1'), 6)}\n")
            f.write(f"Final Val G-L1 Center: {fmt_num(val_global_l1_center_curve[-1] if val_global_l1_center_curve else None, 6)}\n")
            f.write(f"Final Val G-L1 Middle: {fmt_num(val_global_l1_middle_curve[-1] if val_global_l1_middle_curve else None, 6)}\n")
            f.write(f"Final Val G-L1 Edge  : {fmt_num(val_global_l1_edge_curve[-1] if val_global_l1_edge_curve else None, 6)}\n")

            f.write(f"Visual Count         : {visual_stats.get('count', 0)}\n")
            f.write(f"Visual PSNR          : {fmt_num(visual_stats.get('psnr'), 4)}\n")
            f.write(f"Visual SSIM          : {fmt_num(visual_stats.get('ssim'), 4)}\n")
            f.write(f"Visual L1            : {fmt_num(visual_stats.get('l1'), 6)}\n")
            f.write(f"Visual Pred Dir      : {visual_stats.get('pred_dir', 'N/A')}\n")
            f.write(f"Visual Triplet Dir   : {visual_stats.get('triplet_dir', 'N/A')}\n")
            f.write(f"Visual Metrics CSV   : {visual_stats.get('csv', 'N/A')}\n")

            f.write(f"Time Elapsed (s)     : {time.time() - start_time:.2f}\n")
            f.write(f"PSNR Curve           : {fmt_list(psnr_curve, 4)}\n")
            f.write(f"SSIM Curve           : {fmt_list(ssim_curve, 4)}\n")
            f.write(f"RawInput PSNR Curve  : {fmt_list(raw_input_psnr_curve, 4)}\n")
            f.write(f"RawInput SSIM Curve  : {fmt_list(raw_input_ssim_curve, 4)}\n")
            f.write(f"RawInput L1 Curve    : {fmt_list(raw_input_l1_curve, 6)}\n")
            f.write(f"Val G-L1 Center Curve: {fmt_list(val_global_l1_center_curve, 6)}\n")
            f.write(f"Val G-L1 Middle Curve: {fmt_list(val_global_l1_middle_curve, 6)}\n")
            f.write(f"Val G-L1 Edge Curve  : {fmt_list(val_global_l1_edge_curve, 6)}\n\n")

            f.write("Last Epoch Train Stats:\n")
            for k, v in train_stats.items():
                if isinstance(v, (float, int)):
                    f.write(f"  {k:<30}: {v:.8f}\n")
                else:
                    f.write(f"  {k:<30}: {v}\n")
            f.write("\n")
            _append_aux_to_file(f, train_stats)
            f.write(f"[MODEL DONE] {time.strftime('%Y-%m-%d %H:%M:%S')} | {model_name}\n")
            f.write("-" * 100 + "\n")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    accelerator.end_training()

    return {
        "best_psnr": best_psnr,
        "best_ssim": best_ssim,
        "best_epoch": best_epoch,
        "best_grid9_psnr": best_grid9_psnr,
        "best_grid9_ssim": best_grid9_ssim,
        "best_grid9_epoch": best_grid9_epoch,
        "best_checkpoint": str(ckpt_for_eval) if ckpt_for_eval else None,
        "best_grid9_checkpoint": str(best_grid9_ckpt_path) if best_grid9_ckpt_path.is_file() else None,
        "result_file": str(result_path),
        "eval_full": eval_full_res,
        "eval_grid9": eval_grid9_res,
        "raw_full": raw_full,
        "raw_grid9": raw_grid9,
        "visual": visual_stats,
    }


# Backward-friendly alias for a new run script.
train_variant = train_variant_6500
