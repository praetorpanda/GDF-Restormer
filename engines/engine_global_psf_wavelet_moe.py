"""Independent training engine for Restormer_GlobalPSFScaleRetinexWaveletMoE."""
# -*- coding: utf-8 -*-
"""
engine_Global.py
================

Strict global-coordinate PSF-prior training engine.

Save path suggestion:
    /home/ipprlab/Documents/work/lowlevelvision/metalens/2/engines/engine_Global.py

Purpose:
    Train global-coordinate compatible PSF-prior models from models/Res_Global.py:
        - Restormer_GlobalPSFLikeDegField
        - Restormer_GlobalPSFLikeLowRankBasis

Important differences from ordinary PSF/DWT engines:
    - Uses build_globalmeta_rectroi_datasets(...).
    - Dataset returns already-cropped clean_crop, meta_crop, global_coord, global_radius, filename.
    - The input/target are directly 256x256 crops from a moderate central rectangular ROI.
    - No PH:-PH center crop is performed in the engine.
    - global_coord is passed explicitly into model(inp, coord=coord, ...).
    - If the model is a strict global model and coord is missing, it should fail loudly.
"""

import os
import time
import warnings
import socket
import gc
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from accelerate import Accelerator, DistributedDataParallelKwargs

from torchmetrics.functional import peak_signal_noise_ratio
from torchmetrics.functional import structural_similarity_index_measure as ssim

import numpy as np
from tqdm import tqdm

from config import Config
from loss import SSIMLoss
from utils import seed_everything, save_checkpoint
from data.dataset_GlobalMeta import build_globalmeta_rectroi_datasets

try:
    from muon import MuonWithAuxAdam
except Exception:
    MuonWithAuxAdam = None

warnings.filterwarnings("ignore")

_CREATED_PG = False
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# Rectangular ROI / coordinate settings
# ============================================================
# This engine uses dataset_GlobalMeta.py. The dataset returns already-cropped
# 256x256 patches sampled from a moderate central rectangular ROI.
PATCH_SIZE = 256
ROI_W = 768
ROI_H = 640
VAL_CROP_MODE = "grid9"

# RectROI radius bins for spatial error monitoring.
# Previous edge threshold was too strict for the current RectROI setting
# because observed global_radius_max is around 0.70. Therefore we use:
#   center: r < 0.30
#   middle: 0.30 <= r < 0.55
#   edge/outer: r >= 0.55
GLOBAL_CENTER_RADIUS_THR = 0.30
GLOBAL_EDGE_RADIUS_THR = 0.55


# ============================================================
# Distributed helpers
# ============================================================
def _find_free_port():
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _ensure_dist_initialized(use_muon: bool):
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


def cleanup_dist_if_needed():
    global _CREATED_PG
    if _CREATED_PG and dist.is_available() and dist.is_initialized():
        print("[INFO] Destroying process group...")
        dist.destroy_process_group()
        _CREATED_PG = False


# ============================================================
# Dataset builder
# ============================================================
def build_all_datasets_for_globalcoord(
    train_gt_dir,
    train_meta_dir,
    test_gt_dir,
    test_meta_dir,
    train_repeat=1,
    patch_size=PATCH_SIZE,
    roi_w=ROI_W,
    roi_h=ROI_H,
    val_crop_mode=VAL_CROP_MODE,
):
    return build_globalmeta_rectroi_datasets(
        train_gt_dir=train_gt_dir,
        train_meta_dir=train_meta_dir,
        test_gt_dir=test_gt_dir,
        test_meta_dir=test_meta_dir,
        patch_size=patch_size,
        roi_w=roi_w,
        roi_h=roi_h,
        train_repeat=train_repeat,
        val_crop_mode=val_crop_mode,
    )


# ============================================================
# Model statistics
# ============================================================
def compute_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
    }


def _model_forward_for_stats(model, dummy, coord):
    try:
        return model(dummy, coord=coord)
    except TypeError:
        return model(dummy)


def compute_inference_time(model, opt, device, runs=50):
    dummy = torch.randn(1, 3, opt.TESTING.PS_H, opt.TESTING.PS_W).to(device)
    coord = torch.zeros(1, 2, opt.TESTING.PS_H, opt.TESTING.PS_W).to(device)
    model.eval()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(10):
            _ = _model_forward_for_stats(model, dummy, coord)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    timings = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.time()
            _ = _model_forward_for_stats(model, dummy, coord)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            timings.append((time.time() - t0) * 1000)

    timings = np.array(timings)
    return {
        "infer_time_avg": float(timings.mean()),
        "infer_time_std": float(timings.std()),
    }


def analyze_model_statistics(model, opt, device):
    stats_params = compute_model_params(model)
    latency = compute_inference_time(model, opt, device)
    stats = {
        **stats_params,
        "flops": -1,
        **latency,
        "resolution": (opt.TESTING.PS_W, opt.TESTING.PS_H),
    }

    print("========== Model Statistics ==========")
    print(f"Total Params        : {stats['total_params']:,}")
    print(f"Trainable Params    : {stats['trainable_params']:,}")
    print("FLOPs (MACs*2)      : N/A for strict global-coordinate PSF model")
    print(f"Infer Time Avg (ms) : {stats['infer_time_avg']:.3f}")
    print(f"Infer Time Std (ms) : {stats['infer_time_std']:.3f}")
    print(f"Resolution(Patch)   : {stats['resolution']}")
    print("======================================")
    return stats


# ============================================================
# Optimizer
# ============================================================
def build_optimizer(model, base_lr, weight_decay, use_muon, muon_lr=None, muon_lr_mult=5):
    if not use_muon:
        return optim.AdamW(
            model.parameters(),
            lr=base_lr,
            betas=(0.9, 0.999),
            weight_decay=weight_decay,
        )

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


# ============================================================
# Forward / model helpers
# ============================================================
def _unwrap_model_if_needed(model):
    if hasattr(model, "module"):
        return model.module
    return model


def _forward_with_aux(model, inp, coord):
    """
    Strict global models should receive coord explicitly.
    Prefer return_aux=True; fall back to return_monitor=True if needed.
    """
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


def _optional_prior_regularization(
    model,
    lambda_basis=0.0,
    lambda_deg_smooth=0.0,
    lambda_coeff_smooth=0.0,
    lambda_coeff_entropy=0.0,
    coeff_entropy_target=0.65,
    coeff_entropy_mode="min",
    lambda_deg_radial=0.0,
    deg_radial_min_corr=0.10,
    deg_radial_min_gap=0.0,
    deg_radial_require_edge_larger=False,
):
    raw_model = _unwrap_model_if_needed(model)
    reg = None
    reg_terms = {}

    def _add(name, value, weight):
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
# Metrics / spatial stats
# ============================================================
def compute_pair_metrics(pred, tar):
    pred_metric = torch.clamp(pred, 0.0, 1.0)
    tar_metric = torch.clamp(tar, 0.0, 1.0)
    return {
        "psnr": float(peak_signal_noise_ratio(pred_metric, tar_metric, data_range=1.0).detach().item()),
        "ssim": float(ssim(pred_metric, tar_metric, data_range=1.0).detach().item()),
        "l1": float(torch.mean(torch.abs(pred_metric - tar_metric)).detach().item()),
    }


def _safe_masked_mean(x, mask):
    if mask is None or mask.sum().item() <= 0:
        return None
    return x[mask].mean()


def compute_global_radius_error_stats(pred, tar, global_radius):
    """
    Spatial L1 statistics over RectROI global-radius bins.

    The old center/middle/edge split used a larger edge threshold, which produced
    GlobalEdgeCount=0 under the current RectROI setting. Here, "edge" should be
    interpreted as the outer valid RectROI region rather than the extreme image
    border.

    Bins:
        center : r < 0.30
        middle : 0.30 <= r < 0.55
        edge   : r >= 0.55
    """
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


def _prepare_meta_clean_global_batch(clean_crop, meta_crop, global_coord, device):
    """
    dataset_GlobalMeta.py already returns 256x256 crops.
    Do NOT apply the old PH:-PH center crop here.
    """
    clean_crop = clean_crop.to(device)
    meta_crop = meta_crop.to(device)
    global_coord = global_coord.to(device)

    inp = meta_crop
    tar = clean_crop
    coord = global_coord
    return inp, tar, coord


# ============================================================
# Logging helpers
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


def _short_aux_string(stats, max_items=14):
    preferred = [
        "deg_score_mean", "deg_score_std", "deg_score_edge_center_gap", "deg_score_radial_corr",
        "deg_feat_abs_mean", "prior_feat_std", "coeff_entropy_norm", "coeff_top_prob_mean",
        "coeff_spatial_std", "basis_usage_std", "global_coord_std", "global_radius_mean",
        "shallow_delta_ratio", "enc1_delta_ratio", "latent_delta_ratio", "dec1_delta_ratio",
        "reg_deg_smooth", "reg_coeff_smooth", "reg_basis_diversity", "reg_coeff_entropy", "reg_deg_radial",
    ]
    keys = [k for k in preferred if k in stats]
    if not keys:
        keys = [k for k in stats.keys() if isinstance(stats[k], (float, int))]
    keys = keys[:max_items]
    if not keys:
        return ""
    return " | " + " | ".join([f"{k}={stats[k]:.6f}" for k in keys])


def _append_aux_to_file(f, stats):
    if stats is None or len(stats) == 0:
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
# Train one model variant
# ============================================================
def train_variant(
    model_name,
    model_class,
    batch_size,
    model_args,
    *,
    config_path="config.yml",
    checkpoint_root=None,
    result_file=None,
    train_repeat=1,
    train_gt_dir=None,
    train_meta_dir=None,
    test_gt_dir=None,
    test_meta_dir=None,
    patch_size=PATCH_SIZE,
    roi_w=ROI_W,
    roi_h=ROI_H,
    val_crop_mode=VAL_CROP_MODE,
    prior_monitor_every=10,
    lambda_basis_diversity=0.0,
    lambda_deg_smoothness=0.0,
    lambda_coeff_smoothness=0.0,
    lambda_coeff_entropy=0.0,
    coeff_entropy_target=0.65,
    coeff_entropy_mode="min",
    lambda_deg_radial=0.0,
    deg_radial_min_corr=0.10,
    deg_radial_min_gap=0.0,
    deg_radial_require_edge_larger=False,
):
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
        checkpoint_root = str(PROJECT_ROOT / "checkpoints" / "global_psf")
    checkpoint_root = str((PROJECT_ROOT / checkpoint_root).resolve()) if not os.path.isabs(checkpoint_root) else checkpoint_root

    model_save_dir = os.path.join(checkpoint_root, model_name)
    os.makedirs(model_save_dir, exist_ok=True)
    if result_file is None:
        result_file = os.path.join(model_save_dir, "result.txt")

    print(f"\n=== Training {model_name} [GlobalCoord PSF-prior | meta -> clean] ===")
    print(f"[DATA] Train GT   : {train_gt_dir}")
    print(f"[DATA] Train Meta : {train_meta_dir}")
    print(f"[DATA] Test GT    : {test_gt_dir}")
    print(f"[DATA] Test Meta  : {test_meta_dir}")
    print(f"[DATA] RectROI     : ROI_W={roi_w}, ROI_H={roi_h}, PATCH={patch_size}, VAL={val_crop_mode}")

    start_time = time.time()
    accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    device = accelerator.device

    if use_muon:
        _ensure_dist_initialized(True)

    datasets = build_all_datasets_for_globalcoord(
        train_gt_dir=train_gt_dir,
        train_meta_dir=train_meta_dir,
        test_gt_dir=test_gt_dir,
        test_meta_dir=test_meta_dir,
        train_repeat=train_repeat,
        patch_size=patch_size,
        roi_w=roi_w,
        roi_h=roi_h,
        val_crop_mode=val_crop_mode,
    )
    train_dataset = datasets["train"]
    val_dataset = datasets["val"]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    model = model_class(**model_args).to(device)
    model_stats = analyze_model_statistics(model, opt, device)

    optimizer = build_optimizer(model, base_lr, weight_decay, use_muon, muon_lr=muon_lr, muon_lr_mult=muon_lr_mult)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.OPTIM.NUM_EPOCHS, eta_min=lr_min)

    train_loader, val_loader, model, optimizer, scheduler = accelerator.prepare(
        train_loader, val_loader, model, optimizer, scheduler
    )

    l1_criterion = torch.nn.L1Loss()
    ssim_criterion = SSIMLoss()
    lambda_l1 = 1.0
    lambda_ssim = 0.2

    best_psnr, best_ssim = 0.0, 0.0
    psnr_curve, ssim_curve = [], []
    raw_input_psnr_curve, raw_input_ssim_curve, raw_input_l1_curve = [], [], []
    val_global_l1_center_curve, val_global_l1_middle_curve, val_global_l1_edge_curve = [], [], []

    total_epochs = opt.OPTIM.NUM_EPOCHS
    train_stats = {}

    for epoch in range(1, total_epochs + 1):
        model.train()
        meter = _init_meter()

        for clean_crop, meta_crop, global_coord, global_radius, _ in tqdm(
            train_loader,
            desc=f"{model_name} Epoch {epoch} [global-psf]",
        ):
            global_radius = global_radius.to(device)
            inp, tar, coord = _prepare_meta_clean_global_batch(clean_crop, meta_crop, global_coord, device)

            optimizer.zero_grad(set_to_none=True)
            pred, aux = _forward_with_aux(model, inp, coord=coord)

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
            f"[{model_name}] Epoch {epoch}: "
            f"TrainLoss={train_stats['TrainLoss']:.6f} | "
            f"L1={train_stats['L1']:.6f} | "
            f"SSIMLoss={train_stats['SSIMLoss']:.6f} | "
            f"PredMean={train_stats['PredMean']:.4f} | "
            f"PredStd={train_stats['PredStd']:.4f} | "
            f"TarMean={train_stats['TarMean']:.4f} | "
            f"TarStd={train_stats['TarStd']:.4f} | "
            f"G-L1-C={train_stats['GlobalL1Center']:.6f} | "
            f"G-L1-M={train_stats['GlobalL1Middle']:.6f} | "
            f"G-L1-E={train_stats['GlobalL1Edge']:.6f}"
            f"{_short_aux_string(train_stats)}"
        )

        if epoch % opt.TRAINING.VAL_AFTER_EVERY == 0:
            model.eval()
            psnr_total, ssim_total = 0.0, 0.0
            raw_psnr_total, raw_ssim_total, raw_l1_total = 0.0, 0.0, 0.0
            val_l1_center_sum, val_l1_middle_sum, val_l1_edge_sum = 0.0, 0.0, 0.0
            val_center_count, val_middle_count, val_edge_count = 0, 0, 0

            with torch.no_grad():
                for clean_crop, meta_crop, global_coord, global_radius, _ in val_loader:
                    global_radius = global_radius.to(device)
                    inp, tar, coord = _prepare_meta_clean_global_batch(clean_crop, meta_crop, global_coord, device)

                    raw_metrics = compute_pair_metrics(inp, tar)
                    raw_psnr_total += raw_metrics["psnr"]
                    raw_ssim_total += raw_metrics["ssim"]
                    raw_l1_total += raw_metrics["l1"]

                    pred, _ = _forward_with_aux(model, inp, coord=coord)
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
            avg_val_l1_center = val_l1_center_sum / max(val_center_count, 1)
            avg_val_l1_middle = val_l1_middle_sum / max(val_middle_count, 1)
            avg_val_l1_edge = val_l1_edge_sum / max(val_edge_count, 1)

            psnr_curve.append(float(avg_psnr.item()))
            ssim_curve.append(float(avg_ssim.item()))
            raw_input_psnr_curve.append(float(avg_raw_psnr))
            raw_input_ssim_curve.append(float(avg_raw_ssim))
            raw_input_l1_curve.append(float(avg_raw_l1))
            val_global_l1_center_curve.append(float(avg_val_l1_center))
            val_global_l1_middle_curve.append(float(avg_val_l1_middle))
            val_global_l1_edge_curve.append(float(avg_val_l1_edge))

            if float(avg_psnr) > float(best_psnr):
                best_psnr = float(avg_psnr)
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": accelerator.unwrap_model(model).state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    epoch,
                    model_save_dir,
                )
            if float(avg_ssim) > float(best_ssim):
                best_ssim = float(avg_ssim)

            print(
                f"[{model_name}] Epoch {epoch}: "
                f"PSNR={float(avg_psnr):.4f} (Best={float(best_psnr):.4f}), "
                f"SSIM={float(avg_ssim):.4f} (Best={float(best_ssim):.4f}) | "
                f"RawInputPSNR={avg_raw_psnr:.4f} | RawInputSSIM={avg_raw_ssim:.4f} | RawInputL1={avg_raw_l1:.6f} | "
                f"Val-G-L1-C={avg_val_l1_center:.6f} | Val-G-L1-M={avg_val_l1_middle:.6f} | Val-G-L1-E={avg_val_l1_edge:.6f}"
            )

    if accelerator.is_local_main_process:
        with open(result_file, "a", encoding="utf-8") as f:
            f.write(f"{'=' * 24} Summary: {model_name} {'=' * 24}\n")
            f.write("Task                 : meta -> clean restoration\n")
            f.write("Engine Type          : Strict GlobalCoord PSF-prior\n")
            f.write(f"Train GT Dir         : {train_gt_dir}\n")
            f.write(f"Train Meta Dir       : {train_meta_dir}\n")
            f.write(f"Test GT Dir          : {test_gt_dir}\n")
            f.write(f"Test Meta Dir        : {test_meta_dir}\n")
            f.write(f"Dataset Mode         : RectROI GlobalMeta\n")
            f.write(f"Patch Size           : {patch_size}\n")
            f.write(f"ROI W                : {roi_w}\n")
            f.write(f"ROI H                : {roi_h}\n")
            f.write(f"Val Crop Mode        : {val_crop_mode}\n")
            f.write(f"Global Center Thr    : {GLOBAL_CENTER_RADIUS_THR}\n")
            f.write(f"Global Edge/Outer Thr: {GLOBAL_EDGE_RADIUS_THR}\n")
            f.write(f"Model Name           : {model_name}\n")
            f.write(f"Model Class          : {model_class.__name__}\n")
            f.write(f"Total Params         : {model_stats['total_params']:,}\n")
            f.write(f"Trainable Params     : {model_stats['trainable_params']:,}\n")
            f.write("FLOPs (MACs*2)       : N/A\n")
            f.write(f"Resolution(FLOPs)    : {model_stats['resolution']}\n")
            f.write(f"InferTime Avg (ms)   : {model_stats['infer_time_avg']:.3f}\n")
            f.write(f"InferTime Std (ms)   : {model_stats['infer_time_std']:.3f}\n")
            f.write(f"Num Epochs           : {total_epochs}\n")
            f.write(f"Loss                 : L1 + {lambda_ssim} * SSIMLoss\n")
            f.write(f"Lambda Basis Div     : {lambda_basis_diversity}\n")
            f.write(f"Lambda Deg Smooth    : {lambda_deg_smoothness}\n")
            f.write(f"Lambda Coeff Smooth  : {lambda_coeff_smoothness}\n")
            f.write(f"Lambda Coeff Entropy : {lambda_coeff_entropy}\n")
            f.write(f"Lambda Deg Radial    : {lambda_deg_radial}\n")

            if len(psnr_curve) > 0:
                f.write(f"Best PSNR            : {float(best_psnr):.4f}\n")
                f.write(f"Final PSNR           : {psnr_curve[-1]:.4f}\n")
                f.write(f"Best SSIM            : {float(best_ssim):.4f}\n")
                f.write(f"Final SSIM           : {ssim_curve[-1]:.4f}\n")
                f.write(f"Final RawInput PSNR  : {raw_input_psnr_curve[-1]:.4f}\n")
                f.write(f"Final RawInput SSIM  : {raw_input_ssim_curve[-1]:.4f}\n")
                f.write(f"Final RawInput L1    : {raw_input_l1_curve[-1]:.6f}\n")
                f.write(f"Final Val G-L1 Center: {val_global_l1_center_curve[-1]:.6f}\n")
                f.write(f"Final Val G-L1 Middle: {val_global_l1_middle_curve[-1]:.6f}\n")
                f.write(f"Final Val G-L1 Edge  : {val_global_l1_edge_curve[-1]:.6f}\n")

            f.write(f"Time Elapsed (s)     : {time.time() - start_time:.2f}\n")
            f.write(f"PSNR Curve           : {[f'{v:.4f}' for v in psnr_curve]}\n")
            f.write(f"SSIM Curve           : {[f'{v:.4f}' for v in ssim_curve]}\n")
            f.write(f"RawInput PSNR Curve  : {[f'{v:.4f}' for v in raw_input_psnr_curve]}\n")
            f.write(f"RawInput SSIM Curve  : {[f'{v:.4f}' for v in raw_input_ssim_curve]}\n")
            f.write(f"RawInput L1 Curve    : {[f'{v:.6f}' for v in raw_input_l1_curve]}\n")
            f.write(f"Val G-L1 Center Curve: {[f'{v:.6f}' for v in val_global_l1_center_curve]}\n")
            f.write(f"Val G-L1 Middle Curve: {[f'{v:.6f}' for v in val_global_l1_middle_curve]}\n")
            f.write(f"Val G-L1 Edge Curve  : {[f'{v:.6f}' for v in val_global_l1_edge_curve]}\n\n")

            f.write("Last Epoch Train Stats:\n")
            for k, v in train_stats.items():
                if isinstance(v, (float, int)):
                    f.write(f"  {k:<30}: {v:.8f}\n")
                else:
                    f.write(f"  {k:<30}: {v}\n")
            f.write("\n")
            _append_aux_to_file(f, train_stats)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    accelerator.end_training()
