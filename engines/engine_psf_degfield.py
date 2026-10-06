"""Independent training engine for Restormer_PSFLikeDegField."""
import os
import time
import warnings
import socket
import gc
from pathlib import Path
from typing import Tuple

import cv2
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator, DistributedDataParallelKwargs

from torchmetrics.functional import peak_signal_noise_ratio
from torchmetrics.functional import structural_similarity_index_measure as ssim

from ptflops import get_model_complexity_info
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.functional as TF

from config import Config
from loss import SSIMLoss
from utils import seed_everything, save_checkpoint

try:
    from muon import MuonWithAuxAdam
except Exception:
    MuonWithAuxAdam = None


warnings.filterwarnings("ignore")

_CREATED_PG = False

# engine_Psfbasis.py is placed in:
#   project_root/engines/engine_Psfbasis.py
#
# This engine is adapted for models/Res_Psfbasis.py:
#   - Restormer_PSFLikeDegField
#   - Restormer_PSFLikeLowRankBasis
# It keeps the same static meta -> clean data pipeline, while adding
# PSF-like regularization and optional degradation-level contrastive loss.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
# Static processed dataset
# ============================================================
def _list_image_files(root):
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
    return sorted([f for f in os.listdir(root) if f.lower().endswith(exts)])


def _build_pairs_from_two_dirs(gt_dir, meta_dir):
    gt_files = _list_image_files(gt_dir)
    meta_files = _list_image_files(meta_dir)

    gt_set = set(gt_files)
    meta_set = set(meta_files)
    common = sorted(list(gt_set & meta_set))

    if len(common) == 0:
        raise RuntimeError(
            f"No paired files found by identical names.\n"
            f"GT dir  : {gt_dir}\n"
            f"Meta dir: {meta_dir}"
        )

    return [(os.path.join(gt_dir, name), os.path.join(meta_dir, name)) for name in common]


class StaticMetaPairDataset(Dataset):
    """
    Static processed meta/clean dataset for restoration.

    Directory structure:
        root/train/ground_truth
        root/train/meta
        root/test/ground_truth
        root/test/meta

    Task:
        input  = meta / metalens degraded image
        target = clean / clear image

    Return style:
        tar_tensor, inp_tensor, filename

    Here:
        tar_tensor = clean
        inp_tensor = meta

    Important:
        This dataset does NOT do alignment.
        Alignment has already been done by run_data.py when generating
        the temporary static dataset.
    """

    def __init__(
        self,
        gt_dir,
        meta_dir,
        img_options=None,
        split="train",
        repeat_factor=1,
    ):
        self.pairs = _build_pairs_from_two_dirs(gt_dir, meta_dir)

        self.split = split
        self.repeat_factor = max(1, int(repeat_factor)) if split == "train" else 1

        self.img_options = img_options or {"w": 256, "h": 256}
        self.ps_w = int(self.img_options["w"])
        self.ps_h = int(self.img_options["h"])

    def __len__(self):
        return len(self.pairs) * self.repeat_factor

    def _index_map(self, index):
        return index % len(self.pairs)

    def _pad_if_needed(self, img, min_h, min_w):
        h, w = img.shape[:2]

        pad_h = max(min_h - h, 0)
        pad_w = max(min_w - w, 0)

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        if pad_h > 0 or pad_w > 0:
            img = cv2.copyMakeBorder(
                img,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                borderType=cv2.BORDER_REFLECT_101,
            )

        return img

    def _crop_pair(self, clean_img, meta_img):
        """
        Synchronously crop clean/meta.
        Train:
            reflection padding + random crop + synchronized flips/rotations

        Val:
            reflection padding + center crop
        """
        if meta_img.shape[:2] != clean_img.shape[:2]:
            meta_img = cv2.resize(
                meta_img,
                (clean_img.shape[1], clean_img.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        clean_pad = self._pad_if_needed(clean_img, self.ps_h, self.ps_w)
        meta_pad = self._pad_if_needed(meta_img, self.ps_h, self.ps_w)

        padded_h, padded_w = clean_pad.shape[:2]

        max_top = padded_h - self.ps_h
        max_left = padded_w - self.ps_w

        if max_top < 0 or max_left < 0:
            raise ValueError(
                f"Padded image smaller than patch. "
                f"padded=({padded_h},{padded_w}), patch=({self.ps_h},{self.ps_w})"
            )

        if self.split == "train":
            crop_top = np.random.randint(0, max_top + 1) if max_top > 0 else 0
            crop_left = np.random.randint(0, max_left + 1) if max_left > 0 else 0
        else:
            crop_top = max_top // 2
            crop_left = max_left // 2

        clean_crop = clean_pad[
            crop_top:crop_top + self.ps_h,
            crop_left:crop_left + self.ps_w,
            :
        ]

        meta_crop = meta_pad[
            crop_top:crop_top + self.ps_h,
            crop_left:crop_left + self.ps_w,
            :
        ]

        if self.split == "train":
            if np.random.rand() < 0.3:
                clean_crop = np.ascontiguousarray(np.fliplr(clean_crop))
                meta_crop = np.ascontiguousarray(np.fliplr(meta_crop))

            if np.random.rand() < 0.3:
                clean_crop = np.ascontiguousarray(np.flipud(clean_crop))
                meta_crop = np.ascontiguousarray(np.flipud(meta_crop))

            if np.random.rand() < 0.3:
                k = int(np.random.randint(1, 4))
                clean_crop = np.ascontiguousarray(np.rot90(clean_crop, k))
                meta_crop = np.ascontiguousarray(np.rot90(meta_crop, k))

        return clean_crop, meta_crop

    def __getitem__(self, index):
        idx = self._index_map(index)
        gt_path, meta_path = self.pairs[idx]

        clean_img = np.array(Image.open(gt_path).convert("RGB"))
        meta_img = np.array(Image.open(meta_path).convert("RGB"))

        clean_img, meta_img = self._crop_pair(clean_img, meta_img)

        tar_tensor = TF.to_tensor(clean_img)  # clean target
        inp_tensor = TF.to_tensor(meta_img)   # meta input

        filename = os.path.splitext(os.path.basename(gt_path))[0]

        return tar_tensor, inp_tensor, filename


def build_all_datasets_for_data_compare(
    opt,
    *,
    train_gt_dir,
    train_meta_dir,
    test_gt_dir,
    test_meta_dir,
    train_repeat=1,
):
    """
    Build datasets from preprocessed static folders.

    Task:
        meta -> clean restoration
    """
    img_options_train = {"w": opt.TRAINING.PS_W, "h": opt.TRAINING.PS_H}
    img_options_val = {"w": opt.TESTING.PS_W, "h": opt.TESTING.PS_H}

    train_dataset = StaticMetaPairDataset(
        gt_dir=train_gt_dir,
        meta_dir=train_meta_dir,
        img_options=img_options_train,
        split="train",
        repeat_factor=train_repeat,
    )

    val_dataset = StaticMetaPairDataset(
        gt_dir=test_gt_dir,
        meta_dir=test_meta_dir,
        img_options=img_options_val,
        split="val",
        repeat_factor=1,
    )

    print(f"[StaticMetaPairDataset] Train pairs: {len(train_dataset.pairs)}")
    print(f"[StaticMetaPairDataset] Val pairs  : {len(val_dataset.pairs)}")
    print(f"[StaticMetaPairDataset] Train GT   : {train_gt_dir}")
    print(f"[StaticMetaPairDataset] Train Meta : {train_meta_dir}")
    print(f"[StaticMetaPairDataset] Test GT    : {test_gt_dir}")
    print(f"[StaticMetaPairDataset] Test Meta  : {test_meta_dir}")
    print(f"[StaticMetaPairDataset] Task       : meta -> clean restoration")

    return {
        "train": train_dataset,
        "val": val_dataset,
    }


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


def compute_model_flops(model, opt):
    input_h = opt.TESTING.PS_H
    input_w = opt.TESTING.PS_W

    try:
        with torch.cuda.amp.autocast(enabled=False):
            macs, _ = get_model_complexity_info(
                model,
                (3, input_h, input_w),
                as_strings=False,
                print_per_layer_stat=False,
                verbose=False,
            )
        if macs is None:
            return -1
        return macs * 2
    except Exception:
        return -1


def compute_inference_time(model, opt, device, runs=100):
    dummy = torch.randn(1, 3, opt.TESTING.PS_H, opt.TESTING.PS_W).to(device)
    model.eval()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    timings = []

    with torch.no_grad():
        for _ in range(runs):
            t0 = time.time()
            _ = model(dummy)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.time()
            timings.append((t1 - t0) * 1000)

    timings = np.array(timings)
    return {
        "infer_time_avg": float(timings.mean()),
        "infer_time_std": float(timings.std()),
    }


def analyze_model_statistics(model, opt, device):
    stats_params = compute_model_params(model)
    flops = -1 if hasattr(model, "__ptflops_ignore__") else compute_model_flops(model, opt)
    latency = compute_inference_time(model, opt, device)

    stats = {
        **stats_params,
        "flops": flops,
        **latency,
        "resolution": (opt.TESTING.PS_W, opt.TESTING.PS_H),
    }

    print("========== Model Statistics ==========")
    print(f"Total Params        : {stats['total_params']:,}")
    print(f"Trainable Params    : {stats['trainable_params']:,}")
    if stats["flops"] is not None and stats["flops"] > 0:
        print(f"FLOPs (MACs*2)      : {stats['flops']:,}  (~{stats['flops']/1e9:.4f} GFLOPs)")
    else:
        print("FLOPs (MACs*2)      : N/A")
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

    muon_lr = muon_lr if (muon_lr is not None) else base_lr * muon_lr_mult

    param_groups = []

    if conv_kernels:
        param_groups.append(
            dict(
                params=conv_kernels,
                use_muon=True,
                lr=muon_lr,
                weight_decay=weight_decay,
            )
        )

    if others:
        param_groups.append(
            dict(
                params=others,
                use_muon=False,
                lr=base_lr,
                betas=(0.9, 0.999),
                weight_decay=weight_decay,
            )
        )

    return MuonWithAuxAdam(param_groups)


# ============================================================
# Metrics / log helpers
# ============================================================
def compute_pair_metrics(pred, tar):
    pred_metric = torch.clamp(pred, 0.0, 1.0)
    tar_metric = torch.clamp(tar, 0.0, 1.0)

    return {
        "psnr": float(
            peak_signal_noise_ratio(
                pred_metric,
                tar_metric,
                data_range=1.0,
            ).detach().item()
        ),
        "ssim": float(
            ssim(
                pred_metric,
                tar_metric,
                data_range=1.0,
            ).detach().item()
        ),
        "l1": float(torch.mean(torch.abs(pred_metric - tar_metric)).detach().item()),
    }


def _init_train_log_meter():
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
        "count": 0,
    }


def _update_train_log_meter(meter, pred, tar, inp, loss_total, loss_l1, loss_ssim):
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

        meter["count"] += 1


def _format_train_log_meter(meter):
    cnt = max(meter["count"], 1)
    return {
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
    }


# ============================================================
# Prior / module monitor helpers
# ============================================================
def _unwrap_model_if_needed(model):
    if hasattr(model, "module"):
        return model.module
    return model


def _supports_prior_monitor(model):
    raw_model = _unwrap_model_if_needed(model)
    if hasattr(raw_model, "get_last_monitor"):
        return True

    cls_name = raw_model.__class__.__name__
    return cls_name in [
        "Restormer_DegField",
        "Restormer_LowRankBasis",
        "Restormer_PSFLikeDegField",
        "Restormer_PSFLikeLowRankBasis",
    ]


def _model_forward_maybe_monitor(model, inp, monitor=False):
    """
    Keep baseline data processing unchanged.
    Only changes model call when monitor=True and model supports it.
    """
    if monitor and _supports_prior_monitor(model):
        out = model(inp, return_monitor=True)
        if isinstance(out, tuple) and len(out) == 2:
            return out[0], out[1]
        return out, None

    return model(inp), None


def _init_prior_monitor_meter():
    return {
        "sum": {},
        "count": 0,
    }


def _update_prior_monitor_meter(meter, monitor_dict):
    if monitor_dict is None:
        return

    with torch.no_grad():
        for k, v in monitor_dict.items():
            if torch.is_tensor(v):
                value = float(v.detach().float().mean().cpu().item())
            else:
                try:
                    value = float(v)
                except Exception:
                    continue

            if k not in meter["sum"]:
                meter["sum"][k] = 0.0
            meter["sum"][k] += value

        meter["count"] += 1


def _format_prior_monitor_meter(meter):
    cnt = max(meter["count"], 1)
    return {k: v / cnt for k, v in meter["sum"].items()}


def _short_prior_monitor_string(stats, max_items=12):
    if not stats:
        return ""

    preferred_keys = [
        # DegField / PSF-like degradation field
        "deg_score_mean",
        "deg_score_std",
        "deg_score_edge_center_gap",
        "deg_score_radial_corr",
        "deg_feat_abs_mean",
        "deg_feat_std",
        # LowRank / PSF-like basis coefficient field
        "prior_feat_std",
        "coeff_entropy_norm",
        "coeff_top_prob_mean",
        "coeff_spatial_std",
        "coeff_edge_center_gap",
        "coeff_radial_corr",
        "basis_usage_std",
        "basis_usage_min",
        "basis_usage_max",
        "basis_cos_abs_offdiag",
        "basis_diversity_loss",
        # Layer-wise modulation strength
        "shallow_delta_ratio",
        "enc1_delta_ratio",
        "enc2_delta_ratio",
        "latent_delta_ratio",
        "dec1_delta_ratio",
    ]

    keys = [k for k in preferred_keys if k in stats]
    if len(keys) == 0:
        keys = list(stats.keys())[:max_items]

    keys = keys[:max_items]
    return " | " + " | ".join([f"{k}={stats[k]:.6f}" for k in keys])


def _append_prior_monitor_to_file(f, prior_stats):
    if prior_stats is None or len(prior_stats) == 0:
        return

    f.write("\\nLast Epoch Prior/Module Monitor:\\n")
    for k in sorted(prior_stats.keys()):
        f.write(f"  {k:<34}: {prior_stats[k]:.8f}\\n")
    f.write("\\n")


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
    """
    Optional PSF-like auxiliary regularization.

    Supported by Restormer_PSFLikeDegField:
        - deg_smoothness_loss()
        - deg_radial_prior_loss()

    Supported by Restormer_PSFLikeLowRankBasis:
        - basis_diversity_loss()
        - coeff_smoothness_loss()
        - coeff_entropy_loss()

    All lambdas default to 0.0, so baseline behavior is unchanged.
    """
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
            reg_entropy = raw_model.coeff_entropy_loss(
                target_entropy=coeff_entropy_target,
                mode=coeff_entropy_mode,
            )
        except TypeError:
            reg_entropy = raw_model.coeff_entropy_loss()
        _add("coeff_entropy", reg_entropy, lambda_coeff_entropy)

    if lambda_deg_radial > 0 and hasattr(raw_model, "deg_radial_prior_loss"):
        try:
            reg_radial = raw_model.deg_radial_prior_loss(
                min_corr=deg_radial_min_corr,
                min_gap=deg_radial_min_gap,
                require_edge_larger=deg_radial_require_edge_larger,
            )
        except TypeError:
            reg_radial = raw_model.deg_radial_prior_loss(min_corr=deg_radial_min_corr)
        _add("deg_radial", reg_radial, lambda_deg_radial)

    if reg is None:
        device = next(raw_model.parameters()).device
        reg = torch.tensor(0.0, device=device)

    return reg, reg_terms


def _supervised_nt_xent(z, labels, temperature=0.1, eps=1e-8):
    """
    Supervised NT-Xent for PSF-like degradation embeddings.

    z:      [N, D]
    labels: [N]
    """
    if z is None or z.dim() != 2:
        raise ValueError(f"z must be [N,D], got {None if z is None else tuple(z.shape)}")

    labels = labels.view(-1).to(device=z.device)
    valid = labels >= 0
    z = z[valid]
    labels = labels[valid]

    n = z.shape[0]
    if n <= 1:
        return z.sum() * 0.0

    z = F.normalize(z.float(), dim=1)
    logits = (z @ z.t()) / float(temperature)
    logits = logits - logits.detach().max(dim=1, keepdim=True).values

    eye = torch.eye(n, device=z.device, dtype=torch.bool)
    pos_mask = (labels[:, None] == labels[None, :]) & (~eye)

    if pos_mask.sum() == 0:
        return z.sum() * 0.0

    logits_for_denom = logits.masked_fill(eye, -1e9)
    log_prob = logits - torch.logsumexp(logits_for_denom, dim=1, keepdim=True)

    pos_count = pos_mask.sum(dim=1)
    valid_anchor = pos_count > 0
    mean_log_prob_pos = (log_prob * pos_mask.float()).sum(dim=1) / (pos_count.float() + eps)
    return -mean_log_prob_pos[valid_anchor].mean()


def _make_psf_field_labels(
    batch_size,
    grid_size=(9, 9),
    device=None,
    mode="radial",
    coarse_factor=2,
    radial_bins=5,
):
    """
    Create labels for grid degradation embeddings.

    mode="cell":
        Same grid cell across different images is positive.
        Useful when batch_size > 1.

    mode="coarse":
        Neighboring cells in coarse blocks are positives.
        More useful when batch_size is small.

    mode="radial":
        Cells with similar field radius are positives.
        Recommended when batch_size=1 and metalens degradation is field/radius-related.
    """
    gh, gw = int(grid_size[0]), int(grid_size[1])
    yy = torch.arange(gh, device=device).view(gh, 1).expand(gh, gw)
    xx = torch.arange(gw, device=device).view(1, gw).expand(gh, gw)

    if mode == "cell":
        labels = yy * gw + xx
    elif mode == "coarse":
        cf = max(1, int(coarse_factor))
        labels = (yy // cf) * ((gw + cf - 1) // cf) + (xx // cf)
    elif mode == "radial":
        y = (yy.float() + 0.5) / gh * 2.0 - 1.0
        x = (xx.float() + 0.5) / gw * 2.0 - 1.0
        r = torch.sqrt(torch.clamp(x * x + y * y, min=1e-12))
        r = r / (r.max() + 1e-6)
        labels = torch.clamp((r * int(radial_bins)).long(), 0, int(radial_bins) - 1)
    else:
        raise ValueError(f"Unknown contrastive label mode: {mode}")

    labels = labels.reshape(1, gh, gw).expand(int(batch_size), -1, -1).reshape(-1)
    return labels


def _optional_psf_contrastive_loss(
    model,
    batch_size,
    lambda_contrastive=0.0,
    source="auto",
    grid_size=(9, 9),
    temperature=0.1,
    label_mode="radial",
    coarse_factor=2,
    radial_bins=5,
):
    """
    Optional degradation-level contrastive loss for Res_Psfbasis models.

    source="auto": prefer coeff embedding if available, otherwise deg embedding.
    source="coeff": only use get_coeff_grid_embedding().
    source="deg":   only use get_deg_grid_embedding().
    """
    raw_model = _unwrap_model_if_needed(model)
    if lambda_contrastive <= 0:
        device = next(raw_model.parameters()).device
        return torch.tensor(0.0, device=device), {}

    z = None
    used_source = None

    if source in ["auto", "coeff"] and hasattr(raw_model, "get_coeff_grid_embedding"):
        try:
            z = raw_model.get_coeff_grid_embedding(grid_size=grid_size, normalize=True, detach=False)
            used_source = "coeff"
        except Exception:
            if source == "coeff":
                raise

    if z is None and source in ["auto", "deg"] and hasattr(raw_model, "get_deg_grid_embedding"):
        try:
            z = raw_model.get_deg_grid_embedding(grid_size=grid_size, normalize=True, detach=False)
            used_source = "deg"
        except Exception:
            if source == "deg":
                raise

    if z is None:
        device = next(raw_model.parameters()).device
        return torch.tensor(0.0, device=device), {}

    labels = _make_psf_field_labels(
        batch_size=batch_size,
        grid_size=grid_size,
        device=z.device,
        mode=label_mode,
        coarse_factor=coarse_factor,
        radial_bins=radial_bins,
    )

    loss_ctr = _supervised_nt_xent(z, labels, temperature=temperature)
    stats = {
        "psf_ctr_loss": float(loss_ctr.detach().float().cpu().item()),
        "psf_ctr_source": used_source,
        "psf_ctr_embed_abs_mean": float(z.detach().float().abs().mean().cpu().item()),
    }
    return loss_ctr * lambda_contrastive, stats


# ============================================================
# Train one variant
# ============================================================
def train_variant(
    model_name,
    model_class,
    batch_size,
    model_args,
    *,
    data_mode="raw_none",
    train_gt_dir,
    train_meta_dir,
    test_gt_dir,
    test_meta_dir,
    weak_method="orb_homography",
    flow_method="dis",
    flow_strength=0.5,
    config_path="config.yml",
    checkpoint_root=None,
    result_file=None,
    prior_monitor_every=50,
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
    lambda_psf_contrastive=0.0,
    contrastive_source="auto",
    contrastive_grid_size=(9, 9),
    contrastive_temperature=0.1,
    contrastive_label_mode="radial",
    contrastive_coarse_factor=2,
    contrastive_radial_bins=5,
):
    """
    Train one restoration variant using preprocessed static dataset.

    Task:
        meta -> clean

    Model input is always 3-channel RGB meta image.
    For CoordRestormer, coordinate channels are generated inside the model.
    """
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
        checkpoint_root = str(PROJECT_ROOT / "checkpoints" / "data_compare")
    checkpoint_root = str((PROJECT_ROOT / checkpoint_root).resolve()) if not os.path.isabs(checkpoint_root) else checkpoint_root

    model_save_dir = os.path.join(checkpoint_root, data_mode, model_name)
    os.makedirs(model_save_dir, exist_ok=True)

    if result_file is None:
        result_file = os.path.join(model_save_dir, "result.txt")

    print(
        f"\n=== Training {model_name} [meta -> clean | static_data={data_mode}] "
        f"| USE_MUON={use_muon} ==="
    )

    start_time = time.time()

    accelerator = Accelerator(
        kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=True)
        ]
    )
    device = accelerator.device

    if use_muon:
        _ensure_dist_initialized(True)

    datasets = build_all_datasets_for_data_compare(
        opt,
        train_gt_dir=train_gt_dir,
        train_meta_dir=train_meta_dir,
        test_gt_dir=test_gt_dir,
        test_meta_dir=test_meta_dir,
        train_repeat=1,
    )

    train_dataset = datasets["train"]
    val_dataset = datasets["val"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = model_class(**model_args).to(device)
    model_stats = analyze_model_statistics(model, opt, device)

    optimizer = build_optimizer(
        model,
        base_lr,
        weight_decay,
        use_muon,
        muon_lr=muon_lr,
        muon_lr_mult=muon_lr_mult,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=opt.OPTIM.NUM_EPOCHS,
        eta_min=lr_min,
    )

    train_loader, val_loader, model, optimizer, scheduler = accelerator.prepare(
        train_loader,
        val_loader,
        model,
        optimizer,
        scheduler,
    )

    l1_criterion = torch.nn.L1Loss()
    # ssim_criterion = SSIMLoss()

    lambda_l1 = 1.0
    lambda_ssim = 0.2

    best_psnr, best_ssim = 0.0, 0.0
    psnr_curve, ssim_curve = [], []
    raw_input_psnr_curve, raw_input_ssim_curve, raw_input_l1_curve = [], [], []

    total_epochs = opt.OPTIM.NUM_EPOCHS

    for epoch in range(1, total_epochs + 1):
        model.train()
        log_meter = _init_train_log_meter()
        prior_meter = _init_prior_monitor_meter()

        for tar, inp, _ in tqdm(
            train_loader,
            desc=f"{model_name} Epoch {epoch} [data-{data_mode}]",
        ):
            inp = inp.to(device)   # meta
            tar = tar.to(device)   # clean

            optimizer.zero_grad(set_to_none=True)

            monitor_dict = None
            do_prior_monitor = (
                prior_monitor_every is not None
                and int(prior_monitor_every) > 0
                and (log_meter["count"] % int(prior_monitor_every) == 0)
            )

            pred, monitor_dict = _model_forward_maybe_monitor(
                model,
                inp,
                monitor=do_prior_monitor,
            )

            _update_prior_monitor_meter(prior_meter, monitor_dict)

            if pred.shape != tar.shape:
                pred = torch.nn.functional.interpolate(
                    pred,
                    size=tar.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )

            # loss_l1 = l1_criterion(pred, tar)
            # loss_ssim = 1.0 - ssim_criterion(torch.clamp(pred, 0.0, 1.0), tar)
            # loss = lambda_l1 * loss_l1 + lambda_ssim * loss_ssim
            # ------------------------------------------------

            # Restoration loss
            # ------------------------------------------------
            # Clamp prediction before computing pixel/perceptual-structure losses.
            # This avoids unstable supervision when residual-style models produce
            # values outside [0, 1].
            pred_clamp = torch.clamp(pred, 0.0, 1.0)

            # L1 reconstruction loss.
            loss_l1 = l1_criterion(pred_clamp, tar)

            # Use torchmetrics SSIM directly to avoid ambiguity of SSIMLoss().
            # This is a true loss term:
            #     lower is better, 0 means perfect structural similarity.
            loss_ssim = 1.0 - ssim(
                pred_clamp,
                tar,
                data_range=1.0,
            )

            loss = lambda_l1 * loss_l1 + lambda_ssim * loss_ssim

            prior_reg, prior_reg_terms = _optional_prior_regularization(
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

            psf_ctr_reg, psf_ctr_terms = _optional_psf_contrastive_loss(
                model,
                batch_size=inp.shape[0],
                lambda_contrastive=lambda_psf_contrastive,
                source=contrastive_source,
                grid_size=contrastive_grid_size,
                temperature=contrastive_temperature,
                label_mode=contrastive_label_mode,
                coarse_factor=contrastive_coarse_factor,
                radial_bins=contrastive_radial_bins,
            )
            if torch.is_tensor(psf_ctr_reg) and psf_ctr_reg.detach().abs().item() > 0:
                loss = loss + psf_ctr_reg

            # Store auxiliary-loss statistics in the same monitor meter.
            aux_monitor = {}
            for _k, _v in prior_reg_terms.items():
                aux_monitor[f"reg_{_k}"] = torch.as_tensor(_v, device=inp.device)
            for _k, _v in psf_ctr_terms.items():
                if isinstance(_v, (int, float)):
                    aux_monitor[f"reg_{_k}"] = torch.as_tensor(_v, device=inp.device)
            if aux_monitor:
                _update_prior_monitor_meter(prior_meter, aux_monitor)

            accelerator.backward(loss)
            optimizer.step()

            _update_train_log_meter(
                log_meter,
                pred=pred,
                tar=tar,
                inp=inp,
                loss_total=loss,
                loss_l1=loss_l1,
                loss_ssim=loss_ssim,
            )

        scheduler.step()

        train_stats = _format_train_log_meter(log_meter)
        prior_stats = _format_prior_monitor_meter(prior_meter)

        print(
            f"[{model_name} | {data_mode}] Epoch {epoch}: "
            f"TrainLoss={train_stats['TrainLoss']:.6f} | "
            f"L1={train_stats['L1']:.6f} | "
            f"SSIMLoss={train_stats['SSIMLoss']:.6f} | "
            f"PredMean={train_stats['PredMean']:.4f} | "
            f"PredStd={train_stats['PredStd']:.4f} | "
            f"TarMean={train_stats['TarMean']:.4f} | "
            f"TarStd={train_stats['TarStd']:.4f} | "
            f"PredResMean={train_stats['PredResMean']:.4f} | "
            f"PredResStd={train_stats['PredResStd']:.4f} | "
            f"TarResMean={train_stats['TarResMean']:.4f} | "
            f"TarResStd={train_stats['TarResStd']:.4f}"
            f"{_short_prior_monitor_string(prior_stats)}"
        )

        if epoch % opt.TRAINING.VAL_AFTER_EVERY == 0:
            model.eval()

            psnr_total, ssim_total = 0.0, 0.0
            raw_psnr_total, raw_ssim_total, raw_l1_total = 0.0, 0.0, 0.0

            with torch.no_grad():
                for tar, inp, _ in val_loader:
                    inp = inp.to(device)   # meta
                    tar = tar.to(device)   # clean

                    raw_metrics = compute_pair_metrics(inp, tar)
                    raw_psnr_total += raw_metrics["psnr"]
                    raw_ssim_total += raw_metrics["ssim"]
                    raw_l1_total += raw_metrics["l1"]

                    pred, _ = _model_forward_maybe_monitor(
                        model,
                        inp,
                        monitor=False,
                    )

                    if pred.shape != tar.shape:
                        pred = torch.nn.functional.interpolate(
                            pred,
                            size=tar.shape[2:],
                            mode="bilinear",
                            align_corners=False,
                        )

                    pred_metric = torch.clamp(pred, 0.0, 1.0)

                    psnr_total += peak_signal_noise_ratio(
                        pred_metric,
                        tar,
                        data_range=1.0,
                    )

                    ssim_total += ssim(
                        pred_metric,
                        tar,
                        data_range=1.0,
                    )

            avg_psnr = psnr_total / len(val_loader)
            avg_ssim = ssim_total / len(val_loader)

            avg_raw_psnr = raw_psnr_total / len(val_loader)
            avg_raw_ssim = raw_ssim_total / len(val_loader)
            avg_raw_l1 = raw_l1_total / len(val_loader)

            psnr_curve.append(avg_psnr.item())
            ssim_curve.append(avg_ssim.item())
            raw_input_psnr_curve.append(float(avg_raw_psnr))
            raw_input_ssim_curve.append(float(avg_raw_ssim))
            raw_input_l1_curve.append(float(avg_raw_l1))

            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": accelerator.unwrap_model(model).state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    epoch,
                    model_save_dir,
                )

            if avg_ssim > best_ssim:
                best_ssim = avg_ssim

            print(
                f"[{model_name} | {data_mode}] Epoch {epoch}: "
                f"PSNR={avg_psnr:.4f} (Best={best_psnr:.4f}), "
                f"SSIM={avg_ssim:.4f} (Best={best_ssim:.4f}) | "
                f"RawInputPSNR={avg_raw_psnr:.4f} | "
                f"RawInputSSIM={avg_raw_ssim:.4f} | "
                f"RawInputL1={avg_raw_l1:.6f}"
            )

    if accelerator.is_local_main_process:
        with open(result_file, "a", encoding="utf-8") as f:
            f.write(f"{'=' * 24} Summary: {model_name} | data={data_mode} {'=' * 24}\n")
            f.write(f"Task                 : meta -> clean restoration\n")
            f.write(f"Data Compare Mode    : {data_mode}\n")
            f.write(f"Weak Method          : {weak_method}\n")
            f.write(f"Flow Method          : {flow_method}\n")
            f.write(f"Flow Strength        : {flow_strength}\n")
            f.write(f"Train GT Dir         : {train_gt_dir}\n")
            f.write(f"Train Meta Dir       : {train_meta_dir}\n")
            f.write(f"Test GT Dir          : {test_gt_dir}\n")
            f.write(f"Test Meta Dir        : {test_meta_dir}\n")
            f.write(f"Model Name           : {model_name}\n")
            f.write(f"Total Params         : {model_stats['total_params']:,}\n")
            f.write(f"Trainable Params     : {model_stats['trainable_params']:,}\n")

            if model_stats["flops"] is not None and model_stats["flops"] > 0:
                f.write(f"FLOPs (MACs*2)       : {model_stats['flops']:,}\n")
            else:
                f.write(f"FLOPs (MACs*2)       : N/A\n")

            f.write(f"Resolution(FLOPs)    : {model_stats['resolution']}\n")
            f.write(f"InferTime Avg (ms)   : {model_stats['infer_time_avg']:.3f}\n")
            f.write(f"InferTime Std (ms)   : {model_stats['infer_time_std']:.3f}\n")
            f.write(f"Num Epochs           : {opt.OPTIM.NUM_EPOCHS}\n")
            f.write(f"Prior Monitor Every  : {prior_monitor_every}\n")
            f.write(f"Lambda Basis Div     : {lambda_basis_diversity}\n")
            f.write(f"Lambda Deg Smooth    : {lambda_deg_smoothness}\n")

            if len(psnr_curve) > 0:
                f.write(f"Best PSNR            : {best_psnr:.4f}\n")
                f.write(f"Final PSNR           : {psnr_curve[-1]:.4f}\n")
                f.write(f"Best SSIM            : {best_ssim:.4f}\n")
                f.write(f"Final SSIM           : {ssim_curve[-1]:.4f}\n")
                f.write(f"Final RawInput PSNR  : {raw_input_psnr_curve[-1]:.4f}\n")
                f.write(f"Final RawInput SSIM  : {raw_input_ssim_curve[-1]:.4f}\n")
                f.write(f"Final RawInput L1    : {raw_input_l1_curve[-1]:.6f}\n")

            f.write(f"Time Elapsed (s)     : {time.time() - start_time:.2f}\n")
            f.write(f"PSNR Curve           : {[f'{v:.4f}' for v in psnr_curve]}\n")
            f.write(f"SSIM Curve           : {[f'{v:.4f}' for v in ssim_curve]}\n")
            f.write(f"RawInput PSNR Curve  : {[f'{v:.4f}' for v in raw_input_psnr_curve]}\n")
            f.write(f"RawInput SSIM Curve  : {[f'{v:.4f}' for v in raw_input_ssim_curve]}\n")
            f.write(f"RawInput L1 Curve    : {[f'{v:.6f}' for v in raw_input_l1_curve]}\n\n")

            f.write("Last Epoch Train Stats:\n")
            f.write(f"  TrainLoss          : {train_stats['TrainLoss']:.6f}\n")
            f.write(f"  L1                 : {train_stats['L1']:.6f}\n")
            f.write(f"  SSIMLoss           : {train_stats['SSIMLoss']:.6f}\n")
            f.write(f"  PredMean           : {train_stats['PredMean']:.6f}\n")
            f.write(f"  PredStd            : {train_stats['PredStd']:.6f}\n")
            f.write(f"  TarMean            : {train_stats['TarMean']:.6f}\n")
            f.write(f"  TarStd             : {train_stats['TarStd']:.6f}\n")
            f.write(f"  PredResMean        : {train_stats['PredResMean']:.6f}\n")
            f.write(f"  PredResStd         : {train_stats['PredResStd']:.6f}\n")
            f.write(f"  TarResMean         : {train_stats['TarResMean']:.6f}\n")
            f.write(f"  TarResStd          : {train_stats['TarResStd']:.6f}\n\n")

            if "prior_stats" in locals() and len(prior_stats) > 0:
                _append_prior_monitor_to_file(f, prior_stats)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gc.collect()
    accelerator.end_training()