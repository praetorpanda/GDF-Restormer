# -*- coding: utf-8 -*-
"""Train Restormer_GlobalPSFScaleRetinexWaveletMoE. Historical protocol: early global: synth4000 RectROI."""

import argparse as _argparse
_CLI_DATA_ROOT = None
if __name__ == "__main__":
    _parser = _argparse.ArgumentParser(description="Train Restormer_GlobalPSFScaleRetinexWaveletMoE; early global: synth4000 RectROI. See README for protocol.")
    _parser.add_argument("--data_root", help="Paired dataset root containing gt/ and meta/")
    _CLI_DATA_ROOT = _parser.parse_args().data_root


import os
import sys
import shutil
import random
import time
import traceback
from datetime import datetime
from pathlib import Path

from tqdm import tqdm


# ============================================================
# Project path
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
ENGINES_DIR = PROJECT_ROOT / "engines"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ENGINES_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINES_DIR))


# ============================================================
# Engine
# ============================================================
from engines.engine_global_psf_wavelet_moe import train_variant, cleanup_dist_if_needed


# ============================================================
# Models
# ============================================================
from models.Res_Global import Restormer_GlobalPSFScaleRetinexWaveletMoE


# ============================================================
# Synthetic raw meta dataset
# ============================================================
SYNTH_ROOT = Path(_CLI_DATA_ROOT).resolve() if _CLI_DATA_ROOT else PROJECT_ROOT / "datasets"

SRC_GT_DIR = SYNTH_ROOT / "gt"
SRC_META_DIR = SYNTH_ROOT / "meta"


# ============================================================
# Prepared split dataset
# ============================================================
SPLIT_ROOT = PROJECT_ROOT / "tmp_datasets" / "global_psf_wavelet_moe"
REBUILD_SPLIT_DATASET = True

# IMPORTANT: user has limited disk space.
# Always remove the temporary split at the end, even if training fails.
DELETE_SPLIT_AFTER_RUN = True

# Use hardlink to save disk space; fallback to copy2 if hardlink fails.
USE_HARDLINK_IF_POSSIBLE = True


# ============================================================
# Split setting
# ============================================================
SEED = 1234
TRAIN_RATIO = 0.9
MAX_PAIRS = 4000


# ============================================================
# RectROI global-coordinate sampling setting
# ============================================================
PATCH_SIZE = 256
ROI_W = 768
ROI_H = 640
VAL_CROP_MODE = "grid9"  # center / grid5 / grid9, implemented in dataset_GlobalMeta.py


# ============================================================
# Result file control
# ============================================================
USE_SHARED_RESULT_FILE = True
RESET_SHARED_RESULT_FILE_AT_START = False

RESULT_DIR = PROJECT_ROOT / "results" / "global_psf_wavelet_moe"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

SHARED_RESULT_FILE = str(RESULT_DIR / "global_psf_wavelet_moe.txt")
INDIVIDUAL_RESULT_DIR = RESULT_DIR


# ============================================================
# Checkpoint / config
# ============================================================
CONFIG_PATH = str(PROJECT_ROOT / "config.yml")
CHECKPOINT_ROOT = str(PROJECT_ROOT / "checkpoints" / "global_psf_wavelet_moe")


# ============================================================
# Data helpers
# ============================================================
def _list_image_files(root: Path):
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")
    return sorted([p.name for p in root.iterdir() if p.is_file() and p.suffix.lower() in exts])


def _build_pairs(gt_dir: Path, meta_dir: Path):
    gt_files = set(_list_image_files(gt_dir))
    meta_files = set(_list_image_files(meta_dir))
    common = sorted(list(gt_files & meta_files))

    if len(common) == 0:
        raise RuntimeError(
            f"No paired files found.\n"
            f"GT dir  : {gt_dir}\n"
            f"Meta dir: {meta_dir}"
        )

    missing_gt = sorted(list(meta_files - gt_files))
    missing_meta = sorted(list(gt_files - meta_files))
    if missing_gt:
        print(f"[WARNING] meta files without gt: {len(missing_gt)}")
    if missing_meta:
        print(f"[WARNING] gt files without meta: {len(missing_meta)}")

    return [(gt_dir / name, meta_dir / name, name) for name in common]


def _copy_or_link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()

    if USE_HARDLINK_IF_POSSIBLE:
        try:
            os.link(str(src), str(dst))
            return "hardlink"
        except Exception:
            pass

    shutil.copy2(str(src), str(dst))
    return "copy"


def _prepare_split_dataset():
    """
    Output:
        SPLIT_ROOT/train/ground_truth
        SPLIT_ROOT/train/meta
        SPLIT_ROOT/test/ground_truth
        SPLIT_ROOT/test/meta
    """
    if REBUILD_SPLIT_DATASET and SPLIT_ROOT.exists():
        print(f"[PREPARE] Removing existing split dataset: {SPLIT_ROOT}")
        shutil.rmtree(SPLIT_ROOT)

    train_gt_dir = SPLIT_ROOT / "train" / "ground_truth"
    train_meta_dir = SPLIT_ROOT / "train" / "meta"
    test_gt_dir = SPLIT_ROOT / "test" / "ground_truth"
    test_meta_dir = SPLIT_ROOT / "test" / "meta"

    if (
        train_gt_dir.exists()
        and train_meta_dir.exists()
        and test_gt_dir.exists()
        and test_meta_dir.exists()
        and len(_list_image_files(train_gt_dir)) > 0
        and len(_list_image_files(train_meta_dir)) > 0
        and len(_list_image_files(test_gt_dir)) > 0
        and len(_list_image_files(test_meta_dir)) > 0
    ):
        print(f"[PREPARE] Reusing existing split dataset: {SPLIT_ROOT}")
        return {
            "train_gt_dir": str(train_gt_dir),
            "train_meta_dir": str(train_meta_dir),
            "test_gt_dir": str(test_gt_dir),
            "test_meta_dir": str(test_meta_dir),
        }

    pairs = _build_pairs(SRC_GT_DIR, SRC_META_DIR)
    rng = random.Random(SEED)
    rng.shuffle(pairs)

    if MAX_PAIRS is not None:
        pairs = pairs[: int(MAX_PAIRS)]

    total = len(pairs)
    train_count = int(total * TRAIN_RATIO)
    train_pairs = pairs[:train_count]
    test_pairs = pairs[train_count:]

    if len(test_pairs) == 0:
        raise RuntimeError("Test split is empty. Increase MAX_PAIRS or lower TRAIN_RATIO.")

    print("=" * 100)
    print("[PREPARE SYNTHETIC GLOBALMETA RECTROI WAVELET-LLHF-MOE SPLIT DATASET]")
    print(f"SYNTH_ROOT      = {SYNTH_ROOT}")
    print(f"SRC_GT_DIR      = {SRC_GT_DIR}")
    print(f"SRC_META_DIR    = {SRC_META_DIR}")
    print(f"SPLIT_ROOT      = {SPLIT_ROOT}")
    print(f"total pairs     = {total}")
    print(f"train pairs     = {len(train_pairs)}")
    print(f"test pairs      = {len(test_pairs)}")
    print(f"train ratio     = {TRAIN_RATIO}")
    print(f"seed            = {SEED}")
    print(f"max pairs       = {MAX_PAIRS}")
    print(f"patch size      = {PATCH_SIZE}")
    print(f"ROI WxH         = {ROI_W} x {ROI_H}")
    print(f"val crop mode   = {VAL_CROP_MODE}")
    print("=" * 100)

    for gt_path, meta_path, name in tqdm(train_pairs, desc="prepare train split"):
        _copy_or_link(gt_path, train_gt_dir / name)
        _copy_or_link(meta_path, train_meta_dir / name)

    for gt_path, meta_path, name in tqdm(test_pairs, desc="prepare test split"):
        _copy_or_link(gt_path, test_gt_dir / name)
        _copy_or_link(meta_path, test_meta_dir / name)

    manifest_path = SPLIT_ROOT / "split_manifest.txt"
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("Synthetic GlobalMeta RectROI Wavelet-LLHF-MoE split manifest\n")
        f.write("=" * 100 + "\n")
        f.write(f"SYNTH_ROOT={SYNTH_ROOT}\n")
        f.write(f"SRC_GT_DIR={SRC_GT_DIR}\n")
        f.write(f"SRC_META_DIR={SRC_META_DIR}\n")
        f.write(f"SPLIT_ROOT={SPLIT_ROOT}\n")
        f.write(f"total_pairs={total}\n")
        f.write(f"train_pairs={len(train_pairs)}\n")
        f.write(f"test_pairs={len(test_pairs)}\n")
        f.write(f"TRAIN_RATIO={TRAIN_RATIO}\n")
        f.write(f"SEED={SEED}\n")
        f.write(f"MAX_PAIRS={MAX_PAIRS}\n")
        f.write(f"PATCH_SIZE={PATCH_SIZE}\n")
        f.write(f"ROI_W={ROI_W}\n")
        f.write(f"ROI_H={ROI_H}\n")
        f.write(f"VAL_CROP_MODE={VAL_CROP_MODE}\n")

        f.write("\n[TRAIN]\n")
        for _, _, name in train_pairs:
            f.write(name + "\n")
        f.write("\n[TEST]\n")
        for _, _, name in test_pairs:
            f.write(name + "\n")

    return {
        "train_gt_dir": str(train_gt_dir),
        "train_meta_dir": str(train_meta_dir),
        "test_gt_dir": str(test_gt_dir),
        "test_meta_dir": str(test_meta_dir),
    }


def _prepare_result_file(result_file: str):
    result_path = Path(result_file)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    if RESET_SHARED_RESULT_FILE_AT_START and result_path.exists():
        print(f"[RESULT] Removing existing result file: {result_path}")
        result_path.unlink()
    elif result_path.exists():
        print(f"[RESULT] Existing result file found. New logs will be appended: {result_path}")
    else:
        print(f"[RESULT] Creating new result file: {result_path}")

    with open(result_path, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 100 + "\n")
        f.write(f"[NEW GLOBALMETA RECTROI WAVELET-LLHF-MOE RUN] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"SYNTH_ROOT={SYNTH_ROOT}\n")
        f.write(f"SRC_GT_DIR={SRC_GT_DIR}\n")
        f.write(f"SRC_META_DIR={SRC_META_DIR}\n")
        f.write(f"SPLIT_ROOT={SPLIT_ROOT}\n")
        f.write(f"MAX_PAIRS={MAX_PAIRS}\n")
        f.write(f"TRAIN_RATIO={TRAIN_RATIO}\n")
        f.write(f"SEED={SEED}\n")
        f.write(f"PATCH_SIZE={PATCH_SIZE}\n")
        f.write(f"ROI_W={ROI_W}\n")
        f.write(f"ROI_H={ROI_H}\n")
        f.write(f"VAL_CROP_MODE={VAL_CROP_MODE}\n")
        f.write(f"CHECKPOINT_ROOT={CHECKPOINT_ROOT}\n")
        f.write(f"RESULT_FILE={result_file}\n")
        f.write("=" * 100 + "\n")


def _append_run_status(result_file: str, text: str):
    result_path = Path(result_file)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


# ============================================================
# Model args
# ============================================================
RESTORMER_BASE_ARGS = {
    "inp_channels": 3,
    "out_channels": 3,
    "dim": 32,
    "num_blocks": [2, 3, 3],
    "num_refinement_blocks": 1,
    "heads": [1, 2, 4],
    "ffn_expansion_factor": 2.66,
    "bias": False,
    "LayerNorm_type": "BiasFree",
}

# Keep DegField settings here for direct comparison / easy re-enable.
# This experiment is disabled in EXPERIMENTS below.
GLOBAL_PSF_DEGFIELD_ARGS = {
    **RESTORMER_BASE_ARGS,
    "use_coord": True,
    "coord_channels": 4,
    "deg_ch": 16,
    "deg_mid_ch": 32,
    "deg_downsample_factor": 4,
    "modulate_levels": ("shallow", "enc", "latent", "dec"),
}

# Keep Basis settings here for direct comparison / easy re-enable.
# This experiment is disabled in EXPERIMENTS below.
GLOBAL_PSF_LOWRANKBASIS_ARGS = {
    **RESTORMER_BASE_ARGS,
    "use_coord": True,
    "coord_channels": 4,
    "prior_ch": 32,
    "prior_mid_ch": 32,
    "num_basis": 8,
    "coeff_hidden_ch": 32,
    "coeff_downsample_factor": 4,
    "coeff_floor": 0.05,
    "softmax_coeff": True,
    "modulate_levels": ("shallow", "enc", "latent", "dec"),
}

# Scale-aware Spatial MoE settings.
# Keep the same backbone and DegField settings as G1, then replace the single
# deg prior with small/middle/large scale experts and a soft global-coordinate router.
GLOBAL_PSF_SCALEMOE_ARGS = {
    **GLOBAL_PSF_DEGFIELD_ARGS,
    "scale_router_temperature": 1.0,
    "scale_expert_zero_init": True,
    "scale_expert_residual_scale": 0.25,
}

# Retinex-style illumination MoE settings.
# Keep the same backbone / DegField / ScaleMoE settings, then add one
# Retinex-style illumination expert as an additional router candidate.
GLOBAL_PSF_RETINEXMOE_ARGS = {
    **GLOBAL_PSF_SCALEMOE_ARGS,
    "illum_hidden_ch": 16,
}

# Retinex + Wavelet-LL MoE settings.
# This keeps the same backbone / Retinex-MoE setting, then adds one
# router-controlled Haar-like low-frequency wavelet expert.
GLOBAL_PSF_RETINEX_WAVELET_MOE_ARGS = {
    **GLOBAL_PSF_RETINEXMOE_ARGS,
    "wavelet_hidden_ch": 16,
}

# Light version. Keep all original settings; only reduce wavelet expert capacity.
GLOBAL_PSF_RETINEX_WAVELET_LIGHT_H8_MOE_ARGS = {
    **GLOBAL_PSF_RETINEXMOE_ARGS,
    "wavelet_hidden_ch": 8,
}

# Retinex + Wavelet-LL + Wavelet-HF-light MoE settings.
# Keep original G5 h16 Wavelet-LL expert unchanged, then add a lightweight
# directional high-frequency expert with hidden_ch=8.
GLOBAL_PSF_RETINEX_WAVELET_LLHF_MOE_ARGS = {
    **GLOBAL_PSF_RETINEX_WAVELET_MOE_ARGS,
    "wavelet_hf_hidden_ch": 8,
}


# ============================================================
# Experiments:
#   Keep DegField/Basis definitions but disabled.
#   Only ScaleMoE is enabled.
# ============================================================
EXPERIMENTS = [
{
        "enabled": True,
        "model_name": "global_psf_wavelet_moe",
        "model_class": Restormer_GlobalPSFScaleRetinexWaveletMoE,
        "batch_size": 2,
        "prior_monitor_every": 10,

        # Only add the optional Wavelet-LL expert; keep loss and regularization identical.
        "lambda_basis_diversity": 0.0,
        "lambda_deg_smoothness": 1e-4,
        "lambda_coeff_smoothness": 0.0,
        "lambda_coeff_entropy": 0.0,
        "coeff_entropy_target": 0.65,
        "coeff_entropy_mode": "min",

        "lambda_deg_radial": 0.0,
        "deg_radial_min_corr": 0.10,
        "deg_radial_min_gap": 0.0,
        "deg_radial_require_edge_larger": False,

        "model_args": {
            **GLOBAL_PSF_RETINEX_WAVELET_MOE_ARGS,
        },
    }
]


def _print_model_args(model_args):
    print("      Model body:")
    for k in ["dim", "num_blocks", "num_refinement_blocks", "heads", "ffn_expansion_factor", "LayerNorm_type"]:
        print(f"        {k:<24}= {model_args[k]}")
    print("      Extra args:")
    base_keys = set(RESTORMER_BASE_ARGS.keys())
    for k, v in model_args.items():
        if k not in base_keys:
            print(f"        {k:<24}= {v}")


def _status_block_for_exp(exp, result_file, checkpoint_root):
    lines = [
        "",
        "-" * 100,
        f"[MODEL START] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {exp['model_name']}",
        f"Model Class              : {exp['model_class'].__name__}",
        f"Enabled                  : {exp.get('enabled', True)}",
        f"Batch Size               : {exp.get('batch_size', 1)}",
        f"Checkpoint Root          : {checkpoint_root}",
        f"Result File              : {result_file}",
        f"Patch Size               : {PATCH_SIZE}",
        f"ROI W                    : {ROI_W}",
        f"ROI H                    : {ROI_H}",
        f"Val Crop Mode            : {VAL_CROP_MODE}",
    ]
    for k in [
        "prior_monitor_every",
        "lambda_basis_diversity",
        "lambda_deg_smoothness",
        "lambda_coeff_smoothness",
        "lambda_coeff_entropy",
        "coeff_entropy_target",
        "coeff_entropy_mode",
        "lambda_deg_radial",
        "deg_radial_min_corr",
        "deg_radial_min_gap",
        "deg_radial_require_edge_larger",
    ]:
        lines.append(f"{k:<25}: {exp.get(k)}")
    lines.append("-" * 100)
    return "\n".join(lines)


def main():
    if USE_SHARED_RESULT_FILE:
        _prepare_result_file(SHARED_RESULT_FILE)

    try:
        prepare_start = time.time()
        data_paths = _prepare_split_dataset()
        print(f"[PREPARE DONE] Time elapsed: {time.time() - prepare_start:.2f}s")

        for exp in EXPERIMENTS:
            if not exp.get("enabled", True):
                print(f"[SKIP] {exp['model_name']}")
                _append_run_status(SHARED_RESULT_FILE, f"[SKIP] {exp['model_name']}")
                continue

            model_name = exp["model_name"]
            model_class = exp["model_class"]
            model_args = exp["model_args"]
            batch_size = exp.get("batch_size", 1)

            if USE_SHARED_RESULT_FILE:
                result_file = SHARED_RESULT_FILE
            else:
                result_file = str(INDIVIDUAL_RESULT_DIR / f"result_{model_name}.txt")
                _prepare_result_file(result_file)

            print("=" * 100)
            print(f"[RUN] {model_name}")
            print("      task           = synthetic raw meta -> clean restoration")
            print("      engine         = engine_Global")
            print("      dataset        = dataset_GlobalMeta / RectROI random crop")
            print("      model          = strict global PSF-like Retinex-Wavelet LLHF MoE")
            print("      degfield       = defined but disabled")
            print("      basis          = defined but disabled")
            print("      scalemoe       = defined but disabled")
            print("      retinexmoe     = defined but disabled")
            print("      waveletmoe_h16 = defined but disabled")
            print("      wavelet_light  = h8 defined but disabled")
            print("      wavelet_llhf   = enabled for G6")
            print("      coord mode     = external global coord")
            print(f"      patch_size     = {PATCH_SIZE}")
            print(f"      ROI WxH        = {ROI_W} x {ROI_H}")
            print(f"      val_crop_mode  = {VAL_CROP_MODE}")
            print("      cleanup split  = True")
            print(f"      model_class    = {model_class.__name__}")
            print(f"      batch_size     = {batch_size}")
            print(f"      train_gt_dir   = {data_paths['train_gt_dir']}")
            print(f"      train_meta_dir = {data_paths['train_meta_dir']}")
            print(f"      test_gt_dir    = {data_paths['test_gt_dir']}")
            print(f"      test_meta_dir  = {data_paths['test_meta_dir']}")
            print(f"      config_path    = {CONFIG_PATH}")
            print(f"      checkpoint_root= {CHECKPOINT_ROOT}")
            print(f"      result_file    = {result_file}")
            _print_model_args(model_args)
            print("=" * 100)

            _append_run_status(result_file, _status_block_for_exp(exp, result_file, CHECKPOINT_ROOT))

            try:
                train_variant(
                    model_name=model_name,
                    model_class=model_class,
                    batch_size=batch_size,
                    model_args=model_args,
                    config_path=CONFIG_PATH,
                    checkpoint_root=CHECKPOINT_ROOT,
                    result_file=result_file,
                    train_gt_dir=data_paths["train_gt_dir"],
                    train_meta_dir=data_paths["train_meta_dir"],
                    test_gt_dir=data_paths["test_gt_dir"],
                    test_meta_dir=data_paths["test_meta_dir"],
                    patch_size=PATCH_SIZE,
                    roi_w=ROI_W,
                    roi_h=ROI_H,
                    val_crop_mode=VAL_CROP_MODE,
                    prior_monitor_every=exp.get("prior_monitor_every", 10),
                    lambda_basis_diversity=exp.get("lambda_basis_diversity", 0.0),
                    lambda_deg_smoothness=exp.get("lambda_deg_smoothness", 0.0),
                    lambda_coeff_smoothness=exp.get("lambda_coeff_smoothness", 0.0),
                    lambda_coeff_entropy=exp.get("lambda_coeff_entropy", 0.0),
                    coeff_entropy_target=exp.get("coeff_entropy_target", 0.65),
                    coeff_entropy_mode=exp.get("coeff_entropy_mode", "min"),
                    lambda_deg_radial=exp.get("lambda_deg_radial", 0.0),
                    deg_radial_min_corr=exp.get("deg_radial_min_corr", 0.10),
                    deg_radial_min_gap=exp.get("deg_radial_min_gap", 0.0),
                    deg_radial_require_edge_larger=exp.get("deg_radial_require_edge_larger", False),
                )
                _append_run_status(
                    result_file,
                    f"[MODEL DONE] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {model_name}\n"
                    + "-" * 100,
                )
            except Exception as e:
                err_text = traceback.format_exc()
                print(f"[ERROR] {model_name} failed: {e}")
                print(err_text)
                _append_run_status(
                    result_file,
                    f"[MODEL FAILED] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {model_name}\n"
                    f"Error: {repr(e)}\n"
                    "Traceback:\n"
                    f"{err_text}\n"
                    + "-" * 100,
                )
                continue
            finally:
                cleanup_dist_if_needed()

    finally:
        if SPLIT_ROOT.exists():
            print(f"[CLEANUP] Removing split dataset: {SPLIT_ROOT}")
            shutil.rmtree(SPLIT_ROOT)
        else:
            print(f"[CLEANUP] Split dataset already removed or not created: {SPLIT_ROOT}")

        _append_run_status(
            SHARED_RESULT_FILE,
            "[RUN END] " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n" + "=" * 100,
        )


if __name__ == "__main__":
    main()
