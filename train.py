#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train GDF-Restormer (StrictGlobalDegField) on the released fixed split.

Default dataset layout::

    split_6500_1729/
        train/
            gt/
            meta/
        val/
            gt/
            meta/

This public entry point intentionally keeps logging compact. Release-preflight
diagnostics such as source hashes, git status, pip freeze, GPU dumps, metadata
JSON, and console tee logs are not generated here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import Config
from engines.engine_strict_global_degfield import (
    cleanup_dist_if_needed,
    train_variant_6500,
)
from models.Res_Strict import STRICT_MODEL_REGISTRY


MODEL_KEY = "StrictGlobalDegField_4L"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "split_6500_1729"
DEFAULT_CONFIG = PROJECT_ROOT / "config.yml"

ROI_W = 768
ROI_H = 640

COMMON_REG = {
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
}

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"
}


def image_names(directory: Path) -> list[str]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing dataset directory: {directory}")
    return sorted(
        p.name
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def resolve_dataset(data_root: Path, expected_train: int, expected_val: int):
    root = data_root.expanduser().resolve()
    paths = {
        "train_gt": root / "train" / "gt",
        "train_meta": root / "train" / "meta",
        "val_gt": root / "val" / "gt",
        "val_meta": root / "val" / "meta",
    }

    train_gt = image_names(paths["train_gt"])
    train_meta = image_names(paths["train_meta"])
    val_gt = image_names(paths["val_gt"])
    val_meta = image_names(paths["val_meta"])

    if train_gt != train_meta:
        raise RuntimeError("train/gt and train/meta filenames do not match.")
    if val_gt != val_meta:
        raise RuntimeError("val/gt and val/meta filenames do not match.")

    if len(train_gt) != expected_train:
        raise RuntimeError(
            f"Expected {expected_train} training pairs, found {len(train_gt)}."
        )
    if len(val_gt) != expected_val:
        raise RuntimeError(
            f"Expected {expected_val} validation pairs, found {len(val_gt)}."
        )

    overlap = set(train_gt) & set(val_gt)
    if overlap:
        raise RuntimeError(
            f"Train/val overlap detected ({len(overlap)} filenames)."
        )

    # Decode one pair from each split to catch path/corruption mistakes early.
    for gt_dir, meta_dir, names, split in (
        (paths["train_gt"], paths["train_meta"], train_gt, "train"),
        (paths["val_gt"], paths["val_meta"], val_gt, "val"),
    ):
        name = names[0]
        with Image.open(gt_dir / name) as gt, Image.open(meta_dir / name) as meta:
            if gt.size != meta.size:
                raise RuntimeError(
                    f"{split} sample size mismatch for {name}: "
                    f"gt={gt.size}, meta={meta.size}"
                )

    return root, paths


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train GDF-Restormer on Metalens-HyperKvasir."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Dataset root containing train/ and val/.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Training YAML configuration.",
    )
    parser.add_argument(
        "--name",
        default=MODEL_KEY,
        help="Output/checkpoint subdirectory name.",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=PROJECT_ROOT / "checkpoints",
    )
    parser.add_argument(
        "--result-file",
        type=Path,
        default=PROJECT_ROOT / "results" / "strict_global_degfield.txt",
    )
    parser.add_argument("--train-workers", type=int, default=4)
    parser.add_argument("--expected-train", type=int, default=6500)
    parser.add_argument("--expected-val", type=int, default=1729)

    # Preserve the validated training/evaluation protocol.
    parser.add_argument("--grid9-val-every", type=int, default=20)
    parser.add_argument("--max-eval-images", type=int, default=-1)
    parser.add_argument(
        "--eval-grid9",
        action="store_true",
        help="Also run final RectROI-grid9 evaluation.",
    )
    parser.add_argument(
        "--no-final-eval",
        action="store_true",
        help="Skip unified post-training evaluation.",
    )
    parser.add_argument(
        "--pretrained-ckpt",
        type=Path,
        default=None,
        help="Optional model-only warm start checkpoint.",
    )
    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=0,
        help="Optional pre-training latency benchmark. Default 0 keeps release startup concise.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate dataset/config/model construction without training.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    opt = Config(str(config_path))
    data_root, paths = resolve_dataset(
        args.data_root,
        expected_train=args.expected_train,
        expected_val=args.expected_val,
    )

    exp = STRICT_MODEL_REGISTRY[MODEL_KEY]
    model_class = exp["model_class"]
    model_args = exp["model_args"]
    requires_external_coord = bool(exp["requires_external_coord"])

    # Construct once for an immediate architecture/parameter sanity check.
    model = model_class(**model_args)
    total_params = sum(p.numel() for p in model.parameters())
    del model

    print("=" * 72)
    print("GDF-Restormer / StrictGlobalDegField")
    print("=" * 72)
    print(f"Data root      : {data_root}")
    print(
        f"Dataset        : {args.expected_train} train / "
        f"{args.expected_val} val pairs"
    )
    print(f"Model          : {model_class.__name__}")
    print(f"Parameters     : {total_params:,}")
    print(f"Epochs         : {int(opt.OPTIM.NUM_EPOCHS)}")
    print(f"Batch size     : {int(opt.OPTIM.BATCH_SIZE)}")
    print(
        f"Learning rate  : {float(opt.OPTIM.LR_INITIAL):.2e} -> "
        f"{float(opt.OPTIM.LR_MIN):.2e}"
    )
    print(f"Patch size     : {int(opt.TRAINING.PS_W)}")
    print(f"Checkpoint dir : {args.checkpoint_root / args.name}")
    print(f"Result file    : {args.result_file}")
    print("=" * 72)

    if total_params != 25_971_477:
        raise RuntimeError(
            "Unexpected model parameter count: "
            f"{total_params:,} (expected 25,971,477)."
        )

    if args.dry_run:
        print("[OK] Dataset, config, and model construction checks passed.")
        return

    try:
        summary = train_variant_6500(
            model_name=args.name,
            model_class=model_class,
            batch_size=int(opt.OPTIM.BATCH_SIZE),
            model_args=model_args,
            config_path=str(config_path),
            checkpoint_root=str(args.checkpoint_root.expanduser().resolve()),
            result_file=str(args.result_file.expanduser().resolve()),
            train_repeat=1,
            train_gt_dir=str(paths["train_gt"]),
            train_meta_dir=str(paths["train_meta"]),
            val_gt_dir=str(paths["val_gt"]),
            val_meta_dir=str(paths["val_meta"]),
            patch_size=int(opt.TRAINING.PS_W),
            roi_w=ROI_W,
            roi_h=ROI_H,
            val_crop_mode="center",
            train_num_workers=max(0, int(args.train_workers)),
            val_batch_size=1,
            val_num_workers=2,
            grid9_val_every=max(0, int(args.grid9_val_every)),
            grid9_val_batch_size=1,
            grid9_val_num_workers=2,
            eval_ckpt_mode="center",
            prior_monitor_every=10,
            lambda_basis_diversity=COMMON_REG["lambda_basis_diversity"],
            lambda_deg_smoothness=COMMON_REG["lambda_deg_smoothness"],
            lambda_coeff_smoothness=COMMON_REG["lambda_coeff_smoothness"],
            lambda_coeff_entropy=COMMON_REG["lambda_coeff_entropy"],
            coeff_entropy_target=COMMON_REG["coeff_entropy_target"],
            coeff_entropy_mode=COMMON_REG["coeff_entropy_mode"],
            lambda_deg_radial=COMMON_REG["lambda_deg_radial"],
            deg_radial_min_corr=COMMON_REG["deg_radial_min_corr"],
            deg_radial_min_gap=COMMON_REG["deg_radial_min_gap"],
            deg_radial_require_edge_larger=COMMON_REG[
                "deg_radial_require_edge_larger"
            ],
            run_unified_eval=not args.no_final_eval,
            eval_full=True,
            eval_grid9=bool(args.eval_grid9),
            eval_raw=True,
            max_eval_val_images=int(args.max_eval_images),
            full_eval_tile=256,
            full_eval_overlap=32,
            window_size=8,
            visual_gt_dir=None,
            visual_meta_dir=None,
            visual_out_root=None,
            max_visual_images=-1,
            skip_visual=True,
            benchmark_runs=max(0, int(args.benchmark_runs)),
            save_latest=True,
            pretrained_ckpt=(
                str(args.pretrained_ckpt.expanduser().resolve())
                if args.pretrained_ckpt is not None
                else None
            ),
            pass_external_coord=requires_external_coord,
        )
    finally:
        cleanup_dist_if_needed()

    print("\n" + "=" * 72)
    print("Training completed")
    print("=" * 72)
    print(f"Best epoch      : {summary.get('best_epoch')}")
    print(f"Best PSNR       : {summary.get('best_psnr')}")
    print(f"Best SSIM       : {summary.get('best_ssim')}")
    print(f"Best checkpoint : {summary.get('best_checkpoint')}")
    full = summary.get("eval_full") or {}
    if full.get("count", 0):
        print(
            "Full validation : "
            f"PSNR={full.get('psnr'):.4f}, "
            f"SSIM={full.get('ssim'):.4f}, "
            f"L1={full.get('l1'):.6f}"
        )
    print(f"Result file     : {summary.get('result_file')}")
    print("=" * 72)


if __name__ == "__main__":
    main()
