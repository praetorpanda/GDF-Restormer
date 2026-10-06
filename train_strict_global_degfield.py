"""Train the paper StrictGlobalDegField_4L model on the fixed split."""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


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
# Engine / model registry
# ============================================================


# ============================================================
# Constants
# ============================================================
MODEL_KEY = "StrictGlobalDegField_4L"

DEFAULT_DATA_ROOT = PROJECT_ROOT / "datasets"
DEFAULT_SPLIT_NAME = "split_6500_1729"

PATCH_SIZE = 256
ROI_W = 768
ROI_H = 640
VAL_CROP_MODE = "center"

RESULT_DIR_NAME = "results"
CHECKPOINT_SUBDIR = "checkpoints"
RESULT_TXT_NAME = "strict_global_degfield.txt"

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# Same prior regularization as the current G5 fullcanvas run by default.
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


# ============================================================
# Helpers
# ============================================================
def _list_image_files(root: Path) -> List[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {root}")
    # Match GlobalMetaFullCanvasDataset: direct children, identical filenames
    # including extensions. Nested files are not consumed by the dataset.
    files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    files.sort(key=lambda p: p.name)
    return files


def _count_pairs(gt_dir: Path, meta_dir: Path) -> int:
    gt_files = {p.name for p in _list_image_files(gt_dir)}
    meta_files = {p.name for p in _list_image_files(meta_dir)}
    return len(gt_files & meta_files)


def _resolve_fixed_split_dirs(data_root: Path, split_name: str) -> Dict[str, Path]:
    split_root = data_root / split_name
    return {
        "split_root": split_root,
        "train_gt_dir": split_root / "train" / "gt",
        "train_meta_dir": split_root / "train" / "meta",
        "val_gt_dir": split_root / "val" / "gt",
        "val_meta_dir": split_root / "val" / "meta",
        "manifest": split_root / "split_manifest_fixed_sorted.csv",
    }


def _validate_fixed_split(paths: Dict[str, Path], expected_train: int, expected_val: int) -> Tuple[int, int]:
    for k in ["train_gt_dir", "train_meta_dir", "val_gt_dir", "val_meta_dir"]:
        if not paths[k].is_dir():
            raise FileNotFoundError(f"Required fixed split directory missing: {paths[k]}")

    train_count = _count_pairs(paths["train_gt_dir"], paths["train_meta_dir"])
    val_count = _count_pairs(paths["val_gt_dir"], paths["val_meta_dir"])

    if train_count != expected_train or val_count != expected_val:
        raise RuntimeError(
            "Fixed split count mismatch.\n"
            f"Expected train/val: {expected_train}/{expected_val}\n"
            f"Actual train/val  : {train_count}/{val_count}\n"
            f"Split root        : {paths['split_root']}"
        )
    return train_count, val_count


def _print_model_args(model_args: Dict[str, object]) -> None:
    base_keys = {
        "inp_channels", "out_channels", "dim", "num_blocks", "num_refinement_blocks",
        "heads", "ffn_expansion_factor", "bias", "LayerNorm_type",
    }
    print("      Model body:")
    for k in ["dim", "num_blocks", "num_refinement_blocks", "heads", "ffn_expansion_factor", "LayerNorm_type"]:
        if k in model_args:
            print(f"        {k:<24}= {model_args[k]}")
    print("      Extra args:")
    for k, v in model_args.items():
        if k not in base_keys:
            print(f"        {k:<24}= {v}")


def _append_run_header(result_file: Path, args: argparse.Namespace, paths: Dict[str, Path], train_count: int, val_count: int, exp: Dict[str, object]) -> None:
    result_file.parent.mkdir(parents=True, exist_ok=True)
    with result_file.open("a", encoding="utf-8") as f:
        f.write("\n" + "=" * 100 + "\n")
        f.write(f"[NEW MAIN MODEL FULLCANVAS FIXED6500 RUN] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"MODEL_KEY={MODEL_KEY}\n")
        f.write(f"MODEL_CLASS={exp['model_class'].__name__}\n")
        f.write(f"DESCRIPTION={exp.get('description', '')}\n")
        f.write(f"REQUIRES_EXTERNAL_COORD={exp.get('requires_external_coord', True)}\n")
        f.write(f"DATA_ROOT={args.data_root}\n")
        f.write(f"SPLIT_ROOT={paths['split_root']}\n")
        f.write(f"TRAIN_GT_DIR={paths['train_gt_dir']}\n")
        f.write(f"TRAIN_META_DIR={paths['train_meta_dir']}\n")
        f.write(f"VAL_GT_DIR={paths['val_gt_dir']}\n")
        f.write(f"VAL_META_DIR={paths['val_meta_dir']}\n")
        f.write(f"TRAIN_COUNT={train_count}\n")
        f.write(f"VAL_COUNT={val_count}\n")
        f.write(f"PATCH_SIZE={args.patch_size}\n")
        f.write(f"ROI_W={args.roi_w}\n")
        f.write(f"ROI_H={args.roi_h}\n")
        f.write(f"VAL_CROP_MODE={args.val_crop_mode}\n")
        f.write(f"VAL_BATCH_SIZE={args.val_batch_size}\n")
        f.write(f"GRID9_VAL_EVERY={args.grid9_val_every}\n")
        f.write(f"GRID9_VAL_BATCH_SIZE={args.grid9_val_batch_size}\n")
        f.write(f"EVAL_CKPT_MODE={args.eval_ckpt_mode}\n")
        f.write(f"RUN_UNIFIED_EVAL={not args.no_unified_eval}\n")
        f.write(f"EVAL_FULL={not args.no_eval_full}\n")
        f.write(f"EVAL_GRID9={args.eval_grid9}\n")
        f.write(f"CHECKPOINT_ROOT={args.checkpoint_root}\n")
        f.write(f"RESULT_FILE={result_file}\n")
        f.write("=" * 100 + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the paper StrictGlobalDegField model on fixed6500/1729 fullcanvas split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_root", type=str, default=str(DEFAULT_DATA_ROOT))
    p.add_argument("--split_name", type=str, default=DEFAULT_SPLIT_NAME)
    p.add_argument("--expected_train", type=int, default=6500)
    p.add_argument("--expected_val", type=int, default=1729)

    p.add_argument("--name", type=str, default=MODEL_KEY)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--config", type=str, default=str(PROJECT_ROOT / "config.yml"))
    p.add_argument("--checkpoint_root", type=str, default=str(PROJECT_ROOT / CHECKPOINT_SUBDIR))
    p.add_argument("--result_txt", type=str, default=str(PROJECT_ROOT / RESULT_DIR_NAME / RESULT_TXT_NAME))

    p.add_argument("--patch_size", type=int, default=PATCH_SIZE)
    p.add_argument("--roi_w", type=int, default=ROI_W)
    p.add_argument("--roi_h", type=int, default=ROI_H)
    p.add_argument("--val_crop_mode", type=str, default=VAL_CROP_MODE, choices=["center", "grid5", "grid9"])
    p.add_argument("--val_batch_size", type=int, default=1)
    p.add_argument("--val_num_workers", type=int, default=2)
    p.add_argument("--grid9_val_every", type=int, default=20)
    p.add_argument("--grid9_val_batch_size", type=int, default=1)
    p.add_argument("--grid9_val_num_workers", type=int, default=2)
    p.add_argument("--eval_ckpt_mode", type=str, default="center", choices=["center", "grid9", "latest", "auto"])

    p.add_argument("--train_repeat", type=int, default=1)
    p.add_argument("--prior_monitor_every", type=int, default=10)

    p.add_argument("--no_unified_eval", action="store_true")
    p.add_argument("--no_eval_full", action="store_true")
    p.add_argument("--eval_grid9", action="store_true", help="Run final RectROI-grid9 eval. Default off to save time.")
    p.add_argument("--no_eval_raw", action="store_true")
    p.add_argument("--max_eval_val_images", type=int, default=-1)
    p.add_argument("--full_eval_tile", type=int, default=256)
    p.add_argument("--full_eval_overlap", type=int, default=32)
    p.add_argument("--window_size", type=int, default=8)
    p.add_argument("--benchmark_runs", type=int, default=50)

    p.add_argument("--pretrained_ckpt", type=str, default=None, help="Optional model-only warm-start checkpoint. Leave empty to train from scratch, as in the paper log.")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset_module = PROJECT_ROOT / "data" / "dataset_GlobalMeta_FullCanvas.py"
    if not dataset_module.is_file():
        raise FileNotFoundError(
            "Required dataset module is missing: data/dataset_GlobalMeta_FullCanvas.py. "
            "Please use a complete checkout of this project."
        )
    from engines.engine_strict_global_degfield import train_variant_6500, cleanup_dist_if_needed
    from models.Res_Strict import STRICT_MODEL_REGISTRY as MODEL_REGISTRY

    if MODEL_KEY not in MODEL_REGISTRY:
        raise KeyError(
            f"MODEL_KEY={MODEL_KEY} is not found in MODEL_REGISTRY. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    exp = MODEL_REGISTRY[MODEL_KEY]
    model_class = exp["model_class"]
    model_args = exp["model_args"]
    requires_external_coord = bool(exp.get("requires_external_coord", True))

    data_root = Path(args.data_root).resolve()
    paths = _resolve_fixed_split_dirs(data_root, args.split_name)
    train_count, val_count = _validate_fixed_split(paths, args.expected_train, args.expected_val)

    result_file = Path(args.result_txt).resolve()
    checkpoint_root = Path(args.checkpoint_root).resolve()
    result_file.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "logs").mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("[MAIN MODEL 4LEVEL FULLCANVAS WARMSTART FIXED6500 RUNNER]")
    print(f"time:                 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"project_root:         {PROJECT_ROOT}")
    print(f"data_root:            {data_root}")
    print(f"split_root:           {paths['split_root']}")
    print(f"manifest:             {paths['manifest'] if paths['manifest'].exists() else 'not found'}")
    print(f"train_count:          {train_count}")
    print(f"val_count:            {val_count}")
    print(f"model_key:            {MODEL_KEY}")
    print(f"model_name:           {args.name}")
    print(f"model_class:          {model_class.__name__}")
    print(f"description:          {exp.get('description', '')}")
    print(f"pass_external_coord:  {requires_external_coord}")
    print(f"batch_size:           {args.batch_size}")
    print(f"patch_size:           {args.patch_size}")
    print(f"train_crop_mode:      fullcanvas")
    print(f"ROI WxH:              {args.roi_w} x {args.roi_h}  # final RectROI-grid9 eval only")
    print(f"val_crop_mode:        {args.val_crop_mode}")
    print(f"grid9_val_every:      {args.grid9_val_every}")
    print(f"eval_ckpt_mode:       {args.eval_ckpt_mode}")
    print(f"unified_eval:         {not args.no_unified_eval}")
    print(f"eval_full:            {not args.no_eval_full}")
    print(f"eval_grid9:           {args.eval_grid9}")
    print(f"checkpoint_root:      {checkpoint_root}")
    print(f"result_file:          {result_file}")
    _print_model_args(model_args)
    print("=" * 100)

    _append_run_header(result_file, args, paths, train_count, val_count, exp)

    if args.dry_run:
        print("[DRY RUN] Training was not launched.")
        print("[DRY RUN] Path check passed and model imports are available.")
        return

    start = time.time()
    try:
        summary = train_variant_6500(
            model_name=args.name,
            model_class=model_class,
            batch_size=args.batch_size,
            model_args=model_args,
            config_path=args.config,
            checkpoint_root=str(checkpoint_root),
            result_file=str(result_file),
            train_repeat=args.train_repeat,
            train_gt_dir=str(paths["train_gt_dir"]),
            train_meta_dir=str(paths["train_meta_dir"]),
            val_gt_dir=str(paths["val_gt_dir"]),
            val_meta_dir=str(paths["val_meta_dir"]),
            patch_size=args.patch_size,
            roi_w=args.roi_w,
            roi_h=args.roi_h,
            val_crop_mode=args.val_crop_mode,
            val_batch_size=args.val_batch_size,
            val_num_workers=args.val_num_workers,
            grid9_val_every=args.grid9_val_every,
            grid9_val_batch_size=args.grid9_val_batch_size,
            grid9_val_num_workers=args.grid9_val_num_workers,
            eval_ckpt_mode=args.eval_ckpt_mode,
            prior_monitor_every=args.prior_monitor_every,
            lambda_basis_diversity=COMMON_REG["lambda_basis_diversity"],
            lambda_deg_smoothness=COMMON_REG["lambda_deg_smoothness"],
            lambda_coeff_smoothness=COMMON_REG["lambda_coeff_smoothness"],
            lambda_coeff_entropy=COMMON_REG["lambda_coeff_entropy"],
            coeff_entropy_target=COMMON_REG["coeff_entropy_target"],
            coeff_entropy_mode=COMMON_REG["coeff_entropy_mode"],
            lambda_deg_radial=COMMON_REG["lambda_deg_radial"],
            deg_radial_min_corr=COMMON_REG["deg_radial_min_corr"],
            deg_radial_min_gap=COMMON_REG["deg_radial_min_gap"],
            deg_radial_require_edge_larger=COMMON_REG["deg_radial_require_edge_larger"],
            run_unified_eval=not args.no_unified_eval,
            eval_full=not args.no_eval_full,
            eval_grid9=args.eval_grid9,
            eval_raw=not args.no_eval_raw,
            max_eval_val_images=args.max_eval_val_images,
            full_eval_tile=args.full_eval_tile,
            full_eval_overlap=args.full_eval_overlap,
            window_size=args.window_size,
            visual_gt_dir=None,
            visual_meta_dir=None,
            visual_out_root=None,
            max_visual_images=-1,
            skip_visual=True,
            benchmark_runs=args.benchmark_runs,
            save_latest=True,
            pretrained_ckpt=args.pretrained_ckpt,
            pass_external_coord=requires_external_coord,
        )
        print("=" * 100)
        print("[RUN DONE]")
        print(f"elapsed_sec:      {time.time() - start:.2f}")
        print(f"model_key:        {MODEL_KEY}")
        print(f"best_psnr:        {summary.get('best_psnr')}")
        print(f"best_ssim:        {summary.get('best_ssim')}")
        print(f"best_epoch:       {summary.get('best_epoch')}")
        print(f"best_checkpoint:  {summary.get('best_checkpoint')}")
        print(f"result_file:      {summary.get('result_file')}")
        print("=" * 100)
    except Exception as exc:
        err = traceback.format_exc()
        print(f"[ERROR] Main model run failed: {MODEL_KEY} | {exc}")
        print(err)
        with result_file.open("a", encoding="utf-8") as f:
            f.write(f"\n[MODEL FAILED] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {MODEL_KEY}\n")
            f.write(f"Error: {repr(exc)}\n")
            f.write("Traceback:\n")
            f.write(err + "\n")
            f.write("-" * 100 + "\n")
        raise
    finally:
        cleanup_dist_if_needed()


if __name__ == "__main__":
    main()
