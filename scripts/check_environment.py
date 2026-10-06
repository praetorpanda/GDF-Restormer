#!/usr/bin/env python3
"""Verify the minimal GDF-Restormer runtime before downloading/training."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

PACKAGES = [
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("accelerate", "accelerate"),
    ("torchmetrics", "torchmetrics"),
    ("numpy", "numpy"),
    ("cv2", "opencv-python"),
    ("PIL", "Pillow"),
    ("yaml", "PyYAML"),
    ("yacs", "yacs"),
    ("tqdm", "tqdm"),
    ("einops", "einops"),
    ("huggingface_hub", "huggingface_hub"),
]


def version_of(module):
    return getattr(module, "__version__", "unknown")


def main() -> int:
    print("=" * 72)
    print("GDF-Restormer environment check")
    print("=" * 72)
    print(f"Python         : {sys.version.split()[0]}")

    missing = []
    loaded = {}

    for import_name, package_name in PACKAGES:
        try:
            module = importlib.import_module(import_name)
            loaded[import_name] = module
            print(f"{package_name:<15}: {version_of(module)}")
        except Exception as exc:
            missing.append((package_name, exc))
            print(f"{package_name:<15}: MISSING ({exc})")

    if missing:
        print("\n[FAIL] Missing/incompatible packages:")
        for package, exc in missing:
            print(f"  - {package}: {exc}")
        return 1

    torch = loaded["torch"]
    print("-" * 72)
    print(f"torch CUDA     : {torch.version.cuda}")
    print(f"CUDA available : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU            : {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"GPU memory     : {props.total_memory / (1024**3):.2f} GiB")
    else:
        print("[WARN] CUDA is not available. Training will be impractically slow on CPU.")

    # Import the released model and confirm the checkpoint-compatible size.
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from models.Res_Strict import STRICT_MODEL_REGISTRY

    exp = STRICT_MODEL_REGISTRY["StrictGlobalDegField_4L"]
    model = exp["model_class"](**exp["model_args"])
    params = sum(p.numel() for p in model.parameters())

    print("-" * 72)
    print(f"Model          : {model.__class__.__name__}")
    print(f"Parameters     : {params:,}")

    if params != 25_971_477:
        print(
            f"[FAIL] Parameter count mismatch: "
            f"{params:,} != 25,971,477"
        )
        return 2

    print("[PASS] Minimal runtime and model construction are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
