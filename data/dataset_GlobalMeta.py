# -*- coding: utf-8 -*-
"""
dataset_GlobalMeta.py
=====================

Additive dataset module for global-coordinate metalens / meta->clean restoration.

Purpose
-------
This file provides a safer alternative to the previous 320-canvas center-crop
global-coordinate dataset. Instead of always taking the center 256x256 region,
it uses a moderate central rectangular ROI and samples 256x256 patches inside it.

Recommended first setting for Hyper-Kvasir-like endoscopic frames:
    roi_w      = 768
    roi_h      = 640
    patch_size = 256

Key behavior
------------
- No valid-mask filtering is used.
- A central rectangular ROI is used to avoid overly aggressive black-border sampling.
- Training samples random 256x256 crops inside the ROI.
- Validation can use center / grid5 / grid9 crops inside the same ROI.
- Global coordinates are computed from the original full image coordinates,
  not re-normalized inside the crop.
- Return format is directly compatible with a global-coordinate engine:

    clean_crop_tensor : [3, patch_size, patch_size]
    meta_crop_tensor  : [3, patch_size, patch_size]
    global_coord      : [2, patch_size, patch_size]
    global_radius     : [1, patch_size, patch_size]
    filename          : str

Important
---------
This dataset returns already-cropped 256x256 patches. The engine should NOT do
the old PH:-PH center crop again. Use:

    inp   = meta_crop
    tar   = clean_crop
    coord = global_coord

instead of cropping a canvas.
"""

import os
import random
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset


# ============================================================
# File / pair helpers
# ============================================================

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def _list_image_files(root: str) -> List[str]:
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Directory not found: {root}")
    return sorted([f for f in os.listdir(root) if f.lower().endswith(_IMAGE_EXTS)])


def _build_pairs_from_two_dirs(gt_dir: str, meta_dir: str) -> List[Tuple[str, str]]:
    gt_files = set(_list_image_files(gt_dir))
    meta_files = set(_list_image_files(meta_dir))
    common = sorted(list(gt_files & meta_files))

    if len(common) == 0:
        raise RuntimeError(
            f"No paired files found by identical names.\n"
            f"GT dir  : {gt_dir}\n"
            f"Meta dir: {meta_dir}"
        )

    return [(os.path.join(gt_dir, name), os.path.join(meta_dir, name)) for name in common]


# ============================================================
# Geometry helpers
# ============================================================

def _pad_to_min_size_rgb(
    img: np.ndarray,
    min_h: int,
    min_w: int,
    border_mode: int = cv2.BORDER_REFLECT_101,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Reflection-pad an RGB image to at least min_h x min_w.

    Return:
        padded_img
        pad_info: top/bottom/left/right
    """
    h, w = img.shape[:2]
    pad_h = max(int(min_h) - h, 0)
    pad_w = max(int(min_w) - w, 0)

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
            borderType=border_mode,
        )

    return img, {
        "top": int(pad_top),
        "bottom": int(pad_bottom),
        "left": int(pad_left),
        "right": int(pad_right),
    }


def _central_rect_roi(
    padded_h: int,
    padded_w: int,
    roi_h: int,
    roi_w: int,
    patch_size: int,
) -> Tuple[int, int, int, int]:
    """
    Return central rectangular ROI in padded image coordinates:
        roi_top, roi_left, roi_h_eff, roi_w_eff

    roi_h/roi_w are clipped to padded size, but never smaller than patch_size.
    The caller should already ensure padded_h/w >= patch_size.
    """
    patch_size = int(patch_size)
    roi_h_eff = min(int(roi_h), int(padded_h))
    roi_w_eff = min(int(roi_w), int(padded_w))

    if roi_h_eff < patch_size or roi_w_eff < patch_size:
        raise ValueError(
            f"ROI smaller than patch. "
            f"roi=({roi_h_eff},{roi_w_eff}), patch={patch_size}, "
            f"padded=({padded_h},{padded_w})"
        )

    roi_top = max(0, (int(padded_h) - roi_h_eff) // 2)
    roi_left = max(0, (int(padded_w) - roi_w_eff) // 2)
    return roi_top, roi_left, roi_h_eff, roi_w_eff


def _grid_position_to_offset(
    position: str,
    roi_h: int,
    roi_w: int,
    patch_size: int,
) -> Tuple[int, int]:
    """
    Fixed crop offset inside ROI.

    Supported:
        center
        grid9 positions:
            top_left, top_center, top_right,
            middle_left, center, middle_right,
            bottom_left, bottom_center, bottom_right
        grid5 positions:
            center, top_center, middle_left, middle_right, bottom_center
    """
    max_top = int(roi_h) - int(patch_size)
    max_left = int(roi_w) - int(patch_size)

    if max_top < 0 or max_left < 0:
        raise ValueError(
            f"ROI smaller than patch. roi=({roi_h},{roi_w}), patch={patch_size}"
        )

    row_map = {
        "top": 0,
        "middle": max_top // 2,
        "bottom": max_top,
    }
    col_map = {
        "left": 0,
        "center": max_left // 2,
        "right": max_left,
    }

    if position == "center":
        return max_top // 2, max_left // 2

    if "_" not in position:
        raise ValueError(f"Unknown crop position: {position}")

    row, col = position.split("_", 1)
    if row not in row_map or col not in col_map:
        raise ValueError(f"Unknown crop position: {position}")

    return row_map[row], col_map[col]


def _build_global_coord_for_crop(
    full_h: int,
    full_w: int,
    crop_top_padded: int,
    crop_left_padded: int,
    pad_top: int,
    pad_left: int,
    patch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build global coordinates for a crop.

    Coordinates are computed against the original full image:
        x_global = 2 * x_original / (full_w - 1) - 1
        y_global = 2 * y_original / (full_h - 1) - 1

    Padded coordinates are mapped back to original coordinates by subtracting
    pad_top / pad_left. Values outside original image are clipped.
    """
    patch_size = int(patch_size)

    ys = np.arange(patch_size, dtype=np.float32) + int(crop_top_padded) - int(pad_top)
    xs = np.arange(patch_size, dtype=np.float32) + int(crop_left_padded) - int(pad_left)

    ys = np.clip(ys, 0, max(int(full_h) - 1, 0))
    xs = np.clip(xs, 0, max(int(full_w) - 1, 0))

    if int(full_h) <= 1:
        y_norm = np.zeros_like(ys, dtype=np.float32)
    else:
        y_norm = 2.0 * ys / float(int(full_h) - 1) - 1.0

    if int(full_w) <= 1:
        x_norm = np.zeros_like(xs, dtype=np.float32)
    else:
        x_norm = 2.0 * xs / float(int(full_w) - 1) - 1.0

    yy, xx = np.meshgrid(y_norm, x_norm, indexing="ij")

    coord = np.stack([xx, yy], axis=0).astype(np.float32)
    radius = np.sqrt(xx ** 2 + yy ** 2) / np.sqrt(2.0)
    radius = np.clip(radius, 0.0, 1.0).astype(np.float32)[None, ...]

    return torch.from_numpy(coord), torch.from_numpy(radius)


# ============================================================
# Dataset
# ============================================================



# ============================================================
# On-the-fly degradation synthesis for MetaMix-OTF
# ============================================================
def synthesize_otf_meta(
    clean_img: np.ndarray,
    global_coord: torch.Tensor,
    blur_strength: float = 1.0,
    illum_strength: float = 0.1,
    color_shift: float = 0.05,
    noise_std: float = 0.0,
) -> np.ndarray:
    """
    On-the-fly synthetic degradation for a single clean image.

    Parameters
    ----------
    clean_img : np.ndarray
        Input clean image (H x W x 3)
    global_coord : torch.Tensor
        [2, H, W] global coordinate map
    blur_strength : float
        Strength of radius-aware Gaussian blur
    illum_strength : float
        Low-frequency illumination variation magnitude
    color_shift : float
        Small per-channel gain
    noise_std : float
        Optional Gaussian noise

    Returns
    -------
    meta_img : np.ndarray
        Dynamically degraded image
    """
    import cv2
    import numpy as np

    H, W, _ = clean_img.shape
    meta_img = clean_img.astype(np.float32)

    # 1. Radius-aware blur
    radius_map = np.sqrt(global_coord[0].cpu().numpy() ** 2 + global_coord[1].cpu().numpy() ** 2) / np.sqrt(2)
    sigma_map = 0.5 + blur_strength * radius_map  # center small, outer larger

    # Approximate radius-dependent blur with uniform average for efficiency
    sigma = np.mean(sigma_map)
    ksize = int(2 * round(3 * sigma) + 1)
    if ksize % 2 == 0:
        ksize += 1
    meta_img = cv2.GaussianBlur(meta_img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)

    # 2. Low-frequency illumination modulation
    y = 0.299 * meta_img[:, :, 0] + 0.587 * meta_img[:, :, 1] + 0.114 * meta_img[:, :, 2]
    y_blur = cv2.GaussianBlur(y, (31, 31), 0)
    y_blur = (y_blur - y_blur.mean()) * illum_strength
    meta_img += y_blur[:, :, None]

    # 3. Color shift
    meta_img *= (1.0 + (np.random.randn(1, 1, 3) * color_shift))

    # 4. Optional Gaussian noise
    if noise_std > 0:
        meta_img += np.random.randn(*meta_img.shape).astype(np.float32) * noise_std

    # Clip to [0,255] and uint8
    meta_img = np.clip(meta_img, 0, 255).astype(np.uint8)

    return meta_img

class GlobalMetaRectROIDataset(Dataset):
    """
    Global-coordinate paired meta/clean dataset using a central rectangular ROI.

    Compared with GlobalCoordPSFPriorDataset in dataset_RGB.py:
    - This dataset directly returns 256x256 crops.
    - It does not return a larger canvas.
    - It uses a moderate central rectangular ROI instead of fixed 320->256 center crop.
    - It preserves global coordinates based on original full image positions.

    Return:
        clean_crop_tensor : [3, patch_size, patch_size]
        meta_crop_tensor  : [3, patch_size, patch_size]
        global_coord      : [2, patch_size, patch_size]
        global_radius     : [1, patch_size, patch_size]
        filename          : str
    """

    GRID9_POSITIONS = [
        "top_left", "top_center", "top_right",
        "middle_left", "center", "middle_right",
        "bottom_left", "bottom_center", "bottom_right",
    ]

    GRID5_POSITIONS = [
        "top_center",
        "middle_left",
        "center",
        "middle_right",
        "bottom_center",
    ]

    def __init__(
        self,
        gt_dir: str,
        meta_dir: str,
        split: str = "train",
        repeat_factor: int = 1,
        patch_size: int = 256,
        roi_w: int = 768,
        roi_h: int = 640,
        num_limit: Optional[int] = None,
        val_crop_mode: str = "grid9",
        random_seed: Optional[int] = None,
    ):
        super().__init__()

        self.pairs = _build_pairs_from_two_dirs(gt_dir, meta_dir)

        if num_limit is not None:
            num_limit = int(num_limit)
            if 0 < num_limit < len(self.pairs):
                self.pairs = self.pairs[:num_limit]

        self.split = str(split)
        if self.split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got {self.split}")

        self.repeat_factor = max(1, int(repeat_factor)) if self.split == "train" else 1
        self.patch_size = int(patch_size)
        self.roi_w = int(roi_w)
        self.roi_h = int(roi_h)
        self.val_crop_mode = str(val_crop_mode)
        self.random_seed = random_seed

        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive")

        if self.roi_w < self.patch_size or self.roi_h < self.patch_size:
            raise ValueError(
                f"ROI must be no smaller than patch. "
                f"roi=({self.roi_w},{self.roi_h}), patch={self.patch_size}"
            )

        if self.val_crop_mode not in ("center", "grid5", "grid9"):
            raise ValueError(
                f"val_crop_mode must be center/grid5/grid9, got {self.val_crop_mode}"
            )

    def __len__(self):
        if self.split == "train":
            return len(self.pairs) * self.repeat_factor

        if self.val_crop_mode == "grid9":
            return len(self.pairs) * len(self.GRID9_POSITIONS)
        if self.val_crop_mode == "grid5":
            return len(self.pairs) * len(self.GRID5_POSITIONS)
        return len(self.pairs)

    def _index_map(self, index: int) -> Tuple[int, Optional[int]]:
        if self.split == "train":
            return index % len(self.pairs), None

        if self.val_crop_mode == "grid9":
            pair_idx = index // len(self.GRID9_POSITIONS)
            crop_idx = index % len(self.GRID9_POSITIONS)
            return pair_idx, crop_idx

        if self.val_crop_mode == "grid5":
            pair_idx = index // len(self.GRID5_POSITIONS)
            crop_idx = index % len(self.GRID5_POSITIONS)
            return pair_idx, crop_idx

        return index % len(self.pairs), None

    def _select_crop(
        self,
        roi_top: int,
        roi_left: int,
        roi_h_eff: int,
        roi_w_eff: int,
        crop_idx: Optional[int],
        index: int,
    ) -> Tuple[int, int, str]:
        """
        Return crop_top_padded, crop_left_padded, position_name.
        """
        max_top_offset = int(roi_h_eff) - self.patch_size
        max_left_offset = int(roi_w_eff) - self.patch_size

        if self.split == "train":
            if self.random_seed is None:
                top_offset = random.randint(0, max_top_offset)
                left_offset = random.randint(0, max_left_offset)
            else:
                # Deterministic option for debugging.
                rng = random.Random(int(self.random_seed) + int(index))
                top_offset = rng.randint(0, max_top_offset)
                left_offset = rng.randint(0, max_left_offset)

            return (
                int(roi_top) + top_offset,
                int(roi_left) + left_offset,
                "random_rectroi",
            )

        if self.val_crop_mode == "center":
            position = "center"
        elif self.val_crop_mode == "grid9":
            position = self.GRID9_POSITIONS[int(crop_idx)]
        elif self.val_crop_mode == "grid5":
            position = self.GRID5_POSITIONS[int(crop_idx)]
        else:
            raise ValueError(f"Unknown val_crop_mode: {self.val_crop_mode}")

        top_offset, left_offset = _grid_position_to_offset(
            position=position,
            roi_h=roi_h_eff,
            roi_w=roi_w_eff,
            patch_size=self.patch_size,
        )

        return int(roi_top) + top_offset, int(roi_left) + left_offset, position

    def __getitem__(self, index: int):
        pair_idx, crop_idx = self._index_map(index)
        gt_path, meta_path = self.pairs[pair_idx]

        clean_img = np.array(Image.open(gt_path).convert("RGB"))
        meta_img = np.array(Image.open(meta_path).convert("RGB"))

        full_h, full_w = clean_img.shape[:2]

        if meta_img.shape[:2] != clean_img.shape[:2]:
            meta_img = cv2.resize(
                meta_img,
                (full_w, full_h),
                interpolation=cv2.INTER_LINEAR,
            )

        # Pad only if needed, mainly for robustness. For 1215x971 frames and
        # roi_w=768 / roi_h=640, this should not be triggered.
        min_h = max(self.roi_h, self.patch_size)
        min_w = max(self.roi_w, self.patch_size)
        clean_pad, pad_info = _pad_to_min_size_rgb(clean_img, min_h=min_h, min_w=min_w)
        meta_pad, _ = _pad_to_min_size_rgb(meta_img, min_h=min_h, min_w=min_w)

        padded_h, padded_w = clean_pad.shape[:2]

        roi_top, roi_left, roi_h_eff, roi_w_eff = _central_rect_roi(
            padded_h=padded_h,
            padded_w=padded_w,
            roi_h=self.roi_h,
            roi_w=self.roi_w,
            patch_size=self.patch_size,
        )

        crop_top, crop_left, position = self._select_crop(
            roi_top=roi_top,
            roi_left=roi_left,
            roi_h_eff=roi_h_eff,
            roi_w_eff=roi_w_eff,
            crop_idx=crop_idx,
            index=index,
        )

        clean_crop = clean_pad[
            crop_top:crop_top + self.patch_size,
            crop_left:crop_left + self.patch_size,
            :
        ]

        meta_crop = meta_pad[
            crop_top:crop_top + self.patch_size,
            crop_left:crop_left + self.patch_size,
            :
        ]

        if clean_crop.shape[0] != self.patch_size or clean_crop.shape[1] != self.patch_size:
            raise RuntimeError(
                f"Unexpected clean_crop size: {clean_crop.shape}, "
                f"patch_size={self.patch_size}"
            )

        global_coord, global_radius = _build_global_coord_for_crop(
            full_h=full_h,
            full_w=full_w,
            crop_top_padded=crop_top,
            crop_left_padded=crop_left,
            pad_top=pad_info["top"],
            pad_left=pad_info["left"],
            patch_size=self.patch_size,
        )

        clean_tensor = TF.to_tensor(clean_crop)
        meta_tensor = TF.to_tensor(meta_crop)

        base = os.path.splitext(os.path.basename(gt_path))[0]
        if self.split == "train":
            filename = base
        else:
            filename = f"{base}_{position}"

        return clean_tensor, meta_tensor, global_coord, global_radius, filename


# ============================================================
# Builder
# ============================================================

def build_globalmeta_rectroi_datasets(
    train_gt_dir: str,
    train_meta_dir: str,
    test_gt_dir: str,
    test_meta_dir: str,
    patch_size: int = 256,
    roi_w: int = 768,
    roi_h: int = 640,
    train_repeat: int = 1,
    val_crop_mode: str = "grid9",
    train_num_limit: Optional[int] = None,
    val_num_limit: Optional[int] = None,
):
    """
    Build train/val datasets for rectangular-ROI global-coordinate experiments.

    This is intended to replace build_globalcoord_psfprior_datasets(...) when
    you want larger global-coordinate coverage without using a valid-mask sampler.
    """
    train_dataset = GlobalMetaRectROIDataset(
        gt_dir=train_gt_dir,
        meta_dir=train_meta_dir,
        split="train",
        repeat_factor=train_repeat,
        patch_size=patch_size,
        roi_w=roi_w,
        roi_h=roi_h,
        num_limit=train_num_limit,
        val_crop_mode="center",
    )

    val_dataset = GlobalMetaRectROIDataset(
        gt_dir=test_gt_dir,
        meta_dir=test_meta_dir,
        split="val",
        repeat_factor=1,
        patch_size=patch_size,
        roi_w=roi_w,
        roi_h=roi_h,
        num_limit=val_num_limit,
        val_crop_mode=val_crop_mode,
    )

    print(f"[GlobalMetaRectROI] Train pairs : {len(train_dataset.pairs)}")
    print(f"[GlobalMetaRectROI] Val pairs   : {len(val_dataset.pairs)}")
    print(f"[GlobalMetaRectROI] Val samples : {len(val_dataset)}")
    print(f"[GlobalMetaRectROI] Patch size  : {patch_size}")
    print(f"[GlobalMetaRectROI] ROI WxH     : {roi_w} x {roi_h}")
    print(f"[GlobalMetaRectROI] Val crop    : {val_crop_mode}")
    print("[GlobalMetaRectROI] Return      : clean_crop, meta_crop, global_coord, global_radius, filename")

    return {
        "train": train_dataset,
        "val": val_dataset,
    }


# Backward-friendly alias names.
build_globalcoord_rectroi_datasets = build_globalmeta_rectroi_datasets
GlobalCoordRectROIDataset = GlobalMetaRectROIDataset


if __name__ == "__main__":
    # Minimal smoke-test example. Fill paths manually when needed.
    print("dataset_GlobalMeta.py loaded successfully.")
    print("Use build_globalmeta_rectroi_datasets(...) in engine_Global.py.")