# -*- coding: utf-8 -*-
"""
dataset_GlobalMeta_FullCanvas.py
================================

Full-canvas crop dataset for strict global-coordinate metalens / meta->clean restoration.

This file is a full-canvas counterpart of dataset_GlobalMeta.py:
- training samples random patch_size x patch_size crops from the whole padded image;
- validation center/grid5/grid9 crops are distributed over the whole padded image;
- global_coord/global_radius are computed from the original full-image coordinate system;
- return format is unchanged: clean_crop, meta_crop, global_coord, global_radius, filename.

Recommended usage in a new engine/run pair:
    from dataset_GlobalMeta_FullCanvas import build_globalmeta_fullcanvas_datasets

For quick drop-in tests, this file also aliases:
    build_globalmeta_rectroi_datasets = build_globalmeta_fullcanvas_datasets
but that means an old RectROI engine will actually receive full-canvas crops.
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


def _pad_to_min_size_rgb(
    img: np.ndarray,
    min_h: int,
    min_w: int,
    border_mode: int = cv2.BORDER_REFLECT_101,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Reflection-pad an RGB image to at least min_h x min_w."""
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


def _grid_position_to_offset(
    position: str,
    region_h: int,
    region_w: int,
    patch_size: int,
) -> Tuple[int, int]:
    """Fixed crop offset inside a region."""
    max_top = int(region_h) - int(patch_size)
    max_left = int(region_w) - int(patch_size)
    if max_top < 0 or max_left < 0:
        raise ValueError(
            f"Region smaller than patch. region=({region_h},{region_w}), patch={patch_size}"
        )

    row_map = {"top": 0, "middle": max_top // 2, "bottom": max_top}
    col_map = {"left": 0, "center": max_left // 2, "right": max_left}

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
    Build global coordinates for a crop in original full-image coordinates.
    Padded coordinates are mapped back by subtracting pad_top/pad_left and clipped.
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


class GlobalMetaFullCanvasDataset(Dataset):
    """
    Global-coordinate paired meta/clean dataset using full-canvas crop sampling.

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
        self.roi_w = int(roi_w)  # accepted only for compatibility; ignored
        self.roi_h = int(roi_h)  # accepted only for compatibility; ignored
        self.val_crop_mode = str(val_crop_mode)
        self.random_seed = random_seed

        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive")
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
            return index // len(self.GRID9_POSITIONS), index % len(self.GRID9_POSITIONS)
        if self.val_crop_mode == "grid5":
            return index // len(self.GRID5_POSITIONS), index % len(self.GRID5_POSITIONS)
        return index % len(self.pairs), None

    def _select_crop(
        self,
        padded_h: int,
        padded_w: int,
        crop_idx: Optional[int],
        index: int,
    ) -> Tuple[int, int, str]:
        """Return crop_top_padded, crop_left_padded, position_name."""
        max_top_offset = int(padded_h) - self.patch_size
        max_left_offset = int(padded_w) - self.patch_size

        if max_top_offset < 0 or max_left_offset < 0:
            raise ValueError(
                f"Padded image smaller than patch. padded=({padded_h},{padded_w}), "
                f"patch={self.patch_size}"
            )

        if self.split == "train":
            if self.random_seed is None:
                top_offset = random.randint(0, max_top_offset)
                left_offset = random.randint(0, max_left_offset)
            else:
                rng = random.Random(int(self.random_seed) + int(index))
                top_offset = rng.randint(0, max_top_offset)
                left_offset = rng.randint(0, max_left_offset)
            return int(top_offset), int(left_offset), "random_fullcanvas"

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
            region_h=padded_h,
            region_w=padded_w,
            patch_size=self.patch_size,
        )
        return int(top_offset), int(left_offset), position

    def __getitem__(self, index: int):
        pair_idx, crop_idx = self._index_map(index)
        gt_path, meta_path = self.pairs[pair_idx]

        clean_img = np.array(Image.open(gt_path).convert("RGB"))
        meta_img = np.array(Image.open(meta_path).convert("RGB"))
        full_h, full_w = clean_img.shape[:2]

        if meta_img.shape[:2] != clean_img.shape[:2]:
            meta_img = cv2.resize(meta_img, (full_w, full_h), interpolation=cv2.INTER_LINEAR)

        clean_pad, pad_info = _pad_to_min_size_rgb(
            clean_img,
            min_h=self.patch_size,
            min_w=self.patch_size,
        )
        meta_pad, _ = _pad_to_min_size_rgb(
            meta_img,
            min_h=self.patch_size,
            min_w=self.patch_size,
        )

        padded_h, padded_w = clean_pad.shape[:2]
        crop_top, crop_left, position = self._select_crop(
            padded_h=padded_h,
            padded_w=padded_w,
            crop_idx=crop_idx,
            index=index,
        )

        clean_crop = clean_pad[crop_top:crop_top + self.patch_size, crop_left:crop_left + self.patch_size, :]
        meta_crop = meta_pad[crop_top:crop_top + self.patch_size, crop_left:crop_left + self.patch_size, :]

        if clean_crop.shape[0] != self.patch_size or clean_crop.shape[1] != self.patch_size:
            raise RuntimeError(
                f"Unexpected clean_crop size: {clean_crop.shape}, patch_size={self.patch_size}"
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
        filename = base if self.split == "train" else f"{base}_{position}"
        return clean_tensor, meta_tensor, global_coord, global_radius, filename


def build_globalmeta_fullcanvas_datasets(
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
    Build train/val datasets for full-canvas global-coordinate crop experiments.

    roi_w/roi_h are accepted for compatibility with existing engine signatures,
    but they are not used for crop selection in this full-canvas dataset.
    """
    train_dataset = GlobalMetaFullCanvasDataset(
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

    val_dataset = GlobalMetaFullCanvasDataset(
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

    print(f"[GlobalMetaFullCanvas] Train pairs : {len(train_dataset.pairs)}")
    print(f"[GlobalMetaFullCanvas] Val pairs   : {len(val_dataset.pairs)}")
    print(f"[GlobalMetaFullCanvas] Val samples : {len(val_dataset)}")
    print(f"[GlobalMetaFullCanvas] Patch size  : {patch_size}")
    print(f"[GlobalMetaFullCanvas] Crop region : full padded image")
    print(f"[GlobalMetaFullCanvas] ROI WxH     : {roi_w} x {roi_h} (ignored; compatibility only)")
    print(f"[GlobalMetaFullCanvas] Val crop    : {val_crop_mode}")
    print("[GlobalMetaFullCanvas] Return      : clean_crop, meta_crop, global_coord, global_radius, filename")

    return {"train": train_dataset, "val": val_dataset}


build_globalcoord_fullcanvas_datasets = build_globalmeta_fullcanvas_datasets
GlobalCoordFullCanvasDataset = GlobalMetaFullCanvasDataset

# Drop-in compatibility aliases. If this file is copied over dataset_GlobalMeta.py,
# old engines importing build_globalmeta_rectroi_datasets will receive full-canvas crops.
build_globalmeta_rectroi_datasets = build_globalmeta_fullcanvas_datasets
build_globalcoord_rectroi_datasets = build_globalmeta_fullcanvas_datasets
GlobalMetaRectROIDataset = GlobalMetaFullCanvasDataset
GlobalCoordRectROIDataset = GlobalMetaFullCanvasDataset


if __name__ == "__main__":
    print("dataset_GlobalMeta_FullCanvas.py loaded successfully.")
    print("Use build_globalmeta_fullcanvas_datasets(...) for explicit full-canvas experiments.")
