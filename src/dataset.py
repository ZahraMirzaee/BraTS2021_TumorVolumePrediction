# This module defines the dataset class and helper functions for loading and processing BraTS MRI data.

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import List, Optional, Tuple
import pandas as pd
from pathlib import Path

@dataclass(frozen=True)
class Sample:
    image_paths: List[str]
    seg_path: Optional[str]
    y: float

def _load_nii(path: str) -> np.ndarray:
    img = nib.load(path)
    return np.asanyarray(img.dataobj).astype(np.float32, copy=False)

def _zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    m = float(x.mean())
    s = float(x.std())
    return (x - m) / (s + eps)

def _center_crop_or_pad_3d(vol: np.ndarray, target: Tuple[int, int, int]) -> np.ndarray:
    td, th, tw = target
    d, h, w = vol.shape
    out = vol
    # crop
    if d > td: out = out[(d - td)//2 : (d - td)//2 + td, :, :]
    if h > th: out = out[:, (h - th)//2 : (h - th)//2 + th, :]
    if w > tw: out = out[:, :, (w - tw)//2 : (w - tw)//2 + tw]
    # pad
    d2, h2, w2 = out.shape
    pd0, pd1 = max(0, (td - d2)//2), max(0, td - d2 - max(0, (td - d2)//2))
    ph0, ph1 = max(0, (th - h2)//2), max(0, th - h2 - max(0, (th - h2)//2))
    pw0, pw1 = max(0, (tw - w2)//2), max(0, tw - w2 - max(0, (tw - w2)//2))
    if any([pd0, pd1, ph0, ph1, pw0, pw1]):
        out = np.pad(out, ((pd0, pd1), (ph0, ph1), (pw0, pw1)), mode="constant", constant_values=0)
    return out

class BratsScalarDataset(Dataset):
    def __init__(self, samples: List[Sample], patch_size: Tuple[int, int, int], mask_with_seg: bool = False,
                 y_mean: float = 0.0, y_std: float = 1.0) -> None:
        self.samples = samples
        self.patch_size = patch_size
        self.mask_with_seg = mask_with_seg
        self.y_mean = y_mean
        self.y_std = y_std

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        vols = [_load_nii(p) for p in s.image_paths]
        x_chw = np.stack(vols, axis=0)  # (C, D, H, W)
        seg = None
        if self.mask_with_seg and s.seg_path:
            seg = _load_nii(s.seg_path)
            seg = (seg > 0).astype(np.float32)
            seg = _center_crop_or_pad_3d(seg, self.patch_size)
        c = x_chw.shape[0]
        x_out = np.zeros((c, *self.patch_size), dtype=np.float32)
        for ci in range(c):
            v = _center_crop_or_pad_3d(x_chw[ci], self.patch_size)
            if seg is not None:
                v *= seg
            v = _zscore(v)
            x_out[ci] = v
        y_raw = float(s.y)
        y_norm = (np.log1p(y_raw) - self.y_mean) / self.y_std if self.y_std != 0 else np.log1p(y_raw)
        return torch.from_numpy(x_out), torch.tensor([y_norm], dtype=torch.float32)

def read_samples_from_csv(csv_path: str, brats_root: str, label_column: str = "wt_volume_cm3") -> List[Sample]:
    df = pd.read_csv(csv_path)
    samples = []
    if "case_id" in df.columns and label_column in df.columns:
        root = Path(brats_root)
        for _, row in df.iterrows():
            case_id = str(row["case_id"])
            case_dir = root / case_id
            flair_path = str(case_dir / f"{case_id}_flair.nii.gz")
            t1_path = str(case_dir / f"{case_id}_t1.nii.gz")
            t1ce_path = str(case_dir / f"{case_id}_t1ce.nii.gz")
            t2_path = str(case_dir / f"{case_id}_t2.nii.gz")
            seg_path = str(case_dir / f"{case_id}_seg.nii.gz")
            y = float(row[label_column])
            samples.append(Sample(image_paths=[flair_path, t1_path, t1ce_path, t2_path], seg_path=seg_path, y=y))
    return samples