"""Histogram intensitas RGB untuk analisis dataset. Hanya membaca berkas citra."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image


def rgb_histogram(path: str) -> np.ndarray:
    """Histogram 256 bin per channel satu citra, bentuk (3, 256), nilai piksel asli 0-255."""
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"))
    return np.stack([np.bincount(arr[..., c].ravel(), minlength=256) for c in range(3)])


def histograms_by_group(
    root: Path, rel_paths: list[str], groups: list[str], workers: int = 6
) -> dict[str, np.ndarray]:
    """Jumlahkan rgb_histogram per grup (mis. "train/1_cumulus"); urutan groups mengikuti rel_paths."""
    full = [str(Path(root) / p) for p in rel_paths]
    out: dict[str, np.ndarray] = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for group, hist in zip(groups, ex.map(rgb_histogram, full, chunksize=200)):
            out[group] = out[group] + hist if group in out else hist.astype(np.int64)
    return out


def histogram_stats(hist: np.ndarray) -> dict[str, np.ndarray]:
    """Mean, std, min, dan max per channel (skala 0-255) dari histogram berbentuk (3, 256)."""
    levels = np.arange(256, dtype=np.float64)
    total = hist.sum(axis=1)
    mean = (hist * levels).sum(axis=1) / total
    var = (hist * levels ** 2).sum(axis=1) / total - mean ** 2
    present = hist > 0
    return {
        "mean": mean,
        "std": np.sqrt(var),
        "min": present.argmax(axis=1),
        "max": 255 - present[:, ::-1].argmax(axis=1),
    }
