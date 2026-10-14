"""Statistik per citra untuk analisis dataset. Hanya membaca berkas citra."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def image_stats(path: str) -> dict:
    """Statistik warna dan tekstur satu citra, dengan piksel pada rentang [0, 1]."""
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    mean = arr.mean(axis=(0, 1))
    mean_sq = (arr ** 2).mean(axis=(0, 1))
    gray = arr @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    mx, mn = arr.max(axis=2), arr.min(axis=2)
    saturation = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    gy, gx = np.gradient(gray)
    return {
        "mean_r": mean[0], "mean_g": mean[1], "mean_b": mean[2],
        "mean_sq_r": mean_sq[0], "mean_sq_g": mean_sq[1], "mean_sq_b": mean_sq[2],
        "brightness": gray.mean(),
        "contrast": gray.std(),
        "saturation": saturation.mean(),
        "edge_density": np.hypot(gx, gy).mean(),
        "red_blue_ratio": mean[0] / max(mean[2], 1e-6),
    }


def compute_stats(root: Path, rel_paths: list[str], workers: int = 6) -> pd.DataFrame:
    """Hitung image_stats untuk setiap path relatif terhadap root."""
    full = [str(Path(root) / p) for p in rel_paths]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(image_stats, full, chunksize=200))
    df = pd.DataFrame(rows).astype("float64")
    df.insert(0, "rel_path", rel_paths)
    return df


def channel_mean_std(stats: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Mean dan std per channel atas seluruh piksel (semua citra berukuran sama)."""
    mean = stats[["mean_r", "mean_g", "mean_b"]].mean().to_numpy()
    mean_sq = stats[["mean_sq_r", "mean_sq_g", "mean_sq_b"]].mean().to_numpy()
    return mean, np.sqrt(mean_sq - mean ** 2)
