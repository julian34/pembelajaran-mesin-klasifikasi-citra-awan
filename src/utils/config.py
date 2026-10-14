"""Pemuatan konfigurasi eksperimen dan utilitas reprodusibilitas.

Konfigurasi eksperimen di `configs/experiments/*.yaml` hanya menuliskan nilai
yang BERBEDA dari `configs/base.yaml`; `load_config` menggabungkan keduanya
secara rekursif (nilai eksperimen menimpa base). Lihat `configs/base.yaml`.
"""
from __future__ import annotations

import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_BASE = Path("configs/base.yaml")


def _deep_merge(base: dict, override: dict) -> dict:
    """Gabungkan `override` ke atas `base` secara rekursif (tidak mengubah input).

    Dict digabung per kunci; nilai non-dict (termasuk list) ditimpa utuh.
    """
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_yaml(path: str | Path) -> dict:
    """Baca satu berkas YAML menjadi dict (dict kosong bila berkas kosong)."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def load_config(
    experiment: str | Path, base: str | Path = DEFAULT_BASE
) -> dict[str, Any]:
    """Muat config eksperimen di atas `configs/base.yaml` (merge rekursif).

    Args:
        experiment: path ke berkas YAML eksperimen (mis.
            `configs/experiments/exp_006_alexnet.yaml`).
        base: path ke konfigurasi dasar yang ditimpa.

    Returns:
        dict konfigurasi hasil penggabungan.
    """
    base_cfg = load_yaml(base)
    exp_cfg = load_yaml(experiment)
    return _deep_merge(base_cfg, exp_cfg)


def set_seed(seed: int) -> None:
    """Set seed untuk `random`, NumPy, dan (bila terpasang) PyTorch.

    torch diimpor di dalam fungsi agar util ini bisa dipakai tanpa torch.
    """
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy selalu ada di runtime proyek
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover - torch mungkin belum terpasang
        pass


def resolve_device(prefer: str = "auto") -> str:
    """Kembalikan "cuda" bila tersedia (atau diminta), selain itu "cpu"."""
    import torch

    if prefer == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"
