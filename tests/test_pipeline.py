"""Uji pipa baseline yang tidak membutuhkan PyTorch.

Mencakup resolusi manifes (`src/datasets/manifest.py`) dan metrik
(`src/evaluation/metrics.py`). Pemuatan model & loop pelatihan membutuhkan torch
sehingga tidak diuji di sini. Jalankan: `python -m pytest tests/ -q`.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.manifest import (  # noqa: E402
    label_from_class_dir,
    resolve_sample,
)
from src.evaluation.metrics import compute_metrics  # noqa: E402

CLASS_DIRS = [
    "1_cumulus", "2_altocumulus", "3_cirrus", "4_clearsky",
    "5_stratocumulus", "6_cumulonimbus", "7_mixed",
]


def test_label_mapping_is_prefix_minus_one():
    assert [label_from_class_dir(c) for c in CLASS_DIRS] == list(range(7))


def test_label_mapping_rejects_non_numeric_prefix():
    try:
        label_from_class_dir("cumulus")
    except ValueError:
        return
    raise AssertionError("seharusnya ValueError untuk nama tanpa prefiks numerik")


def test_resolve_train_row_uses_filename_column():
    # Manifes train punya kolom filename; path = split_dir/class/filename.
    row = {
        "class": "1_cumulus",
        "filename": "1_cumulus_000002.jpg",
        "kind": "original",
        "split": "train",
        "rel_path": "train/1_cumulus/1_cumulus_000002.jpg",
    }
    sample = resolve_sample(row, "data/combined_balanced/train")
    assert sample.label == 0
    assert sample.path == Path("data/combined_balanced/train/1_cumulus/1_cumulus_000002.jpg")


def test_resolve_val_row_ignores_source_prefix_in_rel_path():
    # rel_path val berprefiks train/ (lokasi SUMBER), tapi file fisik di val_dir.
    row = {
        "dataset": "GCD",
        "split": "val",
        "class": "1_cumulus",
        "src_class": "1_cumulus",
        "rel_path": "train/1_cumulus/1_cumulus_000001.jpg",
    }
    sample = resolve_sample(row, "data/combined/val")
    assert sample.path == Path("data/combined/val/1_cumulus/1_cumulus_000001.jpg")


def test_compute_metrics_perfect_prediction():
    y = [0, 1, 2, 3, 4, 5, 6]
    m = compute_metrics(y, y, num_classes=7)
    assert m["macro_f1"] == 1.0
    assert m["accuracy"] == 1.0
    assert len(m["confusion_matrix"]) == 7


def test_compute_metrics_includes_empty_classes_without_error():
    # Hanya 2 kelas muncul; metrik tetap 7x7 tanpa error pembagian nol.
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    m = compute_metrics(y_true, y_pred, num_classes=7)
    assert len(m["per_class"]) == 7
    assert len(m["confusion_matrix"]) == 7
    assert 0.0 <= m["macro_f1"] <= 1.0
