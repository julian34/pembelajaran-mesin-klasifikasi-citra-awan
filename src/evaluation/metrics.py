"""Metrik evaluasi klasifikasi 7 kelas (selaras docs/EXPERIMENT_PLAN §5 & §8).

Metrik utama macro-F1; dilengkapi balanced accuracy, accuracy, laporan per kelas
(precision/recall/F1 + support), dan confusion matrix 7x7. Semua pakai
`zero_division=0` dan label eksplisit 0..num_classes-1 agar kelas tanpa prediksi
tetap muncul.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def compute_metrics(
    y_true: Sequence[int], y_pred: Sequence[int], num_classes: int
) -> dict:
    """Hitung metrik ringkas + per kelas + confusion matrix.

    Returns dict berisi: macro_f1, balanced_accuracy, accuracy, per_class
    (list of {label, precision, recall, f1, support}), dan confusion_matrix
    (list of list, baris=true, kolom=pred).
    """
    labels = list(range(num_classes))
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = [
        {
            "label": int(lab),
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, lab in enumerate(labels)
    ]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def classification_report_text(
    y_true: Sequence[int], y_pred: Sequence[int], class_names: Sequence[str]
) -> str:
    """Laporan klasifikasi sklearn sebagai teks (untuk classification_report.txt)."""
    labels = list(range(len(class_names)))
    return classification_report(
        y_true, y_pred, labels=labels, target_names=list(class_names), zero_division=0
    )


def save_confusion_matrix(
    confusion: Sequence[Sequence[int]],
    class_names: Sequence[str],
    path: str | Path,
) -> None:
    """Simpan confusion matrix sebagai PNG (anotasi hitungan per sel)."""
    import matplotlib

    matplotlib.use("Agg")  # backend non-interaktif untuk penyimpanan file
    import matplotlib.pyplot as plt

    cm = np.asarray(confusion)
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(1.2 * n + 1, 1.2 * n + 1))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n), labels=list(class_names), rotation=45, ha="right")
    ax.set_yticks(range(n), labels=list(class_names))
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Sebenarnya")
    ax.set_title("Confusion Matrix")

    threshold = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=8,
            )
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
