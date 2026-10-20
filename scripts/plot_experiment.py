"""Buat grafik ringkas hasil eksperimen (gaya exp_006 AlexNet).

Menghasilkan tiga PNG di dalam folder eksperimen:
- plot_loss.png          : train_loss vs val_loss per epoch + garis epoch terbaik.
- plot_val_metrics.png   : val_macro_f1 / val_balanced_accuracy / val_accuracy per epoch.
- plot_per_class_f1.png  : precision / recall / F1 per kelas pada test (urut F1 turun).

Sumber data: training_history.csv dan metrics.json di dalam folder eksperimen.

Contoh:
    .venv/Scripts/python scripts/plot_experiment.py \
        experiments/exp_005_densenet121 --exp exp_005 --model DenseNet121
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Urutan label (indeks 0..6) sesuai classification_report.txt / metrics.json.
CLASS_NAMES = ["Cu", "Ac", "Ci", "Cl", "Sc", "Cb", "Mi"]


def _label_name(label: int) -> str:
    return CLASS_NAMES[label] if 0 <= label < len(CLASS_NAMES) else str(label)


def plot_loss(hist: pd.DataFrame, best_epoch: int, title: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(hist.index, hist["train_loss"], color="tab:blue", label="train_loss")
    ax.plot(hist.index, hist["val_loss"], color="tab:red", label="val_loss")
    ax.axvline(best_epoch, color="green", linestyle="--",
               label=f"epoch terbaik ({best_epoch})")
    ax.set_title(title + " — kurva loss per epoch")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)


def plot_val_metrics(hist: pd.DataFrame, best_epoch: int, best_val: float,
                     title: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(hist.index, hist["val_macro_f1"], color="tab:purple", label="val_macro_f1")
    ax.plot(hist.index, hist["val_balanced_accuracy"], color="tab:cyan",
            label="val_balanced_accuracy")
    ax.plot(hist.index, hist["val_accuracy"], color="tab:orange", label="val_accuracy")
    ax.axvline(best_epoch, color="green", linestyle="--")
    ax.plot([best_epoch], [best_val], "o", color="green", markersize=9,
            label=f"best macro-F1 {best_val:.4f} @ep {best_epoch}")
    ax.set_title(title + " — metrik validasi per epoch")
    ax.set_xlabel("epoch")
    ax.set_ylabel("skor")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)


def plot_per_class_f1(per_class: list[dict], macro_f1: float, support_total: int,
                      title: str, out: Path) -> None:
    rows = sorted(per_class, key=lambda r: r["f1"], reverse=True)
    names = [_label_name(r["label"]) for r in rows]
    prec = [r["precision"] for r in rows]
    rec = [r["recall"] for r in rows]
    f1 = [r["f1"] for r in rows]

    x = range(len(rows))
    width = 0.27
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar([i - width for i in x], prec, width, color="#9ecae1", label="precision")
    ax.bar(list(x), rec, width, color="#fb9a99", label="recall")
    bars_f1 = ax.bar([i + width for i in x], f1, width, color="blueviolet", label="F1")
    ax.axhline(macro_f1, color="green", linestyle="--", label=f"macro-F1 {macro_f1:.3f}")

    for rect, val in zip(bars_f1, f1):
        ax.annotate(f"{val:.2f}", (rect.get_x() + rect.get_width() / 2, val),
                    textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=9)

    support_str = f"{support_total:,}".replace(",", ".")
    ax.set_title(f"{title} — precision / recall / F1 per kelas (test GCD {support_str})")
    ax.set_ylabel("skor")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp_dir", type=Path, help="folder eksperimen")
    ap.add_argument("--exp", default=None, help="label eksperimen untuk judul, mis. exp_005")
    ap.add_argument("--model", default=None, help="nama model untuk judul, mis. DenseNet121")
    args = ap.parse_args()

    exp_dir: Path = args.exp_dir
    metrics = json.loads((exp_dir / "metrics.json").read_text(encoding="utf-8"))
    hist = pd.read_csv(exp_dir / "training_history.csv").set_index("epoch")

    exp = args.exp or metrics.get("experiment", exp_dir.name)
    model = args.model or ""
    title = f"{exp} {model}".strip()

    best_epoch = int(metrics["best_epoch"])
    best_val = float(metrics["best_val_metric"])
    test = metrics["test"]
    support_total = sum(r["support"] for r in test["per_class"])

    plot_loss(hist, best_epoch, title, exp_dir / "plot_loss.png")
    plot_val_metrics(hist, best_epoch, best_val, title, exp_dir / "plot_val_metrics.png")
    plot_per_class_f1(test["per_class"], float(test["macro_f1"]), support_total,
                      title, exp_dir / "plot_per_class_f1.png")

    print(f"Grafik tersimpan di {exp_dir}:")
    for f in ("plot_loss.png", "plot_val_metrics.png", "plot_per_class_f1.png"):
        print(f"  - {f}")


if __name__ == "__main__":
    main()
