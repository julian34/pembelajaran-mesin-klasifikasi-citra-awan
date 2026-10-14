"""Runner pelatihan baseline (transfer learning) untuk klasifikasi citra awan.

Merangkai config -> data -> model -> pelatihan -> evaluasi test, lalu menulis
artefak eksperimen ke `experiments/<nama>/` dan checkpoint terbaik ke
`checkpoints/<nama>/best.pt`. Test dievaluasi SEKALI di akhir memakai checkpoint
terbaik (dipilih dari val), bukan untuk memilih model. Lihat docs/BASELINE_ALEXNET.md
Bagian 6 dan docs/TRAINING_GUIDE.md.

Contoh:
    python scripts/train.py --config configs/experiments/exp_006_alexnet.yaml
    python scripts/train.py --config configs/experiments/exp_006_alexnet.yaml --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path

# Pastikan akar repo ada di sys.path saat dijalankan sebagai skrip (python scripts/train.py).
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.manifest import load_samples, missing_files  # noqa: E402
from src.utils.config import load_config, resolve_device, set_seed  # noqa: E402


def write_outputs(exp_dir: Path, config: dict, result, test_metrics: dict, y_true, y_pred) -> None:
    """Tulis seluruh artefak evaluasi ke `exp_dir` (docs/BASELINE_ALEXNET.md §6)."""
    import yaml

    from src.evaluation.metrics import classification_report_text, save_confusion_matrix

    exp_dir.mkdir(parents=True, exist_ok=True)
    class_names = config["data"]["class_names"]
    monitor = config.get("checkpoint", {}).get("monitor", "val_macro_f1")

    # config efektif (hasil merge) untuk reprodusibilitas.
    with open(exp_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    # metrik: ringkasan val terbaik + test resmi.
    metrics = {
        "experiment": config.get("name"),
        "best_epoch": result.best_epoch,
        "monitor": monitor,
        "best_val_metric": result.best_metric,
        "test": test_metrics,
    }
    with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # riwayat per epoch.
    with open(exp_dir / "training_history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["epoch", "train_loss", "val_loss", "val_macro_f1",
             "val_balanced_accuracy", "val_accuracy"]
        )
        for r in result.history:
            writer.writerow(
                [r.epoch, r.train_loss, r.val_loss, r.val_macro_f1,
                 r.val_balanced_accuracy, r.val_accuracy]
            )

    # laporan klasifikasi test (teks) + confusion matrix.
    (exp_dir / "classification_report.txt").write_text(
        classification_report_text(y_true, y_pred, class_names), encoding="utf-8"
    )
    save_confusion_matrix(
        test_metrics["confusion_matrix"], class_names, exp_dir / "confusion_matrix.png"
    )

    # catatan ringkas.
    notes = (
        f"# Catatan {config.get('name')}\n\n"
        f"Tanggal: {date.today().isoformat()}\n\n"
        f"- Arsitektur: {config['model'].get('architecture')} "
        f"(finetune={config['model'].get('finetune')}, pretrained={config['model'].get('pretrained')})\n"
        f"- Epoch terbaik (val): {result.best_epoch}, metrik val terbaik ({monitor}): {result.best_metric:.4f}\n"
        f"- Test macro-F1: {test_metrics['macro_f1']:.4f} | "
        f"balanced acc: {test_metrics['balanced_accuracy']:.4f} | acc: {test_metrics['accuracy']:.4f}\n"
    )
    (exp_dir / "notes.md").write_text(notes, encoding="utf-8")


def check_data(config: dict) -> None:
    """Verifikasi manifes & keberadaan sebagian file sebelum pelatihan panjang."""
    data = config["data"]
    for split, (mkey, dkey) in {
        "train": ("train_manifest", "train_dir"),
        "val": ("val_manifest", "val_dir"),
        "test": ("test_manifest", "test_dir"),
    }.items():
        if mkey not in data or dkey not in data:
            raise KeyError(f"config.data kekurangan '{mkey}'/'{dkey}' untuk split {split!r}.")
        samples = load_samples(data[mkey], data[dkey])
        if not samples:
            raise ValueError(f"manifes split {split!r} kosong: {data[mkey]}")
        missing = missing_files(samples[:50])  # cek cuplikan agar cepat
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} file split {split!r} tidak ditemukan, mis. {missing[0]}"
            )
        print(f"  {split}: {len(samples)} sampel OK (dir {data[dkey]})")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--config", type=Path,
        default=Path("configs/experiments/exp_006_alexnet.yaml"),
        help="path config eksperimen (override terhadap configs/base.yaml).",
    )
    ap.add_argument("--base", type=Path, default=Path("configs/base.yaml"))
    ap.add_argument("--epochs", type=int, default=None, help="override training.epochs.")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument(
        "--dry-run", action="store_true",
        help="hanya validasi config & data (tanpa membangun model / melatih).",
    )
    args = ap.parse_args()

    config = load_config(args.config, args.base)
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = args.epochs
    set_seed(int(config.get("seed", 42)))

    name = config.get("name") or args.config.stem
    print(f"[config] {args.config} -> eksperimen '{name}'")
    print("[data] verifikasi manifes & file ...")
    check_data(config)

    if args.dry_run:
        print("[dry-run] config & data valid; berhenti sebelum membangun model.")
        return

    # Impor berat ditunda sampai setelah dry-run agar validasi tak butuh torch.
    from src.datasets.cloud_dataset import build_dataloader
    from src.evaluation.metrics import compute_metrics
    from src.models.baseline import build_model, count_parameters
    from src.training.trainer import build_criterion, evaluate, fit

    device = resolve_device(args.device)
    print(f"[device] {device}")

    model_cfg = config["model"]
    model = build_model(
        model_cfg["architecture"],
        num_classes=int(config["data"]["num_classes"]),
        pretrained=bool(model_cfg.get("pretrained")),
        finetune=model_cfg.get("finetune", "feature_extraction"),
    )
    total, trainable = count_parameters(model)
    print(f"[model] {model_cfg['architecture']} | param total {total:,} | trainable {trainable:,}")

    train_loader = build_dataloader("train", config)
    val_loader = build_dataloader("val", config)
    test_loader = build_dataloader("test", config)

    ckpt = Path(config.get("checkpoint", {}).get("dir", "checkpoints")) / name / "best.pt"
    print(f"[train] mulai; checkpoint terbaik -> {ckpt}")
    result = fit(model, train_loader, val_loader, config, device, ckpt)

    # Evaluasi TEST SEKALI memakai checkpoint terbaik (dipilih dari val).
    print("[test] memuat checkpoint terbaik untuk evaluasi akhir ...")
    import torch

    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    criterion = build_criterion(config, device)
    _, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    test_metrics = compute_metrics(y_true, y_pred, int(config["data"]["num_classes"]))
    print(f"[test] macro-F1 {test_metrics['macro_f1']:.4f}")

    exp_dir = Path(config.get("output", {}).get("experiments_dir", "experiments")) / name
    write_outputs(exp_dir, config, result, test_metrics, y_true, y_pred)
    print(f"[done] artefak ditulis ke {exp_dir}")


if __name__ == "__main__":
    main()
