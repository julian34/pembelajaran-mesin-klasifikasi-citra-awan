"""Loop pelatihan & evaluasi baseline (transfer learning).

Memilih checkpoint berdasarkan `checkpoint.monitor` (default val_macro_f1, mode
max; test TIDAK pernah dipakai untuk memilih model). Mendukung optimizer
adam/adamw/sgd dari config. Untuk train yang sudah seimbang
(`training.imbalance: none`) tidak ada class weights (hindari koreksi ganda).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_metrics


def build_criterion(config: dict, device: str) -> nn.Module:
    """Bangun fungsi rugi. imbalance 'none' -> CrossEntropy tanpa bobot.

    Mode 'class_weights' memuat bobot dari `training.class_weights_file` (kolom
    `weight_inverse_freq`, urut indeks kelas). Untuk exp_006 train sudah seimbang
    sehingga mode 'none' yang dipakai.
    """
    training = config.get("training", {})
    mode = training.get("imbalance", "none")
    weight = None
    if mode == "class_weights":
        import csv

        path = training["class_weights_file"]
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        weights = [float(r["weight_inverse_freq"]) for r in rows]
        weight = torch.tensor(weights, dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=weight)


def build_optimizer(config: dict, model: nn.Module) -> torch.optim.Optimizer:
    """Bangun optimizer dari config (hanya parameter trainable yang dioptimasi)."""
    training = config.get("training", {})
    name = str(training.get("optimizer", "adam")).lower()
    lr = float(training.get("learning_rate", 1e-3))
    weight_decay = float(training.get("weight_decay", 0.0))
    params = [p for p in model.parameters() if p.requires_grad]
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"optimizer={name!r} tidak dikenal; pilih adam, adamw, atau sgd.")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    on_batch=None,
) -> float:
    """Latih satu epoch; kembalikan rata-rata rugi (ditimbang jumlah sampel).

    Args:
        on_batch: callable opsional `(step: int, loss: float)` dipanggil tiap
            batch — dipakai untuk memperbarui progress bar (mis. pkbar).
    """
    model.train()
    total_loss, total_n = 0.0, 0
    for step, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        batch_loss = loss.item()
        total_loss += batch_loss * labels.size(0)
        total_n += labels.size(0)
        if on_batch is not None:
            on_batch(step, batch_loss)
    return total_loss / max(total_n, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, list[int], list[int]]:
    """Evaluasi; kembalikan (rata-rata rugi, y_true, y_pred)."""
    model.eval()
    total_loss, total_n = 0.0, 0
    y_true: list[int] = []
    y_pred: list[int] = []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        total_n += labels.size(0)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().tolist())
    return total_loss / max(total_n, 1), y_true, y_pred


@dataclass
class EpochRecord:
    """Satu baris riwayat pelatihan (-> training_history.csv)."""

    epoch: int
    train_loss: float
    val_loss: float
    val_macro_f1: float
    val_balanced_accuracy: float
    val_accuracy: float


@dataclass
class FitResult:
    """Hasil akhir pelatihan."""

    history: list[EpochRecord] = field(default_factory=list)
    best_epoch: int = -1
    best_metric: float = float("-inf")
    best_checkpoint: Path | None = None


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    device: str,
    checkpoint_path: str | Path,
    on_epoch_end=None,
    progress: bool = False,
) -> FitResult:
    """Latih model; simpan checkpoint terbaik menurut `checkpoint.monitor`.

    Hanya metrik validasi yang memilih checkpoint. Mengembalikan FitResult berisi
    riwayat per epoch dan info checkpoint terbaik.

    Args:
        on_epoch_end: callable opsional dipanggil setelah tiap epoch dengan
            `(record: EpochRecord, result: FitResult)` — dipakai notebook untuk
            menampilkan tabel progres langsung.
        progress: bila True, tampilkan progress bar per-batch gaya Keras (pkbar)
            yang diperbarui tiap batch dan diakhiri metrik validasi — pengganti
            print per-epoch. Butuh paket `pkbar`; jika tak ada, otomatis fallback
            ke print biasa.
    """
    num_classes = int(config["data"]["num_classes"])
    epochs = int(config.get("training", {}).get("epochs", 50))
    monitor = config.get("checkpoint", {}).get("monitor", "val_macro_f1")
    metric_key = monitor.removeprefix("val_")  # "val_macro_f1" -> "macro_f1"

    kbar_mod = None
    if progress:
        try:
            import pkbar as kbar_mod
        except ImportError:
            print("pkbar tidak terpasang; memakai print per-epoch. Pasang: pip install pkbar")

    model.to(device)
    criterion = build_criterion(config, device)
    optimizer = build_optimizer(config, model)

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    result = FitResult()
    n_train_batches = len(train_loader)

    for epoch in range(1, epochs + 1):
        bar = None
        on_batch = None
        if kbar_mod is not None:
            bar = kbar_mod.Kbar(
                target=n_train_batches, epoch=epoch - 1, num_epochs=epochs,
                width=16, always_stateful=False,
            )
            on_batch = lambda step, loss, _b=bar: _b.update(step, values=[("loss", loss)])

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, on_batch=on_batch
        )
        val_loss, y_true, y_pred = evaluate(model, val_loader, criterion, device)
        metrics = compute_metrics(y_true, y_pred, num_classes)

        result.history.append(
            EpochRecord(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_macro_f1=metrics["macro_f1"],
                val_balanced_accuracy=metrics["balanced_accuracy"],
                val_accuracy=metrics["accuracy"],
            )
        )
        current = metrics[metric_key]
        is_best = current > result.best_metric
        if bar is not None:
            bar.add(
                1,
                values=[
                    ("train_loss", train_loss),
                    ("val_loss", val_loss),
                    ("val_macro_f1", metrics["macro_f1"]),
                    ("val_bal_acc", metrics["balanced_accuracy"]),
                    ("val_acc", metrics["accuracy"]),
                ],
            )
            if is_best:
                print(f"   ^ best sejauh ini (epoch {epoch}, val_macro_f1 {current:.4f})")
        else:
            print(
                f"epoch {epoch:3d}/{epochs} | train_loss {train_loss:.4f} | "
                f"val_loss {val_loss:.4f} | macro_f1 {metrics['macro_f1']:.4f} | "
                f"bal_acc {metrics['balanced_accuracy']:.4f} | acc {metrics['accuracy']:.4f}"
                + ("  <- best" if is_best else "")
            )

        if is_best:
            result.best_metric = current
            result.best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "monitor": monitor,
                    "metric": current,
                    "config": config,
                },
                checkpoint_path,
            )
            result.best_checkpoint = checkpoint_path

        if on_epoch_end is not None:
            on_epoch_end(result.history[-1], result)

    return result
