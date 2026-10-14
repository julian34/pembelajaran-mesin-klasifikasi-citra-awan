"""Dataset & DataLoader PyTorch untuk citra awan 7 kelas.

Membaca manifes via `src.datasets.manifest`, memuat citra RGB, dan menerapkan
transform. Normalisasi mengikuti `configs/base.yaml`: skala /255 ke [0, 1]
dengan mean 0 / std 1 (tanpa standardisasi ImageNet) — pilihan praproses eksplisit
proyek ini, lihat docs/BASELINE_ALEXNET.md Bagian 3.
"""
from __future__ import annotations

from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .manifest import Sample, load_samples

# Split -> (kunci manifes di config.data, kunci direktori di config.data).
_SPLIT_KEYS = {
    "train": ("train_manifest", "train_dir"),
    "val": ("val_manifest", "val_dir"),
    "test": ("test_manifest", "test_dir"),
}


class ManifestImageDataset(Dataset):
    """Dataset citra berbasis manifes CSV.

    Setiap item = (tensor citra CxHxW float32, label int). Transform menerima
    objek PIL.Image dan mengembalikan tensor.
    """

    def __init__(self, samples: list[Sample], transform: Any) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        with Image.open(sample.path) as im:
            image = im.convert("RGB")
            tensor = self.transform(image)
        return tensor, sample.label


def build_transforms(
    image_size: int,
    scale: float = 255.0,
    mean: list[float] | None = None,
    std: list[float] | None = None,
) -> Any:
    """Bangun transform evaluasi-deterministik: resize -> tensor [0,1] -> normalize.

    `torchvision.transforms.ToTensor` sudah membagi piksel dengan 255 (menghasilkan
    [0, 1]); karena itu `scale` lain dari 255 dikoreksi relatif terhadap ToTensor.
    Dengan mean=[0,0,0], std=[1,1,1] (default config) hasil akhir tetap [0, 1].
    """
    from torchvision import transforms

    mean = [0.0, 0.0, 0.0] if mean is None else mean
    std = [1.0, 1.0, 1.0] if std is None else std

    steps = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),  # HxWxC [0,255] uint8 -> CxHxW [0,1] float32
    ]
    # ToTensor membagi dengan 255. Bila config meminta skala berbeda, sesuaikan.
    if scale != 255.0:
        factor = 255.0 / float(scale)
        steps.append(transforms.Lambda(lambda t: t * factor))
    steps.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(steps)


def build_dataset(split: str, config: dict) -> ManifestImageDataset:
    """Bangun dataset satu split ("train"/"val"/"test") dari config gabungan."""
    if split not in _SPLIT_KEYS:
        raise ValueError(f"split={split!r} tidak dikenal; pilih dari {tuple(_SPLIT_KEYS)}.")
    manifest_key, dir_key = _SPLIT_KEYS[split]
    data = config["data"]
    if manifest_key not in data or dir_key not in data:
        raise KeyError(
            f"config.data harus memuat '{manifest_key}' dan '{dir_key}' untuk split {split!r}."
        )
    samples = load_samples(data[manifest_key], data[dir_key])

    pre = config.get("preprocess", {})
    norm = pre.get("normalize", {})
    transform = build_transforms(
        image_size=int(data.get("image_size", 512)),
        scale=float(norm.get("scale", 255)),
        mean=norm.get("mean"),
        std=norm.get("std"),
    )
    return ManifestImageDataset(samples, transform)


def build_dataloader(
    split: str, config: dict, *, shuffle: bool | None = None
) -> DataLoader:
    """Bangun DataLoader satu split. `shuffle` default True hanya untuk train."""
    dataset = build_dataset(split, config)
    training = config.get("training", {})
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(
        dataset,
        batch_size=int(training.get("batch_size", 32)),
        shuffle=shuffle,
        num_workers=int(training.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
