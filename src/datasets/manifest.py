"""Pembacaan manifes CSV dan resolusi path citra untuk pelatihan.

Modul ini sengaja bebas dari PyTorch supaya logika resolusi path & label bisa
diuji tanpa framework. Dua skema manifes dipakai di proyek ini:

Train seimbang (`combined_balanced_train_manifest.csv`):
    kolom: class, filename, kind, split, rel_path
    contoh: 1_cumulus, 1_cumulus_000002.jpg, original, train,
            train/1_cumulus/1_cumulus_000002.jpg

Val/test (`combined_{val,test}_manifest.csv`):
    kolom: dataset, split, class, src_class, rel_path, width, height, pixel_sha256
    contoh val: GCD, val, 1_cumulus, 1_cumulus,
                train/1_cumulus/1_cumulus_000001.jpg, 512, 512, <sha>

PENTING: pada manifes val/test, `rel_path` mencatat lokasi SUMBER asli (val
berasal dari split train GCD sehingga prefiksnya `train/`), BUKAN lokasi fisik
file yang sudah dimaterialisasi. File fisik selalu berpola
`<split_dir>/<class>/<basename>` (lihat scripts/materialize_combined.py), jadi
path diselesaikan dari `class` + basename `rel_path`, bukan dari `rel_path` utuh.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import NamedTuple

# Nama direktori kelas berpola "<indeks1>_<nama>"; indeks label = angka - 1.
# Urutan ini konsisten dengan configs/base.yaml class_names [Cu, Ac, Ci, Cl, Sc, Cb, Mi].


class Sample(NamedTuple):
    """Satu entri citra: path ke file dan indeks label 0-based."""

    path: Path
    label: int
    class_name: str


def label_from_class_dir(class_dir: str) -> int:
    """Petakan nama direktori kelas (mis. "1_cumulus") ke indeks label 0-based.

    Prefiks numerik 1..7 pada nama kelas GCD menentukan urutan; label = n - 1.

    Raises:
        ValueError: nama kelas tidak diawali prefiks numerik.
    """
    prefix = class_dir.split("_", 1)[0]
    if not prefix.isdigit():
        raise ValueError(
            f"nama kelas {class_dir!r} tidak diawali indeks numerik (mis. '1_cumulus')."
        )
    return int(prefix) - 1


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    """Baca manifes CSV menjadi list of dict (urutan baris dipertahankan)."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def resolve_sample(row: dict[str, str], split_dir: str | Path) -> Sample:
    """Bangun `Sample` dari satu baris manifes dan direktori split fisiknya.

    Path fisik = `split_dir / class / basename(rel_path)`. Kolom `filename`
    dipakai bila ada (manifes train), selain itu basename `rel_path`.
    """
    class_name = row["class"]
    filename = row.get("filename") or Path(row["rel_path"]).name
    path = Path(split_dir) / class_name / filename
    return Sample(path=path, label=label_from_class_dir(class_name), class_name=class_name)


def load_samples(manifest: str | Path, split_dir: str | Path) -> list[Sample]:
    """Muat seluruh baris manifes menjadi list `Sample`.

    Tidak memeriksa keberadaan file di disk (agar cepat & mudah diuji); runner
    pelatihan memakai `missing_files` bila verifikasi disk dibutuhkan.
    """
    return [resolve_sample(row, split_dir) for row in read_manifest(manifest)]


def missing_files(samples: list[Sample]) -> list[Path]:
    """Kembalikan path yang tidak ada di disk (untuk pemeriksaan pra-pelatihan)."""
    return [s.path for s in samples if not s.path.exists()]
