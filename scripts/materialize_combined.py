"""Salin split gabungan (GCD + CCSN_v2) ke data/combined/{train,val,test}/<kelas>/.

Membaca `combined_{train,val,test}_manifest.csv` lalu menyalin tiap citra dari
folder raw sesuai kolom `dataset` (GCD -> data/raw/GCD, CCSN -> data/raw/CCSN_v2).
Kelas memakai skema 7 kelas GCD (kolom `class`). Nama berkas dipertahankan apa
adanya (tidak ada tabrakan nama antar sumber dalam satu kelas). `data/raw/` tidak
diubah; `.gitkeep` dipertahankan.

Contoh:
    python scripts/materialize_combined.py --clean
"""
from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path

SPLITS = ["train", "val", "test"]
RAW = {"GCD": Path("data/raw/GCD"), "CCSN": Path("data/raw/CCSN_v2")}


def read_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", type=Path, default=Path("reports/tables"))
    ap.add_argument("--dest", type=Path, default=Path("data/combined"), help="akar tujuan")
    ap.add_argument("--clean", action="store_true", help="kosongkan folder tujuan dulu (selain .gitkeep)")
    a = ap.parse_args()

    total = 0
    for split in SPLITS:
        rows = read_rows(a.tables / f"combined_{split}_manifest.csv")
        dest_split = a.dest / split
        if a.clean and dest_split.exists():
            for p in dest_split.rglob("*"):
                if p.is_file() and p.name != ".gitkeep":
                    p.unlink()

        per_class: Counter = Counter()
        per_source: Counter = Counter()
        for r in rows:
            raw_root = RAW[r["dataset"]]
            src = raw_root / r["rel_path"]
            dst = dest_split / r["class"] / Path(r["rel_path"]).name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            per_class[r["class"]] += 1
            per_source[r["dataset"]] += 1

        n = sum(per_class.values())
        total += n
        print(f"{split:5s}: {n:,} citra -> {dest_split.as_posix()}/  (sumber: {dict(per_source)})")
        for cls in sorted(per_class):
            print(f"    {cls:16s} {per_class[cls]:>6,}")

    print(f"\nTotal disalin: {total:,} citra. Sumber data/raw/ tidak diubah.")


if __name__ == "__main__":
    main()
