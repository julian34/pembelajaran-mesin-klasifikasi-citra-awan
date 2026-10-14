"""Salin citra ke struktur folder split: data/{train,val,test}/<kelas>/.

Skrip ini MEMBACA manifest split (`*_split_{train,val,test}_manifest.csv`) dan
menyalin setiap citra dari `data/raw/GCD/` ke folder tujuan sesuai split dan
kelasnya. Sumber di `data/raw/` tidak diubah. Berkas `.gitkeep` dipertahankan.

Train/validation berasal dari train resmi bersih (hasil penanganan leakage &
duplikat); test adalah test resmi apa adanya. Tujuan ditentukan dari `class_dir`
dan nama berkas, bukan dari awalan split pada `rel_path`.

Contoh:
    python scripts/materialize_split.py --name gcd_raw
"""
from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path

SPLITS = ["train", "val", "test"]


def read_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="awalan nama manifest (mis. gcd_raw)")
    ap.add_argument("--raw", type=Path, default=Path("data/raw/GCD"), help="akar citra sumber")
    ap.add_argument("--dest", type=Path, default=Path("data"), help="akar tujuan (berisi train/val/test)")
    ap.add_argument("--tables", type=Path, default=Path("reports/tables"))
    ap.add_argument("--clean", action="store_true", help="kosongkan folder tujuan dulu (selain .gitkeep)")
    a = ap.parse_args()

    total = 0
    for split in SPLITS:
        rows = read_rows(a.tables / f"{a.name}_split_{split}_manifest.csv")
        dest_split = a.dest / split
        if a.clean and dest_split.exists():
            for p in dest_split.rglob("*"):
                if p.is_file() and p.name != ".gitkeep":
                    p.unlink()

        per_class = Counter()
        for r in rows:
            src = a.raw / r["rel_path"]
            dst = dest_split / r["class_dir"] / Path(r["rel_path"]).name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            per_class[r["class_dir"]] += 1

        n = sum(per_class.values())
        total += n
        print(f"{split:5s}: {n:,} citra -> {dest_split.as_posix()}/")
        for cls in sorted(per_class):
            print(f"    {cls:16s} {per_class[cls]:>5,}")

    print(f"\nTotal disalin: {total:,} citra. Sumber {a.raw.as_posix()}/ tidak diubah.")


if __name__ == "__main__":
    main()
