"""Bangun daftar pengecualian (exclusion list) dari citra identik.

Skrip ini MEMBACA manifest hasil audit (`*_manifest.csv`) dan menuliskan daftar
citra yang harus DIKELUARKAN saat preprocessing, untuk dua kasus:

1. Cross-split leakage (`find_exclusions`): citra yang sama ada di lebih dari satu
   split. Citra dipertahankan pada split berprioritas tertinggi dan dikeluarkan
   dari split berprioritas lebih rendah. Prioritas default: test > val > train
   (test resmi utuh; validation diutamakan atas train).
2. Duplikat dalam-split (`find_within_split_duplicates`): citra yang sama muncul
   lebih dari sekali di DALAM satu split. Pada split yang dideduplikasi (train,
   val) satu representatif dipertahankan dan salinannya dikeluarkan; pada split
   lain (test) duplikat hanya dicatat, tidak dikeluarkan (test resmi utuh).

Dua citra dianggap "sama" bila hash pikselnya (`pixel_sha256`) identik. Berkas di
`data/raw/` tidak disentuh; penanganan dilakukan di tahap preprocessing dengan
membaca daftar ini.

Contoh:
    python scripts/build_leakage_exclusion.py --name gcd_raw
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

# Split berprioritas lebih tinggi dipertahankan; duplikatnya di split lain dikeluarkan.
SPLIT_PRIORITY = ["test", "val", "validation", "train"]
# Split yang dideduplikasi saat ada duplikat DI DALAM split; lainnya hanya dicatat.
DEDUP_SPLITS = ("train", "val")
SPLIT_ALIASES = {"training": "train", "valid": "val", "validation": "val", "testing": "test"}


def read_manifest(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["split"] = SPLIT_ALIASES.get(r["split"].lower(), r["split"].lower())
    return [r for r in rows if r.get("readable") == "1"]


def priority(split: str) -> int:
    return SPLIT_PRIORITY.index(split) if split in SPLIT_PRIORITY else len(SPLIT_PRIORITY)


def find_exclusions(rows: list[dict]) -> list[dict]:
    """Untuk tiap grup citra identik yang menjangkau >1 split, keluarkan salinan
    dari split berprioritas lebih rendah. Piksel identik dipakai sebagai 'citra
    yang sama'; byte identik dicatat sebagai keterangan tambahan."""
    by_pixel: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("pixel_sha256"):
            by_pixel[r["pixel_sha256"]].append(r)

    exclusions = []
    for digest, group in by_pixel.items():
        splits = {r["split"] for r in group}
        if len(splits) < 2:
            continue  # bukan lintas split
        keep_split = min(splits, key=priority)
        kept = [r for r in group if r["split"] == keep_split]
        keeper = sorted(kept, key=lambda r: r["rel_path"])[0]
        for r in group:
            if r["split"] == keep_split:
                continue
            same_bytes = int(r.get("sha256", "") == keeper.get("sha256", ""))
            exclusions.append({
                "rel_path": r["rel_path"],
                "split": r["split"],
                "class_dir": r["class_dir"],
                "reason": "cross_split_leakage",
                "match_type": "identical_bytes" if same_bytes else "identical_pixels",
                "keep_split": keep_split,
                "keep_rel_path": keeper["rel_path"],
                "pixel_sha256": digest,
            })
    return sorted(exclusions, key=lambda e: (e["split"], e["rel_path"]))


def find_within_split_duplicates(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Duplikat citra identik DI DALAM satu split.

    Mengembalikan (exclusions, kept): `exclusions` adalah salinan yang dikeluarkan
    dari split yang dideduplikasi (DEDUP_SPLITS), `kept` adalah duplikat pada split
    lain (mis. test) yang hanya dicatat. Untuk tiap grup, representatif = rel_path
    terkecil secara leksikografis; sisanya salinan."""
    by_pixel: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("pixel_sha256"):
            by_pixel[r["pixel_sha256"]].append(r)

    exclusions, kept = [], []
    for digest, group in by_pixel.items():
        splits = {r["split"] for r in group}
        if len(splits) != 1 or len(group) < 2:
            continue  # bukan duplikat dalam-split (cross-split ditangani find_exclusions)
        split = next(iter(splits))
        members = sorted(group, key=lambda r: r["rel_path"])
        keeper = members[0]
        for r in members[1:]:
            rec = {
                "rel_path": r["rel_path"],
                "split": split,
                "class_dir": r["class_dir"],
                "reason": "within_split_duplicate",
                "keep_rel_path": keeper["rel_path"],
                "pixel_sha256": digest,
            }
            (exclusions if split in DEDUP_SPLITS else kept).append(rec)
    key = lambda e: (e["split"], e["rel_path"])
    return sorted(exclusions, key=key), sorted(kept, key=key)


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="awalan nama berkas (mis. gcd_raw)")
    ap.add_argument("--tables", type=Path, default=Path("reports/tables"))
    ap.add_argument("--metrics", type=Path, default=Path("reports/metrics"))
    a = ap.parse_args()

    manifest = a.tables / f"{a.name}_manifest.csv"
    rows = read_manifest(manifest)
    exclusions = find_exclusions(rows)
    within_excl, within_kept = find_within_split_duplicates(rows)

    leak_fields = ["rel_path", "split", "class_dir", "reason", "match_type",
                   "keep_split", "keep_rel_path", "pixel_sha256"]
    dup_fields = ["rel_path", "split", "class_dir", "reason", "keep_rel_path", "pixel_sha256"]
    leak_csv = a.tables / f"{a.name}_leakage_exclusion.csv"
    dup_csv = a.tables / f"{a.name}_within_split_duplicate_exclusion.csv"
    kept_csv = a.tables / f"{a.name}_test_duplicates.csv"
    _write_csv(leak_csv, leak_fields, exclusions)
    _write_csv(dup_csv, dup_fields, within_excl)
    _write_csv(kept_csv, dup_fields, within_kept)

    by_class = Counter((e["split"], e["class_dir"]) for e in exclusions)
    kept_test = {e["keep_rel_path"] for e in exclusions if e["keep_split"] == "test"}
    summary = {
        "manifest": manifest.as_posix(),
        "split_priority": SPLIT_PRIORITY,
        "dedup_splits": list(DEDUP_SPLITS),
        "cross_split_leakage": {
            "total_excluded": len(exclusions),
            "excluded_by_split": dict(Counter(e["split"] for e in exclusions)),
            "excluded_by_split_class": {f"{s}/{c}": n for (s, c), n in sorted(by_class.items())},
            "test_images_with_copy_elsewhere": len(kept_test),
        },
        "within_split_duplicates": {
            "excluded": len(within_excl),
            "excluded_by_split_class": {f"{s}/{c}": n for (s, c), n in sorted(
                Counter((e["split"], e["class_dir"]) for e in within_excl).items())},
            "recorded_not_excluded": len(within_kept),
            "recorded_by_split_class": {f"{s}/{c}": n for (s, c), n in sorted(
                Counter((e["split"], e["class_dir"]) for e in within_kept).items())},
        },
        "note": "data/raw/ tidak diubah; daftar ini dibaca saat preprocessing.",
    }
    out_json = a.metrics / f"{a.name}_exclusion_summary.json"
    a.metrics.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    for label, path in [("Leakage lintas split", leak_csv), ("Duplikat dalam-split (dikeluarkan)", dup_csv),
                        ("Duplikat test (dicatat)", kept_csv), ("Ringkasan", out_json)]:
        print(f"{label:34s}: {path.as_posix()}")


if __name__ == "__main__":
    main()
