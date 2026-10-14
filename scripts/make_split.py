"""Bagi train resmi menjadi train + validation secara stratified berseed.

Skrip ini MEMBACA manifest train bersih (hasil penanganan leakage & duplikat) dan
daftar pasangan yang harus tetap satu sisi (duplikat/near-duplicate DI DALAM
train), lalu membuat validation dari train tanpa memecah pasangan tersebut
(mencegah leakage train<->val baru). Test resmi tidak disentuh; hanya manifest
split yang ditulis ke `reports/tables/`. Berkas di `data/raw/` tidak diubah.

Validation diambil sebesar `--val-fraction` dari train bersih (default 0.15),
stratified per kelas, dengan `--seed` tetap (default 42).

Contoh:
    python scripts/make_split.py --name gcd_raw
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def grouped_stratified_split(rows: list[dict], edges: list[tuple[str, str]],
                             val_fraction: float, seed: int) -> set[str]:
    """Kembalikan himpunan rel_path untuk validation.

    `rows`: daftar dict minimal berisi `rel_path` dan `class_dir` (train bersih).
    `edges`: pasangan rel_path yang harus berada di sisi yang sama (grup near-dup).
    Pembagian stratified per kelas; tiap grup diperlakukan sebagai satu unit utuh.
    Deterministik untuk (rows, edges, val_fraction, seed) yang sama."""
    parent = {r["rel_path"]: r["rel_path"] for r in rows}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    present = set(parent)
    for a, b in edges:
        if a in present and b in present:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

    units: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        units[find(r["rel_path"])].append(r)

    def unit_class(members: list[dict]) -> str:
        counts = Counter(m["class_dir"] for m in members)
        top = max(counts.values())
        return sorted(c for c, n in counts.items() if n == top)[0]

    by_class: dict[str, list[list[dict]]] = defaultdict(list)
    for members in units.values():
        by_class[unit_class(members)].append(members)

    rng = random.Random(seed)
    val: set[str] = set()
    for cls in sorted(by_class):
        group_units = sorted(by_class[cls], key=lambda u: min(m["rel_path"] for m in u))
        rng.shuffle(group_units)
        target = round(val_fraction * sum(len(u) for u in group_units))
        taken = 0
        for u in group_units:
            if taken >= target:
                break
            val.update(m["rel_path"] for m in u)
            taken += len(u)
    return val


def read_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def within_train_edges(near_dup_path: Path) -> list[tuple[str, str]]:
    if not near_dup_path.exists():
        return []
    edges = []
    for r in read_rows(near_dup_path):
        if r.get("location") == "dalam train":
            edges.append((r["file_a"], r["file_b"]))
    return edges


def write_rows(path: Path, fields: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="awalan nama berkas (mis. gcd_raw)")
    ap.add_argument("--val-fraction", type=float, default=0.15, help="bagian validation dari train bersih")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tables", type=Path, default=Path("reports/tables"))
    ap.add_argument("--metrics", type=Path, default=Path("reports/metrics"))
    a = ap.parse_args()

    clean = read_rows(a.tables / f"{a.name}_train_clean_manifest.csv")
    for r in clean:
        r["split"] = "train"
    edges = within_train_edges(a.tables / f"{a.name}_eda_near_duplicate_candidates.csv")
    val_paths = grouped_stratified_split(clean, edges, a.val_fraction, a.seed)

    train_rows = [r for r in clean if r["rel_path"] not in val_paths]
    val_rows = [dict(r, split="val") for r in clean if r["rel_path"] in val_paths]

    manifest = read_rows(a.tables / f"{a.name}_manifest.csv")
    test_rows = [r for r in manifest if str(r.get("split", "")).lower() in {"test", "testing"} and r.get("readable") == "1"]

    fields = list(clean[0]) if clean else []
    write_rows(a.tables / f"{a.name}_split_train_manifest.csv", fields, train_rows)
    write_rows(a.tables / f"{a.name}_split_val_manifest.csv", list(val_rows[0]) if val_rows else fields, val_rows)
    if test_rows:
        write_rows(a.tables / f"{a.name}_split_test_manifest.csv", list(test_rows[0]), test_rows)

    # Verifikasi
    train_set, val_set = {r["rel_path"] for r in train_rows}, {r["rel_path"] for r in val_rows}
    assert train_set.isdisjoint(val_set), "train dan val beririsan."
    assert train_set | val_set == {r["rel_path"] for r in clean}, "train+val tidak sama dengan train bersih."
    split_by_group_ok = all(
        not ((a_ in train_set and b_ in val_set) or (a_ in val_set and b_ in train_set))
        for a_, b_ in edges if a_ in (train_set | val_set) and b_ in (train_set | val_set)
    )
    assert split_by_group_ok, "ada pasangan near-duplicate dalam train yang terpisah train/val."

    per_class = {}
    for cls in sorted({r["class_dir"] for r in clean}):
        nt = sum(r["class_dir"] == cls for r in train_rows)
        nv = sum(r["class_dir"] == cls for r in val_rows)
        per_class[cls] = {"train": nt, "val": nv, "val_pct": round(100 * nv / (nt + nv), 2)}

    summary = {
        "seed": a.seed,
        "val_fraction_of_clean_train": a.val_fraction,
        "stratified": True,
        "grouped_keep_together_edges": len(edges),
        "counts": {"clean_train": len(clean), "train": len(train_rows), "val": len(val_rows), "test": len(test_rows)},
        "per_class": per_class,
        "checks": {"train_val_disjoint": True, "union_equals_clean_train": True,
                   "near_dup_pairs_not_split": split_by_group_ok, "test_untouched": True},
        "note": "test resmi tidak dibagi ulang; data/raw/ tidak diubah.",
    }
    a.metrics.mkdir(parents=True, exist_ok=True)
    with open(a.metrics / f"{a.name}_split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
