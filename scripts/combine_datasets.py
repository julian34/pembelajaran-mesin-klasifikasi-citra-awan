"""Gabungkan dataset CCSN_v2 ke dalam skema 7 kelas GCD.

Memetakan 11 genera CCSN ke 7 kelas GCD, melakukan deduplikasi internal CCSN
(dengan menolak citra berlabel ambigu lintas-kelas), membuang citra CCSN yang
identik dengan citra GCD (hindari duplikasi/kebocoran), lalu membentuk split
gabungan: train/validation dari (GCD train bersih + CCSN) secara stratified
berseed, sedangkan TEST = test resmi GCD apa adanya (tidak ditambah CCSN).

Hanya menulis manifest ke reports/tables/. Berkas di data/raw/ tidak diubah.

Contoh:
    python scripts/combine_datasets.py
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # agar make_split dapat diimpor via CLI maupun import
from make_split import grouped_stratified_split

# Pemetaan genera CCSN (WMO) -> kelas GCD. Ct (contrails) -> 3_cirrus (keputusan proyek).
CCSN_TO_GCD = {
    "Cu": "1_cumulus",
    "Ac": "2_altocumulus", "Cc": "2_altocumulus",
    "Ci": "3_cirrus", "Cs": "3_cirrus", "Ct": "3_cirrus",
    "Sc": "5_stratocumulus", "St": "5_stratocumulus", "As": "5_stratocumulus",
    "Cb": "6_cumulonimbus", "Ns": "6_cumulonimbus",
}
VAL_FRACTION = 0.15
SEED = 42
KEEP_COLS = ["dataset", "split", "class", "src_class", "rel_path", "width", "height", "pixel_sha256"]


def dedup_ccsn(ccsn: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Deduplikasi internal CCSN berdasarkan pixel_sha256.

    - Grup yang semua anggotanya memetakan ke kelas GCD sama -> sisakan satu.
    - Grup dengan label (kelas GCD) berbeda -> buang semua (ambigu)."""
    keep_idx, dropped_dup, dropped_ambig = [], 0, 0
    for _, g in ccsn.groupby("pixel_sha256"):
        if g["class"].nunique() > 1:
            dropped_ambig += len(g)
            continue
        first = g.sort_values("rel_path").index[0]
        keep_idx.append(first)
        dropped_dup += len(g) - 1
    kept = ccsn.loc[keep_idx].copy()
    return kept, {"dropped_internal_duplicate": dropped_dup, "dropped_ambiguous_label": dropped_ambig}


def build_combined(T: Path, metrics: Path, val_fraction: float = VAL_FRACTION,
                   seed: int = SEED) -> tuple[dict, dict]:
    """Bangun split gabungan GCD+CCSN, tulis manifest/tabel ke T dan ringkasan ke
    metrics, lalu kembalikan (splits, summary). Dipakai oleh CLI dan notebook."""
    # --- GCD: train bersih (pool) dan test resmi ---
    gcd_train = pd.read_csv(T / "gcd_raw_train_clean_manifest.csv", dtype=str)
    gcd_full = pd.read_csv(T / "gcd_raw_manifest.csv", dtype=str)
    gcd_full = gcd_full[gcd_full["readable"] == "1"]
    gcd_test = gcd_full[gcd_full["split"].str.lower() == "test"].copy()

    for df in (gcd_train, gcd_test):
        df["dataset"] = "GCD"
        df["class"] = df["class_dir"]
        df["src_class"] = df["class_dir"]

    # --- CCSN: map label, dedup, buang yang identik dengan GCD ---
    ccsn_manifest = T / "ccsn_v2_raw_manifest.csv"       # keluaran notebooks/01 (konsisten dgn gcd_raw_*)
    ccsn = pd.read_csv(ccsn_manifest, dtype=str)
    ccsn = ccsn[ccsn["readable"] == "1"].copy()
    ccsn["dataset"] = "CCSN"
    ccsn["src_class"] = ccsn["class_dir"]
    ccsn["class"] = ccsn["class_dir"].map(CCSN_TO_GCD)
    unmapped = ccsn[ccsn["class"].isna()]
    assert unmapped.empty, f"Genera CCSN tak terpetakan: {sorted(unmapped['src_class'].unique())}"

    ccsn, dedup_stats = dedup_ccsn(ccsn)

    gcd_all_px = set(gcd_full["pixel_sha256"])
    gcd_test_px = set(gcd_test["pixel_sha256"])
    dup_with_gcd = ccsn["pixel_sha256"].isin(gcd_all_px)
    leak_with_test = int(ccsn["pixel_sha256"].isin(gcd_test_px).sum())
    ccsn_kept = ccsn[~dup_with_gcd].copy()

    # --- Pool train+val = GCD train bersih + CCSN kept ---
    pool = pd.concat([gcd_train, ccsn_kept], ignore_index=True)
    pool["uid"] = pool["dataset"] + "::" + pool["rel_path"]

    # Grup near-duplicate DALAM train GCD agar tidak terpecah train/val
    edges = []
    nd_path = T / "gcd_raw_eda_near_duplicate_candidates.csv"
    if nd_path.exists():
        nd = pd.read_csv(nd_path)
        for _, r in nd[nd["location"] == "dalam train"].iterrows():
            edges.append((f"GCD::{r['file_a']}", f"GCD::{r['file_b']}"))

    rows = [{"rel_path": u, "class_dir": c} for u, c in zip(pool["uid"], pool["class"])]
    val_uids = grouped_stratified_split(rows, edges, val_fraction, seed)
    pool["split"] = pool["uid"].map(lambda u: "val" if u in val_uids else "train")

    gcd_test["split"] = "test"
    splits = {
        "train": pool[pool["split"] == "train"],
        "val": pool[pool["split"] == "val"],
        "test": gcd_test,
    }
    for name, df in splits.items():
        df[KEEP_COLS].to_csv(T / f"combined_{name}_manifest.csv", index=False, encoding="utf-8")

    # --- Tabel pemetaan ---
    map_rows = [{"ccsn_genus": k, "gcd_class": v} for k, v in sorted(CCSN_TO_GCD.items())]
    pd.DataFrame(map_rows).to_csv(T / "combined_class_mapping.csv", index=False, encoding="utf-8")

    # --- Bobot kelas dari train gabungan ---
    classes = sorted(splits["train"]["class"].unique())
    counts = splits["train"]["class"].value_counts().reindex(classes)
    N, K = int(counts.sum()), len(classes)
    weights = (N / (K * counts)).round(4)
    cw = pd.DataFrame({"index": range(K), "class": classes, "n_train": counts.astype(int).to_numpy(),
                       "weight_inverse_freq": weights.to_numpy()})
    cw.to_csv(T / "combined_class_weights.csv", index=False, encoding="utf-8")

    # --- Verifikasi ---
    tr, va = set(splits["train"]["rel_path"]), set(splits["val"]["rel_path"])
    assert tr.isdisjoint(va)
    assert ccsn_kept["pixel_sha256"].isin(gcd_test_px).sum() == 0

    def per_src(df):
        return dict(Counter(df["dataset"]))

    summary = {
        "seed": seed, "val_fraction_of_pool": val_fraction,
        "mapping": CCSN_TO_GCD,
        "ccsn_raw": int(len(pd.read_csv(ccsn_manifest))),
        "ccsn_dedup": dedup_stats,
        "ccsn_duplicate_with_gcd": int(dup_with_gcd.sum()),
        "ccsn_leakage_with_gcd_test": leak_with_test,
        "ccsn_kept": int(len(ccsn_kept)),
        "counts": {k: len(v) for k, v in splits.items()},
        "source_per_split": {k: per_src(v) for k, v in splits.items()},
        "train_class_distribution": {c: int(n) for c, n in counts.items()},
        "note": "test = GCD resmi saja; CCSN hanya menambah train/val; data/raw/ tidak diubah.",
    }
    metrics.mkdir(parents=True, exist_ok=True)
    with open(metrics / "combined_split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return splits, summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", type=Path, default=Path("reports/tables"))
    ap.add_argument("--metrics", type=Path, default=Path("reports/metrics"))
    a = ap.parse_args()
    _, summary = build_combined(a.tables, a.metrics)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
