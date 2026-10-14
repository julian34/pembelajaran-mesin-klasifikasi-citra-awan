"""Audit dataset citra tanpa mengubah data.

Skrip ini hanya MEMBACA berkas di folder dataset. Hasil ditulis ke reports/.

Contoh:
    python scripts/audit_dataset.py --root data/raw/GCD --name gcd_raw --zip data/raw/GCD.zip
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import zipfile
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
HASH_SIZE = 16  # dHash 16x16 = 256 bit
NEAR_DUP_MAX_DISTANCE = 10  # dari 256 bit
FIELDS = [
    "split", "class_dir", "filename", "rel_path", "bytes", "sha256", "crc32",
    "readable", "error", "format", "mode", "channels", "width", "height",
    "has_exif_orientation", "pixel_sha256", "dhash",
]


def inspect(args: tuple[str, str]) -> dict:
    root, rel = args
    path = Path(root) / rel
    parts = Path(rel).parts
    row = dict.fromkeys(FIELDS, "")
    row.update(
        split=parts[0] if len(parts) >= 3 else "",
        class_dir=parts[-2] if len(parts) >= 2 else "",
        filename=parts[-1],
        rel_path=Path(rel).as_posix(),
    )
    data = path.read_bytes()
    row["bytes"] = len(data)
    row["sha256"] = hashlib.sha256(data).hexdigest()
    row["crc32"] = zlib.crc32(data) & 0xFFFFFFFF
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im.load()  # decoding penuh, mendeteksi berkas terpotong
            row["format"] = im.format
            row["mode"] = im.mode
            row["channels"] = len(im.getbands())
            row["width"], row["height"] = im.size
            row["has_exif_orientation"] = int(0x0112 in im.getexif())
            header = f"{im.mode}|{im.size[0]}x{im.size[1]}|".encode()
            row["pixel_sha256"] = hashlib.sha256(header + im.tobytes()).hexdigest()
            gray = im.convert("L").resize((HASH_SIZE + 1, HASH_SIZE), Image.BILINEAR)
            arr = np.asarray(gray, dtype=np.int16)
            bits = (arr[:, 1:] > arr[:, :-1]).flatten()
            row["dhash"] = np.packbits(bits).tobytes().hex()
        row["readable"] = 1
    except Exception as exc:  # berkas rusak dicatat, bukan diperbaiki
        row["readable"] = 0
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def groups(rows: list[dict], key: str) -> list[list[dict]]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r[key]:
            buckets[r[key]].append(r)
    return [g for g in buckets.values() if len(g) > 1]


def describe_groups(found: list[list[dict]]) -> dict:
    cross_split = [g for g in found if len({r["split"] for r in g}) > 1]
    cross_class = [g for g in found if len({r["class_dir"] for r in g}) > 1]
    return {
        "groups": len(found),
        "files_in_groups": sum(len(g) for g in found),
        "redundant_files": sum(len(g) - 1 for g in found),
        "groups_across_splits": len(cross_split),
        "groups_across_classes": len(cross_class),
    }


def near_duplicates(rows: list[dict], split_a: str, split_b: str) -> list[dict]:
    """Pasangan lintas split dengan jarak Hamming dHash kecil (kandidat, bukan bukti)."""
    a = [r for r in rows if r["split"] == split_a and r["dhash"]]
    b = [r for r in rows if r["split"] == split_b and r["dhash"]]
    if not a or not b:
        return []
    popcount = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)
    ha = np.array([list(bytes.fromhex(r["dhash"])) for r in a], dtype=np.uint8)
    hb = np.array([list(bytes.fromhex(r["dhash"])) for r in b], dtype=np.uint8)
    pairs = []
    for start in range(0, len(ha), 64):
        chunk = ha[start:start + 64]
        dist = popcount[chunk[:, None, :] ^ hb[None, :, :]].sum(axis=2)
        for i, j in zip(*np.nonzero(dist <= NEAR_DUP_MAX_DISTANCE)):
            ra, rb = a[start + i], b[j]
            pairs.append({
                "distance": int(dist[i, j]),
                "file_a": ra["rel_path"], "class_a": ra["class_dir"],
                "file_b": rb["rel_path"], "class_b": rb["class_dir"],
                "same_file_bytes": int(ra["sha256"] == rb["sha256"]),
                "same_pixels": int(ra["pixel_sha256"] == rb["pixel_sha256"]),
            })
    return sorted(pairs, key=lambda p: (p["distance"], p["file_a"]))


def check_zip(zip_path: Path, root: Path, rows: list[dict]) -> dict:
    """Bandingkan isi arsip dengan berkas terekstrak lewat CRC32, tanpa mengekstrak."""
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
    by_suffix = {r["rel_path"]: r for r in rows}
    prefix = root.name + "/"
    matched = mismatched = 0
    missing, non_image = [], []
    seen = set()
    for m in members:
        name = m.filename[len(prefix):] if m.filename.startswith(prefix) else m.filename
        if Path(name).suffix.lower() not in IMAGE_EXTS:
            non_image.append(m.filename)
            continue
        seen.add(name)
        row = by_suffix.get(name)
        if row is None:
            missing.append(m.filename)
        elif int(row["crc32"]) == m.CRC and int(row["bytes"]) == m.file_size:
            matched += 1
        else:
            mismatched += 1
    extra = sorted(set(by_suffix) - seen)
    return {
        "zip": zip_path.as_posix(),
        "zip_bytes": zip_path.stat().st_size,
        "zip_sha256": hashlib.sha256(zip_path.read_bytes()).hexdigest(),
        "image_members": len(seen),
        "non_image_members": non_image[:20],
        "matched_crc_and_size": matched,
        "mismatched": mismatched,
        "in_zip_not_extracted": len(missing),
        "in_zip_not_extracted_examples": missing[:10],
        "extracted_not_in_zip": len(extra),
        "extracted_not_in_zip_examples": extra[:10],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path, help="folder dataset: <root>/<split>/<kelas>/<citra>")
    ap.add_argument("--name", required=True, help="awalan nama berkas keluaran")
    ap.add_argument("--zip", type=Path, default=None, help="arsip sumber untuk dicocokkan (opsional)")
    ap.add_argument("--out", type=Path, default=Path("reports"))
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()

    all_files = sorted(p for p in a.root.rglob("*") if p.is_file())
    images = [p for p in all_files if p.suffix.lower() in IMAGE_EXTS]
    others = [p.relative_to(a.root).as_posix() for p in all_files if p.suffix.lower() not in IMAGE_EXTS]
    jobs = [(str(a.root), str(p.relative_to(a.root))) for p in images]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(inspect, jobs, chunksize=200))

    tables, metrics = a.out / "tables", a.out / "metrics"
    tables.mkdir(parents=True, exist_ok=True)
    metrics.mkdir(parents=True, exist_ok=True)
    with open(tables / f"{a.name}_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    splits = sorted({r["split"] for r in rows})
    classes = sorted({r["class_dir"] for r in rows})
    counts = Counter((r["split"], r["class_dir"]) for r in rows)
    with open(tables / f"{a.name}_class_distribution.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", *splits, "total"])
        for c in classes:
            per = [counts[(s, c)] for s in splits]
            w.writerow([c, *per, sum(per)])
        totals = [sum(counts[(s, c)] for c in classes) for s in splits]
        w.writerow(["TOTAL", *totals, sum(totals)])

    ok = [r for r in rows if r["readable"] == 1]
    byte_dups, pixel_dups = groups(rows, "sha256"), groups(ok, "pixel_sha256")
    with open(tables / f"{a.name}_duplicates.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["kind", "group", "split", "class_dir", "rel_path"])
        for kind, found in (("identical_bytes", byte_dups), ("identical_pixels", pixel_dups)):
            for i, g in enumerate(found, 1):
                for r in g:
                    w.writerow([kind, i, r["split"], r["class_dir"], r["rel_path"]])

    near = []
    for i, sa in enumerate(splits):
        for sb in splits[i + 1:]:
            near += near_duplicates(ok, sa, sb)
    if near:
        with open(tables / f"{a.name}_near_duplicates_across_splits.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(near[0]))
            w.writeheader()
            w.writerows(near)

    # Kesesuaian nama berkas dengan folder kelas dan penomoran antar split
    pat = re.compile(r"^(?P<prefix>.+)_(?P<num>\d+)\.[A-Za-z]+$")
    prefix_mismatch, unparsed = [], []
    numbers: dict[tuple[str, str], list[int]] = defaultdict(list)
    for r in rows:
        m = pat.match(r["filename"])
        if not m:
            unparsed.append(r["rel_path"])
        elif m["prefix"] != r["class_dir"]:
            prefix_mismatch.append(r["rel_path"])
        else:
            numbers[(r["split"], r["class_dir"])].append(int(m["num"]))
    numbering = {
        f"{s}/{c}": {"min": min(v), "max": max(v), "count": len(v), "gaps": max(v) - min(v) + 1 - len(set(v))}
        for (s, c), v in sorted(numbers.items())
    }
    names = Counter(r["filename"] for r in rows)

    def dist(key):
        return {str(k): v for k, v in Counter(r[key] for r in ok).most_common()}

    per_split = {s: {c: counts[(s, c)] for c in classes} for s in splits}
    imbalance = {
        s: round(max(v.values()) / min(x for x in v.values() if x), 2) if any(v.values()) else None
        for s, v in per_split.items()
    }
    sizes = [int(r["bytes"]) for r in rows]
    summary = {
        "root": a.root.as_posix(),
        "images": len(rows),
        "non_image_files": others[:50],
        "total_bytes": sum(sizes),
        "splits": splits,
        "classes": classes,
        "class_dirs_per_split": {s: sorted({r["class_dir"] for r in rows if r["split"] == s}) for s in splits},
        "counts": per_split,
        "split_totals": {s: sum(v.values()) for s, v in per_split.items()},
        "imbalance_ratio_max_over_min": imbalance,
        "unreadable": [{"file": r["rel_path"], "error": r["error"]} for r in rows if r["readable"] != 1],
        "format": dist("format"),
        "mode": dist("mode"),
        "channels": dist("channels"),
        "resolution": {k: v for k, v in Counter(f"{r['width']}x{r['height']}" for r in ok).most_common(20)},
        "extension": {k: v for k, v in Counter(Path(r["filename"]).suffix for r in rows).most_common()},
        "with_exif_orientation": sum(int(r["has_exif_orientation"] or 0) for r in ok),
        "file_bytes": {"min": min(sizes), "median": int(np.median(sizes)), "max": max(sizes)} if sizes else {},
        "zero_byte_files": [r["rel_path"] for r in rows if int(r["bytes"]) == 0],
        "duplicates_identical_bytes": describe_groups(byte_dups),
        "duplicates_identical_pixels": describe_groups(pixel_dups),
        "near_duplicates_across_splits": {
            "method": f"dHash {HASH_SIZE}x{HASH_SIZE}, jarak Hamming <= {NEAR_DUP_MAX_DISTANCE} dari {HASH_SIZE * HASH_SIZE} bit",
            "pairs": len(near),
            "pairs_distance_0": sum(p["distance"] == 0 for p in near),
            "pairs_same_class": sum(p["class_a"] == p["class_b"] for p in near),
            "pairs_by_class_a": dict(Counter(p["class_a"] for p in near).most_common()),
            "distinct_files_involved": {
                "a": len({p["file_a"] for p in near}), "b": len({p["file_b"] for p in near}),
            },
        },
        "filename_prefix_mismatch": prefix_mismatch[:50],
        "filename_unparsed": unparsed[:50],
        "filenames_reused": sum(1 for v in names.values() if v > 1),
        "numbering": numbering,
    }
    if a.zip:
        summary["zip_check"] = check_zip(a.zip, a.root, rows)
    with open(metrics / f"{a.name}_audit_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
