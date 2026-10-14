# Dokumentasi Split Dataset — GCD & Gabungan GCD+CCSN

Tanggal: 13 Oktober 2026 (Asia/Jakarta). Skema: 7 kelas GCD.

Dokumen ini merinci **dua set split** yang tersedia:

1. **GCD-only** (bagian 1–6) — hanya dataset GCD. **train 85% / validation 15% dari train resmi bersih**, test resmi utuh. Dibuat `notebooks/02_preprocessing.ipynb` (bagian 6) via `scripts/make_split.py`, lalu dimaterialisasi ke folder `data/{train,val,test}/` dengan `scripts/materialize_split.py`.
2. **Gabungan GCD+CCSN_v2** (bagian 7) — CCSN_v2 dipetakan ke 7 kelas GCD lalu ditambahkan ke train/validation; test tetap test resmi GCD. Dibuat `notebooks/02_preprocessing.ipynb` (bagian 8) via `scripts/combine_datasets.py`, lalu dimaterialisasi ke folder `data/combined/{train,val,test}/` dengan `scripts/materialize_combined.py`.

Konteks penanganan kualitas: [`DATASET_QUALITY_SUMMARY.md`](DATASET_QUALITY_SUMMARY.md); penggabungan: [`PENGGABUNGAN_DATASET.md`](PENGGABUNGAN_DATASET.md); deskripsi dataset: [`DATASET_DOCUMENTATION.md`](DATASET_DOCUMENTATION.md).

---

## 1. Ringkasan (split GCD-only)

Bagian 1–6 membahas split **GCD-only**. Split gabungan GCD+CCSN ada di **bagian 7**.

| Parameter | Nilai |
|---|---|
| Basis pembagian train/val | Train resmi **bersih** (9.811 citra) |
| Rasio | train **85%** / validation **15%** dari train bersih |
| Test | Test resmi, **tidak dibagi ulang** (9.000 citra) |
| Metode | Stratified per kelas, seed tetap |
| Seed | 42 (`configs/base.yaml` `seed`) |
| Stratifikasi | Ya (proporsi 7 kelas dijaga di train & val) |
| Grup near-duplicate dalam train | Dijaga satu sisi (tidak terpecah train/val) |
| Resolusi | 512 × 512 RGB, seragam (resize ke 224 saat training) |

**Hasil akhir:** train **8.340** · validation **1.471** · test **9.000** · total **18.811** citra.

---

## 2. Asal data (sebelum split)

Train & validation diambil dari **train resmi bersih**, yaitu train resmi (10.000) setelah penanganan kebocoran dan duplikat (lihat `DATASET_QUALITY_SUMMARY.md`):

| Tahap | Dibuang dari train | Sisa |
|---|---:|---:|
| Train resmi (data/raw/GCD/train) | — | 10.000 |
| − Leakage lintas split (piksel identik dengan test) | 156 | 9.844 |
| − Near-duplicate lintas split terkonfirmasi (dHash ≤ 2) | 32 | 9.812 |
| − Duplikat di dalam train | 1 | 9.811 |
| **Train bersih (basis split)** | **189** | **9.811** |

Test resmi (9.000) **tidak diubah sama sekali** — termasuk citra test yang salinannya dibuang dari train. Pembuangan selalu dari sisi train/val, tidak pernah dari test.

---

## 3. Komposisi per kelas

Jumlah citra per kelas di tiap split. `val%` dihitung terhadap train+val (yaitu train bersih per kelas).

| Kelas | Train | Validation | Test | Train+Val | val% |
|---|---:|---:|---:|---:|---:|
| 1_cumulus (Cu) | 659 | 116 | 750 | 775 | 15,0 |
| 2_altocumulus (Ac) | 617 | 108 | 750 | 725 | 14,9 |
| 3_cirrus (Ci) | 980 | 173 | 753 | 1.153 | 15,0 |
| 4_clearsky (Cl) | 1.669 | 294 | 1.589 | 1.963 | 15,0 |
| 5_stratocumulus (Sc) | 1.568 | 278 | 1.790 | 1.846 | 15,1 |
| 6_cumulonimbus (Cb) | 2.553 | 450 | 2.761 | 3.003 | 15,0 |
| 7_mixed (Mi) | 294 | 52 | 607 | 346 | 15,0 |
| **Total** | **8.340** | **1.471** | **9.000** | **9.811** | **15,0** |

Validation konsisten ≈15% di setiap kelas (14,9–15,1%), sehingga distribusi kelas train dan val sepadan.

---

## 4. Struktur folder hasil

Citra disalin dari `data/raw/GCD/` (sumber tetap utuh) ke struktur berikut; tiap kelas punya subfolder sendiri:

```
data/
├── train/   8.340 citra
│   ├── 1_cumulus/        659
│   ├── 2_altocumulus/    617
│   ├── 3_cirrus/         980
│   ├── 4_clearsky/     1.669
│   ├── 5_stratocumulus/ 1.568
│   ├── 6_cumulonimbus/ 2.553
│   └── 7_mixed/          294
├── val/     1.471 citra
│   └── <7 kelas yang sama>
└── test/    9.000 citra
    └── <7 kelas yang sama>
```

Nama berkas dipertahankan apa adanya (mis. `4_clearsky_000894.jpg`). Folder `data/` tidak dilacak Git (`.gitignore`). **Folder `data/{train,val,test}/` ini berisi split GCD-only**; split gabungan ada di `data/combined/{train,val,test}/` (bagian 7).

---

## 5. Aturan pembagian

1. **Stratified per kelas.** Proporsi ketujuh kelas di train dipertahankan di validation (≈15% tiap kelas).
2. **Seed tetap (42).** Pembagian deterministik dan dapat direproduksi.
3. **Grup near-duplicate dalam train tidak terpecah.** 3 pasangan citra sangat mirip di dalam train dipaksa berada di sisi yang sama (semua train atau semua val), supaya tidak muncul kebocoran train↔validation.
4. **Test tidak disentuh.** Test resmi tetap 9.000; tidak dibagi ulang dan tidak diseimbangkan.
5. **Tanpa kebocoran lintas split.** Tidak ada citra train/val yang identik (piksel) dengan test; near-duplicate lintas split yang terbukti sama sudah dibuang lebih dulu.

---

## 6. Verifikasi

Diperiksa otomatis (`reports/metrics/gcd_raw_split_summary.json` dan assert di notebook bagian 6):

- `train ∩ val = ∅` (tidak ada citra di dua sisi) — **OK**
- `train ∪ val = train bersih` (9.811) — **OK**
- 0 citra validation identik (piksel) dengan test — **OK**
- 0 pasangan near-duplicate dalam train yang terpecah train/val — **OK**
- Test tidak berubah (9.000) — **OK**

---

## 7. Split gabungan (GCD + CCSN_v2)

Dataset kedua **CCSN_v2** (2.543 citra, 11 genera WMO) dipetakan ke 7 kelas GCD (contrails Ct → `3_cirrus`), dideduplikasi, lalu **2.523 citra** ditambahkan ke train/validation. **Test tetap test resmi GCD (9.000), tidak ditambah.** Dibuat `notebooks/02_preprocessing.ipynb` bagian 8 / `scripts/combine_datasets.py`. Rincian & pemetaan: [`PENGGABUNGAN_DATASET.md`](PENGGABUNGAN_DATASET.md).

| Parameter | Nilai |
|---|---|
| Basis train/val | GCD train bersih (9.811) + CCSN dipakai (2.523) = **12.334** |
| Rasio | train 85% / validation 15%, stratified, seed 42 |
| Test | GCD resmi **9.000** (murni, 0 CCSN) |

**Hasil akhir:** train **10.483** · validation **1.851** · test **9.000** · total **21.334** citra.

### 7.1 Komposisi per kelas (skema GCD)

| Kelas | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| 1_cumulus (Cu) | 809 | 143 | 750 | 1.702 |
| 2_altocumulus (Ac) | 1.027 | 182 | 750 | 1.959 |
| 3_cirrus (Ci) | 1.511 | 267 | 753 | 2.531 |
| 4_clearsky (Cl) | 1.669 | 294 | 1.589 | 3.552 |
| 5_stratocumulus (Sc) | 2.187 | 386 | 1.790 | 4.363 |
| 6_cumulonimbus (Cb) | 2.986 | 527 | 2.761 | 6.274 |
| 7_mixed (Mi) | 294 | 52 | 607 | 953 |
| **Total** | **10.483** | **1.851** | **9.000** | **21.334** |

### 7.2 Rincian per kelas menurut sumber (GCD vs CCSN)

Pecahan tiap kelas di train dan validation menurut dataset asal (`dataset` pada manifest). Test seluruhnya GCD (lihat §7.1).

| Kelas | Train GCD | Train CCSN | Val GCD | Val CCSN | Train+Val GCD | Train+Val CCSN |
|---|---:|---:|---:|---:|---:|---:|
| 1_cumulus (Cu) | 650 | 159 | 125 | 18 | 775 | 177 |
| 2_altocumulus (Ac) | 626 | 401 | 99 | 83 | 725 | 484 |
| 3_cirrus (Ci) | 993 | 518 | 160 | 107 | 1.153 | 625 |
| 4_clearsky (Cl) | 1.669 | 0 | 294 | 0 | 1.963 | 0 |
| 5_stratocumulus (Sc) | 1.561 | 626 | 285 | 101 | 1.846 | 727 |
| 6_cumulonimbus (Cb) | 2.545 | 441 | 458 | 69 | 3.003 | 510 |
| 7_mixed (Mi) | 294 | 0 | 52 | 0 | 346 | 0 |
| **Total** | **8.338** | **2.145** | **1.473** | **378** | **9.811** | **2.523** |

`4_clearsky` dan `7_mixed` tidak mendapat tambahan CCSN (CCSN tak punya kategori tersebut); `7_mixed` tetap kelas terkecil. Tambahan CCSN terbesar pada `5_stratocumulus` (+727) dan `3_cirrus` (+625).

### 7.3 Kontribusi genus CCSN per kelas GCD

Jumlah citra CCSN **yang dipakai** (setelah dedup/buang ambigu) per genus, dikelompokkan ke kelas GCD. Angka mentah CCSN per genus ada di `ccsn_v2_raw_eda_class_distribution.csv`.

| Kelas GCD | Genus CCSN (jumlah dipakai) | Subtotal |
|---|---|---:|
| 1_cumulus (Cu) | Cu 177 | 177 |
| 2_altocumulus (Ac) | Ac 219, Cc 265 | 484 |
| 3_cirrus (Ci) | Ci 139, Cs 286, **Ct 200** | 625 |
| 5_stratocumulus (Sc) | As 185, Sc 340, St 202 | 727 |
| 6_cumulonimbus (Cb) | Cb 236, Ns 274 | 510 |
| 4_clearsky, 7_mixed | — (tidak ada di CCSN) | 0 |
| **Total** | | **2.523** |

Contrails (Ct, 200) masuk `3_cirrus` sesuai keputusan proyek. Selisih genus vs angka mentah CCSN (mis. Cu 182→177, Cb 242→236) adalah citra yang dibuang karena duplikat internal atau label ambigu (total 20).

### 7.4 Asal data per split

| Split | GCD | CCSN | Total |
|---|---:|---:|---:|
| train | 8.338 | 2.145 | 10.483 |
| validation | 1.473 | 378 | 1.851 |
| test | 9.000 | 0 | 9.000 |

### 7.5 Verifikasi (terhadap artefak)

- train ∩ val = ∅ — **OK**
- 0 citra validation identik (piksel) dengan test — **OK**
- Porsi GCD pada train+val gabungan = train bersih **9.811** (identik dengan basis GCD-only) — **OK**
- CCSN dipakai **2.523** dari 2.543 (dibuang 14 duplikat internal + 6 label-ambigu) — **OK**
- Test = 9.000 (0 citra CCSN) — **OK**

Manifest: `reports/tables/combined_{train,val,test}_manifest.csv`; bobot kelas: `combined_class_weights.csv`; ringkasan: `reports/metrics/combined_split_summary.json`.

**Dimaterialisasi ke folder** `data/combined/{train,val,test}/<kelas>/` (via `scripts/materialize_combined.py`); diverifikasi terhadap disk: train **10.483** / val **1.851** / test **9.000** (cocok manifest, 7 kelas). Tiap citra disalin dari raw-nya (GCD → `data/raw/GCD`, CCSN → `data/raw/CCSN_v2`) dengan nama berkas dipertahankan; `data/raw/` tidak diubah.

---

## 8. Reproduksi

```powershell
# 1. Audit + EDA (menghasilkan manifest & tabel di reports/)
#    notebooks/01_dataset_analysis.ipynb

# 2. Penanganan kualitas + split (manifest di reports/tables/)
#    notebooks/02_preprocessing.ipynb   (atau skrip di bawah)
python scripts/build_leakage_exclusion.py --name gcd_raw
python scripts/make_split.py --name gcd_raw --val-fraction 0.15 --seed 42

# 3. Salin split GCD-only ke folder data/train|val|test
python scripts/materialize_split.py --name gcd_raw --clean

# 4. (Opsional) Split gabungan GCD+CCSN (butuh ccsn_v2_raw_manifest.csv dari langkah 1)
python scripts/combine_datasets.py

# 5. (Opsional) Salin split gabungan ke data/combined/{train,val,test}
python scripts/materialize_combined.py --clean
```

Manifest sumber kebenaran: `gcd_raw_split_*` (GCD-only) dan `combined_*` (gabungan) di `reports/tables/`. Jika manifest berubah, jalankan ulang langkah 3 (materialisasi GCD-only).

---

## 9. Rujukan

- Penanganan kualitas data: [`DATASET_QUALITY_SUMMARY.md`](DATASET_QUALITY_SUMMARY.md)
- Penggabungan dataset: [`PENGGABUNGAN_DATASET.md`](PENGGABUNGAN_DATASET.md)
- Deskripsi dataset: [`DATASET_DOCUMENTATION.md`](DATASET_DOCUMENTATION.md)
- Notebook: `notebooks/02_preprocessing.ipynb`
- Skrip: `scripts/make_split.py`, `scripts/materialize_split.py`, `scripts/combine_datasets.py`, `scripts/materialize_combined.py`
- Ringkasan: `reports/metrics/gcd_raw_split_summary.json`, `combined_split_summary.json`
- Manifest: `reports/tables/gcd_raw_split_{train,val,test}_manifest.csv`, `combined_{train,val,test}_manifest.csv`
