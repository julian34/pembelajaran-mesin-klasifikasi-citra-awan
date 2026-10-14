# Penggabungan Dataset — GCD + CCSN_v2

Tanggal: 13 Oktober 2026. Terkait: [`DATASET_SPLIT.md`](DATASET_SPLIT.md), [`DATASET_QUALITY_SUMMARY.md`](DATASET_QUALITY_SUMMARY.md), [`PENANGANAN_KUALITAS_DATA.md`](PENANGANAN_KUALITAS_DATA.md).

Dokumen ini merinci cara menggabungkan dua dataset mentah — **GCD** (19.000 citra, 7 kelas) dan **CCSN_v2** (2.543 citra, 11 genera WMO) — menjadi satu dataset berskema **7 kelas GCD**. Implementasi: `scripts/combine_datasets.py`. Berkas di `data/raw/` tidak diubah; semua hasil berupa manifest di `reports/`.

---

## 1. Ringkasan

Kedua dataset punya **skema label berbeda**: GCD menggabungkan beberapa genera menjadi satu kelas, sedangkan CCSN memakai genera terpisah (plus *contrails*). Penggabungan dilakukan dengan **memetakan 11 genera CCSN ke 7 kelas GCD** (GCD sebagai skema acuan karena jauh lebih besar dan sudah terstruktur), bukan sekadar menyatukan folder.

**Keputusan desain:**
- **Contrails (Ct) → `3_cirrus`** (contrails adalah awan tipis ketinggian tinggi, mirip cirrus / *cirrus homogenitus*).
- **Test = test resmi GCD saja (9.000)**; CCSN hanya menambah **train/validation**, agar test tetap sebanding dengan literatur GCD dan tidak tercampur domain lain.

**Hasil:** dataset gabungan **21.334 citra** — train 10.483 / validation 1.851 / test 9.000.

---

## 2. Pemetaan label CCSN → GCD

| Kelas GCD | Genera CCSN | Dasar |
|---|---|---|
| 1_cumulus | Cu | Cumulus |
| 2_altocumulus | Ac, Cc | Altocumulus + cirrocumulus |
| 3_cirrus | Ci, Cs, **Ct** | Cirrus + cirrostratus + contrails |
| 5_stratocumulus | Sc, St, As | Stratocumulus + stratus + altostratus |
| 6_cumulonimbus | Cb, Ns | Cumulonimbus + nimbostratus |
| 4_clearsky | — | Tidak ada di CCSN |
| 7_mixed | — | Tidak ada di CCSN |

Tabel pemetaan tersimpan di `reports/tables/combined_class_mapping.csv`. Kelas `4_clearsky` dan `7_mixed` tidak mendapat tambahan dari CCSN (CCSN tidak memiliki kategori tersebut).

---

## 3. Tahapan penggabungan

1. **Audit/EDA CCSN_v2** oleh `notebooks/01_dataset_analysis.ipynb` → manifest `reports/tables/ccsn_v2_raw_manifest.csv` (hash piksel, dHash, ukuran) + tabel `ccsn_v2_raw_eda_*` (dianalisis sebagai satu split). Temuan: 2.543 citra, JPEG, ukuran campuran (400×400 dan 256×256). Manifest inilah yang dibaca penggabungan.
2. **Pemetaan label** genera → kelas GCD (Bagian 2).
3. **Deduplikasi internal CCSN** (berdasarkan hash piksel):
   - 14 salinan duplikat dibuang (sisakan satu per grup);
   - 6 citra berlabel **ambigu** dibuang — yaitu 3 grup citra identik yang muncul di dua folder genus berbeda yang memetakan ke kelas GCD berbeda (konflik label).
4. **Cek duplikasi/kebocoran lintas-dataset:** tidak ada citra CCSN yang identik (piksel) dengan citra GCD — 0 duplikat, 0 kebocoran ke test GCD. CCSN dipakai: **2.523**.
5. **Pembentukan split gabungan:** pool train+val = **GCD train bersih (9.811) + CCSN (2.523) = 12.334**, dibagi **stratified** per kelas (85% train / 15% val), seed 42. Pasangan near-duplicate di dalam train GCD dijaga satu sisi. **Test = test resmi GCD (9.000)**, tidak ditambah.
6. **Bobot kelas** dihitung ulang dari train gabungan → `reports/tables/combined_class_weights.csv`.

---

## 4. Data awal dan data akhir

| Butir | GCD | CCSN_v2 | Gabungan |
|---|---:|---:|---:|
| Citra mentah | 19.000 | 2.543 | 21.543 |
| Dibuang (leakage/duplikat/ambigu) | 189 | 20 | 209 |
| Dipakai | 18.811 | 2.523 | **21.334** |
| Skema kelas | 7 | 11 → dipetakan ke 7 | 7 |

CCSN dibuang 20: 14 duplikat internal + 6 ambigu lintas-kelas.

---

## 5. Komposisi split gabungan

Jumlah citra per kelas (skema GCD) di tiap split:

| Kelas | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| 1_cumulus | 809 | 143 | 750 | 1.702 |
| 2_altocumulus | 1.027 | 182 | 750 | 1.959 |
| 3_cirrus | 1.511 | 267 | 753 | 2.531 |
| 4_clearsky | 1.669 | 294 | 1.589 | 3.552 |
| 5_stratocumulus | 2.187 | 386 | 1.790 | 4.363 |
| 6_cumulonimbus | 2.986 | 527 | 2.761 | 6.274 |
| 7_mixed | 294 | 52 | 607 | 953 |
| **Total** | **10.483** | **1.851** | **9.000** | **21.334** |

Asal data per split (kolom `dataset` pada manifest):

| Split | GCD | CCSN | Total |
|---|---:|---:|---:|
| train | 8.338 | 2.145 | 10.483 |
| val | 1.473 | 378 | 1.851 |
| test | 9.000 | 0 | 9.000 |

Manifest: `reports/tables/combined_{train,val,test}_manifest.csv` — tiap baris memuat `dataset`, `split`, `class` (skema GCD), `src_class` (genus asli), `rel_path`, ukuran, dan `pixel_sha256`.

---

## 6. Catatan & batasan

- **Test murni GCD.** Model gabungan diuji pada domain GCD saja; performa pada domain CCSN tidak diukur oleh test ini. Jika ingin menilai lintas-domain, perlu test CCSN terpisah (lihat opsi di luar cakupan dokumen ini).
- **Domain shift.** GCD dan CCSN berbeda sumber (kamera, lokasi, ukuran). Kolom `dataset` disimpan agar pengaruh asal data dapat dianalisis; pertimbangkan analisis performa per sumber.
- **Kelas `4_clearsky` dan `7_mixed` tetap bersumber GCD saja** — tidak terbantu CCSN; `7_mixed` tetap kelas terkecil (bobot kelas menanganinya).
- **Pemetaan Ct → cirrus** adalah keputusan proyek; contrails bukan genus alami. Bila ingin memisahkannya, ubah `CCSN_TO_GCD` di `scripts/combine_datasets.py`.
- **Near-duplicate internal CCSN** (selain yang piksel-identik) belum dihitung; hanya duplikat eksak yang dibuang. Near-duplicate lintas-dataset GCD↔CCSN juga belum ditinjau visual (hanya piksel-identik yang dicek).
- **Resolusi campuran** (CCSN 400/256, GCD 512) diseragamkan saat pemuatan lewat resize 224 (lihat `preprocess`); tidak ada penulisan ulang citra.

---

## 7. Reproduksi

```powershell
# 1. EDA kedua dataset (menghasilkan ccsn_v2_raw_manifest.csv & gcd_raw_*)
#    notebooks/01_dataset_analysis.ipynb   (data/raw/GCD + data/raw/CCSN_v2 harus ada)

# 2. Penanganan kualitas GCD + split bersih
#    notebooks/02_preprocessing.ipynb bagian 1-7

# 3. Gabungkan (notebooks/02 bagian 8, atau CLI berikut)
python scripts/combine_datasets.py
```

Artefak: `reports/tables/combined_{train,val,test}_manifest.csv`, `combined_class_mapping.csv`, `combined_class_weights.csv`, `reports/metrics/combined_split_summary.json`.

Materialisasi ke folder `data/combined/{train,val,test}/<kelas>/` **sudah dilakukan** dengan `scripts/materialize_combined.py --clean` (menyalin dari dua sumber: `data/raw/GCD` dan `data/raw/CCSN_v2`): train 10.483 / val 1.851 / test 9.000, terverifikasi terhadap disk.

---

## 8. Rujukan

- Split GCD-only: [`DATASET_SPLIT.md`](DATASET_SPLIT.md)
- Penanganan kualitas: [`DATASET_QUALITY_SUMMARY.md`](DATASET_QUALITY_SUMMARY.md)
- Skrip: `scripts/combine_datasets.py`, `scripts/audit_dataset.py`, `scripts/make_split.py`
- Ringkasan: `reports/metrics/combined_split_summary.json`
