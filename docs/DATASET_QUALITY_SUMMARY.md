# Dataset Quality Summary dan Penanganannya — GCD & CCSN_v2

Tanggal: 13 Oktober 2026 (Asia/Jakarta). Cakupan: dataset **GCD** (`data/raw/GCD/`) dan **CCSN_v2** (`data/raw/CCSN_v2/`), serta penggabungannya.

Dokumen ini merangkum **Dataset Quality Summary** dari `notebooks/01_dataset_analysis.ipynb` dan menetapkan **penanganan** untuk tiap temuan. Bagian 1–3 membahas **GCD** (dataset utama); Bagian 4 membahas **CCSN_v2** dan **penggabungan** kedua dataset. Semua angka adalah hasil pengukuran, bukan angka literatur; bukti mentah ada di `reports/EDA_INITIAL_REPORT.md` dan tabel `reports/tables/`.

Catatan penting: proses EDA/audit **hanya membaca** berkas. Belum ada berkas yang diubah, dipindah, atau dihapus. Seluruh penanganan dirancang untuk tahap **preprocessing** lewat manifest/daftar split, **tanpa** mengubah isi `data/raw/`.

---

## 1. Ringkasan status

Sumber: `reports/tables/gcd_raw_eda_quality_summary.csv`.

| Aspek | Status | Temuan |
|---|---|---|
| Class distribution | ⚠️ Warning | Rasio terbesar/terkecil tertinggi 8,63 pada Train (6_cumulonimbus 3.003 vs 7_mixed 348) |
| Image dimensions | ✅ OK | Semua citra 512 × 512 |
| Corrupted images | ✅ OK | 0 tidak terbaca, 0 berkas 0 byte, 0 berkas non-citra |
| Duplicate files | ⚠️ Warning | 159 pasangan citra identik; 0 dengan label berbeda |
| Cross-split leakage | ⚠️ Warning | 156 pasangan identik lintas split; 36 kandidat near-duplicate lintas split; 0 nama file sama |
| Channel consistency | ✅ OK | Semua citra RGB 3 channel |
| Image quality | ✅ OK | 117 citra hampir seragam (contrast < 2,55 dari 255), 1 sangat gelap, 0 sangat terang |
| Split availability | ⚠️ Warning | Split tidak tersedia: Validation |

**4 aspek OK, 4 aspek Warning.** Tidak ada temuan berstatus Error/kritis yang memblokir penggunaan dataset, tetapi empat Warning di atas **harus ditangani sebelum training** agar hasil evaluasi tidak bias.

Ambang status mengikuti konvensi notebook (bukan standar baku): rasio imbalance Warning mulai 3,0; near-duplicate = jarak Hamming ≤ 10 dari 256 bit dHash; citra hampir seragam = contrast < 0,01 (skala 0–1); sangat gelap < 0,10; sangat terang > 0,95.

---

## 2. Penanganan per aspek

### 2.1 Cross-split leakage — ⚠️ Warning (prioritas tertinggi)

**Temuan.** 156 pasangan citra **identik** (hash piksel sama) tersebar antara split train dan test — seluruhnya arah train↔test, 0 nama file sama. Ini berarti **156 citra test** (1,73% dari 9.000) punya salinan identik di train: 155 di kelas `4_clearsky` (9,75% dari citra test kelas itu) dan 1 di `7_mixed`. Selain itu ada **36 kandidat near-duplicate lintas split** (jarak Hamming ≤ 10) yang belum ditinjau. Observasi visual atas 6 pasangan `4_clearsky` memperkuat bahwa itu memang citra yang sama (disimpan/dikompresi ulang).

**Mengapa penting.** Citra test yang juga ada di train membuat metrik test **terlalu optimistis** — model dinilai pada data yang sebagian sudah dilihatnya. Ini cacat evaluasi paling serius di dataset ini.

**Penanganan.**
1. **Jangan ubah `data/raw/`.** Tangani lewat daftar pengecualian (exclusion list) yang disimpan di manifest split pada tahap preprocessing.
2. Karena **test adalah acuan resmi yang tidak dibagi ulang**, buang salinan dari sisi **train**: setiap citra train yang identik dengan citra test dikeluarkan dari train. Dasar pasangan: `reports/tables/gcd_raw_eda_cross_split_leakage.csv` (dan `gcd_raw_near_duplicates_across_splits.csv`).
3. **Tinjau** 36 kandidat near-duplicate lintas split secara visual (`reports/tables/gcd_raw_eda_near_duplicate_candidates.csv`, gambar `reports/figures/eda/near_duplicate_candidates.png`). Pasangan yang terbukti sama diperlakukan seperti duplikat identik.
4. Catat keputusan (jumlah citra train yang dibuang, daftar path) agar dapat diverifikasi ulang.

**Kriteria selesai.** Tidak ada lagi citra test yang hash pikselnya sama dengan citra train yang dipakai; keputusan atas kandidat near-duplicate terdokumentasi.

**Status penanganan (13 Oktober 2026).** ✅ Penanganan **diimplementasikan di `notebooks/02_preprocessing.ipynb` (bagian 2–3) dan diverifikasi**. Logika deteksi dipakai ulang dari `scripts/build_leakage_exclusion.py` (`read_manifest`, `find_exclusions`). Hasil:

- **156 citra train dikeluarkan** (155 `4_clearsky`, 1 `7_mixed`), seluruhnya **byte-identik** dengan citra test. Ini menutup 156 citra test (1,73% dari 9.000) yang sebelumnya bocor.
- Kebijakan: pertahankan salinan di **test** (acuan resmi utuh), keluarkan salinan dari **train**. `data/raw/` tidak diubah.
- Keluaran: `reports/tables/gcd_raw_leakage_exclusion.csv` (daftar path lengkap; kolom `rel_path, split, class_dir, reason, match_type, keep_split, keep_rel_path, pixel_sha256`) dan ringkasan `reports/metrics/gcd_raw_exclusion_summary.json`. Leakage saja membuang 156 citra train (10.000 → 9.844); manifest train bersih final dibentuk setelah §2.2 (lihat di bawah).
- Verifikasi (assert di notebook): 0 citra train bersih yang piksel-identik dengan test; daftar cocok dengan tabel EDA `gcd_raw_eda_cross_split_leakage.csv`.
- Konsumsi: tahap preprocessing berikutnya memakai **manifest train bersih** sebagai sumber train, bukan folder `data/raw/GCD/train/` langsung.
- ✅ **Tinjauan visual near-duplicate lintas split (langkah 3) selesai (13 Okt 2026).** 36 kandidat ditinjau (`reports/figures/eda/near_dup_xsplit_ambiguous.png`, `..._lowdist.png`): **32 pasangan jarak dHash ≤2 (semua `4_clearsky`) terbukti citra yang sama** (foto sama disimpan/dikompresi ulang; dua di antaranya dengan silau matahari di posisi persis sama) → **32 citra train dibuang** (`reports/tables/gcd_raw_near_dup_crosssplit_exclusion.csv`). **4 pasangan jarak ≥6 terbukti BUKAN citra sama** (2 lintas kelas dengan isi jelas berbeda; 2 sekelas tapi pola awan berbeda — false positive dHash pada langit datar) → dipertahankan.
- **Total dibuang dari train (gabungan §2.1 piksel + §2.1 near-dup + §2.2): 189** → manifest train bersih **9.811 dari 10.000**.

### 2.2 Duplicate files (dalam split) — ⚠️ Warning

**Temuan.** Total 159 pasangan citra identik (318 berkas terlibat); **0 pasangan berlabel kelas berbeda**. Di luar 156 pasangan lintas split (di atas), ada **3 pasangan di dalam split yang sama**: 1 di train (`7_mixed`), 2 di test (`1_cumulus`, `5_stratocumulus`). Tidak ada pasangan piksel-identik dengan berkas berbeda yang berlabel beda, jadi tidak ada konflik label.

**Mengapa penting.** Duplikat di dalam train memberi bobot berlebih ke beberapa citra; duplikat di dalam test sedikit menggelembungkan ukuran efektif test. Dampaknya kecil (hanya 3 pasangan) tetapi tetap perlu dirapikan.

**Penanganan.**
1. Dalam train: pertahankan satu citra per pasangan, keluarkan salinannya dari manifest train.
2. Dalam test: **jangan hapus** (test resmi utuh); cukup catat sebagai duplikat di test agar diketahui saat membaca metrik.
3. Dasar pasangan: `reports/tables/gcd_raw_eda_duplicate_pairs.csv` / `gcd_raw_duplicates.csv`.

**Kriteria selesai.** Tiap pasangan identik di train tersisa satu representatif; duplikat di test tercatat.

**Status penanganan (13 Oktober 2026).** ✅ Diimplementasikan di `notebooks/02_preprocessing.ipynb` (bagian 3) dengan `find_within_split_duplicates` dari `scripts/build_leakage_exclusion.py`. Kriteria: dua citra "sama" bila `pixel_sha256` identik (karena 0 pasangan piksel-identik-tapi-byte-beda, ini setara dengan byte-identik di tabel EDA). Hasil:

- **Train:** 1 citra dikeluarkan — `train/7_mixed/7_mixed_000291.jpg` (dipertahankan `train/7_mixed/7_mixed_000088.jpg`). → `reports/tables/gcd_raw_within_split_duplicate_exclusion.csv`.
- **Test:** 2 citra **dicatat, tidak dikeluarkan** — `test/1_cumulus/1_cumulus_001420.jpg` dan `test/5_stratocumulus/5_stratocumulus_003369.jpg`. → `reports/tables/gcd_raw_test_duplicates.csv`.
- **Manifest train bersih final** (gabungan §2.1 piksel + §2.1 near-dup + §2.2): `reports/tables/gcd_raw_train_clean_manifest.csv` = **9.811 dari 10.000 citra** (189 dibuang: 156 leakage piksel + 32 near-dup lintas split + 1 duplikat dalam-split). Rincian per kelas: `4_clearsky` 2.150→1.963, `7_mixed` 348→346, lainnya tetap.
- Verifikasi (assert di notebook): 0 citra train identik dengan test **dan** 0 duplikat tersisa di dalam train bersih.

### 2.3 Class distribution (imbalance) — ⚠️ Warning

**Temuan.** Imbalance **sedang**. Rasio kelas terbesar/terkecil: 8,63 di Train (`6_cumulonimbus` 3.003 vs `7_mixed` 348), 4,55 di Test, 6,04 keseluruhan. Proporsi juga sedikit bergeser antar split (selisih terbesar 3,84 poin persentase pada `4_clearsky`). Distribusi lengkap: `reports/tables/gcd_raw_eda_class_distribution.csv`.

**Mengapa penting.** Model cenderung bias ke kelas mayoritas (`6_cumulonimbus`, `4_clearsky`), dan accuracy saja akan menutupi performa buruk di kelas minoritas (`7_mixed`).

**Penanganan.**
1. **Training:** terapkan penanganan imbalance — bobot kelas (class weights) berbanding terbalik dengan frekuensi, atau sampling berimbang. Tetapkan ini di konfigurasi training, bukan di data.
2. **Evaluasi:** laporkan **macro-F1** dan metrik **per kelas** (precision/recall/F1), bukan hanya accuracy; sertakan confusion matrix.
3. Jangan oversample dengan menyalin citra lintas split, dan jangan menyeimbangkan test (test tetap apa adanya).

**Kriteria selesai.** Strategi imbalance tercantum di config training; protokol evaluasi memakai macro-F1 + metrik per kelas.

**Status penanganan (13 Oktober 2026).** ✅ Strategi ditetapkan; bobot kelas dihitung. Penanganan ini bersifat konfigurasi + evaluasi, bukan operasi data.

- **Config** (`configs/base.yaml`): `training.imbalance: class_weights` dengan `training.class_weights_file: reports/tables/gcd_raw_class_weights.csv`; `evaluation.primary_metric: macro_f1`, `evaluation.per_class_metrics: true`, `evaluation.confusion_matrix: true`. `checkpoint.monitor` tetap `val_macro_f1`.
- **Bobot kelas** (`notebooks/02_preprocessing.ipynb` §5 → `reports/tables/gcd_raw_class_weights.csv`): inverse-frequency `w_c = N/(K·n_c)` dari **train bersih saja** (N=9.811, K=7), test tidak diseimbangkan. Hasil (per index kelas untuk `CrossEntropyLoss(weight=...)`):

  | idx | kelas | n (train bersih) | bobot |
  |---:|---|---:|---:|
  | 0 | Cu (1_cumulus) | 775 | 1,8085 |
  | 1 | Ac (2_altocumulus) | 725 | 1,9332 |
  | 2 | Ci (3_cirrus) | 1.153 | 1,2156 |
  | 3 | Cl (4_clearsky) | 1.963 | 0,7140 |
  | 4 | Sc (5_stratocumulus) | 1.846 | 0,7592 |
  | 5 | Cb (6_cumulonimbus) | 3.003 | 0,4667 |
  | 6 | Mi (7_mixed) | 346 | 4,0508 |

  Rasio bobot maks/min 8,68 (Mi terbesar, Cb terkecil). Alternatif `weighted_sampler` disediakan lewat kolom `sampler_weight_per_image`.
- ⏳ **Penerapan** (`CrossEntropyLoss(weight=...)` / sampler) terjadi saat kode training ditulis; config dan bobot sudah siap dikonsumsi.

### 2.4 Split availability — ⚠️ Warning

**Temuan.** Hanya ada split **train (10.000)** dan **test (9.000)**; **tidak ada validation**. Tanpa validation, tidak ada data netral untuk memilih hyperparameter/early stopping tanpa menyentuh test.

**Mengapa penting.** Memakai test untuk tuning akan membocorkan test ke proses pemilihan model — bentuk lain dari leakage.

**Penanganan.**
1. Buat validation **dari train resmi** secara **stratified** (menjaga proporsi 7 kelas) dengan **seed tetap**; test resmi **tidak** dibagi ulang.
2. Lakukan ini **setelah** penanganan duplikat/near-duplicate, dan pastikan pasangan duplikat/near-duplicate di dalam train **tetap di sisi yang sama** (semua di train atau semua di validation) agar tidak muncul leakage train↔val baru.
3. Simpan manifest split (daftar path per split) yang dapat diverifikasi ulang; catat rasio split dan seed.

**Kriteria selesai.** Ada manifest train/validation hasil pembagian stratified berseed, tanpa citra yang sama di kedua sisi, dan test resmi tidak berubah.

**Status penanganan (13 Oktober 2026).** ✅ Validation dibuat dari train bersih. Diimplementasikan di `notebooks/02_preprocessing.ipynb` (bagian 6) dengan `grouped_stratified_split` dari `scripts/make_split.py`. Parameter: `val_fraction_of_train = 0.15`, `seed = 42`, stratified per kelas.

- **Hasil:** **train 8.340 / val 1.471 / test 9.000** (dari train bersih 9.811). Validation ≈15% per kelas. → `reports/tables/gcd_raw_split_{train,val,test}_manifest.csv`, ringkasan `reports/metrics/gcd_raw_split_summary.json`.
- **Grup near-duplicate dijaga:** 3 pasangan near-duplicate di dalam train (termasuk 1 lintas kelas `2_altocumulus`↔`5_stratocumulus`) dipaksa berada di sisi yang sama, sehingga tidak muncul leakage train↔val baru.
- **Test tidak dibagi ulang:** manifest test = salinan baris test resmi apa adanya.
- Verifikasi (assert di notebook): train∩val = ∅; train∪val = train bersih; 0 citra val identik (piksel) dengan test; tidak ada pasangan near-duplicate dalam train yang terpisah train/val.
- Config: `configs/base.yaml` `data.split` diperbarui (`val_fraction_of_train: 0.15`, `stratified: true`, `keep_near_duplicate_groups_together: true`).

### 2.5 Aspek berstatus OK (dipantau, tanpa tindakan khusus)

| Aspek | Alasan OK | Catatan untuk preprocessing |
|---|---|---|
| Image dimensions | Semua 512 × 512, aspect ratio 1,00 — seragam | Cukup satu `resize` ke ukuran input model saat pemuatan (di memori); tidak perlu menulis ulang citra |
| Corrupted images | 0 tidak terbaca, 0 kosong, 0 non-citra, 0 salah ekstensi | Tidak ada pembersihan berkas yang diperlukan |
| Channel consistency | 19.000 citra RGB 3 channel, format JPEG | Pipeline cukup decode sebagai RGB; tidak perlu konversi grayscale/alpha |
| Image quality | 117 citra hampir seragam (82 di `4_clearsky`, 33 `5_stratocumulus`, 2 `6_cumulonimbus`), 1 sangat gelap, 0 sangat terang | Wajar untuk kelas langit datar; **tidak dibuang**. Normalisasi pakai mean/std **split train saja**: mean (0,4827, 0,6154, 0,7672), std (0,1955, 0,1282, 0,1370) — jika tidak memakai statistik bobot pretrained |

Citra hampir seragam di `4_clearsky`/`5_stratocumulus` adalah karakteristik nyata kelas tersebut (langit bersih / lapisan rata), bukan cacat, sehingga dipertahankan.

**Status pipeline praproses (13 Oktober 2026).** ✅ Ditetapkan di `notebooks/02_preprocessing.ipynb` (bagian 7) dan `configs/base.yaml` (`preprocess`). Pipeline **di memori** (citra tidak ditulis ulang, `data/processed/` tidak dipakai): **decode RGB → resize 224 (bilinear) → normalisasi**.

- Statistik normalisasi dihitung ulang dari **split train final** (8.340 citra, bukan 10.000 train asli dan bukan val/test): mean (0,4885, 0,6170, 0,7650), std (0,1918, 0,1297, 0,1403) pada skala 0-1. → `reports/tables/gcd_raw_norm_stats.csv`, dan tersimpan di `configs/base.yaml` `preprocess.normalize`.
- Catatan: angka di tabel baris *Image quality* di atas (0,4827 dst.) adalah statistik train **asli** dari EDA; untuk training dipakai statistik train **final** di atas. Jika memakai bobot pretrained (mis. ImageNet), gunakan statistik bobot itu.
- Penerapan transform (torchvision) terjadi di kode dataset/training; augmentasi hanya pada train.

---

## 3. Urutan tindakan (checklist preprocessing)

Dijalankan di `02_preprocessing.ipynb`, berbasis manifest, tanpa menyentuh `data/raw/`:

1. [x] Bangun daftar pengecualian cross-split: keluarkan 156 citra **train** yang identik dengan test (§2.1). → `reports/tables/gcd_raw_leakage_exclusion.csv` (13 Okt 2026).
2. [x] Tinjau 36 kandidat near-duplicate lintas split; tambahkan yang terbukti sama ke daftar pengecualian (§2.1). → 32 (d≤2) dibuang, 4 (d≥6) dipertahankan (13 Okt 2026).
3. [x] Keluarkan salinan duplikat **di dalam train** (sisakan satu per pasangan); catat duplikat di test (§2.2). → 1 train dibuang, 2 test dicatat (13 Okt 2026).
4. [x] Buat split **validation dari train** (stratified, seed tetap), jaga pasangan duplikat/near-dup di sisi yang sama (§2.4). → train 8.340 / val 1.471 / test 9.000 (13 Okt 2026).
5. [x] Simpan manifest train/validation/test final yang dapat diverifikasi. → `reports/tables/gcd_raw_split_{train,val,test}_manifest.csv`.
6. [x] Tetapkan pipeline di memori: decode RGB → resize 224 → normalisasi (mean/std train final) (§2.5). → `configs/base.yaml` `preprocess` + `gcd_raw_norm_stats.csv` (13 Okt 2026).
7. [x] Konfigurasikan penanganan imbalance (bobot kelas/sampling) + protokol evaluasi macro-F1 & per kelas (§2.3). → `configs/base.yaml` + `gcd_raw_class_weights.csv` (13 Okt 2026); penerapan di kode training.

Preprocessing berbasis manifest **selesai** untuk GCD. Sisa sebelum baseline: penulisan kode training/evaluasi yang mengonsumsi manifest split, `preprocess`, dan `training.imbalance`.

---

## 4. Dataset kedua (CCSN_v2) dan penggabungan

Dataset kedua, **CCSN_v2** (2.543 citra, 11 genera WMO), ditambahkan dan digabung ke dalam skema 7 kelas GCD. Rincian lengkap: [`PENGGABUNGAN_DATASET.md`](PENGGABUNGAN_DATASET.md). Keputusan: **contrails (Ct) → `3_cirrus`**, dan **test = test resmi GCD saja** (CCSN hanya menambah train/validation).

### 4.1 Ringkasan status kualitas CCSN_v2

Sumber: EDA `notebooks/01_dataset_analysis.ipynb` → `reports/tables/ccsn_v2_raw_eda_quality_summary.csv` dan `ccsn_v2_raw_eda_*` (distribusi, dimensi, duplikat, near-duplicate). Penggabungan (`notebooks/02_preprocessing.ipynb` bagian 8 / `scripts/combine_datasets.py`) membaca manifest yang sama, `reports/tables/ccsn_v2_raw_manifest.csv`.

| Aspek | Status | Temuan & penanganan |
|---|---|---|
| Skema label | ⚠️ Warning | 11 genera ≠ 7 kelas GCD → **dipetakan** ke skema GCD (`combined_class_mapping.csv`) |
| Image dimensions | ⚠️ Warning | Campuran 400×400 (2.332) dan 256×256 (211) → diseragamkan via resize 224 saat load |
| Corrupted images | ✅ OK | 0 tidak terbaca |
| Duplicate files | ⚠️ Warning | 14 duplikat internal (piksel) → disisakan satu per grup |
| Label ambiguity | ⚠️ Warning | 3 grup citra identik di dua folder genus berbeda (6 citra) → **dibuang** (konflik label) |
| Cross-dataset leakage | ✅ OK | 0 citra CCSN identik dengan citra GCD (termasuk test) |
| Channel consistency | ✅ OK | RGB |

CCSN dipakai: **2.523** dari 2.543 (dibuang 20: 14 duplikat + 6 ambigu).

### 4.2 Penanganan penggabungan

Berbasis manifest, `data/raw/` tidak diubah (`scripts/combine_datasets.py`):

1. Audit CCSN_v2 → manifest hash/ukuran.
2. Pemetaan label genera → kelas GCD (Ct→`3_cirrus`).
3. Deduplikasi internal CCSN + buang label ambigu.
4. Cek duplikat/kebocoran lintas-dataset terhadap GCD → 0.
5. Split gabungan: pool (GCD train bersih 9.811 + CCSN 2.523 = 12.334) → **train 10.483 / val 1.851** (stratified 85/15, seed 42); **test = GCD 9.000** (murni).
6. Hitung ulang bobot kelas dari train gabungan.

### 4.3 Artefak gabungan (dipakai untuk model dua-dataset)

| Keperluan | GCD-only | Gabungan GCD+CCSN |
|---|---|---|
| Manifest split | `gcd_raw_split_{train,val,test}_manifest.csv` | `combined_{train,val,test}_manifest.csv` |
| Bobot kelas | `gcd_raw_class_weights.csv` | `combined_class_weights.csv` |
| Pemetaan label | — | `combined_class_mapping.csv` |
| Ringkasan | `gcd_raw_split_summary.json` | `combined_split_summary.json` |

Normalisasi: bila memakai bobot **pra-latih ImageNet**, gunakan statistik ImageNet (lihat `ARSITEKTUR_BASELINE.md`); bila dari nol pada data gabungan, hitung mean/std dari train gabungan.

### 4.4 Batasan

- Test hanya domain GCD; performa domain CCSN tidak diukur test ini (*domain shift* — kolom `dataset` disimpan untuk analisis per sumber).
- `4_clearsky` dan `7_mixed` tidak mendapat tambahan dari CCSN; `7_mixed` tetap kelas terkecil.
- Hanya duplikat **piksel-identik** yang dicek lintas-dataset; near-duplicate GCD↔CCSN belum ditinjau visual.

---

## 5. Rujukan

- Laporan lengkap EDA (GCD): [`reports/EDA_INITIAL_REPORT.md`](../reports/EDA_INITIAL_REPORT.md)
- Penggabungan dataset (GCD + CCSN_v2): [`docs/PENGGABUNGAN_DATASET.md`](PENGGABUNGAN_DATASET.md)
- Hasil split (GCD-only): [`docs/DATASET_SPLIT.md`](DATASET_SPLIT.md)
- Dokumentasi dataset: [`docs/DATASET_DOCUMENTATION.md`](DATASET_DOCUMENTATION.md)
- Rencana eksperimen: [`docs/EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md)
- Notebook sumber: `notebooks/01_dataset_analysis.ipynb`, `notebooks/02_preprocessing.ipynb`
- Tabel bukti GCD: `reports/tables/gcd_raw_eda_quality_summary.csv`, `..._cross_split_leakage.csv`, `..._duplicate_pairs.csv`, `..._near_duplicate_candidates.csv`, `..._class_distribution.csv`
- Tabel bukti gabungan: `reports/tables/combined_{train,val,test}_manifest.csv`, `combined_class_mapping.csv`, `combined_class_weights.csv`, `reports/metrics/combined_split_summary.json`
