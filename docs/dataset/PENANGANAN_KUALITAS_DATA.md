# Penanganan Kualitas Data Dataset GCD — Dokumentasi Ilmiah

Dataset: **TJNU Ground-based Cloud Dataset (GCD)** — klasifikasi citra awan berbasis darat, 7 kelas, 19.000 citra JPEG 512 × 512 RGB.
Tanggal: 13 Oktober 2026 (Asia/Jakarta). Sumber angka: `notebooks/01_dataset_analysis.ipynb`, `notebooks/02_preprocessing.ipynb`, dan tabel di `reports/`.

Dokumen ini menyajikan metodologi penanganan masalah kualitas data secara ilmiah untuk **GCD**. Rincian operasional dan checklist ada di [`DATASET_QUALITY_SUMMARY.md`](DATASET_QUALITY_SUMMARY.md); hasil split di [`DATASET_SPLIT.md`](DATASET_SPLIT.md).

> **Catatan (13 Okt 2026).** Dataset kedua **CCSN_v2** kemudian ditambahkan dan digabung ke skema 7 kelas GCD (pemetaan genera, deduplikasi, split gabungan). Metodologi penggabungan dan hasilnya didokumentasikan terpisah di [`PENGGABUNGAN_DATASET.md`](PENGGABUNGAN_DATASET.md); angka di dokumen ini tetap untuk GCD.

---

## 1. Ringkasan

Analisis data eksploratif (EDA) atas 19.000 citra GCD mengidentifikasi empat masalah kualitas yang berpotensi membiaskan evaluasi model: (i) kebocoran antar split (*cross-split leakage*), (ii) duplikasi citra di dalam split, (iii) ketidakseimbangan kelas (*class imbalance*), dan (iv) ketiadaan split validasi. Keempatnya ditangani secara **berbasis manifest** — seluruh keputusan dicatat sebagai daftar berkas yang dapat diverifikasi ulang, tanpa mengubah, memindah, atau menghapus berkas asli di `data/raw/`.

Penanganan menghasilkan: pembuangan **189 citra train** yang terbukti bocor atau duplikat (10.000 → 9.811), pembentukan **split validasi stratified berseed** dari train bersih (train 8.340 / validasi 1.471, rasio 85/15), penetapan **bobot kelas** untuk mengatasi imbalance, dan **pipeline praproses** (resize + normalisasi) dengan statistik dihitung ulang dari train final. Integritas test resmi (9.000 citra) dipertahankan sepenuhnya. Seluruh prosedur deterministik (seed 42) dan dapat direproduksi.

---

## 2. Tahapan Penanganan

Penanganan mengikuti enam tahap berurutan. Tiap tahap memakai kriteria deteksi yang eksplisit dan menulis artefak bukti ke `reports/`.

### Tahap 1 — Audit dan deteksi (EDA)

Setiap citra dipindai untuk memperoleh manifest: SHA-256 isi berkas, hash piksel (`pixel_sha256`), *difference hash* (dHash 16×16 = 256 bit), dimensi, mode warna, dan statistik kanal. Manifest ini menjadi dasar seluruh deteksi berikutnya. Temuan: 0 berkas rusak/korup, seluruh citra 512 × 512 RGB (seragam), sehingga tidak diperlukan pembersihan berkas.

### Tahap 2 — Penanganan cross-split leakage

Kebocoran antar split dideteksi melalui dua kriteria, keduanya ditangani dengan **mempertahankan citra di test** (acuan resmi) dan membuang salinannya dari **train**:

- **Piksel identik:** citra train dengan `pixel_sha256` sama dengan citra test. Ditemukan **156** citra train (155 `4_clearsky`, 1 `7_mixed`).
- **Near-duplicate terkonfirmasi:** pasangan train–test dengan jarak Hamming dHash ≤ 2, yang melalui **tinjauan visual** terbukti merupakan citra yang sama (disimpan/dikompresi ulang sehingga piksel berbeda). Dari 36 kandidat lintas split, **32** (semua `4_clearsky`, jarak ≤ 2) dikonfirmasi sama dan dibuang; 4 kandidat berjarak ≥ 6 ditinjau dan terbukti **bukan** citra sama (false positive dHash pada langit berdetail rendah) sehingga dipertahankan.

Total dibuang pada tahap ini: **188 citra train**. Artefak: `gcd_raw_leakage_exclusion.csv`, `gcd_raw_near_dup_crosssplit_exclusion.csv`.

### Tahap 3 — Penanganan duplikat di dalam split

Grup citra piksel-identik yang berada di dalam satu split: pada **train** satu representatif dipertahankan dan salinannya dibuang (**1** citra, `7_mixed`); pada **test** duplikat **tidak dibuang** (integritas test), hanya dicatat (**2** citra). Tidak ditemukan pasangan identik berlabel kelas berbeda (tidak ada konflik label). Artefak: `gcd_raw_within_split_duplicate_exclusion.csv`, `gcd_raw_test_duplicates.csv`.

### Tahap 4 — Penanganan ketidakseimbangan kelas

Rasio kelas terbesar/terkecil pada train 8,63 (imbalance sedang). Penanganan bersifat konfigurasi, bukan manipulasi data: **bobot kelas** dihitung dengan skema *inverse-frequency* `w_c = N / (K · n_c)` dari **train bersih** (N = 9.811, K = 7), sehingga kelas minoritas (`7_mixed`, w = 4,05) diberi bobot lebih besar daripada mayoritas (`6_cumulonimbus`, w = 0,47). Protokol evaluasi ditetapkan memakai **macro-F1** dan metrik per kelas (bukan accuracy semata). Test tidak diseimbangkan. Artefak: `gcd_raw_class_weights.csv`; konfigurasi: `configs/base.yaml` (`training.imbalance`, `evaluation`).

### Tahap 5 — Pembentukan split validasi

GCD hanya menyediakan train dan test. Split validasi dibentuk dari **train bersih** secara **stratified** (proporsi 7 kelas dijaga) dengan **seed tetap (42)**, rasio **85% train / 15% validasi**. Pasangan near-duplicate di dalam train dijaga berada pada sisi yang sama (train atau validasi) agar tidak timbul kebocoran train↔validasi baru. Test resmi **tidak dibagi ulang**. Hasil: train 8.340 / validasi 1.471 / test 9.000. Artefak: `gcd_raw_split_{train,val,test}_manifest.csv`, `gcd_raw_split_summary.json`.

### Tahap 6 — Pipeline praproses

Karena seluruh citra seragam 512 × 512 RGB, praproses dilakukan **di memori** saat pemuatan (tanpa menulis ulang citra): dekode RGB → *resize* 224 (bilinear) → normalisasi. Statistik normalisasi dihitung ulang dari **split train final** (8.340 citra, bukan dari validasi/test): mean (0,4885; 0,6170; 0,7650), std (0,1918; 0,1297; 0,1403) pada skala 0–1. Artefak: `gcd_raw_norm_stats.csv`; konfigurasi: `configs/base.yaml` (`preprocess`).

---

## 3. Data Awal dan Data Akhir

### 3.1 Data awal (sebelum penanganan)

| Butir | Nilai |
|---|---|
| Total citra | 19.000 (train 10.000, test 9.000) |
| Kelas | 7 (Cu, Ac, Ci, Cl, Sc, Cb, Mi) |
| Resolusi / format | 512 × 512 RGB, JPEG, seragam |
| Split tersedia | train, test (**tidak ada validasi**) |
| Cross-split leakage (piksel identik) | 156 pasangan train–test |
| Near-duplicate lintas split (kandidat) | 36 (32 terkonfirmasi sama) |
| Duplikat di dalam split | 3 pasangan (1 train, 2 test) |
| Rasio imbalance (train) | 8,63 (maks `6_cumulonimbus` : min `7_mixed`) |
| Citra rusak / korup | 0 |

### 3.2 Data akhir (setelah penanganan)

| Butir | Nilai |
|---|---|
| Train resmi bersih | **9.811** (dibuang 189: 156 leakage piksel + 32 near-dup + 1 duplikat dalam-split) |
| Split final | train **8.340** / validasi **1.471** / test **9.000** (total 18.811) |
| Rasio train/validasi | 85 / 15 (stratified, seed 42) |
| Cross-split leakage tersisa | 0 (train/val ∩ test = ∅ secara piksel) |
| Duplikat tersisa di train | 0 |
| Penanganan imbalance | bobot kelas inverse-frequency + evaluasi macro-F1 |
| Normalisasi | mean/std dari train final (`preprocess.normalize`) |
| Integritas test | utuh, 9.000 citra, tidak dibagi ulang |
| `data/raw/GCD/` | tidak berubah (19.000 citra) |

### 3.3 Perubahan jumlah train per kelas

| Kelas | Train awal | Train bersih | Selisih |
|---|---:|---:|---:|
| 1_cumulus | 775 | 775 | 0 |
| 2_altocumulus | 725 | 725 | 0 |
| 3_cirrus | 1.153 | 1.153 | 0 |
| 4_clearsky | 2.150 | 1.963 | −187 |
| 5_stratocumulus | 1.846 | 1.846 | 0 |
| 6_cumulonimbus | 3.003 | 3.003 | 0 |
| 7_mixed | 348 | 346 | −2 |
| **Total** | **10.000** | **9.811** | **−189** |

Pembuangan terkonsentrasi pada `4_clearsky` (−187), konsisten dengan temuan bahwa kebocoran dan near-duplicate didominasi citra langit cerah yang mudah terduplikasi.

---

## 4. Kesimpulan

1. **Keempat masalah kualitas tertangani.** Cross-split leakage dan duplikasi dihapus dari sisi train/validasi (189 citra), ketidakseimbangan kelas ditangani lewat bobot kelas dan protokol evaluasi macro-F1, serta split validasi stratified berseed dibentuk dari train bersih.
2. **Validitas evaluasi terjaga.** Tidak ada lagi citra train/validasi yang identik dengan test, sehingga metrik test tidak lagi bias optimistis akibat kebocoran. Pemilihan model memakai validasi, bukan test.
3. **Integritas dan reprodusibilitas.** Test resmi dipertahankan utuh (9.000), `data/raw/` tidak diubah, dan seluruh keputusan tersimpan sebagai manifest/daftar pengecualian yang deterministik (seed 42) serta dapat diverifikasi ulang.
4. **Siap untuk pelatihan.** Dataset final (train 8.340 / validasi 1.471 / test 9.000) beserta bobot kelas dan statistik normalisasi telah ditetapkan di `configs/base.yaml`, menjadi dasar yang bersih dan terdokumentasi untuk eksperimen klasifikasi berikutnya.

**Batasan.** Kandidat near-duplicate dinilai dengan dHash dan tinjauan visual; metode ini kuat untuk citra identik yang dikompresi ulang, tetapi tidak dirancang mendeteksi kemiripan semantik tingkat tinggi. Ambang jarak Hamming ≤ 2 dipilih berdasarkan tinjauan visual pada dataset ini dan bukan tetapan universal.
