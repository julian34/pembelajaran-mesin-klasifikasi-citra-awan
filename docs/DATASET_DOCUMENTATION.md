# Dataset Documentation

Audit: 12 Oktober 2026 (Asia/Jakarta); diperbarui 13 Oktober 2026. Cakupan: workspace CitraAwan.
Status: audit isi citra selesai untuk GCD di `data/raw/GCD/`. Dokumen ini merinci **GCD** (dataset utama).

**Pembaruan 13 Oktober 2026 — dataset kedua.** Dataset **CCSN_v2** (`data/raw/CCSN_v2/`, 2.543 citra, 11 genera WMO) ditambahkan dan digabung ke skema 7 kelas GCD. EDA kedua dataset dijalankan ulang oleh `notebooks/01_dataset_analysis.ipynb` (tabel `ccsn_v2_raw_eda_*` dan `gcd_raw_eda_*`). Rincian penggabungan: [PENGGABUNGAN_DATASET.md](PENGGABUNGAN_DATASET.md); penanganan kualitas kedua dataset: [DATASET_QUALITY_SUMMARY.md](DATASET_QUALITY_SUMMARY.md) §4. Bagian di bawah (3–6) tetap khusus GCD.

Semua angka di bagian 3, 4, dan 6 adalah **hasil pengukuran lokal** oleh `scripts/audit_dataset.py`, bukan angka literatur. Bukti mentahnya:

- `reports/metrics/gcd_raw_audit_summary.json`: ringkasan audit.
- `reports/tables/gcd_raw_manifest.csv`: satu baris per citra (split, kelas, ukuran, format, mode, resolusi, SHA-256 berkas, hash piksel, dHash).
- `reports/tables/gcd_raw_class_distribution.csv`: distribusi kelas.
- `reports/tables/gcd_raw_duplicates.csv`: daftar citra identik.
- `reports/tables/gcd_raw_near_duplicates_across_splits.csv`: kandidat citra mirip antar split.

## 1. Dataset Overview

Dataset utama: **TJNU Ground-based Cloud Dataset (GCD)**, dataset klasifikasi single-label citra awan berbasis darat dengan 7 kelas — menjadi **skema acuan** untuk pelatihan, pemilihan, dan pengujian eksperimen di [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md). Dataset kedua **CCSN_v2** (11 genera WMO) dipetakan ke skema 7 kelas GCD dan ditambahkan ke train/validation; test tetap test resmi GCD (lihat [PENGGABUNGAN_DATASET.md](PENGGABUNGAN_DATASET.md)).

| Butir | Hasil audit lokal |
|---|---|
| Nama | TJNU Ground-based Cloud Dataset (GCD) |
| Jumlah kelas | 7 |
| Jumlah citra | 19.000 (train 10.000, test 9.000) |
| Resolusi | 512 x 512 piksel, seragam |
| Format | JPEG, ekstensi `.jpg`, seragam |
| Channel | 3 (RGB), seragam |
| Split bawaan | train dan test; **tidak ada validation** |
| Ukuran di disk | 198,8 MB terekstrak; arsip `GCD.zip` 183,7 MB |

Jumlah dan pembagian ini cocok dengan deskripsi repositori resmi GCD (19.000 citra JPEG 512 x 512; 10.000 train dan 9.000 test), yang dicatat pada versi dokumen sebelumnya. [Repositori resmi GCD](https://github.com/shuangliutjnu/TJNU-Ground-based-Cloud-Dataset).

Angka 12.000 citra yang disebut Li dkk. (*Sensors* 2023, Tabel 1) **tidak cocok** dengan data lokal, baik totalnya maupun jumlah per kelasnya. Angka paper itu tidak dipakai.

Sumber unduhan, tanggal unduh, dan ketentuan pemakaian arsip `GCD.zip` belum tercatat; perlu diisi oleh pemilik proyek. SHA-256 arsip: `8c1a1ab93dcac62247e34965f1f118089a4cf21ab12f7c626c54d13662950bd1`.

Dataset lain yang disebut di jurnal acuan (CCSN, SWIMCAT, HBMCD) tidak ada di proyek.

## 2. Dataset Location

Akar proyek: `D:/n8n-logsiswaparalayang/Jurnal/CitraAwan`.

| Lokasi relatif | Isi | Jumlah citra |
|---|---|---:|
| `data/raw/GCD.zip` | Arsip sumber | 19.000 anggota |
| `data/raw/GCD/train/<kelas>/` | Train resmi, hasil ekstraksi | 10.000 |
| `data/raw/GCD/test/<kelas>/` | Test resmi, hasil ekstraksi | 9.000 |
| `data/processed/GCD/test/1_cumulus/` | Salinan sebagian (lihat bagian 6) | 92 |
| `data/raw/CCSN_v2/<genus>/` | Dataset kedua, 11 genera WMO | 2.543 |
| `data/train/`, `data/val/`, `data/test/` | Split GCD final (hasil `scripts/materialize_split.py`) | 8.340 / 1.471 / 9.000 |

Struktur: `data/raw/GCD/<split>/<kelas>/<kelas>_<nomor 6 digit>.jpg`. Ketujuh folder kelas ada di kedua split dengan nama yang sama.

Pemetaan label. Nama folder hanya menyebut satu genus, sedangkan beberapa kelas adalah gabungan genus (definisi kelompok menurut sumber resmi dan Li dkk. 2023, hlm. 3):

| Indeks | Folder | Kelas | Singkatan |
|---:|---|---|---|
| 0 | `1_cumulus` | Cumulus | Cu |
| 1 | `2_altocumulus` | Altocumulus dan cirrocumulus | Ac |
| 2 | `3_cirrus` | Cirrus dan cirrostratus | Ci |
| 3 | `4_clearsky` | Clear sky | Cl |
| 4 | `5_stratocumulus` | Stratocumulus, stratus, dan altostratus | Sc |
| 5 | `6_cumulonimbus` | Cumulonimbus dan nimbostratus | Cb |
| 6 | `7_mixed` | Mixed | Mi |

Urutan alfabet nama folder sama dengan urutan indeks ini, sehingga pemuat berbasis folder akan menghasilkan indeks yang sama di train dan test. Pemetaan tetap harus ditulis eksplisit di konfigurasi.

## 3. Class Distribution

| Class | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| Cumulus (Cu) | 775 | - | 750 | 1.525 |
| Altocumulus dan cirrocumulus (Ac) | 725 | - | 750 | 1.475 |
| Cirrus dan cirrostratus (Ci) | 1.153 | - | 753 | 1.906 |
| Clear sky (Cl) | 2.150 | - | 1.589 | 3.739 |
| Stratocumulus, stratus, dan altostratus (Sc) | 1.846 | - | 1.790 | 3.636 |
| Cumulonimbus dan nimbostratus (Cb) | 3.003 | - | 2.761 | 5.764 |
| Mixed (Mi) | 348 | - | 607 | 955 |
| **Total** | **10.000** | **-** | **9.000** | **19.000** |

Kolom validation kosong karena dataset tidak menyediakannya.

Proporsi per split:

| Class | Train | Test |
|---|---:|---:|
| Cu | 7,75% | 8,33% |
| Ac | 7,25% | 8,33% |
| Ci | 11,53% | 8,37% |
| Cl | 21,50% | 17,66% |
| Sc | 18,46% | 19,89% |
| Cb | 30,03% | 30,68% |
| Mi | 3,48% | 6,74% |

Ketidakseimbangan kelas **ada**:

- Train: kelas terbesar (Cb, 3.003) 8,63 kali kelas terkecil (Mi, 348).
- Test: kelas terbesar (Cb, 2.761) 4,55 kali kelas terkecil (Mi, 607).
- Model yang selalu menebak Cb akan mendapat accuracy 30,68% pada test. Angka ini adalah batas bawah kewajaran untuk accuracy.

## 4. Image Characteristics

| Karakteristik | Hasil audit (19.000 citra) |
|---|---|
| Resolusi | 512 x 512 untuk semua citra |
| Variasi ukuran | Tidak ada; semua persegi, rasio aspek 1:1 |
| Format hasil decode | JPEG untuk semua citra; ekstensi `.jpg` sesuai isi |
| Mode warna | RGB, 3 channel, 8 bit per channel; tidak ada grayscale, RGBA, atau CMYK |
| Orientasi EXIF | Tidak ada citra yang memiliki tag orientasi |
| Ukuran berkas | 5,0 KB sampai 44,0 KB; median 8,7 KB |

Catatan:

- Ukuran berkas kecil untuk citra 512 x 512, artinya kompresi JPEG-nya kuat. Detail tekstur halus mungkin sudah berkurang di data sumber.
- 651 citra (357 train, 294 test) hampir seragam: tidak ada perbedaan kecerahan horizontal ketika dikecilkan ke 17 x 16 piksel. Sebagian besar clear sky (499), tetapi ada juga Sc (113), Cb (29), Ci (8), dan Ac (2). Citra langit mendung yang rata dan langit cerah yang rata berpotensi sulit dibedakan.

Preprocessing yang diperlukan (dilakukan di memori saat pemuatan, tanpa menulis ulang citra):

1. Decode sebagai RGB. Konversi mode tidak diperlukan karena semua citra sudah RGB.
2. Resize penuh dari 512 x 512 ke 224 x 224 (bilinear, antialias). Karena semua citra persegi, resize ini tidak mengubah rasio aspek dan tidak memerlukan crop.
3. Ubah ke tensor dan normalisasi dengan mean/std ImageNet sesuai bobot pretrained.
4. Penanganan orientasi EXIF tidak diperlukan untuk dataset ini.

Tidak ada resize, konversi, augmentasi, atau resampling yang dilakukan pada tahap audit.

## 5. Dataset Split

- **Train dan test berasal dari dataset resmi.** Keduanya sudah terpisah di dalam arsip. Penomoran berkas mendukung hal ini: di setiap kelas, train bernomor 1 sampai N dan test melanjutkan dari N+1 tanpa nomor yang hilang atau berulang.
- **Validation tidak disediakan** oleh dataset.
- **Tidak ada split baru yang dibuat** pada tahap audit. Train dan test resmi dipertahankan apa adanya.
- Validation harus diambil dari **train resmi**, tidak pernah dari test. Rancangannya ada di [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) bagian 2 dan baru dijalankan setelah disetujui.
- `configs/base.yaml` `data.split` sudah diperbarui: `val_fraction_of_train: 0.15` (test resmi tidak dibagi ulang), bukan lagi 70/15/15.
- Validation **sudah dibuat** dari train resmi bersih (stratified, seed 42) — train 8.340 / val 1.471 / test 9.000. Rincian: [DATASET_SPLIT.md](DATASET_SPLIT.md). Folder `data/train/`, `data/val/`, `data/test/` sudah diisi hasil split (`scripts/materialize_split.py`); manifest sumber kebenaran di `reports/tables/gcd_raw_split_*`.

## 6. Data Integrity

| Pemeriksaan | Hasil |
|---|---|
| Missing file terhadap arsip | Tidak ada. 19.000 anggota arsip, 19.000 berkas terekstrak, semua cocok CRC32 dan ukurannya |
| Berkas di luar arsip | Tidak ada di `data/raw/GCD/` |
| Unreadable image | Tidak ada. Semua citra lolos verifikasi header dan decoding penuh |
| Berkas 0 byte | Tidak ada |
| Berkas non-citra di folder dataset | Tidak ada |
| Format atau channel tidak konsisten | Tidak ada |
| Class mismatch (awalan nama berkas vs folder) | Tidak ada |
| Nama berkas berulang antar split | Tidak ada |
| Struktur folder | Konsisten: 7 folder kelas yang sama di train dan test |
| **Duplikat identik** | **159 pasang** (318 berkas); rincian di bawah |
| Duplikat lintas kelas (citra sama, label beda) | Tidak ada |
| Perubahan data saat audit | Tidak ada; jumlah berkas, total byte, dan waktu ubah terakhir di `data/` sama sebelum dan sesudah audit |

Rincian duplikat identik (isi berkas sama byte demi byte, sehingga pikselnya juga sama):

| Letak pasangan | Kelas | Jumlah pasang |
|---|---|---:|
| Train dan test | Clear sky | 155 |
| Train dan test | Mixed | 1 |
| Dalam test | Cumulus | 1 |
| Dalam test | Stratocumulus | 1 |
| Dalam train | Mixed | 1 |
| **Total** | | **159** |

Kandidat citra mirip tetapi tidak identik antar split (dHash 16 x 16, jarak Hamming paling banyak 10 dari 256 bit):

- Metode ini **tidak dapat diandalkan untuk citra yang hampir seragam**: citra langit rata menghasilkan hash yang hampir sama satu dengan lainnya, sehingga muncul 602.371 pasangan yang sebagian besar bukan duplikat.
- Jika dibatasi pada citra yang bertekstur, tersisa **29 pasangan kandidat** (27 clear sky dengan clear sky, 1 Sc dengan Sc, 1 Sc dengan clear sky). Pasangan ini belum ditinjau secara visual dan belum dapat disebut duplikat.
- Citra berurutan dari kamera atau sesi yang sama tidak dapat diperiksa karena tidak ada metadata waktu atau lokasi pengambilan.

`data/processed/GCD/test/1_cumulus/` berisi 92 citra (`1_cumulus_000776.jpg` sampai `1_cumulus_000867.jpg`) yang identik dengan berkas bernama sama di `data/raw/GCD/test/1_cumulus/`. Ini tampak sebagai ekstraksi yang terhenti di folder yang salah, bukan hasil praproses. Folder ini tidak dipakai dan tidak diubah.

## 7. Potential Dataset Issues

1. **Data leakage train-test (terbukti).** 156 citra test juga ada di train dengan label yang sama: 155 clear sky dan 1 mixed. Ini 1,73% dari test dan 9,75% dari kelas clear sky di test. Recall clear sky dan accuracy test akan sedikit terlalu optimistis jika model dilatih dengan salinan tersebut.
2. **Kandidat near-duplicate antar split (belum terbukti).** 29 pasangan perlu ditinjau secara visual. Di luar itu, ketiadaan metadata membuat kebocoran dari citra berurutan tidak dapat disingkirkan.
3. **Class imbalance.** Rasio 8,63 di train. Kelas Mixed hanya 348 citra train (3,48%), sehingga accuracy saja dapat menutupi kinerja buruk pada kelas ini.
4. **Distribusi kelas train dan test berbeda.** Mixed 3,48% di train tetapi 6,74% di test; Cirrus 11,53% di train tetapi 8,37% di test; Clear sky 21,50% tetapi 17,66%. Metrik per kelas dan macro F1 wajib dilaporkan di samping accuracy.
5. **Validation yang diambil dari train akan memperkecil kelas Mixed.** Dengan 20% untuk validation, Mixed hanya sekitar 70 citra di validation dan 278 di train. Validation macro F1 akan peka terhadap segelintir citra kelas ini.
6. **Duplikat di dalam train** (1 pasang Mixed) dapat terpisah ke train dan validation jika pembagian tidak menjaga pasangan duplikat tetap bersama.
7. **Nama folder menyederhanakan definisi kelas.** `2_altocumulus`, `3_cirrus`, `5_stratocumulus`, dan `6_cumulonimbus` adalah kelompok beberapa genus. Jangan menafsirkannya sebagai satu genus saat melaporkan hasil.
8. **Citra hampir seragam.** 651 citra, tersebar di clear sky dan kelas awan berlapis.
9. **Asal arsip belum terdokumentasi.** Sumber unduhan, tanggal, dan izin pemakaian belum dicatat.
10. **Salinan sebagian di `data/processed/GCD/`** dapat terbaca sebagai dataset oleh skrip yang menunjuk ke `data/processed/`.
11. **Perbedaan karakteristik antardataset.** Hanya ada satu dataset lokal, sehingga tidak ada perbandingan yang dapat diukur. Angka dari paper yang memakai CCSN, SWIMCAT, HBMCD, atau GCD versi 12.000 citra tidak dapat dibandingkan langsung dengan hasil di dataset ini.

Tidak ada masalah di atas yang diperbaiki secara otomatis. Usulan penanganan untuk butir 1, 5, dan 6 ada di [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) dan menunggu keputusan.

## 8. Dataset Usage Policy

- **Train** dipakai untuk pembelajaran parameter model dan, bila diperlukan, perhitungan statistik preprocessing atau bobot kelas.
- **Validation** dipakai untuk pemilihan model, hyperparameter, dan checkpoint terbaik, berdasarkan validation macro F1.
- **Test** resmi hanya dipakai pada evaluasi akhir, setelah semua keputusan dikunci.
- Test set **tidak boleh** dipakai untuk menentukan model terbaik, checkpoint terbaik, arsitektur, augmentasi, atau hyperparameter.
- Validation diambil dari train resmi. Test resmi tidak pernah dipakai sebagai validation dan tidak dibagi ulang.
- Split dan pemetaan label yang sama dipakai di semua eksperimen.
- Pemeriksaan terhadap test hanya untuk inventaris dan integritas (jumlah, format, hash), bukan untuk analisis kesalahan prediksi selama pengembangan.
- Isi `data/raw/` tidak dipindahkan, dihapus, diganti namanya, diaugmentasi, atau di-resampling. Pengecualian citra dari pelatihan dilakukan lewat daftar di manifest, bukan dengan menghapus berkas.
- Masalah dataset dicatat dulu di dokumen ini; perbaikannya memerlukan keputusan yang terdokumentasi.
