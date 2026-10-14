# Laporan EDA Awal

Dihasilkan oleh `notebooks/01_dataset_analysis.ipynb` (seed 42) dari isi `data/raw/` apa adanya. Tidak ada berkas dataset yang diubah, dipindah, atau dihapus. Angka di laporan ini adalah hasil perhitungan, kecuali bagian yang ditandai sebagai observasi visual. Tabel lengkap ada di `reports/tables/` dan grafik di `reports/figures/eda/`.

## Dataset yang ditemukan

| Dataset | Folder train | Folder validation | Folder test | Jumlah kelas | Nama kelas | Jumlah citra |
|---|---|---|---|---:|---|---:|
| CCSN_v2 | tidak ada | tidak ada | tidak ada | 11 | Ac, As, Cb, Cc, Ci, Cs, Ct, Cu, Ns, Sc, St | 2.543 |
| GCD | data/raw/GCD/train | tidak ada | data/raw/GCD/test | 7 | 1_cumulus, 2_altocumulus, 3_cirrus, 4_clearsky, 5_stratocumulus, 6_cumulonimbus, 7_mixed | 19.000 |

- **CCSN_v2**: 11 kelas; daftar folder kelas sama di semua split. Split yang tidak tersedia: **Train, Validation, Test**.

- **GCD**: 7 kelas; daftar folder kelas sama di semua split. Split yang tidak tersedia: **Validation**.

## Distribusi data: CCSN_v2

| Class | Semua | Total |
|---|---:|---:|
| Ac | 221 | 221 |
| As | 188 | 188 |
| Cb | 242 | 242 |
| Cc | 268 | 268 |
| Ci | 139 | 139 |
| Cs | 287 | 287 |
| Ct | 200 | 200 |
| Cu | 182 | 182 |
| Ns | 274 | 274 |
| Sc | 340 | 340 |
| St | 202 | 202 |
| Total | 2.543 | 2.543 |

| Split | Jumlah citra | Kelas terbesar | n terbesar | Kelas terkecil | n terkecil | Rasio terbesar/terkecil |
|---|---:|---|---:|---|---:|---:|
| Semua | 2.543 | Sc | 340 | Ci | 139 | 2,45 |
| Semua split | 2.543 | Sc | 340 | Ci | 139 | 2,45 |

## Distribusi data: GCD

| Class | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| 1_cumulus | 775 | 0 | 750 | 1.525 |
| 2_altocumulus | 725 | 0 | 750 | 1.475 |
| 3_cirrus | 1.153 | 0 | 753 | 1.906 |
| 4_clearsky | 2.150 | 0 | 1.589 | 3.739 |
| 5_stratocumulus | 1.846 | 0 | 1.790 | 3.636 |
| 6_cumulonimbus | 3.003 | 0 | 2.761 | 5.764 |
| 7_mixed | 348 | 0 | 607 | 955 |
| Total | 10.000 | 0 | 9.000 | 19.000 |

| Split | Jumlah citra | Kelas terbesar | n terbesar | Kelas terkecil | n terkecil | Rasio terbesar/terkecil |
|---|---:|---|---:|---|---:|---:|
| Train | 10.000 | 6_cumulonimbus | 3.003 | 7_mixed | 348 | 8,63 |
| Test | 9.000 | 6_cumulonimbus | 2.761 | 7_mixed | 607 | 4,55 |
| Semua split | 19.000 | 6_cumulonimbus | 5.764 | 7_mixed | 955 | 6,04 |

## Pemeriksaan file: CCSN_v2

| Pemeriksaan | Jumlah |
|---|---:|
| Total berkas di folder dataset | 2.543 |
| Berkas berekstensi citra | 2.543 |
| Berkas non-citra | 0 |
| Berkas 0 byte (kosong) | 0 |
| Citra tidak dapat dibaca (gagal verifikasi atau decoding penuh) | 0 |
| Ekstensi tidak sesuai format hasil decode | 0 |

| Ekstensi | Jumlah berkas | Total byte | Byte terkecil | Byte median | Byte terbesar |
|---|---:|---:|---:|---:|---:|
| .jpg | 2.543 | 104.175.593 | 5.310 | 44.419 | 124.119 |

Tidak ada berkas bermasalah: tidak ada berkas non-citra, kosong, rusak, atau salah ekstensi.

## Pemeriksaan file: GCD

| Pemeriksaan | Jumlah |
|---|---:|
| Total berkas di folder dataset | 19.000 |
| Berkas berekstensi citra | 19.000 |
| Berkas non-citra | 0 |
| Berkas 0 byte (kosong) | 0 |
| Citra tidak dapat dibaca (gagal verifikasi atau decoding penuh) | 0 |
| Ekstensi tidak sesuai format hasil decode | 0 |

| Ekstensi | Jumlah berkas | Total byte | Byte terkecil | Byte median | Byte terbesar |
|---|---:|---:|---:|---:|---:|
| .jpg | 19.000 | 198.811.777 | 5.025 | 8.650 | 44.045 |

Tidak ada berkas bermasalah: tidak ada berkas non-citra, kosong, rusak, atau salah ekstensi.

## Dimensi citra: CCSN_v2

| Dimensi | Min | Max | Mean | Median |
|---|---:|---:|---:|---:|
| Width (piksel) | 256,00 | 400,00 | 388,05 | 400,00 |
| Height (piksel) | 256,00 | 400,00 | 388,05 | 400,00 |
| Aspect ratio (width/height) | 1,00 | 1,00 | 1,00 | 1,00 |

| Resolusi | Jumlah citra |
|---|---:|
| 400 x 400 | 2.332 |
| 256 x 256 | 211 |

| Mode warna | Jumlah channel | Jumlah citra |
|---|---:|---:|
| RGB | 3 | 2.543 |

- Seluruh citra berukuran sama: **tidak** (2 resolusi unik).
- Variasi resolusi: **ada**.
- Citra grayscale (1 channel): **0**.
- Citra dengan alpha (RGBA/LA): **0**.

## Dimensi citra: GCD

| Dimensi | Min | Max | Mean | Median |
|---|---:|---:|---:|---:|
| Width (piksel) | 512,00 | 512,00 | 512,00 | 512,00 |
| Height (piksel) | 512,00 | 512,00 | 512,00 | 512,00 |
| Aspect ratio (width/height) | 1,00 | 1,00 | 1,00 | 1,00 |

| Resolusi | Jumlah citra |
|---|---:|
| 512 x 512 | 19.000 |

| Mode warna | Jumlah channel | Jumlah citra |
|---|---:|---:|
| RGB | 3 | 19.000 |

- Seluruh citra berukuran sama: **ya** (1 resolusi unik).
- Variasi resolusi: **tidak ada**.
- Citra grayscale (1 channel): **0**.
- Citra dengan alpha (RGBA/LA): **0**.

## Statistik piksel: CCSN_v2

Mean, Std, Min, dan Max pada skala asli 0-255, dari seluruh piksel:

| Cakupan | Channel | Mean | Std | Min | Max | Mean (skala 0-1) | Std (skala 0-1) |
|---|---|---:|---:|---:|---:|---|---|
| Semua split | Red | 122,30 | 66,89 | 0 | 255 | 0,4796 | 0,2623 |
| Semua split | Green | 133,22 | 60,77 | 0 | 255 | 0,5224 | 0,2383 |
| Semua split | Blue | 142,75 | 65,27 | 0 | 255 | 0,5598 | 0,2560 |
| Semua | Red | 122,30 | 66,89 | 0 | 255 | 0,4796 | 0,2623 |
| Semua | Green | 133,22 | 60,77 | 0 | 255 | 0,5224 | 0,2383 |
| Semua | Blue | 142,75 | 65,27 | 0 | 255 | 0,5598 | 0,2560 |

## Statistik piksel: GCD

Mean, Std, Min, dan Max pada skala asli 0-255, dari seluruh piksel:

| Cakupan | Channel | Mean | Std | Min | Max | Mean (skala 0-1) | Std (skala 0-1) |
|---|---|---:|---:|---:|---:|---|---|
| Semua split | Red | 123,05 | 51,97 | 0 | 255 | 0,4826 | 0,2038 |
| Semua split | Green | 156,38 | 32,95 | 0 | 255 | 0,6132 | 0,1292 |
| Semua split | Blue | 193,56 | 35,81 | 0 | 255 | 0,7591 | 0,1404 |
| Train | Red | 123,08 | 49,85 | 0 | 255 | 0,4827 | 0,1955 |
| Train | Green | 156,93 | 32,70 | 0 | 255 | 0,6154 | 0,1282 |
| Train | Blue | 195,63 | 34,94 | 0 | 255 | 0,7672 | 0,1370 |
| Test | Red | 123,03 | 54,24 | 0 | 255 | 0,4825 | 0,2127 |
| Test | Green | 155,76 | 33,21 | 0 | 255 | 0,6108 | 0,1302 |
| Test | Blue | 191,26 | 36,62 | 0 | 255 | 0,7500 | 0,1436 |

## Brightness dan contrast: CCSN_v2

| Class | n | Brightness mean | Brightness std | Brightness median | Contrast mean | Contrast std | Contrast median |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ac | 221 | 132,24 | 33,16 | 131,84 | 44,47 | 14,59 | 44,24 |
| As | 188 | 119,91 | 39,42 | 122,56 | 44,32 | 12,69 | 44,87 |
| Cb | 242 | 110,18 | 30,10 | 112,16 | 51,32 | 12,53 | 51,12 |
| Cc | 268 | 144,57 | 26,48 | 142,28 | 41,03 | 14,19 | 39,99 |
| Ci | 139 | 130,37 | 25,54 | 129,16 | 40,95 | 13,41 | 41,64 |
| Cs | 287 | 142,66 | 30,38 | 142,89 | 47,46 | 17,30 | 48,14 |
| Ct | 200 | 120,17 | 28,56 | 127,34 | 34,36 | 13,83 | 34,19 |
| Cu | 182 | 126,39 | 29,45 | 123,32 | 47,72 | 12,62 | 47,58 |
| Ns | 274 | 132,11 | 32,78 | 132,94 | 49,97 | 13,79 | 50,33 |
| Sc | 340 | 132,91 | 39,08 | 138,79 | 55,31 | 17,60 | 54,04 |
| St | 202 | 144,36 | 33,16 | 145,89 | 52,76 | 16,33 | 52,72 |

- Brightness: rata-rata kelas dari 110,18 (Cb) sampai 144,57 (Cc); η² = 0,096.
- Contrast: rata-rata kelas dari 34,36 (Ct) sampai 55,31 (Sc); η² = 0,135.

## Brightness dan contrast: GCD

| Class | n | Brightness mean | Brightness std | Brightness median | Contrast mean | Contrast std | Contrast median |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1_cumulus | 1.525 | 139,73 | 19,71 | 140,98 | 42,23 | 12,70 | 44,16 |
| 2_altocumulus | 1.475 | 155,40 | 24,56 | 158,07 | 26,14 | 11,06 | 24,56 |
| 3_cirrus | 1.906 | 143,33 | 21,71 | 144,54 | 22,62 | 10,53 | 21,92 |
| 4_clearsky | 3.739 | 122,62 | 24,93 | 122,39 | 12,58 | 7,44 | 11,29 |
| 5_stratocumulus | 3.636 | 171,40 | 19,16 | 170,46 | 11,75 | 6,06 | 10,61 |
| 6_cumulonimbus | 5.764 | 158,64 | 18,45 | 159,92 | 17,99 | 8,51 | 16,51 |
| 7_mixed | 955 | 157,89 | 17,35 | 158,29 | 29,68 | 11,69 | 29,82 |

- Brightness: rata-rata kelas dari 122,62 (4_clearsky) sampai 171,40 (5_stratocumulus); η² = 0,387.
- Contrast: rata-rata kelas dari 11,75 (5_stratocumulus) sampai 42,23 (1_cumulus); η² = 0,475.

Perbandingan train dan test per kelas. `d` = selisih rata-rata (test − train) dibagi simpangan baku gabungan (Cohen's d); |d| ≥ 0,5 berarti pergeseran sedang, |d| ≥ 0,8 besar.

| Class | Brightness train | Brightness test | d brightness | Contrast train | Contrast test | d contrast |
|---|---:|---:|---:|---:|---:|---:|
| 1_cumulus | 141,25 | 138,15 | -0,16 | 47,40 | 36,89 | -0,91 |
| 2_altocumulus | 153,37 | 157,36 | 0,16 | 28,26 | 24,09 | -0,38 |
| 3_cirrus | 147,01 | 137,69 | -0,44 | 21,19 | 24,80 | 0,35 |
| 4_clearsky | 127,26 | 116,34 | -0,45 | 12,57 | 12,60 | 0,00 |
| 5_stratocumulus | 172,71 | 170,06 | -0,14 | 12,21 | 11,28 | -0,15 |
| 6_cumulonimbus | 157,92 | 159,42 | 0,08 | 19,19 | 16,69 | -0,30 |
| 7_mixed | 159,02 | 157,24 | -0,10 | 32,50 | 28,07 | -0,39 |

## Ketidakseimbangan kelas: CCSN_v2

| Class | Samples | Percentage |
|---|---:|---:|
| Ac | 221 | 8,69 |
| As | 188 | 7,39 |
| Cb | 242 | 9,52 |
| Cc | 268 | 10,54 |
| Ci | 139 | 5,47 |
| Cs | 287 | 11,29 |
| Ct | 200 | 7,86 |
| Cu | 182 | 7,16 |
| Ns | 274 | 10,77 |
| Sc | 340 | 13,37 |
| St | 202 | 7,94 |

Proporsi per split:

| Class | Semua n | Semua % |
|---|---:|---:|
| Ac | 221 | 8,69 |
| As | 188 | 7,39 |
| Cb | 242 | 9,52 |
| Cc | 268 | 10,54 |
| Ci | 139 | 5,47 |
| Cs | 287 | 11,29 |
| Ct | 200 | 7,86 |
| Cu | 182 | 7,16 |
| Ns | 274 | 10,77 |
| Sc | 340 | 13,37 |
| St | 202 | 7,94 |

| Split | Kelas terbesar | n terbesar | Kelas terkecil | n terkecil | Rasio terbesar/terkecil | Tingkat imbalance |
|---|---|---:|---|---:|---:|---|
| Semua | Sc | 340 | Ci | 139 | 2,45 | ringan |
| Semua split | Sc | 340 | Ci | 139 | 2,45 | ringan |

## Ketidakseimbangan kelas: GCD

| Class | Samples | Percentage |
|---|---:|---:|
| 1_cumulus | 1.525 | 8,03 |
| 2_altocumulus | 1.475 | 7,76 |
| 3_cirrus | 1.906 | 10,03 |
| 4_clearsky | 3.739 | 19,68 |
| 5_stratocumulus | 3.636 | 19,14 |
| 6_cumulonimbus | 5.764 | 30,34 |
| 7_mixed | 955 | 5,03 |

Proporsi per split:

| Class | Train n | Train % | Test n | Test % |
|---|---:|---:|---:|---:|
| 1_cumulus | 775 | 7,75 | 750 | 8,33 |
| 2_altocumulus | 725 | 7,25 | 750 | 8,33 |
| 3_cirrus | 1.153 | 11,53 | 753 | 8,37 |
| 4_clearsky | 2.150 | 21,50 | 1.589 | 17,66 |
| 5_stratocumulus | 1.846 | 18,46 | 1.790 | 19,89 |
| 6_cumulonimbus | 3.003 | 30,03 | 2.761 | 30,68 |
| 7_mixed | 348 | 3,48 | 607 | 6,74 |

| Split | Kelas terbesar | n terbesar | Kelas terkecil | n terkecil | Rasio terbesar/terkecil | Tingkat imbalance |
|---|---|---:|---|---:|---:|---|
| Train | 6_cumulonimbus | 3.003 | 7_mixed | 348 | 8,63 | sedang |
| Test | 6_cumulonimbus | 2.761 | 7_mixed | 607 | 4,55 | sedang |
| Semua split | 6_cumulonimbus | 5.764 | 7_mixed | 955 | 6,04 | sedang |

## Duplikasi: CCSN_v2

| Pemeriksaan | Status | Pasangan berkas identik | Pasangan piksel identik | Kandidat near-duplicate |
|---|---|---:|---:|---:|
| Dalam Semua | diperiksa | 17 | 17 | 130 |
| Train dan Validation | tidak dapat diperiksa (split Train, Validation tidak ada) | - | - | - |
| Train dan Test | tidak dapat diperiksa (split Train, Test tidak ada) | - | - | - |
| Validation dan Test | tidak dapat diperiksa (split Validation, Test tidak ada) | - | - | - |

Rincian pasangan berkas identik menurut letak dan kelas:

| Letak pasangan | Kelas | Jumlah pasangan |
|---|---|---:|
| dalam all | Ac / As | 2 |
| dalam all | As | 1 |
| dalam all | Cb | 6 |
| dalam all | Cc | 2 |
| dalam all | Cc / Cs | 1 |
| dalam all | Cu | 5 |

- Pasangan berkas identik: **17**; berkas yang terlibat: 34.
- Pasangan piksel identik yang berkasnya berbeda (disimpan ulang): **0**.
- Pasangan identik dengan label kelas berbeda: **3**.
- Citra dengan dHash informatif: 2.456 dari 2.543; sisanya (87) tidak dapat dinilai dengan metode ini.
- Kandidat near-duplicate (jarak Hamming ≤ 10, bukan piksel identik): **130** pasangan, 0 di antaranya lintas split.

Kandidat near-duplicate menurut letak dan kelas:

| Letak pasangan | Kelas | Jumlah pasangan | Jarak Hamming terkecil | Jarak Hamming terbesar |
|---|---|---:|---:|---:|
| dalam all | Ac / As | 1 | 4 | 4 |
| dalam all | Ac / Cc | 6 | 0 | 6 |
| dalam all | Ac / Ns | 1 | 8 | 8 |
| dalam all | Ac / Sc | 1 | 4 | 4 |
| dalam all | As / Cb | 4 | 0 | 5 |
| dalam all | As / Cc | 1 | 2 | 2 |
| dalam all | As / Ci | 2 | 3 | 6 |
| dalam all | As / Cs | 2 | 9 | 10 |
| dalam all | As / Cu | 2 | 3 | 3 |
| dalam all | As / Ns | 1 | 3 | 3 |
| dalam all | As / Sc | 1 | 10 | 10 |
| dalam all | As / St | 3 | 1 | 2 |
| dalam all | Cb / Ci | 1 | 3 | 3 |
| dalam all | Cb / Cu | 10 | 1 | 3 |
| dalam all | Cc | 2 | 4 | 5 |
| dalam all | Cc / Ci | 1 | 2 | 2 |
| dalam all | Cc / Cs | 12 | 0 | 4 |
| dalam all | Cc / Ns | 2 | 0 | 4 |
| dalam all | Cc / St | 3 | 1 | 3 |
| dalam all | Ci / Cs | 4 | 0 | 3 |
| dalam all | Ci / Ct | 1 | 4 | 4 |
| dalam all | Ci / Cu | 2 | 2 | 2 |
| dalam all | Ci / St | 2 | 2 | 4 |
| dalam all | Cs | 7 | 2 | 10 |
| dalam all | Cs / Cu | 4 | 3 | 10 |

*Menampilkan 25 dari 34 baris.*

## Duplikasi: GCD

| Pemeriksaan | Status | Pasangan berkas identik | Pasangan piksel identik | Kandidat near-duplicate |
|---|---|---:|---:|---:|
| Dalam Train | diperiksa | 1 | 1 | 3 |
| Dalam Validation | tidak dapat diperiksa (split Validation tidak ada) | - | - | - |
| Dalam Test | diperiksa | 2 | 2 | 6 |
| Train dan Validation | tidak dapat diperiksa (split Validation tidak ada) | - | - | - |
| Train dan Test | diperiksa | 156 | 156 | 36 |
| Validation dan Test | tidak dapat diperiksa (split Validation tidak ada) | - | - | - |

Rincian pasangan berkas identik menurut letak dan kelas:

| Letak pasangan | Kelas | Jumlah pasangan |
|---|---|---:|
| dalam test | 1_cumulus | 1 |
| dalam test | 5_stratocumulus | 1 |
| dalam train | 7_mixed | 1 |
| train-test | 4_clearsky | 155 |
| train-test | 7_mixed | 1 |

- Pasangan berkas identik: **159**; berkas yang terlibat: 318.
- Pasangan piksel identik yang berkasnya berbeda (disimpan ulang): **0**.
- Pasangan identik dengan label kelas berbeda: **0**.
- Citra dengan dHash informatif: 14.648 dari 19.000; sisanya (4.352) tidak dapat dinilai dengan metode ini.
- Kandidat near-duplicate (jarak Hamming ≤ 10, bukan piksel identik): **45** pasangan, 36 di antaranya lintas split.

Kandidat near-duplicate menurut letak dan kelas:

| Letak pasangan | Kelas | Jumlah pasangan | Jarak Hamming terkecil | Jarak Hamming terbesar |
|---|---|---:|---:|---:|
| dalam test | 1_cumulus | 2 | 3 | 5 |
| dalam test | 4_clearsky / 5_stratocumulus | 2 | 7 | 10 |
| dalam test | 6_cumulonimbus | 1 | 8 | 8 |
| dalam test | 7_mixed | 1 | 10 | 10 |
| dalam train | 1_cumulus | 1 | 5 | 5 |
| dalam train | 2_altocumulus / 5_stratocumulus | 1 | 4 | 4 |
| dalam train | 7_mixed | 1 | 8 | 8 |
| train-test | 3_cirrus | 1 | 8 | 8 |
| train-test | 4_clearsky | 32 | 0 | 2 |
| train-test | 4_clearsky / 5_stratocumulus | 2 | 7 | 9 |
| train-test | 5_stratocumulus | 1 | 6 | 6 |

**Observasi visual** (subjektif; dari 6 pasangan pada gambar di atas, bukan hasil perhitungan): Keenam pasangan lintas split yang ditampilkan (semuanya 4_clearsky, jarak Hamming 0 sampai 1) tampak sebagai citra yang sama, termasuk satu pasangan dengan silau matahari di posisi yang sama. Hash pikselnya berbeda, sehingga kemungkinan besar ini citra yang sama yang disimpan atau dikompresi ulang. Kandidat lain di tabel belum ditinjau.

## Potensi data leakage: CCSN_v2

Hanya satu split tersedia, sehingga pemeriksaan lintas split tidak dapat dilakukan.

**Nama file yang sama antar split: 0 pasangan** (tidak ada).

**Hash berkas (SHA-256) yang sama antar split: 0 pasangan** (tidak ada).

**Citra identik (piksel sama) antar split: 0 pasangan** (tidak ada).

## Potensi data leakage: GCD

**Nama file yang sama antar split: 0 pasangan** (tidak ada).

**Hash berkas (SHA-256) yang sama antar split: 156 pasangan** (train-test: 156).

| File A | Split A | File B | Split B | Detection Method |
|---|---|---|---|---|
| train/4_clearsky/4_clearsky_001900.jpg | train | test/4_clearsky/4_clearsky_003289.jpg | test | SHA-256 isi berkas |
| train/4_clearsky/4_clearsky_001918.jpg | train | test/4_clearsky/4_clearsky_003310.jpg | test | SHA-256 isi berkas |
| train/4_clearsky/4_clearsky_001930.jpg | train | test/4_clearsky/4_clearsky_003324.jpg | test | SHA-256 isi berkas |
| train/4_clearsky/4_clearsky_001098.jpg | train | test/4_clearsky/4_clearsky_003426.jpg | test | SHA-256 isi berkas |
| train/4_clearsky/4_clearsky_000923.jpg | train | test/4_clearsky/4_clearsky_003238.jpg | test | SHA-256 isi berkas |
| train/4_clearsky/4_clearsky_000914.jpg | train | test/4_clearsky/4_clearsky_003225.jpg | test | SHA-256 isi berkas |
| train/4_clearsky/4_clearsky_001940.jpg | train | test/4_clearsky/4_clearsky_003335.jpg | test | SHA-256 isi berkas |
| train/4_clearsky/4_clearsky_002013.jpg | train | test/4_clearsky/4_clearsky_003403.jpg | test | SHA-256 isi berkas |
| train/4_clearsky/4_clearsky_001981.jpg | train | test/4_clearsky/4_clearsky_003370.jpg | test | SHA-256 isi berkas |
| train/4_clearsky/4_clearsky_001249.jpg | train | test/4_clearsky/4_clearsky_003482.jpg | test | SHA-256 isi berkas |

*Menampilkan 10 dari 156 baris.*

**Citra identik (piksel sama) antar split: 156 pasangan** (train-test: 156).

| File A | Split A | File B | Split B | Detection Method |
|---|---|---|---|---|
| train/4_clearsky/4_clearsky_001064.jpg | train | test/4_clearsky/4_clearsky_003411.jpg | test | hash piksel |
| train/4_clearsky/4_clearsky_001252.jpg | train | test/4_clearsky/4_clearsky_003485.jpg | test | hash piksel |
| train/4_clearsky/4_clearsky_000948.jpg | train | test/4_clearsky/4_clearsky_003285.jpg | test | hash piksel |
| train/4_clearsky/4_clearsky_000901.jpg | train | test/4_clearsky/4_clearsky_003201.jpg | test | hash piksel |
| train/4_clearsky/4_clearsky_001217.jpg | train | test/4_clearsky/4_clearsky_003474.jpg | test | hash piksel |
| train/4_clearsky/4_clearsky_001157.jpg | train | test/4_clearsky/4_clearsky_003464.jpg | test | hash piksel |
| train/4_clearsky/4_clearsky_001911.jpg | train | test/4_clearsky/4_clearsky_003303.jpg | test | hash piksel |
| train/4_clearsky/4_clearsky_001247.jpg | train | test/4_clearsky/4_clearsky_003480.jpg | test | hash piksel |
| train/4_clearsky/4_clearsky_001153.jpg | train | test/4_clearsky/4_clearsky_003459.jpg | test | hash piksel |
| train/4_clearsky/4_clearsky_001155.jpg | train | test/4_clearsky/4_clearsky_003462.jpg | test | hash piksel |

*Menampilkan 10 dari 156 baris.*

Citra test yang salinan identiknya ada di split lain: **156** dari 9.000 (1,73% test). Per kelas: 4_clearsky: 155 dari 1.589 citra test kelas itu (9,75%); 7_mixed: 1 dari 607 citra test kelas itu (0,16%).

## Analisis per kelas: CCSN_v2

**Hasil perhitungan** (brightness dan contrast pada skala 0-255; saturation pada skala 0-1; kelas terdekat dihitung dari fitur warna tingkat rendah pada split train):

| Class | Jumlah citra | Bagian (%) | Brightness | Contrast | Saturation | Rasio R/B | Citra hampir seragam | Citra dalam duplikat identik | Citra dalam kandidat near-duplicate | Kelas terdekat (fitur warna) | Jarak centroid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| Ac | 221 | 8,69 | 132,24 | 44,47 | 0,32 | 1,04 | 0 | 2 | 9 | Cs | 0,66 |
| As | 188 | 7,39 | 119,91 | 44,32 | 0,32 | 1,62 | 0 | 4 | 17 | Cb | 1,11 |
| Cb | 242 | 9,52 | 110,18 | 51,32 | 0,29 | 0,97 | 0 | 12 | 15 | Cu | 1,06 |
| Cc | 268 | 10,54 | 144,57 | 41,03 | 0,34 | 0,78 | 0 | 5 | 27 | Cs | 1,04 |
| Ci | 139 | 5,47 | 130,37 | 40,95 | 0,39 | 0,76 | 0 | 0 | 12 | Cu | 0,72 |
| Cs | 287 | 11,29 | 142,66 | 47,46 | 0,29 | 0,90 | 0 | 1 | 38 | St | 0,62 |
| Ct | 200 | 7,86 | 120,17 | 34,36 | 0,46 | 0,67 | 0 | 0 | 7 | Ci | 0,85 |
| Cu | 182 | 7,16 | 126,39 | 47,72 | 0,34 | 0,90 | 0 | 10 | 18 | Ac | 0,71 |
| Ns | 274 | 10,77 | 132,11 | 49,97 | 0,21 | 0,97 | 0 | 0 | 35 | Sc | 0,50 |
| Sc | 340 | 13,37 | 132,91 | 55,31 | 0,23 | 1,01 | 0 | 0 | 25 | Ns | 0,50 |
| St | 202 | 7,94 | 144,36 | 52,76 | 0,22 | 0,97 | 0 | 0 | 41 | Cs | 0,62 |

**Observasi visual**: belum diisi untuk dataset ini.

## Analisis per kelas: GCD

**Hasil perhitungan** (brightness dan contrast pada skala 0-255; saturation pada skala 0-1; kelas terdekat dihitung dari fitur warna tingkat rendah pada split train):

| Class | Jumlah citra | Bagian (%) | Brightness | Contrast | Saturation | Rasio R/B | Citra hampir seragam | Citra dalam duplikat identik | Citra dalam kandidat near-duplicate | Kelas terdekat (fitur warna) | Jarak centroid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 1_cumulus | 1.525 | 8,03 | 139,73 | 42,23 | 0,48 | 0,55 | 0 | 2 | 6 | 7_mixed | 1,87 |
| 2_altocumulus | 1.475 | 7,76 | 155,40 | 26,14 | 0,36 | 0,65 | 0 | 0 | 1 | 7_mixed | 1,08 |
| 3_cirrus | 1.906 | 10,03 | 143,33 | 22,62 | 0,41 | 0,60 | 0 | 0 | 2 | 6_cumulonimbus | 1,43 |
| 4_clearsky | 3.739 | 19,68 | 122,62 | 12,58 | 0,67 | 0,33 | 82 | 310 | 66 | 3_cirrus | 2,24 |
| 5_stratocumulus | 3.636 | 19,14 | 171,40 | 11,75 | 0,17 | 0,84 | 33 | 2 | 5 | 6_cumulonimbus | 1,23 |
| 6_cumulonimbus | 5.764 | 30,34 | 158,64 | 17,99 | 0,21 | 0,80 | 2 | 0 | 2 | 5_stratocumulus | 1,23 |
| 7_mixed | 955 | 5,03 | 157,89 | 29,68 | 0,31 | 0,70 | 0 | 4 | 4 | 2_altocumulus | 1,08 |

**Observasi visual** (subjektif; dari 5 sampel per kelas pada gambar bagian 6, seed 42; bukan hasil perhitungan dan bukan kesimpulan klasifikasi):

| Class | Karakteristik visual umum | Variasi tampilan | Kualitas citra | Potensi overlap (kesan visual) |
|---|---|---|---|---|
| 1_cumulus | Gumpalan putih tebal bertepi tegas di atas langit biru pekat | Cakupan awan dari sekitar sepertiga sampai hampir seluruh bingkai; satu sampel memiliki dasar awan abu-abu | Tajam; tidak terlihat noise atau artefak yang mengganggu | 7_mixed (salah satu sampel mixed tampak seperti cumulus) |
| 2_altocumulus | Gumpalan kecil berulang yang menutupi hampir seluruh bingkai | Dari tipis dan terang sampai tebal dengan celah gelap; satu sampel pucat dan berkontras rendah | Tajam; tidak terlihat noise atau artefak yang mengganggu | 7_mixed; sampel yang pucat mendekati lapisan awan rata |
| 3_cirrus | Serat atau selubung tipis di atas langit biru | Sangat lebar: dari serat jelas sampai dua sampel yang hampir tanpa struktur | Baik; awan sangat tipis sehingga detailnya samar | 4_clearsky dan 5_stratocumulus pada sampel yang hampir polos (mis. 3_cirrus_000576, 3_cirrus_000211) |
| 4_clearsky | Biru polos atau gradasi halus, hampir tanpa tekstur | Rona biru berbeda-beda; satu sampel kebiruan-ungu pucat | Baik; hampir tidak ada detail untuk dinilai | 3_cirrus yang sangat tipis |
| 5_stratocumulus | Lapisan abu-abu atau biru pucat yang menutupi seluruh bingkai | Tiga sampel abu-abu rata nyaris tanpa tekstur; dua sampel kebiruan dengan tekstur halus | Baik; kontras rendah | 6_cumulonimbus pada sampel bertekstur; langit pucat pada sampel rata |
| 6_cumulonimbus | Lapisan tebal abu-biru bertekstur lembut yang menutupi seluruh bingkai | Cukup seragam antar sampel; berbeda pada tingkat kegelapan | Baik; kontras rendah sampai sedang | 5_stratocumulus |
| 7_mixed | Beberapa bentuk awan dalam satu bingkai | Paling beragam; tiap sampel tampak berbeda satu sama lain | Tajam; tidak terlihat noise atau artefak yang mengganggu | 1_cumulus dan 2_altocumulus (mis. 7_mixed_000054 tampak seperti cumulus) |

## Dataset Quality Summary: CCSN_v2

| Aspect | Status | Finding |
|---|---|---|
| Class distribution | OK | Rasio terbesar/terkecil tertinggi 2,45 pada Semua (Sc 340 vs Ci 139) |
| Image dimensions | Warning | 2 resolusi berbeda |
| Corrupted images | OK | 0 tidak terbaca, 0 berkas 0 byte, 0 berkas non-citra |
| Duplicate files | Warning | 17 pasangan citra identik; 3 dengan label berbeda |
| Cross-split leakage | OK | 0 pasangan identik lintas split; 0 kandidat near-duplicate lintas split; 0 nama file sama |
| Channel consistency | OK | Semua citra RGB 3 channel |
| Image quality | OK | 0 citra hampir seragam (contrast < 2,55 dari 255), 2 sangat gelap, 0 sangat terang |
| Split availability | Warning | Split tidak tersedia: Train, Validation, Test |

## Dataset Quality Summary: GCD

| Aspect | Status | Finding |
|---|---|---|
| Class distribution | Warning | Rasio terbesar/terkecil tertinggi 8,63 pada Train (6_cumulonimbus 3.003 vs 7_mixed 348) |
| Image dimensions | OK | Semua citra 512 x 512 |
| Corrupted images | OK | 0 tidak terbaca, 0 berkas 0 byte, 0 berkas non-citra |
| Duplicate files | Warning | 159 pasangan citra identik; 0 dengan label berbeda |
| Cross-split leakage | Warning | 156 pasangan identik lintas split; 36 kandidat near-duplicate lintas split; 0 nama file sama |
| Channel consistency | OK | Semua citra RGB 3 channel |
| Image quality | OK | 117 citra hampir seragam (contrast < 2,55 dari 255), 1 sangat gelap, 0 sangat terang |
| Split availability | Warning | Split tidak tersedia: Validation |

## EDA Conclusion: CCSN_v2

### Dataset Characteristics

- 2.543 citra dalam 11 kelas (semua 2.543).
- Resolusi: 2 resolusi berbeda; mode warna: RGB 3 channel untuk semua citra.
- Mean RGB semua split (skala 0-1): 0,4796, 0,5224, 0,5598; std: 0,2623, 0,2383, 0,2560.
- Kelas terbesar Sc dan terkecil Ci; rasio tertinggi 2,45 pada Semua (imbalance ringan).
- Perbedaan antar kelas: η² brightness 0,096, η² contrast 0,135.

### Issues Found

- Split Train, Validation, Test tidak tersedia di `data/raw/CCSN_v2/`.
- 17 pasangan citra identik berada di dalam split yang sama.
- 3 pasangan citra identik memiliki label berbeda.
- 130 kandidat near-duplicate (0 lintas split) belum dapat dipastikan tanpa tinjauan visual.

### Implications for Preprocessing

- Resize diperlukan karena resolusi tidak konsisten; tentukan cara menjaga aspect ratio.
- Normalisasi: bila tidak memakai statistik bobot pretrained, gunakan mean/std dari semua split saja (mean 0,4796, 0,5224, 0,5598; std 0,2623, 0,2383, 0,2560), bukan dari test.
- Saat validation dibuat dari train, pasangan duplikat dan kandidat near-duplicate di dalam train harus tetap berada di sisi yang sama.
- Split validation perlu dibuat dari train resmi (stratified, seed tetap) pada tahap berikutnya; test resmi tidak dibagi ulang.

### Next Step

- Putuskan penanganan duplikat lintas split dan tinjau kandidat near-duplicate (`reports/tables/ccsn_v2_raw_eda_near_duplicate_candidates.csv`).
- Rancang split train/validation dari train resmi pada tahap preprocessing, dengan manifest split yang tersimpan dan dapat diverifikasi.
- Tetapkan pipeline preprocessing di memori (decode RGB, resize, normalisasi) berdasarkan temuan di atas.
- Baru setelah itu jalankan baseline. Tahap-tahap ini belum dijalankan.

## EDA Conclusion: GCD

### Dataset Characteristics

- 19.000 citra dalam 7 kelas (train 10.000, test 9.000).
- Resolusi: 512 x 512; mode warna: RGB 3 channel untuk semua citra.
- Mean RGB split train (skala 0-1): 0,4827, 0,6154, 0,7672; std: 0,1955, 0,1282, 0,1370.
- Kelas terbesar 6_cumulonimbus dan terkecil 7_mixed; rasio tertinggi 8,63 pada Train (imbalance sedang).
- Perbedaan antar kelas: η² brightness 0,387, η² contrast 0,475.

### Issues Found

- Split Validation tidak tersedia di `data/raw/GCD/`.
- Data leakage lintas split terbukti: 156 pasangan citra identik berada di split berbeda (156 citra test terdampak).
- 3 pasangan citra identik berada di dalam split yang sama.
- 45 kandidat near-duplicate (36 lintas split) belum dapat dipastikan tanpa tinjauan visual.
- Class imbalance sedang: rasio 8,63 pada Train.
- Proporsi kelas berbeda antar split: selisih terbesar 3,84 poin persentase pada 4_clearsky.
- Sebaran brightness atau contrast bergeser antara train dan test (|d| ≥ 0,5) pada kelas: 1_cumulus.
- 117 citra hampir seragam, terbanyak pada 4_clearsky (82), 5_stratocumulus (33), 6_cumulonimbus (2).

### Implications for Preprocessing

- Resize: semua citra 512 x 512, jadi cukup satu resize ke ukuran input model saat pemuatan (di memori); tidak perlu menulis ulang citra.
- Normalisasi: bila tidak memakai statistik bobot pretrained, gunakan mean/std dari split train saja (mean 0,4827, 0,6154, 0,7672; std 0,1955, 0,1282, 0,1370), bukan dari test.
- Class imbalance perlu dipertimbangkan saat training (bobot kelas atau sampling) dan saat evaluasi (macro F1 dan metrik per kelas, bukan accuracy saja).
- Duplikat lintas split harus ditangani sebelum training lewat daftar pengecualian di manifest, tanpa menghapus berkas dari data/raw/.
- Saat validation dibuat dari train, pasangan duplikat dan kandidat near-duplicate di dalam train harus tetap berada di sisi yang sama.
- Split validation perlu dibuat dari train resmi (stratified, seed tetap) pada tahap berikutnya; test resmi tidak dibagi ulang.

### Next Step

- Putuskan penanganan duplikat lintas split dan tinjau kandidat near-duplicate (`reports/tables/gcd_raw_eda_near_duplicate_candidates.csv`).
- Rancang split train/validation dari train resmi pada tahap preprocessing, dengan manifest split yang tersimpan dan dapat diverifikasi.
- Tetapkan pipeline preprocessing di memori (decode RGB, resize, normalisasi) berdasarkan temuan di atas.
- Baru setelah itu jalankan baseline. Tahap-tahap ini belum dijalankan.
