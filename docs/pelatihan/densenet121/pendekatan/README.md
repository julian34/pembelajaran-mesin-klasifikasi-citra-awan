# Pendekatan Pelatihan DenseNet-121

Diperbarui: **22 Oktober 2026 (Asia/Jakarta)**.

Dokumen metode dan analisis dikelompokkan berdasarkan pendekatan.
Nomor folder menunjukkan urutan pendekatan, bukan urutan tahap yang wajib
dijalankan dalam satu pipeline.

| Pendekatan | Informasi metode | Analisis | Status artefak saat pemeriksaan |
|---|---|---|---|
| **1 — Training dari awal** | [Informasi](01-training-dari-awal/training-dari-awal.md) | [Analisis](01-training-dari-awal/analisis-training-dari-awal.md) | 10 epoch selesai; hasil test tersedia |
| **2 — Pretrained ImageNet** | [Informasi](02-pretrained-imagenet/informasi-pretrained-imagenet.md) | [Analisis](02-pretrained-imagenet/analisis-pretrained-imagenet.md) | Konfigurasi/arsitektur tersedia; hasil training belum tersedia |

```text
pendekatan/
├── README.md
├── 01-training-dari-awal/
│   ├── training-dari-awal.md
│   └── analisis-training-dari-awal.md
└── 02-pretrained-imagenet/
    ├── informasi-pretrained-imagenet.md
    └── analisis-pretrained-imagenet.md
```

Folder lama `traning dari awal` telah dirapikan menjadi
`01-training-dari-awal`. Tautan dokumen mengikuti struktur baru.

Pendekatan 3, pretraining pada dataset awan sumber sebelum transfer ke target,
belum diimplementasikan dalam rangkaian eksperimen ini.
Baseline lama `exp_005_densenet121` tetap didokumentasikan di
[folder baseline](../BASELINE_DENSENET121.md); hasilnya tidak dicampurkan dengan
eksperimen `exp_005c_densenet121_imagenet`.

Dokumen analisis harus diperbarui berdasarkan artefak aktual. Ketika hasil
belum ada, status ditulis sebagai belum tersedia tanpa mengarang metrik.
