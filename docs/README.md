# Dokumentasi CitraAwan

Indeks dokumentasi proyek klasifikasi citra awan 7 kelas (GCD + CCSN_v2, PyTorch). Dokumen dikelompokkan ke tiga folder tematik.

## dataset/ — data, kualitas, split, augmentasi

| Dokumen | Isi |
|---|---|
| [DATASET_DOCUMENTATION.md](dataset/DATASET_DOCUMENTATION.md) | Dokumentasi dataset GCD + CCSN_v2, skema 7 kelas, sumber & lisensi |
| [DATASET_QUALITY_SUMMARY.md](dataset/DATASET_QUALITY_SUMMARY.md) | Ringkasan temuan kualitas (leakage, duplikat, imbalance, normalisasi) |
| [PENANGANAN_KUALITAS_DATA.md](dataset/PENANGANAN_KUALITAS_DATA.md) | Dokumentasi ilmiah penanganan kualitas data |
| [DATASET_SPLIT.md](dataset/DATASET_SPLIT.md) | Pembentukan split (test resmi GCD + validation 0,15 dari train) |
| [PENGGABUNGAN_DATASET.md](dataset/PENGGABUNGAN_DATASET.md) | Penggabungan GCD + CCSN_v2 ke skema 7 kelas |
| [PENYETARAAN_SAMPEL.md](dataset/PENYETARAAN_SAMPEL.md) | Penyetaraan train ke 3.000/kelas (balancing + augmentasi) |
| [AUGMENTASI_DATA.md](dataset/AUGMENTASI_DATA.md) | Strategi augmentasi data |

## pelatihan/ — rencana, model, hyperparameter

| Dokumen | Isi |
|---|---|
| [EXPERIMENT_PLAN.md](pelatihan/EXPERIMENT_PLAN.md) | Rencana eksperimen, variabel terkontrol, metrik, reproduksibilitas |
| [TRAINING_GUIDE.md](pelatihan/TRAINING_GUIDE.md) | Struktur folder, aturan eksperimen, checkpoint, kontrak keluaran |
| [ARSITEKTUR_BASELINE.md](pelatihan/ARSITEKTUR_BASELINE.md) | Arsitektur baseline (transfer learning ImageNet) + daftar exp_005–009 |
| [PERSIAPAN_PELATIHAN.md](pelatihan/PERSIAPAN_PELATIHAN.md) | Keputusan praproses (512×512, /255) & hyperparameter awal |
| [BASELINE_ALEXNET.md](pelatihan/BASELINE_ALEXNET.md) | Baseline AlexNet exp_006 (detail arsitektur, data, konfigurasi, run) |
| [ANALISIS_BASELINE_ALEXNET.md](pelatihan/ANALISIS_BASELINE_ALEXNET.md) | **Analisis hasil** exp_006 (waktu latih, semua metrik, F1 per kelas, confusion matrix) |
| [rasional-alexnet-hyperparameter/](pelatihan/rasional-alexnet-hyperparameter/README.md) | **Mengapa** kombinasi AlexNet + hyperparameter yang dipakai |

## referensi/ — tooling & rujukan

| Dokumen | Isi |
|---|---|
| [REFERENSI_EKSTERNAL.md](referensi/REFERENSI_EKSTERNAL.md) | Referensi eksternal untuk persiapan |
| [RELORE.md](referensi/RELORE.md) | Panduan Relore (pembacaan kode lokal, server, indeks GitHub) |

---

**Hyperparameter baseline saat ini** (dari [`configs/base.yaml`](../configs/base.yaml), berlaku untuk exp_005–009): epochs 200, batch_size 32 (8 pada 512×512 di GPU 8 GB), optimizer AdamW, learning_rate 1e-4, weight_decay 0,0, scheduler null, checkpoint by `val_macro_f1`. Rasional untuk AlexNet: [pelatihan/rasional-alexnet-hyperparameter/](pelatihan/rasional-alexnet-hyperparameter/README.md).
