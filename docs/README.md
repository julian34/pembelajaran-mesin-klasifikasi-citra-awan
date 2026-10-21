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

**Dokumen umum (lintas arsitektur):**

| Dokumen | Isi |
|---|---|
| [EXPERIMENT_PLAN.md](pelatihan/EXPERIMENT_PLAN.md) | Rencana eksperimen, variabel terkontrol, metrik, reproduksibilitas |
| [TRAINING_GUIDE.md](pelatihan/TRAINING_GUIDE.md) | Struktur folder, aturan eksperimen, checkpoint, kontrak keluaran |
| [ARSITEKTUR_BASELINE.md](pelatihan/ARSITEKTUR_BASELINE.md) | Arsitektur baseline (transfer learning ImageNet) + daftar exp_005–009 |
| [HYPERPARAMETER.md](pelatihan/HYPERPARAMETER.md) | **Referensi hyperparameter yang digunakan** (nilai efektif `base.yaml` + override exp_005–009; tabel per-eksperimen) |
| [PERSIAPAN_PELATIHAN.md](pelatihan/PERSIAPAN_PELATIHAN.md) | Keputusan praproses (512×512, /255) & hyperparameter awal |

**pelatihan/alexnet/ — baseline AlexNet (exp_006):**

| Dokumen | Isi |
|---|---|
| [BASELINE_ALEXNET.md](pelatihan/alexnet/BASELINE_ALEXNET.md) | Baseline AlexNet exp_006 (detail arsitektur, data, konfigurasi, run) |
| [ANALISIS_BASELINE_ALEXNET.md](pelatihan/alexnet/ANALISIS_BASELINE_ALEXNET.md) | **Analisis hasil** exp_006 (waktu latih, semua metrik, F1 per kelas, confusion matrix) |
| [arsitektur-alexnet-lapisan.html](pelatihan/alexnet/arsitektur-alexnet-lapisan.html) | **Diagram arsitektur animasi** AlexNet awal→akhir (5 konv + 3 FC, ukuran peta fitur, Dropout; buka di browser, dashmotion) |
| [arsitektur-exp006.html](pelatihan/alexnet/arsitektur-exp006.html) | **Diagram pipeline animasi** exp_006 (data → model → latih → evaluasi; buka di browser, dashmotion) |
| [rasional-hyperparameter.md](pelatihan/alexnet/rasional-hyperparameter.md) | **Mengapa** kombinasi AlexNet + hyperparameter yang dipakai |

**pelatihan/densenet121/ — baseline DenseNet-121 (exp_005):**

| Dokumen | Isi |
|---|---|
| [BASELINE_DENSENET121.md](pelatihan/densenet121/BASELINE_DENSENET121.md) | Baseline DenseNet-121 exp_005 (detail arsitektur, data, konfigurasi, run) |
| [Pendekatan pelatihan DenseNet-121](pelatihan/densenet121/pendekatan/README.md) | Informasi dan analisis terpisah untuk training dari awal (01) serta pretrained ImageNet (02) |
| [ANALISIS_BASELINE_DENSENET121.md](pelatihan/densenet121/ANALISIS_BASELINE_DENSENET121.md) | **Analisis hasil** exp_005 (diisi setelah pelatihan dijalankan) |
| [arsitektur-densenet121-koneksi.html](pelatihan/densenet121/arsitektur-densenet121-koneksi.html) | **Diagram arsitektur animasi** — *dense connectivity* (ciri khas DenseNet: concat antar-lapisan dalam satu dense block; buka di browser, dashmotion) |
| [arsitektur-densenet121-lapisan.html](pelatihan/densenet121/arsitektur-densenet121-lapisan.html) | **Diagram lapisan animasi** DenseNet-121 awal→akhir (ukuran peta fitur tiap tahap; buka di browser, dashmotion) |
| [arsitektur-exp005.html](pelatihan/densenet121/arsitektur-exp005.html) | **Diagram pipeline animasi** exp_005 (data → model → latih → evaluasi; buka di browser, dashmotion) |

**pelatihan/resnet18/ — baseline ResNet-18 (exp_008):**

| Dokumen | Isi |
|---|---|
| [BASELINE_RESNET18.md](pelatihan/resnet18/BASELINE_RESNET18.md) | Baseline ResNet-18 exp_008 (detail arsitektur, data, konfigurasi, run) |
| [ANALISIS_BASELINE_RESNET18.md](pelatihan/resnet18/ANALISIS_BASELINE_RESNET18.md) | **Analisis hasil** exp_008 (diisi setelah pelatihan dijalankan) |
| [rasional-hyperparameter.md](pelatihan/resnet18/rasional-hyperparameter.md) | **Mengapa** kombinasi ResNet-18 + hyperparameter (yang khas ResNet: skip connection, BatchNorm) |
| [arsitektur-resnet18-residual.html](pelatihan/resnet18/arsitektur-resnet18-residual.html) | **Diagram arsitektur animasi** — *residual connection* (ciri khas ResNet: y = F(x) + x pada BasicBlock; buka di browser, dashmotion) |
| [arsitektur-resnet18-lapisan.html](pelatihan/resnet18/arsitektur-resnet18-lapisan.html) | **Diagram lapisan animasi** ResNet-18 awal→akhir (ukuran peta fitur tiap tahap; buka di browser, dashmotion) |
| [arsitektur-exp008.html](pelatihan/resnet18/arsitektur-exp008.html) | **Diagram pipeline animasi** exp_008 (data → model → latih → evaluasi; buka di browser, dashmotion) |

## referensi/ — tooling & rujukan

| Dokumen | Isi |
|---|---|
| [REFERENSI_EKSTERNAL.md](referensi/REFERENSI_EKSTERNAL.md) | Referensi eksternal untuk persiapan |
| [RELORE.md](referensi/RELORE.md) | Panduan Relore (pembacaan kode lokal, server, indeks GitHub) |

---

**Hyperparameter baseline saat ini** (dari [`configs/base.yaml`](../configs/base.yaml), berlaku untuk exp_005–009): epochs 200, batch_size 32 (8 pada 512×512 di GPU 8 GB), optimizer AdamW, learning_rate 1e-4, weight_decay 0,0, scheduler null, checkpoint by `val_macro_f1`. Rasional untuk AlexNet: [pelatihan/alexnet/rasional-hyperparameter.md](pelatihan/alexnet/rasional-hyperparameter.md).
