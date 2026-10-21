# Informasi Pendekatan 2 — DenseNet-121 Pretrained ImageNet

Tanggal: **22 Oktober 2026 (Asia/Jakarta)**.  
Eksperimen: `exp_005c_densenet121_imagenet`.  
Hasil dan status: [Analisis pretrained ImageNet](analisis-pretrained-imagenet.md).  
Navigasi: [Daftar pendekatan](../README.md).

## 1. Tujuan

Pendekatan ini memuat DenseNet-121 dengan bobot yang telah dilatih pada
ImageNet, lalu menyesuaikannya untuk klasifikasi tujuh kelas awan GCD + CCSN_v2.
Model menggunakan `DenseNet121_Weights.IMAGENET1K_V1` melalui pembangun model
proyek. Bobot tersedia dari cache PyTorch atau diunduh bila cache belum ada.

Classifier asli `Linear(1024, 1000)` diganti menjadi `Linear(1024, 7)`.
Classifier baru diinisialisasi acak dan dilatih dengan dataset awan.
Implementasi saat ini memakai **feature extraction**, sesuai baseline proyek.

## 2. Arsitektur dan parameter yang diperbarui

| Komponen | Pengaturan |
|---|---|
| Arsitektur | DenseNet-121, torchvision |
| Bobot awal backbone | ImageNet, `IMAGENET1K_V1` |
| Konfigurasi pretrained | `pretrained: imagenet` |
| Mode | `finetune: feature_extraction` |
| Classifier baru | `Linear(1024, 7)` |
| Parameter total | 6.961.031 |
| Parameter trainable | **7.175**, seluruhnya classifier |
| Parameter dibekukan | **6.953.856**, parameter backbone |

Jumlah tersebut dikonfirmasi pada
[arsitektur model tersimpan](../../../../../experiments/exp_005c_densenet121_imagenet/model_architecture.txt).
Optimizer hanya menerima parameter dengan `requires_grad=True`.

**Perilaku BatchNorm:** trainer memanggil `model.train()` pada awal epoch.
Parameter BatchNorm pada backbone dibekukan, tetapi statistik berjalan
(running mean/variance) tetap diperbarui selama training. Dengan demikian,
pembekuan parameter tidak berarti seluruh state backbone tetap identik.
Saat validation/test, trainer menggunakan `model.eval()`.

## 3. Dataset dan praproses

| Split | Jumlah | Tujuan |
|---|---:|---|
| Train | 21.000 | Melatih classifier; 3.000 citra per kelas |
| Validation | 1.851 | Memilih checkpoint dan mengevaluasi keputusan pengembangan |
| Test | 9.000 | Evaluasi akhir pada test resmi GCD |

Urutan kelas: **Cu, Ac, Ci, Cl, Sc, Cb, Mi**.
Train mengikuti `combined_balanced_train_manifest.csv`; validation dan test
mengikuti `combined_val_manifest.csv` serta `combined_test_manifest.csv`
di `reports/tables/`.

Citra dibaca sebagai RGB, diubah ukurannya menjadi **512 × 512**, dan
dinormalisasi dengan **/255** ke [0, 1]. Mean [0, 0, 0] dan std [1, 1, 1]
menghasilkan transformasi identitas setelah pembagian 255.
Standardisasi mean/std ImageNet tidak diterapkan: ini adalah pilihan
praproses eksplisit baseline proyek dan sama dengan pendekatan 1.
Lihat [baseline DenseNet-121](../../BASELINE_DENSENET121.md).

Augmentasi saat pemuatan tidak digunakan. Train telah memuat 10.483 citra asli
dan 10.517 citra augmentasi yang dimaterialisasi sebelumnya. Validation dan
test tidak diseimbangkan. `imbalance: none` menghindari pembobotan kelas
tambahan pada train yang sudah seimbang.
Rincian: [penyetaraan sampel](../../../../dataset/PENYETARAAN_SAMPEL.md).

## 4. Hyperparameter

Sumber nilai efektif:
[snapshot konfigurasi run](../../../../../experiments/exp_005c_densenet121_imagenet/config.yaml).

| Pengaturan | Nilai |
|---|---|
| Epoch yang direncanakan | **10** |
| Batch size | **8** |
| Optimizer | **AdamW** |
| Learning rate | **0,0001 (`1e-4`)** |
| Weight decay | 0 |
| Scheduler | Tidak ada; learning rate tetap |
| Loss | CrossEntropyLoss tanpa bobot kelas |
| Random seed | 42 |
| DataLoader workers | 0 |
| Checkpoint monitor | Validation macro-F1, mode max |
| Early stopping | Tidak diterapkan |

Nilai epoch dalam konfigurasi merupakan rencana run, bukan bukti training
telah menyelesaikan sepuluh epoch. Status hasil dipisahkan pada dokumen analisis.

## 5. Sumber kode dan konfigurasi

| Sumber | Fungsi |
|---|---|
| [Notebook 05c](../../../../../notebooks/tahap_pre_training/05c_densenet121_pretrained_imagenet.ipynb) | Setup, pemuatan ImageNet, training, grafik, dan evaluasi test |
| [Konfigurasi eksperimen](../../../../../configs/experiments/exp_005c_densenet121_imagenet.yaml) | Identitas run, bobot pretrained, mode feature extraction, dan data |
| [Konfigurasi dasar](../../../../../configs/base.yaml) | Nilai bawaan yang diwarisi |
| [Pembangun model](../../../../../src/models/baseline.py) | Memuat ImageNet, membekukan parameter, mengganti classifier |
| [Dataset](../../../../../src/datasets/cloud_dataset.py) | Pemuatan citra dan praproses |
| [Trainer](../../../../../src/training/trainer.py) | Optimizer, training, validation, dan checkpoint |
| [Metrik](../../../../../src/evaluation/metrics.py) | Macro-F1, balanced accuracy, accuracy, laporan kelas, dan matriks |

## 6. Menjalankan notebook

1. Buka notebook 05c menggunakan lingkungan Python proyek.
2. Jalankan setup dan konfigurasi. `EPOCHS=None`, `BATCH_SIZE=None`, serta
   `NUM_WORKERS=None` memakai nilai konfigurasi eksperimen.
3. Siapkan train dan validation, lalu muat model ImageNet. Periksa output:
   total 6.961.031 parameter, trainable 7.175, classifier tujuh keluaran.
4. Jalankan sel training. Konfigurasi dan arsitektur disimpan sebelum `fit`;
   checkpoint terbaik dipilih dari validation macro-F1.
5. Setelah training selesai, periksa `training_history.csv`, `metrics.json`,
   serta grafik riwayat pada notebook.
6. Setelah keputusan model selesai berdasarkan validation, aktifkan
   `EVALUATE_TEST=True` untuk evaluasi akhir menggunakan checkpoint terbaik.

Untuk mengulang dari bobot ImageNet dengan classifier baru, jalankan kembali
sel model sebelum training. Menjalankan sel training saja memakai model yang
masih berada di memori. Jika kehabisan memori, batch dapat diturunkan; perubahan
konfigurasi efektif perlu dicatat sebagai bagian dari run.

## 7. Artefak dan kriteria penyelesaian

Folder hasil: `experiments/exp_005c_densenet121_imagenet/`.  
Checkpoint: `checkpoints/exp_005c_densenet121_imagenet/best.pt`.

| Berkas | Kapan tersedia |
|---|---|
| `config.yaml`, `model_architecture.txt` | Sebelum training |
| `best.pt` | Selama training ketika validation macro-F1 membaik |
| `training_history.csv`, `metrics.json` | Setelah fungsi training selesai |
| `classification_report.txt`, `confusion_matrix.png`, `notes.md` | Setelah evaluasi test |

Metrik test berstatus `not_evaluated` sampai evaluasi akhir dijalankan.
Keberadaan konfigurasi dan arsitektur saja tidak membuktikan penyelesaian
training. Jangan menyamakan tidak adanya `metrics.json` dengan hasil test nol.

## 8. Perbandingan dengan pendekatan 1

| Aspek | Pendekatan 1 | Pendekatan 2 |
|---|---|---|
| Bobot awal backbone | Acak | ImageNet |
| Parameter trainable | 6.961.031 | 7.175 |
| Mode | `full` | `feature_extraction` |
| Classifier | Dilatih | Dilatih |
| Dataset, input, seed, optimizer, lr, batch, rencana epoch | Pengaturan sama pada konfigurasi saat ini | Pengaturan sama pada konfigurasi saat ini |

Perbandingan mencerminkan perubahan bobot awal **dan** cakupan parameter yang
dioptimasi; hasil tidak dapat diatribusikan hanya pada inisialisasi.
Baseline lama `exp_005_densenet121` memakai anggaran 200 epoch dan merupakan
eksperimen berbeda dari `exp_005c_densenet121_imagenet`.

Aturan evaluasi mengikuti [panduan pelatihan](../../../TRAINING_GUIDE.md).
