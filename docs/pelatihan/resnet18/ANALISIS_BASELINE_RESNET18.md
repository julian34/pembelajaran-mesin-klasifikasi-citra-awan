# Analisis Hasil Baseline ResNet-18 (exp_008)

> **Status: menunggu hasil pelatihan.** Dokumen ini adalah **kerangka analisis**
> yang akan diisi **setelah** `exp_008_resnet18` dijalankan. Angka pada tabel
> ditandai `TBD` (*to be determined*) dan harus diisi **hanya** dari artefak nyata
> di `experiments/exp_008_resnet18/` — bukan perkiraan. Bandingkan dengan baseline
> AlexNet di [`../alexnet/ANALISIS_BASELINE_ALEXNET.md`](../alexnet/ANALISIS_BASELINE_ALEXNET.md)
> dan DenseNet-121 di [`../densenet121/ANALISIS_BASELINE_DENSENET121.md`](../densenet121/ANALISIS_BASELINE_DENSENET121.md).

Analisis hasil pelatihan **transfer learning ResNet-18** (bobot ImageNet, mode
`feature_extraction`) untuk klasifikasi 7 kelas awan (GCD + CCSN_v2). Setup baseline
(arsitektur, data, konfigurasi): [`BASELINE_RESNET18.md`](BASELINE_RESNET18.md).

- Eksperimen: `exp_008_resnet18` · tanggal artefak **TBD**
- Sumber (nyata, bukan perkiraan): `experiments/exp_008_resnet18/`
  — `metrics.json`, `classification_report.txt`, `training_history.csv`,
  `confusion_matrix.png`, `config.yaml`, `notes.md`; checkpoint
  `checkpoints/exp_008_resnet18/best.pt`.
- Rasional pilihan hyperparameter (bersama lintas baseline): [`../alexnet/rasional-hyperparameter.md`](../alexnet/rasional-hyperparameter.md).
- Diagram alur arsitektur (animasi, buka di browser): [`arsitektur-exp008.html`](arsitektur-exp008.html).

---

## 1. Ringkasan eksekutif

| Besaran | Nilai |
|---|---|
| Epoch terbaik (dipilih dari validation) | **TBD** dari 200 |
| val macro-F1 terbaik | **TBD** |
| **Test macro-F1** (metrik primer) | **TBD** |
| Test accuracy (= micro-F1 global) | **TBD** |
| Test balanced accuracy | **TBD** |
| Parameter dilatih / total | 3.591 / 11.180.103 (head `Linear(512→7)` saja) |
| Waktu pelatihan | **TBD** |

**Inti temuan:** _(isi setelah run — bandingkan terhadap AlexNet exp_006: test macro-F1 0,5042)._

---

## 2. Konfigurasi yang dievaluasi

Diambil dari `experiments/exp_008_resnet18/config.yaml` (snapshot saat run). Identik dengan AlexNet exp_006 & DenseNet-121 exp_005 kecuali arsitektur:

| Item | Nilai |
|---|---|
| Arsitektur | `resnet18` (torchvision, pretrained ImageNet) |
| Mode fine-tune | `feature_extraction` (backbone beku, hanya `fc` 512→7 dilatih) |
| Input | 512×512 RGB, normalisasi `/255` → [0,1] (mean 0, std 1) |
| Augmentasi | nonaktif |
| Epochs | 200 |
| Batch size | 8 (TBD — konfirmasi dari `notes.md` run) |
| Optimizer | AdamW, lr 1e-4, weight_decay 0,0 |
| Scheduler | — (tidak ada) |
| Penanganan imbalance | `none` (train sudah diseimbangkan 3.000/kelas) |
| Checkpoint | simpan terbaik menurut `val_macro_f1` (mode max) |
| Data | train seimbang 21.000 · val 1.851 · **test resmi GCD 9.000** |

---

## 3. Waktu pelatihan

_(Isi dari stempel waktu artefak / `training_history.csv` setelah run. Loop pelatihan
belum mencatat durasi ke file — lihat catatan metode di analisis AlexNet §3.)_

---

## 4. Dinamika pelatihan (200 epoch)

Sumber: `training_history.csv` (kolom `train_loss`, `val_loss`, `val_macro_f1`,
`val_balanced_accuracy`, `val_accuracy`).

| Epoch | train_loss | val_loss | val_macro_f1 | Keterangan |
|---|---|---|---|---|
| 1 | TBD | TBD | TBD | awal |
| … | … | … | … | … |
| **best** | TBD | TBD | **TBD** | **terbaik → checkpoint** |
| 200 | TBD | TBD | TBD | akhir |

_(Sisipkan grafik setelah run:)_
<!-- ![Kurva loss per epoch](../../../experiments/exp_008_resnet18/plot_loss.png) -->
<!-- ![Metrik validasi per epoch](../../../experiments/exp_008_resnet18/plot_val_metrics.png) -->

---

## 5. Evaluasi test — semua metrik

Test dihitung **sekali** memakai checkpoint terbaik (dipilih dari validation,
**bukan** test). Sumber: `metrics.json` + `classification_report.txt`.

### 5.1 Metrik global

| Metrik | Nilai | Catatan |
|---|---|---|
| Accuracy (= micro-F1) | TBD | |
| **Macro-F1** (primer) | **TBD** | |
| Balanced accuracy | TBD | |
| Weighted-F1 | TBD | |
| Macro precision / recall | TBD / TBD | |

### 5.2 F1 per kelas (precision / recall / F1 / support)

Sumber: `metrics.json → test.per_class`.

| Kelas | Precision | Recall | **F1** | Support |
|---|---|---|---|---|
| Cu | TBD | TBD | TBD | 750 |
| Ac | TBD | TBD | TBD | 750 |
| Ci | TBD | TBD | TBD | 753 |
| Cl | TBD | TBD | TBD | 1.589 |
| Sc | TBD | TBD | TBD | 1.790 |
| Cb | TBD | TBD | TBD | 2.761 |
| Mi | TBD | TBD | TBD | 607 |

_(Sisipkan grafik setelah run:)_
<!-- ![Precision / recall / F1 per kelas](../../../experiments/exp_008_resnet18/plot_per_class_f1.png) -->

### 5.3 Confusion matrix

Baris = kelas sebenarnya, kolom = prediksi (urutan Cu, Ac, Ci, Cl, Sc, Cb, Mi).

<!-- ![Confusion matrix test](../../../experiments/exp_008_resnet18/confusion_matrix.png) -->

---

## 6. Perbandingan antar-baseline

| Metrik | AlexNet exp_006 | DenseNet-121 exp_005 | ResNet-18 exp_008 |
|---|---|---|---|
| Test macro-F1 | 0,5042 | **TBD** | **TBD** |
| Test accuracy | 0,5782 | **TBD** | **TBD** |
| Test balanced accuracy | 0,4865 | **TBD** | **TBD** |
| Parameter trainable | 28.679 | 7.175 | 3.591 |
| Parameter total | 57.032.519 | 6.961.031 | 11.180.103 |

_(Analisis naratif perbandingan diisi setelah run — apakah skip connection + kedalaman
sedang ResNet-18 mengangkat kelas bertekstur halus Sc/Ci/Mi yang lemah di AlexNet,
dan bagaimana posisinya terhadap DenseNet-121 pada jumlah parameter trainable yang lebih kecil.)_

---

## 7. Diagnosis & rekomendasi

_(Isi setelah run.)_

---

## 8. Reproduksibilitas

- Metrik & angka harus seluruhnya berasal dari artefak `experiments/exp_008_resnet18/`
  dan `checkpoints/exp_008_resnet18/best.pt` (seed 42).
- Evaluasi ulang test dari checkpoint:
  `python scripts/train.py --config configs/experiments/exp_008_resnet18.yaml`
  (atau jalankan Sel 6 di [`notebooks/06_resnet18_training.ipynb`](../../../notebooks/06_resnet18_training.ipynb)).
- Protokol split, kontrak keluaran, dan aturan eksperimen:
  [`TRAINING_GUIDE.md`](../TRAINING_GUIDE.md); rencana lengkap: [`EXPERIMENT_PLAN.md`](../EXPERIMENT_PLAN.md).
