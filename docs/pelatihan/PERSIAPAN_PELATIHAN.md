# Persiapan Pelatihan — Baseline Model (Klasifikasi Citra Awan)

Tanggal: 14 Oktober 2026. Framework: PyTorch / torchvision. Terkait: [`ARSITEKTUR_BASELINE.md`](ARSITEKTUR_BASELINE.md), [`BASELINE_ALEXNET.md`](BASELINE_ALEXNET.md), [`PENYETARAAN_SAMPEL.md`](../dataset/PENYETARAAN_SAMPEL.md), [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md), [`TRAINING_GUIDE.md`](TRAINING_GUIDE.md).

Dokumen ini menetapkan **persiapan pelatihan** untuk **baseline model** klasifikasi citra awan 7 kelas. Status: **siap** — `torch`/`torchvision` build CUDA cu121 sudah terpasang (GPU RTX 4060 Laptop); pelatihan penuh belum dijalankan (dipantau sendiri oleh pengguna). Model dibangun dari [`src/models/baseline.py`](../../src/models/baseline.py); konfigurasi di `configs/base.yaml` + `configs/experiments/exp_005…009`.

---

## 1. Keputusan praproses (baru)

| Butir | Nilai |
|---|---|
| **Skala masukan** | **512 × 512 piksel** (resolusi asli; semua sumber sudah 512×512 RGB → tanpa penskalaan) |
| **Normalisasi piksel** | **bagi tiap nilai piksel dengan 255** → rentang **[0, 1]**. **Tanpa** standardisasi mean/std |
| Decode | RGB |
| Interpolasi | bilinear (hanya relevan bila suatu sumber bukan 512×512) |

Di `configs/base.yaml`:

```yaml
data:
  image_size: 512
preprocess:
  decode: RGB
  resize: 512
  normalize:
    scale: 255          # bagi /255 -> [0,1]
    mean: [0.0, 0.0, 0.0]   # identitas: tidak ada standardisasi
    std:  [1.0, 1.0, 1.0]
```

Implementasi pipeline (torchvision): `ToTensor()` sudah membagi /255 ke [0,1]; **jangan** menambahkan `Normalize(mean, std)` ImageNet (atau pakai `mean=0, std=1` = identitas) agar nilai tetap [0,1].

> **Catatan transfer learning.** Bobot pra-latih ImageNet aslinya dilatih dengan standardisasi mean/std ImageNet pada 224×224. Memakai [0,1] pada 512×512 adalah **pilihan praproses eksplisit** proyek ini (menggantikan rencana 224/ImageNet pada dokumen lama). Semua backbone yang dipakai memakai *adaptive pooling* sehingga menerima input 512×512.

---

## 2. Data

| Split | Isi | Sumber |
|---|---|---|
| Train | **combined_balanced** — 3.000/kelas = 21.000 citra (seimbang) | `data/combined_balanced/train/`, `reports/tables/combined_balanced_train_manifest.csv` |
| Validation | 1.851 (distribusi nyata, tidak diseimbangkan) | `reports/tables/combined_val_manifest.csv` |
| Test | 9.000 (distribusi nyata) | `reports/tables/combined_test_manifest.csv` |

Karena train sudah seimbang → `training.imbalance: none` (jangan dipadukan `class_weights`). Val/test tetap apa adanya. Lihat [`PENYETARAAN_SAMPEL.md`](../dataset/PENYETARAAN_SAMPEL.md).

---

## 3. Baseline model

Dibangun via `build_model(architecture, num_classes=7, pretrained=True, finetune=...)` di [`src/models/baseline.py`](../../src/models/baseline.py); semua memakai **bobot ImageNet** dan mengganti classifier ke 7 kelas.

| Config | Arsitektur | Head diganti | Sumber bobot |
|---|---|---|---|
| exp_005 | densenet121 | `classifier` (1024→7) | torchvision |
| exp_006 | alexnet | `classifier[6]` (4096→7) | torchvision |
| exp_007 | densenet264 | `classifier` (2688→7) | timm `densenet264d` |
| exp_008 | resnet18 | `fc` (512→7) | torchvision |
| exp_009 | resnet50 | `fc` (2048→7) | torchvision |

Mode: `feature_extraction` (bekukan backbone) atau `full` (fine-tuning) per config.

---

## 4. Hyperparameter awal (`configs/base.yaml`)

| Parameter | Nilai | Catatan |
|---|---|---|
| epochs | 200 | |
| batch_size | 32 | **lihat §5** — pada 512×512 dipakai 8 (GPU 8 GB) |
| optimizer | adamw | |
| learning_rate | 0.0001 (1e-4) | |
| weight_decay | 0.0 | |
| imbalance | none | train sudah seimbang (override di exp_005…009) |
| monitor checkpoint | val_macro_f1 (max) | test tak dipakai memilih |

---

## 5. Catatan sumber daya (512 × 512)

Input 512×512 ≈ **5,2×** piksel dibanding 224×224, sehingga memori & waktu naik tajam. Pada GPU 8 GB (RTX 4060 Laptop, lihat `EXPERIMENT_PLAN.md` §7):

- `batch_size: 32` kemungkinan **OOM** untuk ResNet-50 / DenseNet. Mulai dari **8–16** dan naikkan bila muat; catat batch efektif yang dipakai.
- Pertimbangkan **mixed precision (AMP)** untuk menghemat memori.
- AlexNet/ResNet-18 lebih ringan; DenseNet-264 paling berat.

Uji kelayakan memori satu batch sebelum run penuh (sesuai `EXPERIMENT_PLAN.md`).

---

## 6. Evaluasi & keluaran

Metrik dan kontrak keluaran mengikuti [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md) §5 & §8: macro-F1 (utama), balanced accuracy, accuracy, per-kelas, confusion matrix 7×7; keluaran per eksperimen di `experiments/<nama>/` + checkpoint `checkpoints/<nama>/best.pt`. Test resmi dievaluasi **sekali** di akhir.

---

## 7. Langkah berikutnya

1. Pasang `torch`/`torchvision` (dan `timm` untuk exp_007).
2. Tulis loader (`src/datasets/`) yang membaca manifest + praproses §1 (decode RGB, 512, /255 → [0,1]).
3. Tulis loop training/evaluasi (`src/training/`, `src/evaluation/`) dengan `build_model`, monitor `val_macro_f1`.
4. Jalankan tiap baseline (exp_005…009), catat hasil di `experiments/`.
