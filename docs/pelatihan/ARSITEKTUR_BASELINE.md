# Arsitektur Baseline — DenseNet-264 (Transfer Learning)

Dataset: GCD, 7 kelas (Cu, Ac, Ci, Cl, Sc, Cb, Mi). Framework: **PyTorch / torchvision**.
Tanggal: 13 Oktober 2026. Terkait: [`TRAINING_GUIDE.md`](TRAINING_GUIDE.md), [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md), [`DATASET_SPLIT.md`](../dataset/DATASET_SPLIT.md).

Dokumen ini menyiapkan arsitektur **baseline awal**: klasifikasi citra awan memakai **Transfer Learning** dengan bobot pra-latih **ImageNet**, lalu mengganti lapisan terakhir (*classifier*) agar keluaran = **7 kelas**. Arsitektur target: **DenseNet-264**.

---

## 1. Ringkasan

| Butir | Nilai |
|---|---|
| Tugas | Klasifikasi citra awan, 7 kelas, single-label |
| Arsitektur | DenseNet-264 |
| Strategi | Transfer Learning (bobot pra-latih ImageNet) |
| Perubahan | Ganti `classifier` → `Linear(in_features, 7)` |
| Input | 512 × 512 RGB (resolusi asli; sumber sudah 512×512, lihat `preprocess`) |
| Framework | PyTorch + torchvision |
| Loss | `CrossEntropyLoss` dengan bobot kelas (lihat `training.imbalance`) |
| Metrik utama | macro-F1 (lihat `evaluation`) |

> **Catatan ketersediaan bobot (penting).** torchvision menyediakan bobot ImageNet hanya untuk DenseNet-121/161/169/201 — **bukan DenseNet-264**. Untuk benar-benar memakai DenseNet-264 **dengan** bobot pra-latih, lihat Bagian 4 (opsi `timm` atau fallback). Ini perlu diputuskan sebelum implementasi.

---

## 2. Arsitektur DenseNet-264

DenseNet (Densely Connected Convolutional Networks) menghubungkan tiap lapisan ke semua lapisan sesudahnya dalam satu blok, sehingga fitur dipakai ulang dan jumlah parameter relatif efisien. Varian untuk ImageNet dibedakan oleh konfigurasi blok (`block_config`):

| Varian | block_config | Bobot ImageNet di torchvision |
|---|---|---|
| DenseNet-121 | (6, 12, 24, 16) | ✅ ada |
| DenseNet-169 | (6, 12, 32, 32) | ✅ ada |
| DenseNet-201 | (6, 12, 48, 32) | ✅ ada |
| **DenseNet-264** | **(6, 12, 64, 48)** | ❌ **tidak ada** |

Parameter umum DenseNet-BC untuk ImageNet: `growth_rate = 32`, `num_init_features = 64`, kompresi 0,5. DenseNet-264 berukuran ~33 juta parameter (perkiraan), lebih dalam dari DenseNet-201.

---

## 3. Strategi Transfer Learning

Langkah inti:

1. **Muat model** dengan bobot pra-latih ImageNet.
2. **Ganti classifier.** Pada DenseNet torchvision, kepala klasifikasi bernama `model.classifier` (sebuah `nn.Linear`). Ganti menjadi `nn.Linear(in_features, 7)`.
3. **Pilih mode pelatihan:**
   - **Feature extraction** — bekukan seluruh backbone (`requires_grad = False`), latih hanya classifier. Cepat, cocok sebagai baseline paling awal.
   - **Fine-tuning** — latih semua (atau sebagian blok akhir) dengan learning rate kecil. Biasanya akurasi lebih tinggi, butuh lebih banyak komputasi.

Rekomendasi baseline: mulai dengan **feature extraction** sebagai titik acuan, lalu bandingkan dengan **fine-tuning penuh**.

### Normalisasi saat memakai bobot pra-latih

> **Pembaruan (praproses baseline).** Pelatihan baseline proyek ini memakai normalisasi **bagi /255 → [0, 1]** (bukan mean/std ImageNet); lihat [`PERSIAPAN_PELATIHAN.md`](PERSIAPAN_PELATIHAN.md). Panduan mean/std di bawah hanya berlaku bila sengaja memilih standardisasi ImageNet.

Karena backbone dilatih di ImageNet, **gunakan normalisasi ImageNet**, bukan statistik dataset:

```
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

Statistik dataset sendiri (`preprocess.normalize` di `configs/base.yaml`: mean [0,4885; 0,6170; 0,7650]) hanya dipakai bila melatih **dari nol** tanpa bobot pra-latih. Konsistensi ini sudah dicatat di `DATASET_QUALITY_SUMMARY.md` §2.5.

---

## 4. Ketersediaan bobot pra-latih DenseNet-264 (keputusan)

torchvision **tidak** memiliki bobot ImageNet untuk DenseNet-264. Tiga opsi:

**Opsi A — `timm` (rekomendasi jika ingin tetap DenseNet-264 + pra-latih).**
Pustaka `timm` menyediakan varian DenseNet-264 (mis. `densenet264d`). Perlu:
- menambah `timm` ke `requirements.txt`;
- **memverifikasi** ketersediaan bobot ImageNet-1k untuk nama model tersebut saat implementasi (jangan diasumsikan).

```python
import timm
model = timm.create_model("densenet264d", pretrained=True, num_classes=7)
```

**Opsi B — DenseNet-201 pra-latih (paling mulus di torchvision).**
DenseNet terdalam yang bobot ImageNet-nya tersedia resmi di torchvision. Cocok sebagai baseline transfer-learning yang langsung jalan.

```python
import torch.nn as nn
from torchvision import models
from torchvision.models import DenseNet201_Weights

model = models.densenet201(weights=DenseNet201_Weights.IMAGENET1K_V1)
model.classifier = nn.Linear(model.classifier.in_features, 7)
```

**Opsi C — DenseNet-264 dari nol (tanpa pra-latih).**
Arsitektur DenseNet-264 dapat dibangun dengan kelas `DenseNet` torchvision, tetapi **tanpa** bobot pra-latih (bukan transfer learning):

```python
from torchvision.models import DenseNet
model = DenseNet(growth_rate=32, block_config=(6, 12, 64, 48),
                 num_init_features=64, num_classes=7)
```

Konsekuensi: butuh data/epoch lebih banyak dan biasanya kalah dari transfer learning pada dataset berukuran ini.

**Saran.** Untuk baseline transfer-learning yang langsung dapat dijalankan, pilih **Opsi A** (jika ingin tetap DenseNet-264 persis) atau **Opsi B** (jika boleh DenseNet terdekat yang bobotnya resmi). Catat pilihan di `configs/base.yaml` `model`.

---

## 5. Integrasi dengan proyek

| Aspek | Sumber | Nilai |
|---|---|---|
| Sumber data (GCD-only) | `reports/tables/gcd_raw_split_{train,val,test}_manifest.csv` atau folder `data/{train,val,test}/` | train 8.340 / val 1.471 / test 9.000 |
| Sumber data (gabungan GCD+CCSN) | `reports/tables/combined_{train,val,test}_manifest.csv` (kolom `rel_path` relatif ke `data/raw/<dataset>`) | train 10.483 / val 1.851 / test 9.000 |
| Input size | `configs/base.yaml` `data.image_size` / `preprocess.resize` | 512 |
| Normalisasi | lihat Bagian 3 | ImageNet (pra-latih) atau statistik dataset (dari nol) |
| Ketidakseimbangan | `configs/base.yaml` `training.imbalance` + `gcd_raw_class_weights.csv` (GCD) / `combined_class_weights.csv` (gabungan) | `CrossEntropyLoss(weight=...)` |
| Pemilihan model | `configs/base.yaml` `checkpoint.monitor` | `val_macro_f1` (bukan test) |
| Evaluasi | `configs/base.yaml` `evaluation` | macro-F1 + per kelas + confusion matrix |

Pakai artefak **GCD-only** (`gcd_raw_*`) untuk eksperimen GCD saja, atau **gabungan** (`combined_*`) untuk model dua-dataset — bobot kelas dan manifest harus dari sumber yang sama.

Isian yang disarankan di `configs/base.yaml` (bagian `model`, saat ini `null`):

```yaml
model:
  architecture: densenet264   # atau densenet201 (Opsi B); samakan dengan kode pemuatan
  pretrained: imagenet        # null bila melatih dari nol (Opsi C)
  source: timm                # timm (Opsi A) | torchvision (Opsi B/C)
  num_classes: 7
  finetune: feature_extraction  # feature_extraction | full
```

---

## 6. Sketsa alur pelatihan (ilustrasi)

```python
import torch, torch.nn as nn
# 1) Model (pilih salah satu opsi di Bagian 4) -> ganti classifier ke 7 kelas
# 2) Bobot kelas untuk loss
import pandas as pd
w = pd.read_csv("reports/tables/gcd_raw_class_weights.csv")["weight_inverse_freq"].to_numpy()
class_weight = torch.tensor(w, dtype=torch.float32)
criterion = nn.CrossEntropyLoss(weight=class_weight)
# 3) Optimizer sesuai configs/base.yaml (adamw, lr 1e-4)
# 4) Loop: train -> validasi -> pilih checkpoint by val_macro_f1
# 5) Evaluasi akhir sekali pada test (macro-F1 + per kelas)
```

Catatan: `torch`/`torchvision` **sudah terpasang** (torch 2.1.1+cu121, torchvision 0.16.1+cu121; CUDA 12.1 aktif di RTX 4060 Laptop), jadi kode di atas langsung dapat dijalankan.

---

## 7. Hyperparameter awal (dari `configs/base.yaml`)

| Parameter | Nilai saat ini |
|---|---|
| epochs | 200 |
| batch_size | 32 |
| optimizer | adamw |
| learning_rate | 0,0001 (1e-4) |
| weight_decay | 0,0 |
| scheduler | null |
| monitor checkpoint | val_macro_f1 (mode max) |

Nilai ini berlaku untuk semua baseline exp_005–009 (diwarisi dari `configs/base.yaml`). Rasional pemilihannya untuk AlexNet dibahas di [`alexnet/rasional-hyperparameter.md`](alexnet/rasional-hyperparameter.md).

---

## 8. Langkah berikutnya

1. Putuskan opsi bobot pra-latih (Bagian 4) dan isi `configs/base.yaml` `model`.
2. Pasang `torch`/`torchvision` (dan `timm` bila Opsi A).
3. Tulis `src/datasets/` (Dataset/DataLoader membaca manifest split + `preprocess`) dan `src/models/` (pemuatan model + ganti classifier).
4. Tulis loop training/evaluasi (`src/training/`, `src/evaluation/`) dengan loss berbobot dan metrik macro-F1.
5. Jalankan baseline, catat hasil di `experiments/` sesuai `EXPERIMENT_PLAN.md`.

---

## 9. Baseline tambahan (transfer learning ImageNet)

Ditambahkan beberapa arsitektur baseline untuk perbandingan. Semua memakai pola transfer learning yang sama (Bagian 3): muat bobot ImageNet → ganti classifier ke `Linear(in_features, 7)` → pilih mode `feature_extraction` (bekukan backbone) atau `full` (fine-tuning). Praproses pelatihan baseline: input **512 × 512** dan normalisasi **bagi /255 → [0, 1]** (tanpa mean/std); lihat [`PERSIAPAN_PELATIHAN.md`](PERSIAPAN_PELATIHAN.md).

| Arsitektur | Classifier (head) | in_features | Sumber bobot ImageNet |
|---|---|---:|---|
| **DenseNet-121** | `model.classifier` (`nn.Linear`) | 1.024 | ✅ torchvision `DenseNet121_Weights.IMAGENET1K_V1` |
| **DenseNet-264** | `model.classifier` (`nn.Linear`) | 2.688 | ✅ **timm** `densenet264d` (torchvision tak punya) |
| **AlexNet** | `model.classifier[6]` (`nn.Linear` dalam `nn.Sequential`) | 4.096 | ✅ torchvision `AlexNet_Weights.IMAGENET1K_V1` |
| **ResNet-18** | `model.fc` (`nn.Linear`) | 512 | ✅ torchvision `ResNet18_Weights.IMAGENET1K_V1` |
| **ResNet-50** | `model.fc` (`nn.Linear`) | 2.048 | ✅ torchvision `ResNet50_Weights.IMAGENET1K_V1` |

Perhatikan **nama head berbeda antar keluarga**: DenseNet memakai `model.classifier`, AlexNet `model.classifier[6]`, dan ResNet `model.fc`. Pabrik model menangani ini otomatis.

### Implementasi

Pabrik model ada di [`src/models/baseline.py`](../../src/models/baseline.py) — fungsi `build_model(architecture, num_classes=7, pretrained=True, finetune="feature_extraction")`:

```python
from src.models.baseline import build_model, count_parameters

# torchvision (bobot ImageNet langsung tersedia)
model = build_model("densenet121", finetune="feature_extraction")
model = build_model("alexnet",     finetune="full")
model = build_model("resnet18",    finetune="feature_extraction")
model = build_model("resnet50",    finetune="full")

# DenseNet-264 pra-latih: dibangun via timm "densenet264d" (butuh `pip install timm`)
model = build_model("densenet264", pretrained=True, finetune="feature_extraction")

print(count_parameters(model))  # (total, trainable)
```

Catatan:
- `feature_extraction` membekukan backbone **sebelum** classifier baru dipasang, sehingga hanya head yang dilatih.
- `densenet264` dengan `pretrained=True` memerlukan paket **`timm`**; dengan `pretrained=False` dibangun dari nol via torchvision (`block_config (6,12,64,48)`), bukan transfer learning. Lihat Bagian 4.

### Konfigurasi eksperimen

| Eksperimen | Config | Arsitektur | Sumber |
|---|---|---|---|
| exp_005 | [`exp_005_densenet121.yaml`](../../configs/experiments/exp_005_densenet121.yaml) | densenet121 | torchvision |
| exp_006 | [`exp_006_alexnet.yaml`](../../configs/experiments/exp_006_alexnet.yaml) | alexnet | torchvision |
| exp_007 | [`exp_007_densenet264.yaml`](../../configs/experiments/exp_007_densenet264.yaml) | densenet264 | timm |
| exp_008 | [`exp_008_resnet18.yaml`](../../configs/experiments/exp_008_resnet18.yaml) | resnet18 | torchvision |
| exp_009 | [`exp_009_resnet50.yaml`](../../configs/experiments/exp_009_resnet50.yaml) | resnet50 | torchvision |

Semua meng-override `model` dan `preprocess.normalize` (ke statistik ImageNet) terhadap `configs/base.yaml`, dan dibandingkan sebagai alternatif baseline — dari AlexNet/ResNet-18 yang ringan hingga DenseNet-264 yang terdalam.

### Dataset baseline: combined_balanced

Kelima eksperimen baseline (exp_005–009) memakai **train seimbang** `data/combined_balanced/train/` (3.000 citra/kelas = 21.000 = 10.483 asli + 10.517 augmentasi; GCD+CCSN, lihat [`PENYETARAAN_SAMPEL.md`](../dataset/PENYETARAAN_SAMPEL.md)), dengan **val/test combined apa adanya** (`combined_val`/`combined_test`, 1.851 / 9.000; tidak diseimbangkan). **Total data pelatihan = 21.000 + 1.851 + 9.000 = 31.851 citra.** Karena train sudah seimbang, `training.imbalance: none` — tidak dipadukan dengan `class_weights` agar tidak terjadi koreksi ganda. Blok `data` dan `training.imbalance` ini ditulis di tiap config exp_005–009.

Dokumen baseline per arsitektur (rinci): **AlexNet** → [`alexnet/BASELINE_ALEXNET.md`](alexnet/BASELINE_ALEXNET.md); **DenseNet-121** → [`densenet121/BASELINE_DENSENET121.md`](densenet121/BASELINE_DENSENET121.md).
