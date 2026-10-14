# Baseline AlexNet — Klasifikasi Citra Awan (Transfer Learning ImageNet)

Dataset train: **combined_balanced** (GCD+CCSN, 3.000/kelas = 21.000 citra). Framework: **PyTorch / torchvision**.
Tanggal: 14 Oktober 2026. Terkait: [`ARSITEKTUR_BASELINE.md`](ARSITEKTUR_BASELINE.md), [`PENYETARAAN_SAMPEL.md`](PENYETARAAN_SAMPEL.md), [`PENGGABUNGAN_DATASET.md`](PENGGABUNGAN_DATASET.md), [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md).

Dokumen ini mendokumentasikan **baseline AlexNet**: transfer learning dengan bobot pra-latih **ImageNet**, lapisan klasifikasi terakhir diganti agar keluaran = **7 kelas** awan. Status: **persiapan** — kode training belum dijalankan (`torch`/`torchvision` belum terpasang). Kode pembuatan model sudah ada di [`src/models/baseline.py`](../src/models/baseline.py); konfigurasi di [`configs/experiments/exp_006_alexnet.yaml`](../configs/experiments/exp_006_alexnet.yaml).

---

## 1. Ringkasan

| Butir | Nilai |
|---|---|
| Arsitektur | AlexNet (8 lapisan berbobot: 5 konvolusi + 3 fully-connected) |
| Strategi | Transfer learning, bobot pra-latih ImageNet |
| Bobot | `torchvision` `AlexNet_Weights.IMAGENET1K_V1` |
| Perubahan head | `classifier[6]`: `Linear(4096, 1000)` → `Linear(4096, 7)` |
| Train | combined_balanced, 3.000/kelas (21.000 citra) |
| Val / test | combined apa adanya (1.851 / 9.000) — **tidak diseimbangkan** |
| Ketidakseimbangan | `imbalance: none` (train sudah seimbang; tak ada koreksi ganda) |
| Input | 512 × 512 RGB (resolusi asli; sumber sudah 512×512) |
| Normalisasi | bagi tiap piksel dengan 255 → rentang [0, 1] (tanpa standardisasi mean/std) |
| Metrik utama | macro-F1 (+ balanced accuracy, accuracy, per kelas, confusion matrix) |

---

## 2. Arsitektur AlexNet (deskripsi asli, Krizhevsky dkk. 2012)

AlexNet terdiri dari **delapan lapisan berbobot**: lima lapisan konvolusi diikuti tiga lapisan fully-connected.

**Lapisan konvolusi:**

1. **Conv-1** — 96 filter berukuran 11×11×3, stride 4 piksel. Keluaran melewati ReLU, lalu max pooling jendela 3×3 stride 2.
2. **Conv-2** — 256 filter 5×5×48. Keluaran dinormalisasi dengan *Local Response Normalization* (LRN), lalu max pooling.
3. **Conv-3** — 384 filter 3×3 (tanpa pooling antara lapisan 3–5).
4. **Conv-4** — 384 filter 3×3.
5. **Conv-5** — 256 filter 3×3.

**Lapisan fully-connected:**

- **FC-6, FC-7** — masing-masing 4.096 neuron.
- **FC-8** — keluaran dilewatkan softmax **1000-way** (1.000 kelas ImageNet).

> Catatan: ukuran filter Conv-2 "5×5×**48**" dan pembagian 96→2×48 mencerminkan rancangan asli yang **dibagi ke dua GPU** (grouped convolution) dan memakai **LRN**.

### 2.1 Varian torchvision yang dipakai (PENTING — berbeda dari deskripsi asli)

Bobot pra-latih yang dipakai di proyek ini berasal dari `torchvision.models.alexnet`, yang mengimplementasikan varian **single-stream** ("One weird trick…", Krizhevsky 2014), **bukan** persis rancangan dua-GPU di atas. Perbedaannya:

| Aspek | AlexNet asli (2012, §2) | torchvision `alexnet` (dipakai) |
|---|---|---|
| Conv-1 | 96 filter 11×11, stride 4 | **64** filter 11×11, stride 4, pad 2 |
| Conv-2 | 256 filter 5×5×48 (grouped) | **192** filter 5×5, pad 2 (tanpa grouping) |
| Conv-3/4/5 | 384 / 384 / 256 | 384 / 256 / 256 |
| LRN | Ya | **Tidak** (dihilangkan) |
| FC-6/7 | 4.096 / 4.096 | 4.096 / 4.096 (sama) |
| FC-8 | 1000-way softmax | `Linear(4096, 1000)` |

Jumlah neuron FC (4.096) dan head 1000-kelas **sama**, sehingga penggantian head ke 7 kelas identik untuk kedua varian. Perbedaan channel/LRN dicatat agar laporan akurat: yang benar-benar dilatih adalah varian torchvision, bukan rancangan dua-GPU asli.

---

## 3. Transfer learning ke 7 kelas

Pola sama dengan [`ARSITEKTUR_BASELINE.md`](ARSITEKTUR_BASELINE.md) Bagian 3:

1. Muat AlexNet dengan bobot ImageNet.
2. Ganti **lapisan terakhir** `model.classifier[6]` (sebuah `nn.Linear(4096, 1000)`) menjadi `nn.Linear(4096, 7)`.
3. Pilih mode: `feature_extraction` (bekukan backbone, latih head saja) atau `full` (fine-tuning seluruh lapisan).

```python
from src.models.baseline import build_model, count_parameters

# classifier[6] (Linear 4096 -> 1000) diganti -> Linear(4096, 7)
model = build_model("alexnet", num_classes=7, pretrained=True, finetune="full")
print(count_parameters(model))  # (total, trainable)
```

Normalisasi memakai **pembagian /255 ke rentang [0, 1]** (tanpa mean/std); lihat [`PERSIAPAN_PELATIHAN.md`](PERSIAPAN_PELATIHAN.md). Catatan: bobot pra-latih ImageNet aslinya dilatih dengan standardisasi mean/std ImageNet, jadi penggunaan [0,1] adalah pilihan praproses eksplisit proyek ini.

---

## 4. Data: combined_balanced

Train memakai set **seimbang 3.000 citra/kelas** (total 21.000) hasil penyetaraan dataset gabungan GCD+CCSN, sudah dimaterialisasi di `data/combined_balanced/train/<kelas>/`:

| Komponen | Nilai | Sumber |
|---|---|---|
| Train (seimbang) | 21.000 (7 × 3.000; 10.483 asli + 10.517 augmentasi) | `reports/tables/combined_balanced_train_manifest.csv` (kolom `class`, `filename`, `kind`, `split`, `rel_path`) |
| Validation | 1.851 (distribusi nyata) | `reports/tables/combined_val_manifest.csv` |
| Test | 9.000 (distribusi nyata) | `reports/tables/combined_test_manifest.csv` |

Prinsip (lihat [`PENYETARAAN_SAMPEL.md`](PENYETARAAN_SAMPEL.md)):
- **Hanya train** yang diseimbangkan; **val/test tetap apa adanya** agar evaluasi jujur dan sebanding.
- Karena train sudah seimbang, `training.imbalance: none` — **jangan** sekaligus memakai `class_weights` (koreksi ganda).
- Kolom `kind` membedakan citra `original` vs `augmented`; augmentasi hanya pada train.

---

## 5. Konfigurasi & hyperparameter

Config: [`configs/experiments/exp_006_alexnet.yaml`](../configs/experiments/exp_006_alexnet.yaml) (override terhadap `configs/base.yaml`).

```yaml
model:
  architecture: alexnet
  pretrained: imagenet          # AlexNet_Weights.IMAGENET1K_V1
  source: torchvision
  num_classes: 7
  finetune: feature_extraction  # feature_extraction | full

data:
  train_dir: data/combined_balanced/train
  train_manifest: reports/tables/combined_balanced_train_manifest.csv
  val_manifest: reports/tables/combined_val_manifest.csv
  test_manifest: reports/tables/combined_test_manifest.csv

training:
  imbalance: none
```

Hyperparameter awal dari `configs/base.yaml`: epochs 50, batch_size 32, optimizer adam, lr 0,001 (untuk fine-tuning penuh pertimbangkan lr lebih kecil, mis. 1e-4). Pemilihan checkpoint: `val_macro_f1` (test tidak dipakai untuk memilih).

---

## 6. Evaluasi (selaras EXPERIMENT_PLAN §5 & §8)

- Metrik: macro-F1 (utama), balanced accuracy, accuracy, precision/recall/F1 per kelas + support, confusion matrix 7×7 (`zero_division=0`, label 0..6).
- Keluaran per eksperimen: `experiments/exp_006_alexnet/` (`config.yaml`, `metrics.json`, `training_history.csv`, `classification_report.txt`, `confusion_matrix.png`, `notes.md`); checkpoint `checkpoints/exp_006_alexnet/best.pt`.
- Test resmi dievaluasi **sekali** di akhir, bukan untuk memilih model.

---

## 7. Catatan & keterbatasan

- Bobot pra-latih yang dipakai adalah **varian torchvision** AlexNet (64/192 channel, tanpa LRN), bukan rancangan dua-GPU asli pada §2 — dicatat agar laporan akurat.
- AlexNet relatif dangkal/klasik; berguna sebagai **baseline ringan** pembanding terhadap DenseNet/ResNet pada data dan metrik yang sama.
- Train seimbang (21.000) berisi ~50% citra augmentasi; variasi augmentasi untuk kelas minoritas (mis. Mi ~10×) terbatas — perhatikan risiko overfitting pola augmentasi (lihat `PENYETARAAN_SAMPEL.md` §4.3).
- Belum dijalankan: perlu `torch`/`torchvision` terpasang.
