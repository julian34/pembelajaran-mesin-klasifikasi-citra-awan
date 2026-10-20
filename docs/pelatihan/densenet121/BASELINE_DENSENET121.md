# Baseline DenseNet-121 — Klasifikasi Citra Awan (Transfer Learning ImageNet)

Dataset train: **combined_balanced** (GCD+CCSN, 3.000/kelas = 21.000 citra). Framework: **PyTorch / torchvision**.
Tanggal: 18 Oktober 2026. Terkait: [`ARSITEKTUR_BASELINE.md`](../ARSITEKTUR_BASELINE.md), [`ANALISIS_BASELINE_DENSENET121.md`](ANALISIS_BASELINE_DENSENET121.md), [`PENYETARAAN_SAMPEL.md`](../../dataset/PENYETARAAN_SAMPEL.md), [`PENGGABUNGAN_DATASET.md`](../../dataset/PENGGABUNGAN_DATASET.md), [`EXPERIMENT_PLAN.md`](../EXPERIMENT_PLAN.md).

Dokumen ini mendokumentasikan **baseline DenseNet-121**: transfer learning dengan bobot pra-latih **ImageNet**, lapisan klasifikasi terakhir diganti agar keluaran = **7 kelas** awan. **Hyperparameter sama persis dengan baseline AlexNet (exp_006)** — hanya arsitekturnya yang berbeda, agar perbandingan antar-model adil (lihat [`ARSITEKTUR_BASELINE.md`](../ARSITEKTUR_BASELINE.md) dan rasional hyperparameter bersama di [`../alexnet/rasional-hyperparameter.md`](../alexnet/rasional-hyperparameter.md)). Status: **pelatihan penuh belum dijalankan** — jalankan & pantau sendiri via notebook. Kode:

- Model: [`src/models/baseline.py`](../../../src/models/baseline.py) (`build_model`, arsitektur `densenet121`).
- Data: [`src/datasets/manifest.py`](../../../src/datasets/manifest.py) (resolusi path/label) + [`src/datasets/cloud_dataset.py`](../../../src/datasets/cloud_dataset.py) (`Dataset`/`DataLoader` + transform /255).
- Pelatihan: [`src/training/trainer.py`](../../../src/training/trainer.py) (`fit`, checkpoint by `val_macro_f1`; opsi `progress=True` → progress bar per-epoch gaya Keras via `pkbar`).
- Evaluasi: [`src/evaluation/metrics.py`](../../../src/evaluation/metrics.py).
- Config: [`configs/experiments/exp_005_densenet121.yaml`](../../../configs/experiments/exp_005_densenet121.yaml); runner: [`scripts/train.py`](../../../scripts/train.py); notebook pemantau: [`notebooks/05_densenet121_training.ipynb`](../../../notebooks/05_densenet121_training.ipynb).

---

## 1. Ringkasan

| Butir | Nilai |
|---|---|
| Arsitektur | DenseNet-121 (4 *dense block*, *growth rate* 32, dengan *transition layer* di antaranya) |
| Strategi | Transfer learning, bobot pra-latih ImageNet |
| Bobot | `torchvision` `DenseNet121_Weights.IMAGENET1K_V1` |
| Perubahan head | `classifier`: `Linear(1024, 1000)` → `Linear(1024, 7)` |
| Regularisasi | — (DenseNet torchvision tidak punya dropout FC bawaan; andalkan `feature_extraction` + train seimbang) |
| Train | combined_balanced, 3.000/kelas (21.000 citra) |
| Val / test | combined apa adanya (1.851 / 9.000) — **tidak diseimbangkan** |
| Ketidakseimbangan | `imbalance: none` (train sudah seimbang; tak ada koreksi ganda) |
| Input | 512 × 512 RGB (resolusi asli; sumber sudah 512×512) |
| Normalisasi | bagi tiap piksel dengan 255 → rentang [0, 1] (tanpa standardisasi mean/std) |
| Metrik utama | macro-F1 (+ balanced accuracy, accuracy, per kelas, confusion matrix) |

---

## 2. Arsitektur DenseNet-121 (Huang dkk., 2017)

DenseNet (*Densely Connected Convolutional Network*) menghubungkan **setiap lapisan ke semua lapisan berikutnya** dalam satu *dense block*: masukan lapisan ke-*l* adalah **gabungan (concatenation)** peta fitur seluruh lapisan sebelumnya. Pola ini memperkuat aliran gradien, mendorong **penggunaan ulang fitur**, dan membuat jumlah parameter relatif hemat dibanding kedalamannya.

Struktur DenseNet-121 (varian torchvision):

1. **Stem** — Conv 7×7 stride 2 (64 filter) + max pool 3×3 stride 2.
2. **Dense Block 1** — 6 *dense layer* → **Transition 1** (Conv 1×1 + avg pool 2×2).
3. **Dense Block 2** — 12 *dense layer* → **Transition 2**.
4. **Dense Block 3** — 24 *dense layer* → **Transition 3**.
5. **Dense Block 4** — 16 *dense layer*.
6. **Classifier** — global average pool → `Linear(1024, 1000)` (head ImageNet).

Angka "121" = total lapisan berbobot (konvolusi + 1 FC): 1 (stem) + (6+12+24+16)×2 lapisan konvolusi dalam *dense layer* + 3 transisi + 1 FC. *Growth rate* k=32 (tiap *dense layer* menambah 32 peta fitur). Setelah Dense Block 4, dimensi fitur = **1024**, itulah `in_features` head.

Diagram (animasi, buka di browser):
- **Arsitektur — dense connectivity** (ciri khas DenseNet: concat antar-lapisan dalam satu dense block): [`arsitektur-densenet121-koneksi.html`](arsitektur-densenet121-koneksi.html).
- **Lapisan awal→akhir** (ukuran peta fitur tiap tahap, input 512×512 → 7 kelas): [`arsitektur-densenet121-lapisan.html`](arsitektur-densenet121-lapisan.html).
- **Pipeline** (data → model → latih → evaluasi): [`arsitektur-exp005.html`](arsitektur-exp005.html).

---

## 3. Transfer learning ke 7 kelas

Pola sama dengan [`ARSITEKTUR_BASELINE.md`](../ARSITEKTUR_BASELINE.md) Bagian 3:

1. Muat DenseNet-121 dengan bobot ImageNet.
2. Ganti **lapisan terakhir** `model.classifier` (sebuah `nn.Linear(1024, 1000)`) menjadi `nn.Linear(1024, 7)`. Lapisan lain dibiarkan utuh.
3. Pilih mode: `feature_extraction` (bekukan backbone, latih head saja) atau `full` (fine-tuning seluruh lapisan).

```python
from src.models.baseline import build_model, count_parameters

# classifier (Linear 1024 -> 1000) diganti -> Linear(1024, 7)
model = build_model("densenet121", num_classes=7, pretrained=True, finetune="feature_extraction")
print(model.classifier)        # Linear(1024, 7) — head keluaran
print(count_parameters(model)) # (total, trainable)
```

### 3.1 Catatan regularisasi

Berbeda dari AlexNet (yang punya Dropout(0.5) bawaan pada dua FC tersembunyi), DenseNet-121 torchvision **tidak** menyisipkan dropout pada head. Untuk baseline ini tidak ada dropout tambahan yang disisipkan; pengendali overfitting adalah:

- **`feature_extraction`** — hanya head `Linear(1024→7)` (**7.175** parameter) yang dilatih; backbone beku sehingga kapasitas belajar sangat terbatas → risiko overfitting rendah.
- **Train sudah diseimbangkan** (3.000/kelas) sehingga tidak ada koreksi bobot ganda.
- **Pemilihan checkpoint by `val_macro_f1`** (bukan early stopping) menjaga model akhir = epoch val-terbaik.

Normalisasi memakai **pembagian /255 ke rentang [0, 1]** (tanpa mean/std); lihat [`PERSIAPAN_PELATIHAN.md`](../PERSIAPAN_PELATIHAN.md). Catatan: bobot pra-latih ImageNet aslinya dilatih dengan standardisasi mean/std ImageNet, jadi penggunaan [0,1] adalah pilihan praproses eksplisit proyek ini (seragam lintas baseline).

---

## 4. Data: combined_balanced

Identik dengan baseline AlexNet (data dikontrol sama agar hanya arsitektur yang berbeda). Train memakai set **seimbang 3.000 citra/kelas** (total 21.000) hasil penyetaraan dataset gabungan GCD+CCSN, sudah dimaterialisasi di `data/combined_balanced/train/<kelas>/`:

| Komponen | Nilai | Sumber |
|---|---|---|
| Train (seimbang) | 21.000 (7 × 3.000; 10.483 asli + 10.517 augmentasi) | `reports/tables/combined_balanced_train_manifest.csv` |
| Validation | 1.851 (distribusi nyata) | `reports/tables/combined_val_manifest.csv` |
| Test | 9.000 (distribusi nyata) | `reports/tables/combined_test_manifest.csv` |
| **Total** | **31.851** | train seimbang + val + test |

Prinsip (lihat [`PENYETARAAN_SAMPEL.md`](../../dataset/PENYETARAAN_SAMPEL.md)):
- **Hanya train** yang diseimbangkan; **val/test tetap apa adanya** agar evaluasi jujur dan sebanding.
- Karena train sudah seimbang, `training.imbalance: none` — **jangan** sekaligus memakai `class_weights` (koreksi ganda).

---

## 5. Konfigurasi & hyperparameter

Config: [`configs/experiments/exp_005_densenet121.yaml`](../../../configs/experiments/exp_005_densenet121.yaml) (override terhadap `configs/base.yaml`).

```yaml
model:
  architecture: densenet121
  pretrained: imagenet          # DenseNet121_Weights.IMAGENET1K_V1
  source: torchvision
  num_classes: 7
  finetune: feature_extraction  # feature_extraction | full

data:
  train_dir: data/combined_balanced/train
  train_manifest: reports/tables/combined_balanced_train_manifest.csv
  val_dir: data/combined/val      # sama dengan exp_006_alexnet
  val_manifest: reports/tables/combined_val_manifest.csv
  test_dir: data/combined/test
  test_manifest: reports/tables/combined_test_manifest.csv

training:
  imbalance: none
```

Hyperparameter efektif (diwarisi dari `configs/base.yaml`, **sama dengan exp_006_alexnet**): epochs 200, batch_size 32, optimizer adamw, lr 0,0001 (1e-4), weight_decay 0,0, scheduler null. Rasional pemilihan kombinasi ini (berlaku lintas baseline exp_005–009) dibahas di [`../alexnet/rasional-hyperparameter.md`](../alexnet/rasional-hyperparameter.md). Pemilihan checkpoint: `val_macro_f1` (test tidak dipakai untuk memilih). Catatan VRAM: DenseNet-121 **lebih berat memori** dari AlexNet pada input 512×512; notebook memakai **batch 8** (turunkan ke 4 bila OOM di GPU 8 GB).

### 5.1 Menjalankan

**Opsi A — notebook (pantau sendiri, disarankan untuk eksplorasi):** buka [`notebooks/05_densenet121_training.ipynb`](../../../notebooks/05_densenet121_training.ipynb), jalankan sel atas→bawah. Sel 5 melatih dengan **progress bar per-epoch (pkbar)** dan Sel 5b menampilkan tabel + grafik riwayat. Untuk uji cepat set `EPOCHS=1`.

**Opsi B — runner terminal (lebih cepat, pemuatan paralel):**
```powershell
# 1) Validasi config + manifes + file tanpa membangun model/melatih:
python scripts/train.py --config configs/experiments/exp_005_densenet121.yaml --dry-run

# 2) Latih baseline (artefak -> experiments/exp_005_densenet121/, checkpoint -> checkpoints/exp_005_densenet121/best.pt):
python scripts/train.py --config configs/experiments/exp_005_densenet121.yaml --batch-size 8
```

`--epochs N` meng-override jumlah epoch; `--batch-size N` meng-override batch; `--device cpu|cuda|auto` memilih perangkat.

**Soal kecepatan (I/O-bound, bukan GPU-bound).** Citra dibaca sebagai JPEG 512×512 dari disk USB eksternal secara serial. Dengan `num_workers=0` (default aman di Jupyter Windows) laju per-epoch lambat; untuk jauh lebih cepat, naikkan `training.num_workers` (mis. 4) dan latih lewat **runner terminal** (Opsi B).

---

## 6. Evaluasi (selaras EXPERIMENT_PLAN §5 & §8)

- Metrik: macro-F1 (utama), balanced accuracy, accuracy, precision/recall/F1 per kelas + support, confusion matrix 7×7 (`zero_division=0`, label 0..6).
- Keluaran per eksperimen: `experiments/exp_005_densenet121/` (`config.yaml`, `metrics.json`, `training_history.csv`, `classification_report.txt`, `confusion_matrix.png`, `notes.md`); checkpoint `checkpoints/exp_005_densenet121/best.pt`.
- Test resmi dievaluasi **sekali** di akhir, bukan untuk memilih model.
- Analisis hasil lengkap ditulis di [`ANALISIS_BASELINE_DENSENET121.md`](ANALISIS_BASELINE_DENSENET121.md) **setelah** pelatihan dijalankan.

---

## 7. Catatan & keterbatasan

- DenseNet-121 jauh lebih dalam dari AlexNet namun **hemat parameter** berkat *dense connectivity* + *bottleneck* 1×1; diharapkan menangkap tekstur awan halus (Sc/Ci) lebih baik daripada AlexNet pada metrik & data yang sama.
- Pada `feature_extraction` hanya head (7.175 parameter) yang belajar; bila macro-F1 val mentok, pertimbangkan `finetune: full` dengan lr tetap kecil.
- Train seimbang (21.000) berisi ~50% citra augmentasi; pantau selisih train vs val macro-F1 sebagai indikator overfitting.
- Pelatihan **I/O-bound** pada disk USB (riwayat *intermittent read error*) — jalankan lewat runner terminal dan pantau.
- Pelatihan penuh **belum dijalankan**; dijalankan & dipantau sendiri oleh pengguna via notebook (progress bar pkbar per epoch).

---

## 8. Referensi

- Huang, Liu, van der Maaten, Weinberger (2017), *Densely Connected Convolutional Networks* (DenseNet). https://arxiv.org/abs/1608.06993
- Loshchilov & Hutter (2019), *Decoupled Weight Decay Regularization* (AdamW). https://arxiv.org/abs/1711.05101
- torchvision DenseNet: https://pytorch.org/vision/stable/models/generated/torchvision.models.densenet121.html
- Dokumen internal: [`ARSITEKTUR_BASELINE.md`](../ARSITEKTUR_BASELINE.md), [`../alexnet/rasional-hyperparameter.md`](../alexnet/rasional-hyperparameter.md), [`EXPERIMENT_PLAN.md`](../EXPERIMENT_PLAN.md), [`PENYETARAAN_SAMPEL.md`](../../dataset/PENYETARAAN_SAMPEL.md).
