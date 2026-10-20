# Baseline DenseNet-264 — Klasifikasi Citra Awan (Transfer Learning ImageNet)

Dataset train: **combined_balanced** (GCD+CCSN, 3.000/kelas = 21.000 citra). Framework: **PyTorch + `timm`**.
Tanggal: 20 Oktober 2026. Terkait: [`ARSITEKTUR_BASELINE.md`](../ARSITEKTUR_BASELINE.md), [`BASELINE_DENSENET121.md`](../densenet121/BASELINE_DENSENET121.md), [`PENYETARAAN_SAMPEL.md`](../../dataset/PENYETARAAN_SAMPEL.md), [`PENGGABUNGAN_DATASET.md`](../../dataset/PENGGABUNGAN_DATASET.md), [`EXPERIMENT_PLAN.md`](../EXPERIMENT_PLAN.md).

Dokumen ini mendokumentasikan **baseline DenseNet-264** (exp_007): transfer learning dengan bobot pra-latih **ImageNet**, lapisan klasifikasi terakhir diganti agar keluaran = **7 kelas** awan. Ini **kelanjutan langsung dari DenseNet-121 (exp_005)** — varian DenseNet **terdalam** pada rencana baseline — dengan **data & hyperparameter dikontrol sama** agar yang berbeda hanya kedalaman arsitektur (lihat [`ARSITEKTUR_BASELINE.md`](../ARSITEKTUR_BASELINE.md) dan rasional hyperparameter bersama di [`../alexnet/rasional-hyperparameter.md`](../alexnet/rasional-hyperparameter.md)). Status: **pelatihan penuh belum dijalankan** — jalankan & pantau sendiri via notebook. Kode:

- Model: [`src/models/baseline.py`](../../../src/models/baseline.py) (`build_model`, arsitektur `densenet264` → `_build_densenet264_timm`).
- Data: [`src/datasets/manifest.py`](../../../src/datasets/manifest.py) (resolusi path/label) + [`src/datasets/cloud_dataset.py`](../../../src/datasets/cloud_dataset.py) (`Dataset`/`DataLoader` + transform /255).
- Pelatihan: [`src/training/trainer.py`](../../../src/training/trainer.py) (`fit`, checkpoint by `val_macro_f1`; opsi `progress=True` → progress bar per-epoch gaya Keras via `pkbar`).
- Evaluasi: [`src/evaluation/metrics.py`](../../../src/evaluation/metrics.py).
- Config: [`configs/experiments/exp_007_densenet264.yaml`](../../../configs/experiments/exp_007_densenet264.yaml); runner: [`scripts/train.py`](../../../scripts/train.py); notebook pemantau: [`notebooks/07_densenet264_training.ipynb`](../../../notebooks/07_densenet264_training.ipynb).

---

## 1. Ringkasan

| Butir | Nilai |
|---|---|
| Arsitektur | DenseNet-264 (4 *dense block* 6/12/64/48, *growth rate* 32, dengan *transition layer* di antaranya) |
| Strategi | Transfer learning, bobot pra-latih ImageNet |
| Bobot | `timm` `densenet264d` (pra-latih ImageNet; **torchvision tidak menyediakan** bobot DenseNet-264) |
| Perubahan head | `classifier`: `Linear(4032, 1000)` → `Linear(4032, 7)` (dibuat timm saat `num_classes=7`) |
| Parameter | total ± **68,7 juta**; `feature_extraction` → hanya head **28.231** parameter yang dilatih (4032×7 + 7) |
| Regularisasi | — (DenseNet tidak punya dropout FC bawaan; andalkan `feature_extraction` + train seimbang) |
| Train | combined_balanced, 3.000/kelas (21.000 citra) |
| Val / test | combined apa adanya (1.851 / 9.000) — **tidak diseimbangkan** |
| Ketidakseimbangan | `imbalance: none` (train sudah seimbang; tak ada koreksi ganda) |
| Input | 512 × 512 RGB (resolusi asli; sumber sudah 512×512) |
| Normalisasi | bagi tiap piksel dengan 255 → rentang [0, 1] (tanpa standardisasi mean/std) |
| Metrik utama | macro-F1 (+ balanced accuracy, accuracy, per kelas, confusion matrix) |

> **Catatan varian `densenet264d`:** timm memakai varian **deep-stem** (stem 3× konvolusi 3×3 menggantikan satu konvolusi 7×7). Dimensi fitur akhir = **4032** (`model.num_features`), itulah `in_features` head — bukan ~2.688 seperti perkiraan awal di [`EXPERIMENT_PLAN.md`](../EXPERIMENT_PLAN.md) §2 (angka itu untuk DenseNet-264 "polos" torchvision yang bobotnya tak tersedia).

---

## 2. Arsitektur DenseNet-264 (Huang dkk., 2017)

DenseNet (*Densely Connected Convolutional Network*) menghubungkan **setiap lapisan ke semua lapisan berikutnya** dalam satu *dense block*: masukan lapisan ke-*l* adalah **gabungan (concatenation)** peta fitur seluruh lapisan sebelumnya. Pola ini memperkuat aliran gradien, mendorong **penggunaan ulang fitur**, dan membuat jumlah parameter relatif hemat dibanding kedalamannya.

DenseNet-264 memakai pola yang sama seperti DenseNet-121 tetapi **jauh lebih dalam**, terutama pada dua blok tengah/akhir:

1. **Stem** — varian `densenet264d`: 3× Conv 3×3 (deep stem) + max pool, menggantikan Conv 7×7 tunggal DenseNet klasik.
2. **Dense Block 1** — 6 *dense layer* → **Transition 1** (Conv 1×1 + avg pool 2×2).
3. **Dense Block 2** — 12 *dense layer* → **Transition 2**.
4. **Dense Block 3** — **64** *dense layer* → **Transition 3**.
5. **Dense Block 4** — **48** *dense layer*.
6. **Classifier** — global average pool → `Linear(4032, 1000)` (head ImageNet, diganti ke 7).

Dibanding DenseNet-121 (block 6/12/24/16), DenseNet-264 memperdalam Dense Block 3 (24 → 64) dan 4 (16 → 48), sehingga fitur akhir membengkak dari 1.024 → **4.032** channel. *Growth rate* tetap k=32. Konsekuensinya: kapasitas representasi jauh lebih besar, tetapi **biaya memori aktivasi** pada input 512×512 juga jauh lebih besar (lihat §5 soal batch).

Perbandingan ringkas keluarga DenseNet pada proyek ini:

| Model | block_config | fitur akhir | sumber bobot |
|---|---|---:|---|
| DenseNet-121 (exp_005) | 6 / 12 / 24 / 16 | 1.024 | torchvision |
| DenseNet-264 (exp_007) | 6 / 12 / 64 / 48 | 4.032 | timm (`densenet264d`) |

---

## 3. Transfer learning ke 7 kelas

Pola sama dengan [`ARSITEKTUR_BASELINE.md`](../ARSITEKTUR_BASELINE.md) Bagian 3 & 4, dengan satu perbedaan: **bobot datang dari timm**, bukan torchvision.

1. Bangun `densenet264d` dengan bobot ImageNet via `timm.create_model(..., pretrained=True, num_classes=7)`. timm langsung membuat head `Linear(4032 → 7)` baru.
2. Mode `feature_extraction`: backbone **dibekukan**, lalu **hanya head** (`model.get_classifier()`) di-*unfreeze* kembali sehingga yang dilatih cuma 28.231 parameter.
3. Alternatif `full`: fine-tuning seluruh parameter (± 68,7 juta) dengan lr tetap kecil.

```python
from src.models.baseline import build_model, count_parameters

# timm densenet264d (ImageNet) -> head Linear(4032 -> 7); backbone beku (feature_extraction)
model = build_model("densenet264", num_classes=7, pretrained=True, finetune="feature_extraction")
print(model.get_classifier())  # Linear(4032, 7) — head keluaran
print(count_parameters(model))  # (total ~68.7e6, trainable 28_231)
```

Ketersediaan bobot: torchvision hanya punya DenseNet-121/161/169/201. Jika `timm` tidak terpasang, `build_model("densenet264", pretrained=True)` melempar `ImportError` yang menyarankan `pip install timm` (sudah dipatok `timm==0.9.12` di `requirements.txt`); fallback yang dicatat di [`EXPERIMENT_PLAN.md`](../EXPERIMENT_PLAN.md) §2 adalah DenseNet-201 torchvision.

### 3.1 Catatan regularisasi

Sama seperti DenseNet-121: **tidak ada dropout FC bawaan** dan tidak ada dropout tambahan yang disisipkan. Pengendali overfitting:

- **`feature_extraction`** — hanya head `Linear(4032→7)` (28.231 parameter) yang dilatih; backbone beku → risiko overfitting rendah meskipun backbone sangat besar.
- **Train sudah diseimbangkan** (3.000/kelas) → tak ada koreksi bobot ganda.
- **Pemilihan checkpoint by `val_macro_f1`** menjaga model akhir = epoch val-terbaik.

Normalisasi memakai **pembagian /255 ke [0, 1]** (tanpa mean/std), seragam lintas baseline; lihat [`PERSIAPAN_PELATIHAN.md`](../PERSIAPAN_PELATIHAN.md).

---

## 4. Data: combined_balanced

Identik dengan baseline AlexNet/DenseNet-121/ResNet-18 (data dikontrol sama agar hanya arsitektur yang berbeda):

| Komponen | Nilai | Sumber |
|---|---|---|
| Train (seimbang) | 21.000 (7 × 3.000; 10.483 asli + 10.517 augmentasi) | `reports/tables/combined_balanced_train_manifest.csv` |
| Validation | 1.851 (distribusi nyata) | `reports/tables/combined_val_manifest.csv` |
| Test | 9.000 (distribusi nyata) | `reports/tables/combined_test_manifest.csv` |
| **Total** | **31.851** | train seimbang + val + test |

Prinsip (lihat [`PENYETARAAN_SAMPEL.md`](../../dataset/PENYETARAAN_SAMPEL.md)): hanya **train** yang diseimbangkan; **val/test tetap apa adanya**. Karena train sudah seimbang, `training.imbalance: none`.

---

## 5. Konfigurasi & hyperparameter

Config: [`configs/experiments/exp_007_densenet264.yaml`](../../../configs/experiments/exp_007_densenet264.yaml) (override terhadap `configs/base.yaml`).

```yaml
model:
  architecture: densenet264
  pretrained: imagenet          # dibangun via timm "densenet264d"
  source: timm                  # torchvision tak punya bobotnya
  num_classes: 7
  finetune: feature_extraction  # feature_extraction | full

data:
  train_dir: data/combined_balanced/train
  train_manifest: reports/tables/combined_balanced_train_manifest.csv
  val_dir: data/combined/val      # WAJIB di-override (sama dengan exp_005); base.yaml -> data/val (GCD-only)
  val_manifest: reports/tables/combined_val_manifest.csv
  test_dir: data/combined/test
  test_manifest: reports/tables/combined_test_manifest.csv

training:
  imbalance: none
```

> **Perbaikan config:** versi awal `exp_007_densenet264.yaml` hanya menuliskan `val_manifest`/`test_manifest` **tanpa** `val_dir`/`test_dir`, sehingga direktori fisik jatuh ke bawaan `base.yaml` (`data/val` GCD-only) dan `--dry-run` gagal `FileNotFoundError`. `val_dir`/`test_dir` kini di-override ke `data/combined/*` (sama persis exp_005). *(Config kembar `exp_009_resnet50.yaml` masih punya bug yang sama — perlu perbaikan serupa bila dijalankan.)*

Hyperparameter efektif (diwarisi dari `configs/base.yaml`, **sama dengan exp_005/006/008**): epochs 200, optimizer adamw, lr 0,0001 (1e-4), weight_decay 0,0, scheduler null. Pemilihan checkpoint: `val_macro_f1`.

**VRAM / batch size (penting).** DenseNet-264 jauh lebih berat memori daripada DenseNet-121. Pada 512×512 di GPU 8 GB (RTX 4060 Laptop), batch 8 (yang dipakai baseline lain) berisiko OOM; notebook default **batch 4** (turunkan ke 2 bila OOM). Perbedaan batch dari baseline lain adalah **keputusan karena kendala memori** dan dicatat sesuai [`EXPERIMENT_PLAN.md`](../EXPERIMENT_PLAN.md) §4; perbandingan antar-arsitektur mempertimbangkan ini.

### 5.1 Menjalankan

**Opsi A — notebook (pantau sendiri):** buka [`notebooks/07_densenet264_training.ipynb`](../../../notebooks/07_densenet264_training.ipynb), jalankan sel atas→bawah. Sel 5 melatih dengan **progress bar per-epoch (pkbar)**; Sel 5b menampilkan tabel + grafik riwayat. Untuk uji cepat set `EPOCHS=1`.

**Opsi B — runner terminal (lebih cepat, pemuatan paralel):**
```powershell
# 1) Validasi config + manifes + file tanpa membangun model/melatih:
python scripts/train.py --config configs/experiments/exp_007_densenet264.yaml --dry-run

# 2) Latih (artefak -> experiments/exp_007_densenet264/, checkpoint -> checkpoints/exp_007_densenet264/best.pt):
python scripts/train.py --config configs/experiments/exp_007_densenet264.yaml --batch-size 4
```

`--epochs N` meng-override jumlah epoch; `--batch-size N` meng-override batch; `--device cpu|cuda|auto` memilih perangkat.

**Soal kecepatan (I/O-bound, bukan GPU-bound).** JPEG 512×512 dibaca serial dari disk USB eksternal. Dengan `num_workers=0` (aman di Jupyter Windows) laju per-epoch lambat; untuk lebih cepat naikkan `training.num_workers` (mis. 4) dan latih lewat runner terminal (Opsi B).

---

## 6. Evaluasi (selaras EXPERIMENT_PLAN §5 & §8)

- Metrik: macro-F1 (utama), balanced accuracy, accuracy, precision/recall/F1 per kelas + support, confusion matrix 7×7 (`zero_division=0`, label 0..6).
- Keluaran: `experiments/exp_007_densenet264/` (`config.yaml`, `metrics.json`, `training_history.csv`, `classification_report.txt`, `confusion_matrix.png`, `notes.md`); checkpoint `checkpoints/exp_007_densenet264/best.pt`.
- Test resmi dievaluasi **sekali** di akhir, bukan untuk memilih model.
- Analisis hasil lengkap ditulis di `docs/pelatihan/densenet264/ANALISIS_BASELINE_DENSENET264.md` **setelah** pelatihan dijalankan (belum dibuat; ikuti format [`../densenet121/ANALISIS_BASELINE_DENSENET121.md`](../densenet121/ANALISIS_BASELINE_DENSENET121.md)).

---

## 7. Catatan & keterbatasan

- DenseNet-264 adalah **batas atas kapasitas** pada keluarga DenseNet proyek ini; hipotesis: menangkap tekstur awan halus (Sc/Ci) lebih baik dari DenseNet-121 — tetapi pada `feature_extraction` (backbone beku) kenaikan belum tentu terwujud karena backbone tidak beradaptasi ke domain awan.
- Jika macro-F1 val mentok dengan `feature_extraction`, pertimbangkan `finetune: full` (lr kecil), dengan konsekuensi memori & waktu jauh lebih besar.
- Backbone besar + batch kecil (4) → estimasi gradien lebih berisik dan laju per-epoch lebih lambat; sabar memantau.
- Bobot `densenet264d` diunduh timm sekali di awal; butuh koneksi internet pada run pertama.
- Pelatihan **I/O-bound** pada disk USB (riwayat *intermittent read error*) — loader retry 3×; jalankan lewat runner terminal dan pantau.
- Pelatihan penuh **belum dijalankan**; dijalankan & dipantau sendiri oleh pengguna via notebook.

---

## 8. Referensi

- Huang, Liu, van der Maaten, Weinberger (2017), *Densely Connected Convolutional Networks* (DenseNet). https://arxiv.org/abs/1608.06993
- Loshchilov & Hutter (2019), *Decoupled Weight Decay Regularization* (AdamW). https://arxiv.org/abs/1711.05101
- timm (PyTorch Image Models), varian `densenet264d`: https://github.com/huggingface/pytorch-image-models
- Dokumen internal: [`ARSITEKTUR_BASELINE.md`](../ARSITEKTUR_BASELINE.md), [`BASELINE_DENSENET121.md`](../densenet121/BASELINE_DENSENET121.md), [`../alexnet/rasional-hyperparameter.md`](../alexnet/rasional-hyperparameter.md), [`EXPERIMENT_PLAN.md`](../EXPERIMENT_PLAN.md), [`PENYETARAAN_SAMPEL.md`](../../dataset/PENYETARAAN_SAMPEL.md).
