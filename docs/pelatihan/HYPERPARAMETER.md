# Hyperparameter yang Digunakan — Baseline exp_005–009

Dokumen **referensi** nilai hyperparameter yang benar-benar dipakai pada pelatihan baseline klasifikasi citra awan 7 kelas. Fokus di sini adalah **apa** nilainya dan **dari mana** asalnya; penjelasan **mengapa** tiap pilihan dibahas di dokumen rasional (lihat §6).

Tanggal: 19 Oktober 2026. Framework: PyTorch / torchvision (+ timm untuk DenseNet-264).
Sumber kebenaran: [`configs/base.yaml`](../../configs/base.yaml) + override [`configs/experiments/exp_005…009`](../../configs/experiments/).
Terkait: [`ARSITEKTUR_BASELINE.md`](ARSITEKTUR_BASELINE.md), [`PERSIAPAN_PELATIHAN.md`](PERSIAPAN_PELATIHAN.md), [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md), [`../dataset/PENYETARAAN_SAMPEL.md`](../dataset/PENYETARAAN_SAMPEL.md).

> **Prinsip inheritance.** `configs/base.yaml` memuat nilai **bersama**; tiap berkas `configs/experiments/exp_00X_*.yaml` hanya menuliskan nilai yang **berbeda** (override). Nilai efektif saat run = hasil *merge* keduanya (`src/utils/config.py` `load_config`). Karena itu hyperparameter inti **sama persis** lintas baseline exp_005–009 — hanya **arsitektur** (dan `imbalance`) yang berbeda — agar perbandingan antar-model adil.

---

## 1. Hyperparameter inti (bersama, dari `configs/base.yaml`)

Berlaku untuk **semua** baseline exp_005–009 kecuali yang di-override (§3).

| Kelompok | Parameter | Nilai | Catatan |
|---|---|---|---|
| Reproduksibilitas | `seed` | **42** | dikunci lintas eksperimen |
| Data | `image_size` | **512 × 512** RGB | resolusi asli; sumber sudah 512×512 |
| Data | `num_classes` | **7** | Cu, Ac, Ci, Cl, Sc, Cb, Mi |
| Praproses | `normalize` | **/255 → [0, 1]** | scale 255, mean 0, std 1 (tanpa standardisasi mean/std) |
| Praproses | `resize` / `interpolation` | 512 / bilinear | tanpa penskalaan (sumber sudah 512) |
| Augmentasi | `augmentation.enabled` | **false** | on-the-fly nonaktif (train sudah pra-seimbang, lihat §4) |
| Training | `epochs` | **200** | |
| Training | `batch_size` | **32** (config) → **8** saat run | 8 dipakai pada 512×512 di GPU 8 GB (lihat §4) |
| Training | `optimizer` | **AdamW** | decoupled weight decay |
| Training | `learning_rate` | **1e-4** (0,0001) | konstan |
| Training | `weight_decay` | **0.0** | |
| Training | `scheduler` | **null** | lr konstan, tanpa penjadwalan |
| Training | `num_workers` | **0** | aman di Jupyter Windows; naikkan via runner terminal |
| Checkpoint | `monitor` / `mode` | **val_macro_f1** / **max** | test **tidak pernah** dipakai memilih model |
| Checkpoint | `save_best_only` | **true** | simpan hanya epoch val-terbaik |
| Evaluasi | `primary_metric` | **macro_f1** | + balanced_accuracy, accuracy |
| Evaluasi | `per_class_metrics` / `confusion_matrix` | true / true | precision·recall·F1 per kelas + matriks 7×7 |

---

## 2. Model (transfer learning, pola bersama)

Semua baseline memakai pola yang sama: muat bobot **ImageNet** → ganti head klasifikasi ke `Linear(in_features → 7)` → mode **`feature_extraction`** (backbone beku, hanya head dilatih). Dibangun oleh [`src/models/baseline.py`](../../src/models/baseline.py) `build_model(...)`.

| Parameter | Nilai |
|---|---|
| `pretrained` | `imagenet` |
| `finetune` | `feature_extraction` (opsi lanjut: `full`) |
| `num_classes` | 7 |

---

## 3. Override per eksperimen (exp_005–009)

Setiap berkas eksperimen hanya mengubah **arsitektur**, **sumber bobot**, dan **`imbalance`**. Hyperparameter inti (§1) & pola model (§2) tetap sama.

| Exp | Config | Arsitektur | Sumber | Head (in_features→7) | Param head dilatih¹ | `imbalance` |
|---|---|---|---|---|---:|---|
| exp_005 | [`exp_005_densenet121.yaml`](../../configs/experiments/exp_005_densenet121.yaml) | DenseNet-121 | torchvision | `classifier` 1024→7 | 7.175 | none |
| exp_006 | [`exp_006_alexnet.yaml`](../../configs/experiments/exp_006_alexnet.yaml) | AlexNet | torchvision | `classifier[6]` 4096→7 | 28.679 | none |
| exp_007 | [`exp_007_densenet264.yaml`](../../configs/experiments/exp_007_densenet264.yaml) | DenseNet-264 | timm | `classifier` 2688→7 | 18.823 | none |
| exp_008 | [`exp_008_resnet18.yaml`](../../configs/experiments/exp_008_resnet18.yaml) | ResNet-18 | torchvision | `fc` 512→7 | 3.591 | none |
| exp_009 | [`exp_009_resnet50.yaml`](../../configs/experiments/exp_009_resnet50.yaml) | ResNet-50 | torchvision | `fc` 2048→7 | 14.343 | none |

¹ Jumlah parameter head yang dilatih pada `feature_extraction` = `in_features × 7 + 7` (bias). Parameter **total** per arsitektur jauh lebih besar (backbone beku) — lihat dokumen baseline masing-masing.

**Data (override bersama exp_005–009):** train = `data/combined_balanced/train` (seimbang 3.000/kelas = 21.000), val/test = combined apa adanya (1.851 / 9.000). Total data pelatihan = **31.851** citra.

---

## 4. Catatan penting (hal yang sering keliru)

1. **`batch_size` 32 vs 8.** `base.yaml` menulis 32 sebagai default bersama, tetapi pada input **512×512** di GPU **RTX 4060 Laptop 8 GB** dipakai **batch 8** (notebook & argumen `--batch-size 8`) agar tidak OOM. Batch efektif yang dipakai **wajib dicatat** di `notes.md` tiap run.
2. **`imbalance: none` pada baseline, bukan `class_weights`.** `base.yaml` default-nya `imbalance: class_weights` (untuk jalur GCD-only dengan train timpang). Namun exp_005–009 memakai train **sudah seimbang** (`combined_balanced`, 3.000/kelas), jadi di-override ke **`none`** agar kelas tidak dikoreksi dua kali (koreksi ganda). Lihat [`PENYETARAAN_SAMPEL.md`](../dataset/PENYETARAAN_SAMPEL.md) §2.3.
3. **Augmentasi `enabled: false`, tapi train tetap "teraugmentasi".** Augmentasi sudah **dibakar ke berkas** saat pra-penyetaraan (Opsi A → 10.517 citra augmentasi di `combined_balanced/train`); augmentasi acak on-the-fly saat training sengaja dimatikan. Lihat [`AUGMENTASI_DATA.md`](../dataset/AUGMENTASI_DATA.md).
4. **Normalisasi [0,1], bukan mean/std ImageNet.** Keputusan praproses eksplisit & seragam lintas baseline; konsekuensinya dibahas di dokumen rasional.
5. **Pemilihan model dari validation.** Checkpoint terbaik dipilih dari `val_macro_f1`; **test resmi dievaluasi sekali** di akhir dan tidak pernah dipakai memilih model.
6. **Tanpa scheduler & tanpa early stopping.** lr konstan 1e-4; "pengaman" overfitting = checkpoint by val-terbaik + `feature_extraction` + train seimbang.

---

## 5. Melihat nilai efektif (verifikasi)

Nilai efektif adalah hasil *merge* `base.yaml` + override; jangan menebak — baca langsung:

```python
from src.utils.config import load_config
cfg = load_config("configs/experiments/exp_008_resnet18.yaml")
print(cfg["training"])   # epochs, batch_size, optimizer, learning_rate, weight_decay, scheduler, imbalance, num_workers
print(cfg["model"])      # architecture, pretrained, finetune, num_classes
print(cfg["checkpoint"]) # monitor, mode, save_best_only
```

Atau validasi pra-run tanpa melatih:

```powershell
python scripts/train.py --config configs/experiments/exp_008_resnet18.yaml --dry-run
```

Saat run sungguhan, konfigurasi efektif lengkap disalin ke `experiments/<exp>/config.yaml` (snapshot reproduksibel).

---

## 6. Rasional ("mengapa" tiap nilai)

Dokumen ini hanya mencatat **nilai**. Alasan pemilihan tiap hyperparameter dan mengapa kombinasinya koheren:

- Rasional bersama (berbasis AlexNet, berlaku lintas baseline): [`alexnet/rasional-hyperparameter.md`](alexnet/rasional-hyperparameter.md)
- Khas ResNet (skip connection, BatchNorm vs Dropout): [`resnet18/rasional-hyperparameter.md`](resnet18/rasional-hyperparameter.md)

---

## 7. Rujukan

- Konfigurasi: [`configs/base.yaml`](../../configs/base.yaml), [`configs/experiments/`](../../configs/experiments/)
- Loader merge: [`src/utils/config.py`](../../src/utils/config.py) (`load_config`)
- Arsitektur baseline: [`ARSITEKTUR_BASELINE.md`](ARSITEKTUR_BASELINE.md)
- Penyetaraan & total data: [`../dataset/PENYETARAAN_SAMPEL.md`](../dataset/PENYETARAAN_SAMPEL.md)
- Rencana & kontrak keluaran: [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md), [`TRAINING_GUIDE.md`](TRAINING_GUIDE.md)
- Loshchilov & Hutter (2019), *Decoupled Weight Decay Regularization* (AdamW): https://arxiv.org/abs/1711.05101

---

## 8. Tabel ringkas untuk laporan/jurnal

Parameter yang umum dicantumkan pada bagian *Experimental Setup / Training Configuration* sebuah jurnal, diisi dengan nilai eksperimen ini (baseline exp_005–009). Siap disalin ke tabel laporan.

| Parameter | Nilai |
|---|---|
| Ukuran citra masukan (*input size*) | 512 × 512 px, RGB |
| Praproses / normalisasi | *Rescale* 1/255 → [0, 1] (tanpa mean/std) |
| Augmentasi data | *Offline* penyetaraan ke 3.000/kelas (translasi, flip horizontal, rotasi ±15°, color/contrast/brightness); *on-the-fly* nonaktif |
| Pembagian data (train/val/test) | 21.000 / 1.851 / 9.000 (total **31.851**) |
| Jumlah kelas | 7 (Cu, Ac, Ci, Cl, Sc, Cb, Mi) |
| Arsitektur / *backbone* | DenseNet-121 · AlexNet · DenseNet-264 · ResNet-18 · ResNet-50 |
| Bobot pra-latih (*pretrained*) | ImageNet (IMAGENET1K_V1) |
| Strategi transfer learning | *Feature extraction* (backbone dibekukan) |
| Fungsi rugi (*loss*) | *Cross-Entropy* (tanpa bobot kelas) |
| *Optimizer* | AdamW |
| *Learning rate* | 1 × 10⁻⁴ (konstan) |
| *Weight decay* | 0 |
| *LR scheduler* | Tidak ada |
| *Batch size* | 8 |
| Jumlah *epoch* | 200 |
| *Early stopping* | Tidak ada (*best checkpoint* via val macro-F1) |
| Kriteria pemilihan model | Validation macro-F1 (maks) |
| Metrik evaluasi | Macro-F1 (utama), *balanced accuracy*, *accuracy*, P/R/F1 per kelas, *confusion matrix* |
| *Framework* | PyTorch 2.1.1 + torchvision 0.16.1 |
| Perangkat / GPU | NVIDIA RTX 4060 Laptop (8 GB), CUDA 12.1 |
| *Random seed* | 42 |

> Catatan: *batch size* yang dilaporkan = **8** (nilai efektif pada 512×512, GPU 8 GB); `configs/base.yaml` mencatat 32 sebagai default bersama. Baris **Arsitektur** diisi sesuai eksperimen yang dilaporkan (satu baris per model bila tabel dipecah per eksperimen).
