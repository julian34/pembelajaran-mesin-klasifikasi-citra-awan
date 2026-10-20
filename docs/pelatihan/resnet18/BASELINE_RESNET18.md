# Baseline ResNet-18 — Klasifikasi Citra Awan (Transfer Learning ImageNet)

Dataset train: **combined_balanced** (GCD+CCSN, 3.000/kelas = 21.000 citra). Framework: **PyTorch / torchvision**.
Tanggal: 18 Oktober 2026. Terkait: [`ARSITEKTUR_BASELINE.md`](../ARSITEKTUR_BASELINE.md), [`ANALISIS_BASELINE_RESNET18.md`](ANALISIS_BASELINE_RESNET18.md), [`PENYETARAAN_SAMPEL.md`](../../dataset/PENYETARAAN_SAMPEL.md), [`PENGGABUNGAN_DATASET.md`](../../dataset/PENGGABUNGAN_DATASET.md), [`EXPERIMENT_PLAN.md`](../EXPERIMENT_PLAN.md).

Dokumen ini mendokumentasikan **baseline ResNet-18**: transfer learning dengan bobot pra-latih **ImageNet**, lapisan klasifikasi terakhir diganti agar keluaran = **7 kelas** awan. **Hyperparameter sama persis dengan baseline AlexNet (exp_006) dan DenseNet-121 (exp_005)** — hanya arsitekturnya yang berbeda, agar perbandingan antar-model adil (lihat [`ARSITEKTUR_BASELINE.md`](../ARSITEKTUR_BASELINE.md) dan rasional hyperparameter bersama di [`../alexnet/rasional-hyperparameter.md`](../alexnet/rasional-hyperparameter.md)). Status: **pelatihan penuh belum dijalankan** — jalankan & pantau sendiri via notebook. Kode:

- Model: [`src/models/baseline.py`](../../../src/models/baseline.py) (`build_model`, arsitektur `resnet18`).
- Data: [`src/datasets/manifest.py`](../../../src/datasets/manifest.py) (resolusi path/label) + [`src/datasets/cloud_dataset.py`](../../../src/datasets/cloud_dataset.py) (`Dataset`/`DataLoader` + transform /255).
- Pelatihan: [`src/training/trainer.py`](../../../src/training/trainer.py) (`fit`, checkpoint by `val_macro_f1`; opsi `progress=True` → progress bar per-epoch gaya Keras via `pkbar`).
- Evaluasi: [`src/evaluation/metrics.py`](../../../src/evaluation/metrics.py).
- Config: [`configs/experiments/exp_008_resnet18.yaml`](../../../configs/experiments/exp_008_resnet18.yaml); runner: [`scripts/train.py`](../../../scripts/train.py); notebook pemantau: [`notebooks/06_resnet18_training.ipynb`](../../../notebooks/06_resnet18_training.ipynb).

---

## 1. Ringkasan

| Butir | Nilai |
|---|---|
| Arsitektur | ResNet-18 (4 tahap *residual*, tiap tahap 2 `BasicBlock` dengan *skip connection* identitas) |
| Strategi | Transfer learning, bobot pra-latih ImageNet |
| Bobot | `torchvision` `ResNet18_Weights.IMAGENET1K_V1` |
| Perubahan head | `fc`: `Linear(512, 1000)` → `Linear(512, 7)` |
| Regularisasi | — (ResNet torchvision tidak punya dropout FC bawaan; andalkan `feature_extraction` + train seimbang; BatchNorm bawaan di tiap blok) |
| Train | combined_balanced, 3.000/kelas (21.000 citra) |
| Val / test | combined apa adanya (1.851 / 9.000) — **tidak diseimbangkan** |
| Ketidakseimbangan | `imbalance: none` (train sudah seimbang; tak ada koreksi ganda) |
| Input | 512 × 512 RGB (resolusi asli; sumber sudah 512×512) |
| Normalisasi | bagi tiap piksel dengan 255 → rentang [0, 1] (tanpa standardisasi mean/std) |
| Parameter | total **11.180.103** · trainable **3.591** (head `Linear(512→7)` saja) |
| Metrik utama | macro-F1 (+ balanced accuracy, accuracy, per kelas, confusion matrix) |

---

## 2. Arsitektur ResNet-18 (He dkk., 2016)

ResNet (*Residual Network*) memperkenalkan **koneksi pintas (skip/residual connection)**: tiap blok mempelajari **residu** `F(x)` lalu menjumlahkannya dengan masukan, `y = F(x) + x`. Penjumlahan identitas ini membuat gradien mengalir langsung ke lapisan awal sehingga jaringan dalam **tetap mudah dilatih** (mengatasi *degradation problem*). ResNet-18 memakai blok dasar **`BasicBlock`** (dua konvolusi 3×3 + BatchNorm + ReLU, lalu dijumlah dengan jalur pintas).

Struktur ResNet-18 (varian torchvision):

1. **Stem** — Conv 7×7 stride 2 (64 filter) + BatchNorm + ReLU + max pool 3×3 stride 2.
2. **Stage 1 (`layer1`)** — 2 `BasicBlock`, 64 filter (peta fitur tetap).
3. **Stage 2 (`layer2`)** — 2 `BasicBlock`, 128 filter (blok pertama *downsample* stride 2; pintas pakai Conv 1×1).
4. **Stage 3 (`layer3`)** — 2 `BasicBlock`, 256 filter (*downsample*).
5. **Stage 4 (`layer4`)** — 2 `BasicBlock`, 512 filter (*downsample*).
6. **Head** — global average pool → `Linear(512, 1000)` (head ImageNet).

Angka "18" = total lapisan berbobot: 1 (stem conv) + 16 konvolusi dalam blok (4 tahap × 2 blok × 2 conv) + 1 FC = **18**. Setelah Stage 4 + global average pool, dimensi fitur = **512**, itulah `in_features` head. ResNet-18 adalah varian ResNet **paling ringan** (BasicBlock, bukan Bottleneck) — pembanding cepat terhadap ResNet-50/DenseNet.

Diagram (animasi, buka di browser):
- **Arsitektur — residual connection** (ciri khas ResNet: `y = F(x) + x` pada tiap blok): [`arsitektur-resnet18-residual.html`](arsitektur-resnet18-residual.html).
- **Lapisan awal→akhir** (ukuran peta fitur tiap tahap, input 512×512 → 7 kelas): [`arsitektur-resnet18-lapisan.html`](arsitektur-resnet18-lapisan.html).
- **Pipeline** (data → model → latih → evaluasi): [`arsitektur-exp008.html`](arsitektur-exp008.html).

---

## 3. Transfer learning ke 7 kelas

Pola sama dengan [`ARSITEKTUR_BASELINE.md`](../ARSITEKTUR_BASELINE.md) Bagian 3:

1. Muat ResNet-18 dengan bobot ImageNet.
2. Ganti **lapisan terakhir** `model.fc` (sebuah `nn.Linear(512, 1000)`) menjadi `nn.Linear(512, 7)`. Lapisan lain dibiarkan utuh.
3. Pilih mode: `feature_extraction` (bekukan backbone, latih head saja) atau `full` (fine-tuning seluruh lapisan).

```python
from src.models.baseline import build_model, count_parameters

# fc (Linear 512 -> 1000) diganti -> Linear(512, 7)
model = build_model("resnet18", num_classes=7, pretrained=True, finetune="feature_extraction")
print(model.fc)                # Linear(512, 7) — head keluaran
print(count_parameters(model)) # (11_180_103, 3_591)
```

### 3.1 Catatan regularisasi

Berbeda dari AlexNet (yang punya Dropout(0.5) bawaan pada dua FC tersembunyi), ResNet-18 torchvision **tidak** menyisipkan dropout pada head. Untuk baseline ini tidak ada dropout tambahan yang disisipkan; pengendali overfitting adalah:

- **`feature_extraction`** — hanya head `Linear(512→7)` (**3.591** parameter) yang dilatih; backbone beku sehingga kapasitas belajar sangat terbatas → risiko overfitting rendah.
- **BatchNorm** di tiap blok backbone (beku saat `feature_extraction`, memakai statistik ImageNet) menstabilkan fitur.
- **Train sudah diseimbangkan** (3.000/kelas) sehingga tidak ada koreksi bobot ganda.
- **Pemilihan checkpoint by `val_macro_f1`** (bukan early stopping) menjaga model akhir = epoch val-terbaik.

Normalisasi memakai **pembagian /255 ke rentang [0, 1]** (tanpa mean/std); lihat [`PERSIAPAN_PELATIHAN.md`](../PERSIAPAN_PELATIHAN.md). Catatan: bobot pra-latih ImageNet aslinya dilatih dengan standardisasi mean/std ImageNet, jadi penggunaan [0,1] adalah pilihan praproses eksplisit proyek ini (seragam lintas baseline).

---

## 4. Data: combined_balanced

Identik dengan baseline AlexNet & DenseNet-121 (data dikontrol sama agar hanya arsitektur yang berbeda). Train memakai set **seimbang 3.000 citra/kelas** (total 21.000) hasil penyetaraan dataset gabungan GCD+CCSN, sudah dimaterialisasi di `data/combined_balanced/train/<kelas>/`:

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

Config: [`configs/experiments/exp_008_resnet18.yaml`](../../../configs/experiments/exp_008_resnet18.yaml) (override terhadap `configs/base.yaml`).

```yaml
model:
  architecture: resnet18
  pretrained: imagenet          # ResNet18_Weights.IMAGENET1K_V1
  source: torchvision
  num_classes: 7
  finetune: feature_extraction  # feature_extraction | full

data:
  train_dir: data/combined_balanced/train
  train_manifest: reports/tables/combined_balanced_train_manifest.csv
  val_dir: data/combined/val      # sama dengan exp_005 & exp_006
  val_manifest: reports/tables/combined_val_manifest.csv
  test_dir: data/combined/test
  test_manifest: reports/tables/combined_test_manifest.csv

training:
  imbalance: none
```

Hyperparameter efektif (diwarisi dari `configs/base.yaml`, **sama dengan exp_005 & exp_006**): epochs 200, batch_size 32, optimizer adamw, lr 0,0001 (1e-4), weight_decay 0,0, scheduler null. Rasional pemilihan kombinasi ini (berlaku lintas baseline exp_005–009) dibahas di [`../alexnet/rasional-hyperparameter.md`](../alexnet/rasional-hyperparameter.md) dan ringkasan khusus ResNet di [`rasional-hyperparameter.md`](rasional-hyperparameter.md). Pemilihan checkpoint: `val_macro_f1` (test tidak dipakai untuk memilih). Catatan VRAM: ResNet-18 **lebih ringan memori** dari DenseNet-121 pada input 512×512; notebook tetap memakai **batch 8** demi perbandingan setara (boleh dinaikkan bila VRAM lega).

### 5.1 Menjalankan

**Opsi A — notebook (pantau sendiri, disarankan untuk eksplorasi):** buka [`notebooks/06_resnet18_training.ipynb`](../../../notebooks/06_resnet18_training.ipynb), jalankan sel atas→bawah. Sel 5 melatih dengan **progress bar per-epoch (pkbar)** dan Sel 5b menampilkan tabel + grafik riwayat. Untuk uji cepat set `EPOCHS=1`.

**Opsi B — runner terminal (lebih cepat, pemuatan paralel):**
```powershell
# 1) Validasi config + manifes + file tanpa membangun model/melatih:
python scripts/train.py --config configs/experiments/exp_008_resnet18.yaml --dry-run

# 2) Latih baseline (artefak -> experiments/exp_008_resnet18/, checkpoint -> checkpoints/exp_008_resnet18/best.pt):
python scripts/train.py --config configs/experiments/exp_008_resnet18.yaml --batch-size 8
```

`--epochs N` meng-override jumlah epoch; `--batch-size N` meng-override batch; `--device cpu|cuda|auto` memilih perangkat.

**Soal kecepatan (I/O-bound, bukan GPU-bound).** Citra dibaca sebagai JPEG 512×512 dari disk USB eksternal secara serial. Dengan `num_workers=0` (default aman di Jupyter Windows) laju per-epoch lambat; untuk jauh lebih cepat, naikkan `training.num_workers` (mis. 4) dan latih lewat **runner terminal** (Opsi B).

---

## 6. Evaluasi (selaras EXPERIMENT_PLAN §5 & §8)

- Metrik: macro-F1 (utama), balanced accuracy, accuracy, precision/recall/F1 per kelas + support, confusion matrix 7×7 (`zero_division=0`, label 0..6).
- Keluaran per eksperimen: `experiments/exp_008_resnet18/` (`config.yaml`, `metrics.json`, `training_history.csv`, `classification_report.txt`, `confusion_matrix.png`, `notes.md`); checkpoint `checkpoints/exp_008_resnet18/best.pt`.
- Test resmi dievaluasi **sekali** di akhir, bukan untuk memilih model.
- Analisis hasil lengkap ditulis di [`ANALISIS_BASELINE_RESNET18.md`](ANALISIS_BASELINE_RESNET18.md) **setelah** pelatihan dijalankan.

---

## 7. Catatan & keterbatasan

- ResNet-18 adalah pembanding **ringan ber-skip-connection**: lebih dalam dari AlexNet namun jauh lebih sedikit parameter total (±11,2 juta vs 57 juta AlexNet), dan lebih dangkal dari DenseNet-121/ResNet-50.
- Pada `feature_extraction` hanya head (3.591 parameter) yang belajar; bila macro-F1 val mentok, pertimbangkan `finetune: full` dengan lr tetap kecil (BatchNorm akan ikut diperbarui).
- Train seimbang (21.000) berisi ~50% citra augmentasi; pantau selisih train vs val macro-F1 sebagai indikator overfitting.
- Pelatihan **I/O-bound** pada disk USB (riwayat *intermittent read error*) — jalankan lewat runner terminal dan pantau.
- Pelatihan penuh **belum dijalankan**; dijalankan & dipantau sendiri oleh pengguna via notebook (progress bar pkbar per epoch).

---

## 8. Referensi

- He, Zhang, Ren, Sun (2016), *Deep Residual Learning for Image Recognition* (ResNet). https://arxiv.org/abs/1512.03385
- Loshchilov & Hutter (2019), *Decoupled Weight Decay Regularization* (AdamW). https://arxiv.org/abs/1711.05101
- torchvision ResNet-18: https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
- Dokumen internal: [`ARSITEKTUR_BASELINE.md`](../ARSITEKTUR_BASELINE.md), [`../alexnet/rasional-hyperparameter.md`](../alexnet/rasional-hyperparameter.md), [`EXPERIMENT_PLAN.md`](../EXPERIMENT_PLAN.md), [`PENYETARAAN_SAMPEL.md`](../../dataset/PENYETARAAN_SAMPEL.md).
