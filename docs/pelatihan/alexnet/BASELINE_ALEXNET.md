# Baseline AlexNet — Klasifikasi Citra Awan (Transfer Learning ImageNet)

Dataset train: **combined_balanced** (GCD+CCSN, 3.000/kelas = 21.000 citra). Framework: **PyTorch / torchvision**.
Tanggal: 14 Oktober 2026. Terkait: [`ARSITEKTUR_BASELINE.md`](../ARSITEKTUR_BASELINE.md), [`ANALISIS_BASELINE_ALEXNET.md`](ANALISIS_BASELINE_ALEXNET.md), [`PENYETARAAN_SAMPEL.md`](../../dataset/PENYETARAAN_SAMPEL.md), [`PENGGABUNGAN_DATASET.md`](../../dataset/PENGGABUNGAN_DATASET.md), [`EXPERIMENT_PLAN.md`](../EXPERIMENT_PLAN.md).

Dokumen ini mendokumentasikan **baseline AlexNet**: transfer learning dengan bobot pra-latih **ImageNet**, lapisan klasifikasi terakhir diganti agar keluaran = **7 kelas** awan. Regularisasi **Dropout(0.5)** adalah bawaan desain AlexNet pada **dua lapisan fully-connected tersembunyi** (FC-6 & FC-7). Status: **pelatihan penuh (200 epoch) sudah dijalankan** — `torch`/`torchvision` build CUDA cu121 terpasang (GPU RTX 4060 Laptop). Hasil: best epoch 63, **test macro-F1 0,5042** (artefak di `experiments/exp_006_alexnet/`); analisis lengkap (waktu latih, semua metrik, F1 per kelas, confusion matrix) di **[`ANALISIS_BASELINE_ALEXNET.md`](ANALISIS_BASELINE_ALEXNET.md)**. Kode:

- Model: [`src/models/baseline.py`](../../../src/models/baseline.py) (`build_model`).
- Data: [`src/datasets/manifest.py`](../../../src/datasets/manifest.py) (resolusi path/label) + [`src/datasets/cloud_dataset.py`](../../../src/datasets/cloud_dataset.py) (`Dataset`/`DataLoader` + transform /255).
- Pelatihan: [`src/training/trainer.py`](../../../src/training/trainer.py) (`fit`, checkpoint by `val_macro_f1`; opsi `progress=True` → progress bar per-epoch gaya Keras via `pkbar`).
- Evaluasi: [`src/evaluation/metrics.py`](../../../src/evaluation/metrics.py).
- Config: [`configs/experiments/exp_006_alexnet.yaml`](../../../configs/experiments/exp_006_alexnet.yaml); runner: [`scripts/train.py`](../../../scripts/train.py); notebook pemantau: [`notebooks/04_baseline_training.ipynb`](../../../notebooks/04_baseline_training.ipynb).

Bagian bebas-torch (resolusi manifes & metrik) sudah diuji di [`tests/test_pipeline.py`](../../../tests/test_pipeline.py) dan resolusi path diverifikasi terhadap manifes nyata (train 21.000 / val 1.851 / test 9.000, 0 file hilang).

---

## 1. Ringkasan

| Butir | Nilai |
|---|---|
| Arsitektur | AlexNet (8 lapisan berbobot: 5 konvolusi + 3 fully-connected) |
| Strategi | Transfer learning, bobot pra-latih ImageNet |
| Bobot | `torchvision` `AlexNet_Weights.IMAGENET1K_V1` |
| Perubahan head | `classifier[6]`: `Linear(4096, 1000)` → `Linear(4096, 7)` |
| Regularisasi | Dropout(0.5) bawaan pada 2 FC tersembunyi (`classifier[0]` sebelum FC-6, `classifier[3]` sebelum FC-7) |
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

Diagram (animasi, buka di browser):
- **Arsitektur lapisan awal→akhir** (5 konvolusi + 3 fully-connected, ukuran peta fitur tiap tahap, Dropout FC): [`arsitektur-alexnet-lapisan.html`](arsitektur-alexnet-lapisan.html).
- **Pipeline** (data → model → latih → evaluasi): [`arsitektur-exp006.html`](arsitektur-exp006.html).

---

## 3. Transfer learning ke 7 kelas

Pola sama dengan [`ARSITEKTUR_BASELINE.md`](../ARSITEKTUR_BASELINE.md) Bagian 3:

1. Muat AlexNet dengan bobot ImageNet.
2. Ganti **lapisan terakhir** `model.classifier[6]` (sebuah `nn.Linear(4096, 1000)`) menjadi `nn.Linear(4096, 7)`. Lapisan lain — termasuk kedua `Dropout(0.5)` — dibiarkan utuh.
3. Pilih mode: `feature_extraction` (bekukan backbone, latih head saja) atau `full` (fine-tuning seluruh lapisan).

```python
from src.models.baseline import build_model, count_parameters

# classifier[6] (Linear 4096 -> 1000) diganti -> Linear(4096, 7)
model = build_model("alexnet", num_classes=7, pretrained=True, finetune="full")
print(model.classifier[0])     # Dropout(p=0.5) — sebelum FC-6
print(model.classifier[3])     # Dropout(p=0.5) — sebelum FC-7
print(model.classifier[6])     # Linear(4096, 7) — head keluaran
print(count_parameters(model)) # (total, trainable)
```

### 3.1 Dropout(0.5)

Dropout(0.5) adalah **regularisasi bawaan AlexNet**, diterapkan pada **dua lapisan fully-connected tersembunyi pertama** (FC-6 dan FC-7). Di `torchvision.models.alexnet` ia berada di `classifier[0]` (sebelum FC-6) dan `classifier[3]` (sebelum FC-7); tidak ada dropout sebelum lapisan keluaran.

Mekanisme:

- **Forward pass:** pada tiap iterasi, **50% neuron** pada lapisan tersebut dimatikan secara acak (keluarannya dinolkan); neuron yang bertahan diskalakan `1/(1−p)` (inverted dropout) agar ekspektasi aktivasi tetap. Ini mencegah ko-adaptasi antar-neuron sehingga model tidak bergantung pada fitur tunggal.
- **Backpropagation:** gradien hanya mengalir mundur melalui **neuron yang aktif** pada iterasi itu; neuron yang dimatikan menerima gradien 0 sehingga bobotnya tidak diperbarui pada langkah tersebut.
- **Train vs eval:** dropout **hanya aktif saat `model.train()`**. Saat `model.eval()` (validasi/test/inferensi) dropout **mati** — seluruh neuron dipakai tanpa penskalaan acak — sehingga prediksi deterministik. `trainer.fit`/`evaluate` sudah mengatur `train()`/`eval()` otomatis.

Dropout tidak punya parameter yang dipelajari, jadi memuat bobot ImageNet tidak mengubah perilakunya. Untuk baseline ini penempatannya **tidak diubah**; regularisasi tambahan (mis. weight decay) diatur lewat `configs/base.yaml`.

Normalisasi memakai **pembagian /255 ke rentang [0, 1]** (tanpa mean/std); lihat [`PERSIAPAN_PELATIHAN.md`](../PERSIAPAN_PELATIHAN.md). Catatan: bobot pra-latih ImageNet aslinya dilatih dengan standardisasi mean/std ImageNet, jadi penggunaan [0,1] adalah pilihan praproses eksplisit proyek ini.

---

## 4. Data: combined_balanced

Train memakai set **seimbang 3.000 citra/kelas** (total 21.000) hasil penyetaraan dataset gabungan GCD+CCSN, sudah dimaterialisasi di `data/combined_balanced/train/<kelas>/`:

| Komponen | Nilai | Sumber |
|---|---|---|
| Train (seimbang) | 21.000 (7 × 3.000; 10.483 asli + 10.517 augmentasi) | `reports/tables/combined_balanced_train_manifest.csv` (kolom `class`, `filename`, `kind`, `split`, `rel_path`) |
| Validation | 1.851 (distribusi nyata) | `reports/tables/combined_val_manifest.csv` |
| Test | 9.000 (distribusi nyata) | `reports/tables/combined_test_manifest.csv` |
| **Total** | **31.851** | train seimbang + val + test |

Prinsip (lihat [`PENYETARAAN_SAMPEL.md`](../../dataset/PENYETARAAN_SAMPEL.md)):
- **Hanya train** yang diseimbangkan; **val/test tetap apa adanya** agar evaluasi jujur dan sebanding.
- Karena train sudah seimbang, `training.imbalance: none` — **jangan** sekaligus memakai `class_weights` (koreksi ganda).
- Kolom `kind` membedakan citra `original` vs `augmented`; augmentasi hanya pada train.

---

## 5. Konfigurasi & hyperparameter

Config: [`configs/experiments/exp_006_alexnet.yaml`](../../../configs/experiments/exp_006_alexnet.yaml) (override terhadap `configs/base.yaml`).

```yaml
model:
  architecture: alexnet
  pretrained: imagenet          # AlexNet_Weights.IMAGENET1K_V1
  source: torchvision
  num_classes: 7
  finetune: feature_extraction  # feature_extraction | full
  # Dropout(0.5) bawaan AlexNet (FC-6 & FC-7); tak perlu disetel — lihat §3.1.

data:
  train_dir: data/combined_balanced/train
  train_manifest: reports/tables/combined_balanced_train_manifest.csv
  val_dir: data/combined/val      # file fisik val (manifes val mencatat path sumber)
  val_manifest: reports/tables/combined_val_manifest.csv
  test_dir: data/combined/test
  test_manifest: reports/tables/combined_test_manifest.csv

training:
  imbalance: none
```

Hyperparameter efektif (diwarisi dari `configs/base.yaml`): epochs 200, batch_size 32, optimizer adamw, lr 0,0001 (1e-4), weight_decay 0,0, scheduler null. Rasional pemilihan kombinasi ini dibahas rinci di [`rasional-hyperparameter.md`](rasional-hyperparameter.md). Pemilihan checkpoint: `val_macro_f1` (test tidak dipakai untuk memilih). Catatan VRAM: input 512×512 di GPU 8 GB (RTX 4060 Laptop) nyaman di **batch 8** — notebook memakai default itu.

### 5.1 Menjalankan

**Lingkungan sudah siap:** `torch 2.1.1+cu121` / `torchvision 0.16.1+cu121` terpasang di `.venv`, `pkbar` terpasang, dan GPU terdeteksi (`torch.cuda.is_available() == True`).

**Opsi A — notebook (pantau sendiri, disarankan untuk eksplorasi):** buka [`notebooks/04_baseline_training.ipynb`](../../../notebooks/04_baseline_training.ipynb), jalankan sel atas→bawah. Sel 5 melatih dengan **progress bar per-epoch (pkbar)** dan Sel 5b menampilkan tabel + grafik riwayat. Untuk uji cepat set `EPOCHS=1`.

**Opsi B — runner terminal (lebih cepat, pemuatan paralel):**
```powershell
# 1) (Sekali) pasang dependensi bila di lingkungan baru:
pip install -r requirements.txt

# 2) Validasi config + manifes + file tanpa membangun model/melatih:
python scripts/train.py --config configs/experiments/exp_006_alexnet.yaml --dry-run

# 3) Latih baseline (artefak -> experiments/exp_006_alexnet/, checkpoint -> checkpoints/exp_006_alexnet/best.pt):
python scripts/train.py --config configs/experiments/exp_006_alexnet.yaml --batch-size 8
```

`--epochs N` meng-override jumlah epoch; `--batch-size N` meng-override batch; `--device cpu|cuda|auto` memilih perangkat.

**Soal kecepatan (I/O-bound, bukan GPU-bound).** Citra dibaca sebagai JPEG 512×512 dari disk USB eksternal secara serial; komputasi GPU hanya ~4% waktu. Dengan `num_workers=0` (default aman di Jupyter Windows): ~18 menit/epoch, ~15 jam untuk 50 epoch. Untuk jauh lebih cepat, naikkan `training.num_workers` (mis. 4) dan latih lewat **runner terminal** (Opsi B) — `DataLoader` worker >0 sering tidak stabil di dalam notebook Windows.

---

## 6. Evaluasi (selaras EXPERIMENT_PLAN §5 & §8)

- Metrik: macro-F1 (utama), balanced accuracy, accuracy, precision/recall/F1 per kelas + support, confusion matrix 7×7 (`zero_division=0`, label 0..6).
- Keluaran per eksperimen: `experiments/exp_006_alexnet/` (`config.yaml`, `metrics.json`, `training_history.csv`, `classification_report.txt`, `confusion_matrix.png`, `notes.md`); checkpoint `checkpoints/exp_006_alexnet/best.pt`.
- Test resmi dievaluasi **sekali** di akhir, bukan untuk memilih model.

---

## 7. Catatan & keterbatasan

- Bobot pra-latih yang dipakai adalah **varian torchvision** AlexNet (64/192 channel, tanpa LRN), bukan rancangan dua-GPU asli pada §2 — dicatat agar laporan akurat.
- AlexNet relatif dangkal/klasik; berguna sebagai **baseline ringan** pembanding terhadap DenseNet/ResNet pada data dan metrik yang sama.
- Train seimbang (21.000) berisi ~50% citra augmentasi; variasi augmentasi untuk kelas minoritas (mis. Mi ~10×) terbatas — perhatikan risiko overfitting pola augmentasi (lihat `PENYETARAAN_SAMPEL.md` §4.3). **Dropout(0.5) bawaan** pada FC-6/FC-7 (§3.1) membantu meredam: meski pada `feature_extraction` bobot FC-6/FC-7 beku, dropout tetap bekerja di *forward pass* mode latih sehingga fitur yang sampai ke head tetap teregularisasi. Karena hanya head yang belajar, pantau selisih train vs val macro-F1 sebagai indikator overfitting.
- Lingkungan: `torch`/`torchvision` build **cu121 (GPU)** terpasang; pelatihan **I/O-bound** pada disk USB (~15 jam/50 epoch, `num_workers=0`) — lihat §5.1 untuk mempercepat lewat runner terminal + `num_workers`.
- Pelatihan penuh **belum dijalankan**; dijalankan & dipantau sendiri oleh pengguna via notebook (progress bar pkbar per epoch).
