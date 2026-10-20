# CitraAwan

Klasifikasi citra awan berbasis darat dengan deep learning (PyTorch).

Status: dua dataset tersedia (**GCD** dan **CCSN_v2**). EDA, penanganan kualitas, pembagian split, dan penggabungan sudah selesai (berbasis manifest di `reports/`). Pipa pelatihan+evaluasi baseline sudah ditulis (`src/`, `scripts/train.py`; baseline pertama: AlexNet exp_006, lihat [docs/pelatihan/alexnet/BASELINE_ALEXNET.md](docs/pelatihan/alexnet/BASELINE_ALEXNET.md)). `torch`/`torchvision` build CUDA cu121 sudah terpasang (GPU RTX 4060 Laptop); **pelatihan penuh belum dijalankan**.

## Menyiapkan lingkungan

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Opsional, untuk menjalankan notebook dari terminal:
pip install -r requirements-dev.txt
```

## Struktur

```
data/          dataset (raw, processed, train, val, test); tidak dilacak Git
configs/       base.yaml dan konfigurasi tiap eksperimen
src/           kode: datasets, models, training, evaluation, utils
notebooks/     notebook bernomor: analisis dataset, praproses, pelatihan, evaluasi
experiments/   hasil tiap eksperimen
checkpoints/   bobot model; tidak dilacak Git
logs/          log pelatihan; tidak dilacak Git
reports/       gambar, metrik, dan tabel untuk laporan
scripts/       skrip yang dijalankan dari terminal
tests/         uji otomatis
docs/          dokumentasi
jurnal/        jurnal acuan
```

## Unduh dataset

Proyek memakai dua dataset publik yang digabung ke skema 7 kelas (lihat [PENGGABUNGAN_DATASET.md](docs/dataset/PENGGABUNGAN_DATASET.md)). Berkas mentah **tidak dilacak Git** — unduh sendiri lalu letakkan di `data/raw/`.

| Dataset | Isi | Unduh |
|---|---|---|
| **GCD** — TJNU Ground-based Cloud Dataset (utama) | 19.000 citra · 7 kelas · 512×512 JPEG (train 10.000 / test 9.000) | [Repositori resmi (GitHub)](https://github.com/shuangliutjnu/TJNU-Ground-based-Cloud-Dataset) |
| **CCSN** — Cirrus Cumulus Stratus Nimbus (kedua) | 2.543 citra · 11 genera WMO · JPEG | Sumber asli: Zhang dkk. 2018, *Geophysical Research Letters* · mirror publik: [HuggingFace `aduuuuuu/CCSN`](https://huggingface.co/datasets/aduuuuuu/CCSN) |

Setelah diunduh, tempatkan sebagai `data/raw/GCD/` dan `data/raw/CCSN_v2/` (struktur folder di [DATASET_DOCUMENTATION.md](docs/dataset/DATASET_DOCUMENTATION.md) §2). SHA-256 arsip `GCD.zip` tercatat di dokumen itu untuk verifikasi.

> Catatan: proyek memakai varian **CCSN_v2** (ukuran campuran 400×400 & 256×256). Sumber unduh persis varian ini belum tercatat di dokumentasi; tautan di atas adalah lokasi publik CCSN yang diketahui — sesuaikan bila pemilik proyek punya sumber resmi.

## Dokumentasi

Indeks lengkap: [docs/README.md](docs/README.md). Dokumen dikelompokkan per folder:

**Dataset** (`docs/dataset/`)
- [Dokumentasi dataset](docs/dataset/DATASET_DOCUMENTATION.md)
- [Ringkasan kualitas dataset](docs/dataset/DATASET_QUALITY_SUMMARY.md)
- [Penanganan kualitas data (dokumentasi ilmiah)](docs/dataset/PENANGANAN_KUALITAS_DATA.md)
- [Hasil split dataset](docs/dataset/DATASET_SPLIT.md)
- [Penggabungan dataset (GCD + CCSN_v2)](docs/dataset/PENGGABUNGAN_DATASET.md)
- [Augmentasi data](docs/dataset/AUGMENTASI_DATA.md)
- [Penyetaraan sampel (balancing 3.000/kelas)](docs/dataset/PENYETARAAN_SAMPEL.md) — implementasi: `notebooks/03_augmentation.ipynb`

**Pelatihan & model** (`docs/pelatihan/`)
- [Rencana eksperimen](docs/pelatihan/EXPERIMENT_PLAN.md)
- [Panduan pelatihan](docs/pelatihan/TRAINING_GUIDE.md)
- [Arsitektur baseline (transfer learning ImageNet)](docs/pelatihan/ARSITEKTUR_BASELINE.md)
- [Hyperparameter yang digunakan (referensi exp_005–009)](docs/pelatihan/HYPERPARAMETER.md)
- [Persiapan pelatihan](docs/pelatihan/PERSIAPAN_PELATIHAN.md)
- [Baseline AlexNet (exp_006)](docs/pelatihan/alexnet/BASELINE_ALEXNET.md) — [rasional hyperparameter](docs/pelatihan/alexnet/rasional-hyperparameter.md) · notebook `notebooks/04_baseline_training.ipynb`
- [Baseline DenseNet-121 (exp_005)](docs/pelatihan/densenet121/BASELINE_DENSENET121.md) — notebook `notebooks/05_densenet121_training.ipynb`
- [Baseline DenseNet-264 (exp_007)](docs/pelatihan/densenet264/BASELINE_DENSENET264.md) — lanjutan DenseNet terdalam (via timm) · notebook `notebooks/07_densenet264_training.ipynb`
- [Baseline ResNet-18 (exp_008)](docs/pelatihan/resnet18/BASELINE_RESNET18.md) — [rasional hyperparameter](docs/pelatihan/resnet18/rasional-hyperparameter.md) · notebook `notebooks/06_resnet18_training.ipynb`

**Referensi** (`docs/referensi/`)
- [Referensi eksternal (persiapan)](docs/referensi/REFERENSI_EKSTERNAL.md)
- [Panduan Relore](docs/referensi/RELORE.md)

## Memori repository untuk coding agent

Relore tersedia melalui skrip PowerShell. Lihat [panduan Relore](docs/referensi/RELORE.md) untuk pembacaan kode lokal, server, dan pengindeksan riwayat GitHub.
