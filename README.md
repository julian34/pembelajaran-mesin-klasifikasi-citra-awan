# CitraAwan

Klasifikasi citra awan berbasis darat dengan deep learning (PyTorch).

Status: dua dataset tersedia (**GCD** dan **CCSN_v2**). EDA, penanganan kualitas, pembagian split, dan penggabungan sudah selesai (berbasis manifest di `reports/`). Pipa pelatihan+evaluasi baseline sudah ditulis (`src/`, `scripts/train.py`; baseline pertama: AlexNet exp_006, lihat [docs/pelatihan/BASELINE_ALEXNET.md](docs/pelatihan/BASELINE_ALEXNET.md)). `torch`/`torchvision` build CUDA cu121 sudah terpasang (GPU RTX 4060 Laptop); **pelatihan penuh belum dijalankan**.

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
- [Persiapan pelatihan](docs/pelatihan/PERSIAPAN_PELATIHAN.md)
- [Baseline AlexNet (exp_006)](docs/pelatihan/BASELINE_ALEXNET.md)
- [Rasional AlexNet + hyperparameter](docs/pelatihan/rasional-alexnet-hyperparameter/README.md)

**Referensi** (`docs/referensi/`)
- [Referensi eksternal (persiapan)](docs/referensi/REFERENSI_EKSTERNAL.md)
- [Panduan Relore](docs/referensi/RELORE.md)

## Memori repository untuk coding agent

Relore tersedia melalui skrip PowerShell. Lihat [panduan Relore](docs/referensi/RELORE.md) untuk pembacaan kode lokal, server, dan pengindeksan riwayat GitHub.
