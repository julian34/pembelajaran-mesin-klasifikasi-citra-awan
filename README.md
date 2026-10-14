# CitraAwan

Klasifikasi citra awan berbasis darat dengan deep learning (PyTorch).

Status: dua dataset tersedia (**GCD** dan **CCSN_v2**). EDA, penanganan kualitas, pembagian split, dan penggabungan sudah selesai (berbasis manifest di `reports/`); kode pelatihan belum ditulis.

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
notebook/      notebook bernomor: analisis dataset, praproses, pelatihan, evaluasi
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

- [Panduan pelatihan](docs/TRAINING_GUIDE.md)
- [Rencana eksperimen](docs/EXPERIMENT_PLAN.md)
- [Dokumentasi dataset](docs/DATASET_DOCUMENTATION.md)
- [Penanganan kualitas dataset](docs/DATASET_QUALITY_SUMMARY.md)
- [Penanganan kualitas data (dokumentasi ilmiah)](docs/PENANGANAN_KUALITAS_DATA.md)
- [Hasil split dataset](docs/DATASET_SPLIT.md)
- [Penggabungan dataset (GCD + CCSN_v2)](docs/PENGGABUNGAN_DATASET.md)
- [Arsitektur baseline (DenseNet-264, transfer learning)](docs/ARSITEKTUR_BASELINE.md)
- [Augmentasi data](docs/AUGMENTASI_DATA.md)
- [Penyetaraan sampel (balancing 3.000/kelas)](docs/PENYETARAAN_SAMPEL.md) — implementasi: `notebooks/03_augmentation.ipynb`
- [Referensi eksternal (persiapan)](docs/REFERENSI_EKSTERNAL.md)

## Memori repository untuk coding agent

Relore tersedia melalui skrip PowerShell. Lihat [panduan Relore](docs/RELORE.md) untuk pembacaan kode lokal, server, dan pengindeksan riwayat GitHub.
