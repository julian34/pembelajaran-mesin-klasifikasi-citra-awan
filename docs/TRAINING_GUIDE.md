# Panduan Pelatihan CitraAwan

Panduan ini menjelaskan di mana setiap bagian pekerjaan pelatihan model klasifikasi citra awan disimpan dan aturan yang harus diikuti agar hasil eksperimen dapat diulang dan dibandingkan.

Kode pelatihan belum ditulis. Dokumen ini menetapkan kerangkanya lebih dulu.

## 1. Tujuan struktur folder

Struktur ini memisahkan empat hal yang sering tercampur: data, kode, konfigurasi, dan hasil. Dengan pemisahan itu:

- setiap eksperimen dapat diulang hanya dari berkas konfigurasinya;
- hasil antar-eksperimen dapat dibandingkan karena disimpan dalam bentuk yang sama;
- eksperimen baru dapat ditambahkan tanpa mengubah eksperimen lama;
- berkas besar (dataset, checkpoint, log) tidak ikut masuk ke Git.

## 2. Fungsi setiap folder

| Folder | Fungsi | Dilacak Git |
|--------|--------|-------------|
| `data/raw/` | Dataset asli, persis seperti saat diunduh. Tidak pernah diubah. | Tidak |
| `data/processed/` | Hasil praproses dari `data/raw/` (dibersihkan, diubah ukurannya). | Tidak |
| `data/train/`, `data/val/`, `data/test/` | Hasil pembagian data, satu subfolder per kelas. | Tidak |
| `configs/base.yaml` | Nilai bawaan semua hyperparameter. | Ya |
| `configs/experiments/` | Satu berkas per eksperimen, berisi nilai yang berbeda dari `base.yaml`. | Ya |
| `src/datasets/` | Pemuatan data dan transformasi. | Ya |
| `src/models/` | Definisi arsitektur model. | Ya |
| `src/training/` | Loop pelatihan, optimizer, scheduler, penyimpanan checkpoint. | Ya |
| `src/evaluation/` | Perhitungan metrik, confusion matrix, classification report. | Ya |
| `src/utils/` | Fungsi bantu: seed, pembacaan konfigurasi, logging. | Ya |
| `notebook/` | Notebook eksplorasi dan pelaporan, bernomor urut. | Ya |
| `experiments/` | Hasil tiap eksperimen (konfigurasi, metrik, grafik). | Ya |
| `checkpoints/` | Bobot model tiap eksperimen. | Tidak |
| `logs/` | Log pelatihan tiap eksperimen. | Tidak |
| `reports/figures/`, `reports/metrics/`, `reports/tables/` | Gambar, metrik, dan tabel gabungan lintas eksperimen untuk laporan. | Ya |
| `scripts/` | Skrip yang dijalankan dari terminal (praproses, pelatihan, evaluasi). | Ya |
| `tests/` | Uji otomatis untuk kode di `src/`. | Ya |
| `docs/` | Dokumentasi proyek. | Ya |
| `jurnal/` | Jurnal acuan (PDF). | Ya |

Notebook hanya memanggil fungsi dari `src/`. Logika yang dipakai lebih dari sekali ditulis di `src/`, bukan disalin antar-notebook.

## 3. Dataset: raw dan processed

- **`data/raw/`** menyimpan dataset asli. Folder ini hanya dibaca. Jika ada citra yang rusak atau salah label, perbaikannya dilakukan di tahap praproses dan dicatat di [DATASET_DOCUMENTATION.md](DATASET_DOCUMENTATION.md), bukan dengan mengedit berkas di `raw/`.
- **`data/processed/`** menyimpan hasil praproses. Isinya harus dapat dibuat ulang seluruhnya dari `data/raw/` dengan menjalankan skrip praproses.

## 4. Pembagian train, validation, dan test

- Pembagian bawaan: **70% train, 15% validation, 15% test**, diatur di `configs/base.yaml` (`data.split`).
- Pembagian bersifat **terstratifikasi**: proporsi tiap kelas sama di ketiga bagian.
- Pembagian dilakukan **satu kali** dengan seed tetap, lalu dipakai oleh semua eksperimen. Eksperimen yang memakai pembagian berbeda tidak dapat dibandingkan.
- Satu citra hanya boleh berada di satu bagian. Citra duplikat atau hampir sama (misalnya bingkai berurutan dari kamera yang sama) harus masuk ke bagian yang sama agar tidak terjadi kebocoran data.
- Jika dataset yang dipakai sudah menyediakan pembagian resmi, pakai pembagian resmi itu dan catat di [DATASET_DOCUMENTATION.md](DATASET_DOCUMENTATION.md).
- Augmentasi hanya diterapkan pada `data/train/`. Statistik normalisasi (rata-rata dan simpangan baku) dihitung hanya dari `data/train/`.

Peran tiap bagian:

| Bagian | Dipakai untuk |
|--------|---------------|
| Train | Memperbarui bobot model. |
| Validation | Memilih checkpoint terbaik, menyetel hyperparameter, membandingkan eksperimen. |
| Test | Mengukur kinerja akhir, **satu kali**, setelah semua keputusan dibuat. |

## 5. Lokasi tiap komponen

| Komponen | Lokasi |
|----------|--------|
| Konfigurasi dasar | `configs/base.yaml` |
| Konfigurasi eksperimen | `configs/experiments/<nama_eksperimen>.yaml` |
| Arsitektur model | `src/models/` |
| Logika pelatihan | `src/training/` |
| Skrip pelatihan (titik masuk) | `scripts/` |
| Kode evaluasi | `src/evaluation/` |
| Checkpoint | `checkpoints/<nama_eksperimen>/` |
| Hasil eksperimen | `experiments/<nama_eksperimen>/` |
| Log | `logs/<nama_eksperimen>/` |
| Rangkuman lintas eksperimen | `reports/` |

Skrip di `scripts/` menerima satu berkas konfigurasi eksperimen sebagai masukan dan memanggil kode di `src/`. Skrip tidak menyimpan nilai hyperparameter di dalam kodenya.

## 6. Aturan penamaan eksperimen

Format: `exp_<nomor 3 digit>_<nama_singkat>`

```
exp_001_baseline
exp_002_augmentation
exp_003_dilated
exp_004_aspp
```

- Nomor berurutan dan tidak pernah dipakai ulang, termasuk untuk eksperimen yang gagal atau dibatalkan.
- Nama singkat memakai huruf kecil dan garis bawah, menyebut **satu perubahan utama** terhadap eksperimen pembandingnya.
- Nama yang sama dipakai di empat tempat: `configs/experiments/<nama>.yaml`, `experiments/<nama>/`, `checkpoints/<nama>/`, dan `logs/<nama>/`.
- Eksperimen yang sudah selesai tidak diubah. Mengubah hyperparameter berarti membuat eksperimen baru dengan nomor baru.
- Mengulang eksperimen yang sama dengan seed berbeda memakai subfolder `seed_<nilai>/` di dalam folder eksperimen itu, bukan nomor eksperimen baru.

## 7. Isi folder setiap eksperimen

Setiap eksperimen minimal menyimpan hal-hal berikut.

| Yang disimpan | Berkas |
|---------------|--------|
| Config lengkap (gabungan `base.yaml` dan berkas eksperimen) | `experiments/<nama>/config.yaml` |
| Arsitektur model (ringkasan lapisan dan jumlah parameter) | `experiments/<nama>/model_architecture.txt` |
| Random seed, optimizer, learning rate, batch size, jumlah epoch | tercatat di `config.yaml` |
| Training loss, validation loss, validation accuracy, validation macro F1 per epoch | `experiments/<nama>/history.csv` |
| Test accuracy dan test macro F1 | `experiments/<nama>/test_metrics.json` |
| Confusion matrix | `experiments/<nama>/confusion_matrix.png` dan `.csv` |
| Classification report | `experiments/<nama>/classification_report.txt` |
| Checkpoint terbaik | `checkpoints/<nama>/best.pt` |

`test_metrics.json`, confusion matrix, dan classification report pada test set baru dibuat pada tahap evaluasi akhir (lihat bagian 11). Selama pemilihan model, yang diisi hanya metrik validation.

## 8. Aturan reproducibility

Sebuah eksperimen dianggap dapat diulang jika orang lain dapat memperoleh angka yang sama hanya dari isi repositori dan dataset yang sama.

1. **Semua nilai berasal dari konfigurasi.** Tidak ada hyperparameter yang ditulis langsung di kode atau diubah lewat sel notebook tanpa dicatat.
2. **Config lengkap disalin** ke folder eksperimen saat pelatihan dimulai, sehingga perubahan `base.yaml` di kemudian hari tidak mengubah catatan eksperimen lama.
3. **Versi pustaka dikunci** di `requirements.txt`. Perubahan versi dicatat sebagai eksperimen baru jika memengaruhi hasil.
4. **Commit Git dicatat** di `config.yaml` eksperimen (hash commit saat pelatihan dijalankan), dan pelatihan dijalankan dari kode yang sudah di-commit.
5. **Pembagian data tetap** untuk semua eksperimen (bagian 4).
6. **Perangkat dicatat** (CPU atau GPU, tipe GPU), karena hasil dapat sedikit berbeda antar-perangkat.

## 9. Random seed

- Seed bawaan: **42**, diatur di `configs/base.yaml` (`seed`).
- Seed yang sama dipasang untuk semua sumber keacakan: `random` Python, NumPy, PyTorch (CPU dan GPU), pengacakan urutan data di DataLoader beserta worker-nya, dan augmentasi.
- Pemasangan seed dilakukan oleh satu fungsi di `src/utils/` yang dipanggil di awal setiap skrip dan notebook.
- Seed untuk pembagian data dan seed untuk pelatihan adalah nilai yang sama secara bawaan, tetapi pembagian data tidak diulang ketika seed pelatihan diganti.
- Operasi GPU tertentu tetap tidak sepenuhnya deterministik. Karena itu, kesimpulan bahwa satu eksperimen lebih baik dari yang lain sebaiknya didukung oleh beberapa seed (misalnya tiga), dengan rata-rata dan simpangan bakunya dilaporkan.

## 10. Konfigurasi hyperparameter

- `configs/base.yaml` memuat nilai bawaan untuk semua eksperimen.
- `configs/experiments/<nama>.yaml` hanya memuat nilai yang **berbeda** dari `base.yaml`. Dengan begitu, perbedaan antar-eksperimen terlihat langsung dari isi berkasnya.
- Satu eksperimen sebaiknya mengubah **satu hal** terhadap pembandingnya, agar perubahan hasil dapat dijelaskan.
- Hyperparameter yang wajib ada: arsitektur model, seed, optimizer, learning rate, weight decay, batch size, jumlah epoch, scheduler, ukuran citra masukan, dan pengaturan augmentasi.
- Penyetelan hyperparameter hanya memakai validation set.

Nilai di `base.yaml` saat ini adalah nilai awal dan perlu ditinjau setelah dataset tersedia dan EDA selesai.

## 11. Penyimpanan checkpoint terbaik

- Checkpoint terbaik dipilih berdasarkan **validation macro F1** tertinggi (`checkpoint.monitor: val_macro_f1`). Macro F1 dipakai karena jumlah citra antar-kelas awan tidak seimbang, sehingga accuracy saja dapat menutupi kinerja buruk pada kelas kecil.
- Berkas disimpan sebagai `checkpoints/<nama_eksperimen>/best.pt` dan ditimpa setiap kali ada epoch dengan nilai yang lebih baik.
- Checkpoint menyimpan bobot model, keadaan optimizer, nomor epoch, nilai metrik saat disimpan, dan config.
- `last.pt` boleh disimpan untuk melanjutkan pelatihan yang terhenti, tetapi tidak dipakai untuk evaluasi.
- Folder `checkpoints/` tidak dilacak Git. Checkpoint yang perlu disimpan jangka panjang dicadangkan di luar repositori.

## 12. Larangan memakai test set selama pemilihan model

Test set **tidak boleh** dipakai untuk keputusan apa pun selama pengembangan. Ini mencakup:

- memilih checkpoint terbaik atau menentukan early stopping;
- menyetel hyperparameter;
- memilih arsitektur atau membandingkan eksperimen;
- menentukan augmentasi atau langkah praproses;
- menghitung statistik normalisasi.

Alasannya: setiap keputusan yang dibuat dengan melihat test set membuat model secara tidak langsung menyesuaikan diri dengan data itu, sehingga angka test tidak lagi mencerminkan kinerja pada data baru.

Urutan yang benar:

1. Latih semua eksperimen; pantau hanya train dan validation.
2. Pilih model terbaik berdasarkan validation macro F1.
3. Jalankan evaluasi pada test set **satu kali** untuk tiap eksperimen yang dilaporkan, dengan `best.pt` masing-masing.
4. Laporkan angka test apa adanya. Jika setelah itu model diubah lagi, angka test lama tidak boleh dijadikan dasar perubahan tersebut.
