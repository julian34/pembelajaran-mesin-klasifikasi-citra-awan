# Informasi Training dari Awal DenseNet-121

Tanggal dokumentasi: 21 Oktober 2026; dirapikan 22 Oktober 2026 (Asia/Jakarta).  
Eksperimen: `exp_005b_densenet121_scratch`.  
Dokumen hasil: [Analisis training dari awal](analisis-training-dari-awal.md).

## 1. Tujuan dan pendekatan

Eksperimen ini melatih DenseNet-121 dari bobot acak pada dataset awan gabungan
GCD dan CCSN_v2 untuk klasifikasi tujuh kelas. Seluruh parameter backbone dan
classifier diperbarui selama pelatihan.

Pengaturan `pretrained: false` membuat pembangun model memakai
`torchvision.models.densenet121(weights=None)`. Pengaturan `finetune: full`
mengaktifkan pelatihan seluruh parameter; istilah `finetune` di sini merupakan
nama opsi pada kode proyek, bukan penanda bahwa model memakai bobot pretrained.

Tidak ada pemuatan checkpoint saat model diinisialisasi. Checkpoint hasil
pelatihan ini baru dimuat pada tahap evaluasi akhir.

## 2. Sumber konfigurasi dan implementasi

| Sumber | Fungsi |
|---|---|
| [Notebook 05b](../../../../../notebooks/tahap_pre_training/05b_densenet121_pre_trainning.ipynb) | Menyiapkan data, membangun model, melatih, dan mengevaluasi |
| [Konfigurasi eksperimen](../../../../../configs/experiments/exp_005b_densenet121_scratch.yaml) | Menentukan pendekatan bobot acak dan pengaturan eksperimen |
| [Konfigurasi dasar](../../../../../configs/base.yaml) | Nilai bawaan optimizer, learning rate, praproses, dan seed |
| [Snapshot konfigurasi run](../../../../../experiments/exp_005b_densenet121_scratch/config.yaml) | Konfigurasi efektif yang disimpan sebelum pelatihan |
| [Pembangun model](../../../../../src/models/baseline.py) | Inisialisasi DenseNet-121 dan classifier |
| [Trainer](../../../../../src/training/trainer.py) | Pembaruan parameter, validasi, dan penyimpanan checkpoint |

Untuk menganalisis suatu run, gunakan snapshot konfigurasi di folder eksperimen.
Override dari notebook dicatat pada snapshot tersebut.

## 3. Arsitektur model

DenseNet-121 memakai empat dense block dengan jumlah dense layer 6, 12, 24, dan
16, dipisahkan oleh tiga transition layer. Fitur akhir diringkas melalui global
average pooling dan diteruskan ke classifier tujuh kelas.

| Komponen | Nilai |
|---|---|
| Arsitektur | DenseNet-121 dari torchvision |
| Inisialisasi | Bobot acak, `weights=None` |
| Classifier | `Linear(1024, 7)` |
| Parameter total | 6.961.031 |
| Parameter dilatih | 6.961.031 (100%) |
| Parameter dibekukan | 0 |

Jumlah parameter dikonfirmasi dari
[arsitektur model yang disimpan](../../../../../experiments/exp_005b_densenet121_scratch/model_architecture.txt).
Parameter BatchNorm yang dapat dipelajari ikut dilatih; statistik berjalan
BatchNorm diperbarui dalam mode training.

## 4. Dataset dan praproses

| Split | Jumlah citra | Penggunaan |
|---|---:|---|
| Train | 21.000 | Pembaruan bobot; 3.000 citra per kelas |
| Validation | 1.851 | Memantau generalisasi dan memilih checkpoint |
| Test | 9.000 | Evaluasi akhir pada test resmi GCD |

Urutan kelas adalah `Cu, Ac, Ci, Cl, Sc, Cb, Mi`. Train memakai
`reports/tables/combined_balanced_train_manifest.csv`, sedangkan validation dan
test memakai `combined_val_manifest.csv` dan `combined_test_manifest.csv`.

Citra dibaca sebagai RGB, diubah ke ukuran 512 × 512, lalu dibagi 255 sehingga
rentangnya [0, 1]. Mean [0, 0, 0] dan standar deviasi [1, 1, 1] tidak mengubah
rentang tersebut.

Augmentasi saat pemuatan tidak digunakan. Namun, train sudah berisi citra
augmentasi yang dibuat sebelumnya: 10.483 citra asli dan 10.517 citra augmentasi.
Rincian tersedia pada [dokumentasi penyetaraan sampel](../../../../dataset/PENYETARAAN_SAMPEL.md).
Validation dan test tidak diseimbangkan. Karena train telah seimbang,
`imbalance: none` digunakan tanpa tambahan class weights atau weighted sampler.

## 5. Hyperparameter run

Nilai berikut berasal dari snapshot konfigurasi run yang tersedia.

| Pengaturan | Nilai |
|---|---|
| Epoch yang direncanakan | 10 |
| Batch size | 8 |
| Optimizer | AdamW |
| Learning rate | **0,0001 (`1e-4`)** |
| Weight decay | 0,0 |
| Scheduler learning rate | Tidak ada; learning rate tetap |
| Fungsi loss | CrossEntropyLoss tanpa bobot kelas |
| Random seed | 42 |
| DataLoader workers | 0 |
| Input | 512 × 512 RGB |
| Pemilihan checkpoint | Validation macro-F1 tertinggi |
| Early stopping | Tidak diterapkan pada loop saat ini |

Output setup notebook yang tersimpan menunjukkan PyTorch `2.1.1+cu121` dan
GPU NVIDIA GeForce RTX 4060 Laptop. Sepuluh epoch merupakan pengaturan run ini;
kecukupannya dinilai dari kurva train dan validation, bukan diasumsikan sudah
mencapai konvergensi.

## 6. Alur pelatihan dan evaluasi

1. Muat konfigurasi, pasang seed, dan siapkan DataLoader train serta validation.
2. Bangun model dari bobot acak dan pastikan jumlah parameter trainable sama
   dengan jumlah parameter total.
3. Simpan konfigurasi efektif dan ringkasan arsitektur.
4. Latih semua parameter menggunakan train set. Pada akhir setiap epoch,
   hitung loss dan metrik validation.
5. Simpan `best.pt` ketika validation macro-F1 meningkat.
6. Setelah seluruh epoch selesai, simpan riwayat pelatihan dan ringkasan metrik.
7. Setelah keputusan model selesai berdasarkan validation, jalankan evaluasi
   test sekali menggunakan checkpoint terbaik.

Sel evaluasi memakai `EVALUATE_TEST=False` secara bawaan. Sel ini perlu diaktifkan
secara eksplisit ketika memasuki evaluasi akhir. Penyelesaian training tidak
otomatis berarti test sudah dievaluasi.

Untuk memulai ulang dari bobot acak, jalankan ulang sel pembentukan model
sebelum sel pelatihan. Menjalankan ulang sel pelatihan saja menggunakan bobot
yang masih berada dalam memori.

## 7. Artefak yang dihasilkan

Folder hasil: `experiments/exp_005b_densenet121_scratch/`.  
Folder checkpoint: `checkpoints/exp_005b_densenet121_scratch/`.

| Artefak | Waktu pembuatan dan isi |
|---|---|
| `config.yaml` | Sebelum training; konfigurasi efektif |
| `model_architecture.txt` | Sebelum training; struktur dan jumlah parameter |
| `best.pt` di folder checkpoint | Selama training; bobot dan informasi epoch validation terbaik |
| `training_history.csv` | Setelah training selesai; loss dan metrik validation tiap epoch |
| `metrics.json` | Setelah training selesai; epoch terbaik dan nilai monitor; test berstatus `not_evaluated` sebelum evaluasi akhir |
| `classification_report.txt` | Setelah evaluasi test; precision, recall, F1, dan support per kelas |
| `confusion_matrix.png` | Setelah evaluasi test; matriks kelas sebenarnya dan prediksi |
| `notes.md` | Setelah evaluasi test; ringkasan run |

Riwayat CSV baru ditulis setelah fungsi `fit` selesai. Keberadaan `best.pt`
sendiri tidak membuktikan semua epoch telah selesai. Checkpoint trainer saat
ini menyimpan bobot, epoch, metrik, dan konfigurasi; state optimizer tidak
disimpan sehingga checkpoint tersebut bukan fasilitas resume lengkap.

## 8. Ketentuan analisis hasil

Analisis akhir ditulis pada [analisis-training-dari-awal.md](analisis-training-dari-awal.md)
setelah artefak penyelesaian training tersedia. Angka final diambil dari
`training_history.csv` dan `metrics.json`; output progres notebook hanya
menjadi catatan sementara.

Analisis mencakup dinamika loss, epoch terbaik, metrik validation, indikasi
overfitting atau kebutuhan tambahan epoch, serta metrik test jika evaluasi akhir
sudah dilakukan. Perbandingan dengan baseline ImageNet perlu menyebut perbedaan
jumlah epoch dan parameter yang dilatih, sehingga selisih hasil tidak langsung
diatribusikan hanya pada bobot awal.

Aturan pemisahan train, validation, dan test mengikuti
[panduan pelatihan proyek](../../../TRAINING_GUIDE.md).
