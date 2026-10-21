# Analisis Pendekatan 2 — DenseNet-121 Pretrained ImageNet

> **Status: hasil training belum tersedia untuk dianalisis.**
> Pemeriksaan 22 Oktober 2026 (Asia/Jakarta) menemukan konfigurasi dan arsitektur
> model, tetapi belum menemukan riwayat pelatihan, metrik, atau checkpoint terbaik.
> Dokumen ini mencatat kesiapan metode dan batas bukti; bukan laporan hasil final.

Eksperimen: `exp_005c_densenet121_imagenet`.  
Metode: [Informasi pretrained ImageNet](informasi-pretrained-imagenet.md).  
Navigasi: [Daftar pendekatan](../README.md).

## 1. Artefak yang diperiksa

| Sumber | Hasil pemeriksaan |
|---|---|
| [Konfigurasi run](../../../../../experiments/exp_005c_densenet121_imagenet/config.yaml) | Tersedia |
| [Arsitektur model](../../../../../experiments/exp_005c_densenet121_imagenet/model_architecture.txt) | Tersedia; total 6.961.031, trainable 7.175 |
| [Notebook 05c](../../../../../notebooks/tahap_pre_training/05c_densenet121_pretrained_imagenet.ipynb) | Sel kode tersedia; tidak ada output hasil training yang tersimpan saat pemeriksaan |
| `training_history.csv` | Belum ditemukan pada folder eksperimen |
| `metrics.json` | Belum ditemukan pada folder eksperimen |
| `checkpoints/exp_005c_densenet121_imagenet/best.pt` | Belum ditemukan |
| Laporan klasifikasi dan confusion matrix test | Belum ditemukan |

Status ini berdasarkan berkas di disk, bukan pemantauan langsung kernel
Jupyter. Training mungkin belum dijalankan, masih berlangsung, atau belum
menghasilkan artefak akhir; penyebabnya tidak dapat ditentukan dari kondisi
berkas tersebut saja. Tidak ada klaim bahwa training gagal.

## 2. Analisis konfigurasi yang dapat dikonfirmasi

Konfigurasi menetapkan bobot **ImageNet**, mode **feature extraction**, dan
classifier **7 kelas**. Seluruh **7.175 parameter trainable** berada pada
classifier; parameter backbone dibekukan. Statistik berjalan BatchNorm tetap
dapat beradaptasi selama training, mengikuti implementasi trainer proyek.

Run direncanakan selama **10 epoch**, batch **8**, optimizer **AdamW**,
learning rate **0,0001**, weight decay **0**, tanpa scheduler, dan seed **42**.
Input **512 × 512 RGB** memakai normalisasi **/255**, sesuai konfigurasi
baseline proyek. Dukungan metode ini dapat diverifikasi dari kode, tetapi
kesesuaiannya terhadap dataset harus dinilai dari metrik hasil run.

Train seimbang berisi **21.000 citra**, validation **1.851**, dan test resmi GCD
**9.000**. Validation macro-F1 merupakan kriteria pemilihan checkpoint.

## 3. Hasil yang belum dapat dilaporkan

| Komponen | Status |
|---|---|
| Jumlah epoch selesai | Belum terverifikasi |
| Train loss dan validation loss per epoch | Menunggu `training_history.csv` |
| Epoch terbaik dan validation macro-F1 tertinggi | Menunggu CSV dan `metrics.json` |
| Validation accuracy dan balanced accuracy | Menunggu CSV |
| Waktu training | Belum ada catatan durasi yang dapat dianalisis |
| Test macro-F1, accuracy, dan balanced accuracy | Menunggu evaluasi akhir |
| Precision, recall, F1, dan support per kelas | Menunggu laporan test |
| Kekeliruan antar kelas | Menunggu confusion matrix |

Tabel ini sengaja tidak berisi angka perkiraan, nol pengganti hasil hilang,
atau metrik baseline lama. Angka `exp_005_densenet121` dan
`exp_005b_densenet121_scratch` bukan hasil run ini.

## 4. Kerangka analisis setelah training selesai

Analisis final akan memakai urutan berikut:

1. **Kelengkapan run:** cocokkan jumlah dan urutan epoch pada CSV dengan
   snapshot konfigurasi. Jika run tidak mencapai rencana, laporkan jumlah
   aktual dan bukti penghentiannya bila tersedia.
2. **Pemilihan checkpoint:** cocokkan `best_epoch` dan `best_val_metric`
   pada JSON dengan validation macro-F1 tertinggi pada CSV.
3. **Dinamika belajar:** sajikan kurva/tabel train loss, validation loss,
   macro-F1, balanced accuracy, dan accuracy. Bahas tren serta fluktuasi;
   satu penurunan metrik tidak cukup untuk menyimpulkan overfitting menetap.
4. **Generalisasi:** bahas selisih train/validation dengan memperhatikan
   perbedaan mode training/evaluasi dan adaptasi BatchNorm.
5. **Evaluasi test:** setelah model dipilih berdasarkan validation, laporkan
   metrik test, kelas terlemah, dan pasangan kelas yang sering tertukar.
   Jika test belum dijalankan, pisahkan status selesai training dari selesai
   evaluasi test.
6. **Perbandingan:** gunakan split dan metrik yang sama, sebutkan perbedaan
   bobot awal serta jumlah parameter trainable; hindari klaim keunggulan
   statistik dari satu seed.

## 5. Pembanding yang sudah tersedia

Hasil pendekatan 1 berikut berasal dari
[artefak training dari awal](../../../../../experiments/exp_005b_densenet121_scratch/metrics.json);
angka ini hanya pembanding dan tidak menggantikan hasil pendekatan 2.

| Metrik | Pendekatan 1: dari awal | Pendekatan 2: ImageNet |
|---|---:|---|
| Epoch terbaik | 10 | Belum tersedia |
| Validation macro-F1 terbaik | 0,779281 | Belum tersedia |
| Test macro-F1 | 0,658259 | Belum tersedia |
| Test accuracy | 0,698333 | Belum tersedia |
| Test balanced accuracy | 0,644180 | Belum tersedia |
| Parameter trainable | 6.961.031 | 7.175 |

[Analisis pendekatan 1](../01-training-dari-awal/analisis-training-dari-awal.md)
sudah mencakup hasil test. Selisih kinerja antarmetode baru dapat dihitung
ketika hasil pendekatan 2 tersedia. Validation tidak dibandingkan langsung
dengan test karena keduanya merupakan split berbeda.

## 6. Kesimpulan saat ini

Artefak yang tersedia mengonfirmasi setup DenseNet-121 pretrained ImageNet
dengan classifier tujuh kelas dan pembekuan parameter backbone.
**Kinerja training, konvergensi, overfitting, serta generalisasi belum dapat
disimpulkan.** Dokumen ini siap dilengkapi dari artefak run setelah tersedia;
tidak ada training atau evaluasi baru yang dijalankan untuk menulis dokumen ini.
