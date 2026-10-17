# Analisis Hasil Baseline AlexNet (exp_006)

Analisis hasil pelatihan **transfer learning AlexNet** (bobot ImageNet, mode
`feature_extraction`) untuk klasifikasi 7 kelas awan (GCD + CCSN_v2). Dokumen ini
membahas **waktu pelatihan**, **seluruh metrik evaluasi**, dan **F1 per kelas**.

- Eksperimen: `exp_006_alexnet` · tanggal artefak **17 Oktober 2026**
- Sumber (nyata, bukan perkiraan): [`experiments/exp_006_alexnet/`](../../../experiments/exp_006_alexnet/)
  — `metrics.json`, `classification_report.txt`, `training_history.csv`,
  `confusion_matrix.png`, `config.yaml`, `notes.md`; checkpoint
  [`checkpoints/exp_006_alexnet/best.pt`](../../../checkpoints/exp_006_alexnet/best.pt).
- Rasional pilihan arsitektur + hyperparameter: [rasional-alexnet-hyperparameter/](rasional-alexnet-hyperparameter/README.md).
  Detail setup baseline: [BASELINE_ALEXNET.md](BASELINE_ALEXNET.md).

---

## 1. Ringkasan eksekutif

| Besaran | Nilai |
|---|---|
| Epoch terbaik (dipilih dari validation) | **63** dari 200 |
| val macro-F1 terbaik | **0,6231** |
| **Test macro-F1** (metrik primer) | **0,5042** |
| Test accuracy (= micro-F1 global) | **0,5782** |
| Test balanced accuracy | **0,4865** |
| Test weighted-F1 | **0,53** |
| Parameter dilatih / total | 28.679 / 57.032.519 (0,05%) |
| Waktu pelatihan (estimasi dari stempel berkas) | **± 13 jam** untuk 200 epoch (~3,9 menit/epoch) |

**Inti temuan:** baseline berjalan wajar (jauh di atas tebakan mayoritas 30,68%),
tetapi kapasitas sangat terbatas karena hanya lapisan keluaran yang dilatih. Model
cenderung **menumpuk prediksi ke kelas mayoritas Cb** dan ke kelas mudah **Cl**,
sehingga kelas bertekstur halus (**Sc, Ci, Mi**) nyaris tak terangkat (recall <
0,26). Ada pula **jurang generalisasi val→test ~0,12** pada macro-F1 (val gabungan
GCD+CCSN vs test resmi GCD murni). Pelatihan **berhenti membaik setelah epoch 63**
— 137 epoch sisanya mubazir; early stopping layak dipertimbangkan.

---

## 2. Konfigurasi yang dievaluasi

Diambil dari `experiments/exp_006_alexnet/config.yaml` (snapshot saat run):

| Item | Nilai |
|---|---|
| Arsitektur | `alexnet` (torchvision, pretrained ImageNet) |
| Mode fine-tune | `feature_extraction` (backbone beku, hanya `classifier[6]` 4096→7 dilatih) |
| Input | 512×512 RGB, normalisasi `/255` → [0,1] (mean 0, std 1) |
| Augmentasi | nonaktif |
| Epochs | 200 |
| Batch size | 8 |
| Optimizer | AdamW, lr 1e-4, weight_decay 0,0 |
| Scheduler | — (tidak ada) |
| Penanganan imbalance | `none` (train sudah diseimbangkan 3.000/kelas) |
| Checkpoint | simpan terbaik menurut `val_macro_f1` (mode max) |
| Data | train seimbang 21.000 · val 1.851 · **test resmi GCD 9.000** |

---

## 3. Waktu pelatihan

> **Catatan metode (penting):** loop pelatihan **tidak** mencatat durasi ke file
> (lihat `EpochRecord` di `src/training/trainer.py` — tidak ada kolom waktu), dan
> folder `logs/` kosong. Angka di bawah **diturunkan dari stempel waktu berkas
> (mtime)**, bukan dari timer di dalam kode, jadi bersifat estimasi.

Dua jangkar waktu yang dapat diukur:

| Peristiwa | Stempel (Asia/Jakarta) |
|---|---|
| `best.pt` terakhir ditulis (akhir epoch 63 — epoch terbaik global) | 2026-10-16 **19:33:17** |
| Artefak test ditulis (akhir run + evaluasi test) | 2026-10-17 **04:26:01** |

- Selang kedua jangkar = **8 jam 52 menit 44 detik**, mencakup **epoch 64–200
  (137 epoch) + satu evaluasi test penuh** (9.000 citra).
- → laju **± 233 detik/epoch (± 3,9 menit/epoch)**.
- Ekstrapolasi untuk 200 epoch: **± 12–13 jam** (waktu mulai epoch 1 tidak terekam,
  jadi total adalah perkiraan).

Angka ini **lebih cepat** dari estimasi di notebook (~18 menit/epoch); estimasi
notebook mengasumsikan `NUM_WORKERS=0` di Jupyter Windows, sedangkan run nyata
(via terminal) tampaknya memakai pemuatan paralel. Pelatihan ini **I/O-bound**
(baca JPEG 512×512 dari disk USB), bukan GPU-bound.

**Rekomendasi:** tambahkan pencatatan `time.perf_counter()` per epoch ke
`EpochRecord`/`training_history.csv` agar waktu latih tercatat pasti pada run
berikut, bukan direkonstruksi dari mtime.

---

## 4. Dinamika pelatihan (200 epoch)

Sumber: `training_history.csv` (kolom `train_loss`, `val_loss`, `val_macro_f1`,
`val_balanced_accuracy`, `val_accuracy`).

| Epoch | train_loss | val_loss | val_macro_f1 | Keterangan |
|---|---|---|---|---|
| 1 | 1,2555 | 1,1299 | 0,4739 | awal |
| 24 | 0,9412 | 0,9703 | 0,6077 | puncak awal |
| **63** | **0,9219** | **0,9333** | **0,6231** | **terbaik → checkpoint** |
| 149 | 0,9185 | 0,9373 | 0,6186 | nyaris menyamai |
| 162 | 0,9134 | 0,9486 | 0,6159 | — |
| 200 | 0,9052 | 0,9583 | 0,5748 | akhir |

**Pengamatan:**

1. **Konvergen cepat lalu datar.** `train_loss` turun tajam 1,26 → ~0,94 dalam
   ~20 epoch, lalu merayap sangat lambat ke ~0,905 di epoch 200. Ini khas
   `feature_extraction`: hanya 28.679 parameter yang dilatih, kapasitas kecil, dan
   backbone beku membatasi loss minimum yang dapat dicapai.
2. **Bukan overfitting — lebih ke underfitting/kapasitas terbatas.** `val_loss`
   (~0,93–0,98) justru berada di sekitar atau sedikit di atas `train_loss`, tanpa
   divergensi yang membesar. Tidak ada tanda overfitting klasik (train turun, val
   naik). Dropout(0,5) bawaan AlexNet + backbone beku menahan overfitting.
3. **Plateau setelah epoch 63.** `val_macro_f1` terbaik 0,6231 di epoch 63; 137
   epoch berikutnya **tidak pernah melampauinya** (hanya mendekat: 0,6186 @149,
   0,6161 @160). Nilai berosilasi ribut di kisaran 0,57–0,62.
   → **200 epoch mubazir**; early stopping (mis. patience 30–40) akan berhenti
   ~epoch 70–100 dan memangkas waktu latih drastis tanpa kehilangan kualitas.

---

## 5. Evaluasi test — penjelasan semua metrik

Test dihitung **sekali** memakai checkpoint terbaik (dipilih dari validation,
**bukan** test). Sumber: `metrics.json` + `classification_report.txt`.

### 5.1 Metrik global

| Metrik | Nilai | Arti & cara baca |
|---|---|---|
| **Accuracy** | **0,5782** | Proporsi prediksi benar atas 9.000 citra. Dalam klasifikasi single-label multikelas, ini **identik dengan micro-F1** (lihat 5.2). Didominasi kelas besar (Cb, Cl). |
| **Macro-F1** | **0,5042** | Rata-rata **tak berbobot** F1 ketujuh kelas — tiap kelas berbobot sama, apa pun jumlahnya. **Metrik primer** proyek karena adil terhadap kelas minoritas. Jauh di bawah accuracy → tanda kelas kecil tertinggal. |
| **Balanced accuracy** | **0,4865** | Rata-rata **recall** per kelas. Rendah → banyak kelas yang sampelnya gagal ditemukan model. |
| **Weighted-F1** | **0,53** | Rata-rata F1 **ditimbang support**; condong ke kelas besar, berada di antara macro dan accuracy. |
| Macro precision / recall | 0,68 / 0,49 | Precision tinggi tapi recall rendah → saat model menyebut sebuah kelas minoritas ia cukup sering benar, tetapi ia **jarang berani menyebutnya**. |

**Batas bawah kewajaran:** tebakan selalu-Cb memberi accuracy 30,68%. Model
(57,82%) jelas belajar sesuatu, tetapi macro-F1 0,50 menunjukkan kualitas tak
merata antar kelas.

### 5.2 Micro vs macro (menjawab "F1 mikro per kelas")

- **Micro-F1** menggabungkan **semua** TP/FP/FN ketujuh kelas lebih dulu, baru
  hitung F1. Pada klasifikasi single-label, micro-precision = micro-recall =
  micro-F1 = **accuracy = 0,5782**. Jadi micro-F1 bersifat **global, bukan
  per-kelas** — tidak ada "micro-F1 per kelas" yang berbeda-beda; angkanya satu.
- **F1 per kelas** (tabel 5.3) adalah F1 yang dihitung terpisah untuk tiap kelas.
- **Macro-F1 (0,5042)** = rata-rata tak berbobot kolom F1 per kelas di bawah.

### 5.3 F1 per kelas (precision / recall / F1 / support)

Diurutkan dari F1 tertinggi. Sumber: `metrics.json → test.per_class`.

| # | Kelas | Precision | Recall | **F1** | Support |
|---|---|---|---|---|---|
| 3 | **Cl** — Clear sky | 0,622 | 0,921 | **0,743** | 1.589 |
| 0 | **Cu** — Cumulus | 0,745 | 0,608 | **0,670** | 750 |
| 5 | **Cb** — Cumulonimbus+nimbostratus | 0,494 | 0,857 | **0,627** | 2.761 |
| 1 | **Ac** — Altocumulus+cirrocumulus | 0,854 | 0,391 | **0,536** | 750 |
| 2 | **Ci** — Cirrus+cirrostratus | 0,742 | 0,256 | **0,381** | 753 |
| 6 | **Mi** — Mixed | 0,550 | 0,198 | **0,291** | 607 |
| 4 | **Sc** — Stratocumulus+stratus+altostratus | 0,730 | 0,175 | **0,282** | 1.790 |

**Interpretasi per kelas:**

- **Cl (F1 0,74)** — terbaik. Recall 0,92 (hampir semua clear sky ketangkap) tapi
  precision 0,62 → model **terlalu mudah menebak Cl**, menyerap citra kelas lain.
- **Cb (F1 0,63)** — recall 0,86 dengan precision hanya 0,49. Sebagai **kelas
  mayoritas (30,7% test)**, Cb menjadi "tempat sampah" prediksi: banyak kelas lain
  salah diklasifikasikan sebagai Cb (lihat 5.4).
- **Cu (F1 0,67)** — paling seimbang (P 0,75 / R 0,61).
- **Ac (F1 0,54)** — precision sangat tinggi (0,85) tapi recall 0,39: kalau model
  menyebut Ac biasanya benar, tapi 61% Ac gagal dikenali.
- **Ci, Mi, Sc (F1 0,38 / 0,29 / 0,28)** — terburuk. Recall sangat rendah
  (0,26 / 0,20 / 0,18): mayoritas sampel kelas ini **"bocor" ke Cb/Cl**. Sc paling
  parah meski support-nya besar (1.790).

### 5.4 Confusion matrix

Berkas: [`confusion_matrix.png`](../../../experiments/exp_006_alexnet/confusion_matrix.png).
Baris = kelas sebenarnya, kolom = prediksi (urutan Cu, Ac, Ci, Cl, Sc, Cb, Mi).

| True \ Pred | Cu | Ac | Ci | Cl | Sc | **Cb** | Mi |
|---|---|---|---|---|---|---|---|
| **Cu** (750) | **456** | 6 | 6 | 4 | 1 | 233 | 44 |
| **Ac** (750) | 13 | **293** | 20 | 11 | 1 | 383 | 29 |
| **Ci** (753) | 48 | 7 | **193** | 126 | 24 | 333 | 22 |
| **Cl** (1.589) | 0 | 0 | 4 | **1.464** | 8 | 113 | 0 |
| **Sc** (1.790) | 0 | 0 | 0 | 434 | **313** | 1.043 | 0 |
| **Cb** (2.761) | 0 | 0 | 3 | 311 | 79 | **2.365** | 3 |
| **Mi** (607) | 95 | 37 | 34 | 3 | 3 | 315 | **120** |

**Pola kebingungan utama:**

1. **Cb sebagai magnet kesalahan.** Kolom Cb menyerap kesalahan dari hampir semua
   kelas: Ac→Cb 383, Ci→Cb 333, Mi→Cb 315, Cu→Cb 233. Kelas mayoritas menjadi
   tebakan default saat model ragu.
2. **Sc runtuh (recall 0,18).** Dari 1.790 Sc: 1.043 → Cb, 434 → Cl, hanya 313
   benar. Sc (stratocumulus/stratus/altostratus) bertekstur mendung merata,
   mirip Cb dan Cl secara global — sulit dipisah oleh fitur ImageNet beku.
3. **Cl & Cb saling tukar.** Cl→Cb 113 dan Cb→Cl 311; keduanya langit "berat"
   berskala besar.
4. **Mi tersebar** ke Cb (315) dan Cu (95) — wajar, karena Mixed memang gabungan.

---

## 6. Diagnosis — mengapa hasilnya begini

1. **Kapasitas `feature_extraction` kecil.** Hanya kepala linear 4096→7 (28.679
   parameter) yang dilatih; backbone AlexNet ImageNet dibekukan. Fitur ImageNet
   bagus untuk objek, tetapi tekstur awan halus (Sc/Ci) kurang terwakili, sehingga
   model tak bisa memisahkannya dan jatuh ke kelas dominan.
2. **Bias kelas mayoritas pada test.** Train diseimbangkan (3.000/kelas), tetapi
   **test mengikuti distribusi alami GCD** (Cb 30,7%, Mi 6,7%). Kepala linear
   belajar batas keputusan yang, saat ragu, condong ke Cb.
3. **Jurang domain val→test.** val macro-F1 0,6231 vs test 0,5042 (selisih ~0,12).
   Train+val memuat **CCSN_v2**, test adalah **GCD resmi murni**; perbedaan
   distribusi sumber + penyeimbangan membuat val lebih optimistis daripada test.
4. **200 epoch tak membantu.** Puncak di epoch 63; sisanya osilasi. Konsisten
   dengan model berkapasitas kecil yang cepat jenuh.

---

## 7. Rekomendasi tindak lanjut

- **Fine-tuning penuh / parsial** (buka beberapa blok konvolusi terakhir, lr lebih
  kecil) untuk menaikkan kapasitas — kandidat perbaikan macro-F1 terbesar.
- **Early stopping** (patience ~30 pada `val_macro_f1`) → hemat ~60% waktu latih.
- **Scheduler** (mis. cosine / ReduceLROnPlateau) untuk menghaluskan plateau ribut.
- Bandingkan dengan arsitektur berkapasitas lebih besar yang sudah direncanakan
  (DenseNet121/264, ResNet18/50 — exp_005/007/008/009) memakai protokol sama.
- **Catat waktu latih** di `training_history.csv` (lihat §3) agar terukur.
- Perhatikan khusus **Sc, Ci, Mi** (recall rendah) saat evaluasi model berikut.

---

## 8. Reproduksibilitas

- Metrik & angka di dokumen ini seluruhnya berasal dari artefak
  `experiments/exp_006_alexnet/` dan `checkpoints/exp_006_alexnet/best.pt`
  (seed 42). Waktu pelatihan adalah estimasi dari mtime berkas (§3).
- Evaluasi ulang test dari checkpoint:
  `python scripts/train.py --config configs/experiments/exp_006_alexnet.yaml`
  (atau jalankan Sel 6 di [`notebooks/04_baseline_training.ipynb`](../../../notebooks/04_baseline_training.ipynb)).
- Protokol split, kontrak keluaran, dan aturan eksperimen:
  [TRAINING_GUIDE.md](TRAINING_GUIDE.md); rencana lengkap:
  [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md).
