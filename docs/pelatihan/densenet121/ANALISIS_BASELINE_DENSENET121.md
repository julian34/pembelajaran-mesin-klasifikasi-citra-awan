# Analisis Hasil Baseline DenseNet-121 (exp_005)

> **Status: selesai.** Semua angka pada dokumen ini diisi **hanya** dari artefak
> nyata di `experiments/exp_005_densenet121/` (`metrics.json`,
> `training_history.csv`, `classification_report.txt`, `notes.md`, `config.yaml`)
> — bukan perkiraan. Bandingkan dengan baseline AlexNet di
> [`../alexnet/ANALISIS_BASELINE_ALEXNET.md`](../alexnet/ANALISIS_BASELINE_ALEXNET.md).

Analisis hasil pelatihan **transfer learning DenseNet-121** (bobot ImageNet, mode
`feature_extraction`) untuk klasifikasi 7 kelas awan (GCD + CCSN_v2). Setup baseline
(arsitektur, data, konfigurasi): [`BASELINE_DENSENET121.md`](BASELINE_DENSENET121.md).

- Eksperimen: `exp_005_densenet121` · tanggal artefak **2026-10-20**
- Sumber (nyata, bukan perkiraan): `experiments/exp_005_densenet121/`
  — `metrics.json`, `classification_report.txt`, `training_history.csv`,
  `confusion_matrix.png`, `config.yaml`, `notes.md`; checkpoint
  `checkpoints/exp_005_densenet121/best.pt`.
- Rasional pilihan hyperparameter (bersama lintas baseline): [`../alexnet/rasional-hyperparameter.md`](../alexnet/rasional-hyperparameter.md).
- Diagram alur arsitektur (animasi, buka di browser): [`arsitektur-exp005.html`](arsitektur-exp005.html).

---

## 1. Ringkasan eksekutif

| Besaran | Nilai |
|---|---|
| Epoch terbaik (dipilih dari validation) | **144** dari 200 |
| val macro-F1 terbaik | **0,7786** |
| **Test macro-F1** (metrik primer) | **0,6798** |
| Test accuracy (= micro-F1 global) | **0,7138** |
| Test balanced accuracy | **0,6586** |
| Parameter dilatih / total | 7.175 / **6.961.031** (head `Linear(1024→7)` saja) |
| Waktu pelatihan | tidak tercatat (loop belum mencatat durasi ke file) |

**Inti temuan:** DenseNet-121 (feature extraction) mengungguli baseline AlexNet
exp_006 secara signifikan pada ketiga metrik: test macro-F1 **0,6798** vs 0,5042
(**+0,1756**), accuracy **0,7138** vs 0,5782 (+0,1356), balanced accuracy **0,6586**
vs 0,4865 (+0,1721) — padahal hanya melatih **7.175** parameter head (¼ dari AlexNet).
Lonjakan terbesar justru pada kelas bertekstur halus yang paling lemah di AlexNet:
Sc 0,28→**0,57**, Mi 0,29→**0,41**, Ci 0,38→**0,70**. Sisa kelemahan terkonsentrasi
pada **Mi** (F1 0,41; recall 0,28 — banyak tertukar jadi Cb) dan kerancuan **Sc↔Cb**.
Terdapat selisih val→test (val macro-F1 0,7786 vs test 0,6798) karena test memakai
distribusi resmi GCD 9.000 yang timpang, bukan karena overfitting (val_loss ≈ train_loss).

---

## 2. Konfigurasi yang dievaluasi

Diambil dari `experiments/exp_005_densenet121/config.yaml` (snapshot saat run). Identik dengan AlexNet exp_006 kecuali arsitektur:

| Item | Nilai |
|---|---|
| Arsitektur | `densenet121` (torchvision, pretrained ImageNet) |
| Mode fine-tune | `feature_extraction` (backbone beku, hanya `classifier` 1024→7 dilatih) |
| Input | 512×512 RGB, normalisasi `/255` → [0,1] (mean 0, std 1) |
| Augmentasi | nonaktif |
| Epochs | 200 |
| Batch size | 8 (dikonfirmasi dari `config.yaml`) |
| Optimizer | AdamW, lr 1e-4, weight_decay 0,0 |
| Scheduler | — (tidak ada) |
| Penanganan imbalance | `none` (train sudah diseimbangkan 3.000/kelas) |
| Checkpoint | simpan terbaik menurut `val_macro_f1` (mode max) |
| Data | train seimbang 21.000 · val 1.851 · **test resmi GCD 9.000** |

---

## 3. Waktu pelatihan

Durasi pelatihan **tidak tercatat**: loop pelatihan belum menulis stempel waktu
mulai/selesai ke file, dan `training_history.csv` hanya memuat metrik per epoch
(bukan timestamp). Tanggal artefak pada `notes.md`: **2026-10-20**. Lihat catatan
metode di analisis AlexNet §3.

---

## 4. Dinamika pelatihan (200 epoch)

Sumber: `training_history.csv` (kolom `train_loss`, `val_loss`, `val_macro_f1`,
`val_balanced_accuracy`, `val_accuracy`).

| Epoch | train_loss | val_loss | val_macro_f1 | Keterangan |
|---|---|---|---|---|
| 1 | 1,2675 | 0,8889 | 0,6416 | awal |
| … | … | … | … | … |
| **144 (best)** | 0,6087 | 0,5686 | **0,7786** | **terbaik → checkpoint** |
| 200 | 0,5920 | 0,5633 | 0,7748 | akhir |

Catatan: `val_loss` minimum justru di epoch 77 (0,5596), tetapi checkpoint dipilih
berdasarkan `val_macro_f1` (mode max) yang puncaknya di epoch 144. Train & val loss
turun berdampingan hingga ~0,6 dan mendatar — `val_loss` ≈ `train_loss` di akhir,
menandakan **tidak ada overfitting** yang berarti pada 200 epoch.

![Kurva loss per epoch](../../../experiments/exp_005_densenet121/plot_loss.png)

![Metrik validasi per epoch](../../../experiments/exp_005_densenet121/plot_val_metrics.png)

---

## 5. Evaluasi test — semua metrik

Test dihitung **sekali** memakai checkpoint terbaik (dipilih dari validation,
**bukan** test). Sumber: `metrics.json` + `classification_report.txt`.

### 5.1 Metrik global

| Metrik | Nilai | Catatan |
|---|---|---|
| Accuracy (= micro-F1) | 0,7138 | 6.424 / 9.000 benar |
| **Macro-F1** (primer) | **0,6798** | rata-rata F1 7 kelas (tak berbobot) |
| Balanced accuracy | 0,6586 | = rata-rata recall per kelas |
| Weighted-F1 | 0,7050 | F1 tertimbang support |
| Macro precision / recall | 0,7589 / 0,6586 | recall jauh < precision → model konservatif di kelas minor |

### 5.2 F1 per kelas (precision / recall / F1 / support)

Sumber: `metrics.json → test.per_class`.

| Kelas | Precision | Recall | **F1** | Support |
|---|---|---|---|---|
| Cu | 0,9256 | 0,6307 | 0,7502 | 750 |
| Ac | 0,8344 | 0,5440 | 0,6586 | 750 |
| Ci | 0,6121 | 0,8194 | 0,7007 | 753 |
| Cl | 0,9182 | 0,9824 | **0,9492** | 1.589 |
| Sc | 0,5813 | 0,5615 | 0,5712 | 1.790 |
| Cb | 0,6544 | 0,7936 | 0,7173 | 2.761 |
| Mi | 0,7860 | 0,2784 | **0,4112** | 607 |

Urut F1: **Cl 0,95** > Cu 0,75 > Cb 0,72 > Ci 0,70 > Ac 0,66 > Sc 0,57 > **Mi 0,41**.
Mi adalah titik terlemah — precision tinggi (0,79) tetapi recall sangat rendah (0,28):
model jarang menebak Mi, dan saat Mi muncul banyak tertukar menjadi Cb (lihat §5.3).

![Precision / recall / F1 per kelas](../../../experiments/exp_005_densenet121/plot_per_class_f1.png)

### 5.3 Confusion matrix

Baris = kelas sebenarnya, kolom = prediksi (urutan Cu, Ac, Ci, Cl, Sc, Cb, Mi).

![Confusion matrix test](../../../experiments/exp_005_densenet121/confusion_matrix.png)

Kerancuan utama (dari `metrics.json → test.confusion_matrix`):

- **Sc ↔ Cb** saling tertukar paling besar: Sc→Cb 619 dan Cb→Sc 500 — dua kelas
  dominan yang teksturnya mirip.
- **Mi → Cb** 226 (lebih banyak daripada Mi yang benar, 169) → penyebab recall Mi rendah.
- **Cl** nyaris sempurna (1.561/1.589 benar), konsisten dengan F1 0,95.

---

## 6. Perbandingan dengan AlexNet (exp_006)

| Metrik | AlexNet exp_006 | DenseNet-121 exp_005 | Selisih |
|---|---|---|---|
| Test macro-F1 | 0,5042 | **0,6798** | **+0,1756** |
| Test accuracy | 0,5782 | **0,7138** | +0,1356 |
| Test balanced accuracy | 0,4865 | **0,6586** | +0,1721 |
| Parameter trainable | 28.679 | 7.175 | ¼× |

**F1 per kelas (AlexNet → DenseNet):**

| Kelas | AlexNet | DenseNet | Δ |
|---|---|---|---|
| Cl | 0,74 | 0,95 | +0,21 |
| Cu | 0,67 | 0,75 | +0,08 |
| Cb | 0,63 | 0,72 | +0,09 |
| Ac | 0,54 | 0,66 | +0,12 |
| Ci | 0,38 | 0,70 | **+0,32** |
| Mi | 0,29 | 0,41 | +0,12 |
| Sc | 0,28 | 0,57 | **+0,29** |

Hipotesis terkonfirmasi: kedalaman + *dense connectivity* DenseNet **mengangkat
kelas bertekstur halus** yang paling lemah di AlexNet — Ci (+0,32) dan Sc (+0,29)
melompat paling jauh, dan **semua** kelas membaik tanpa kecuali. Peningkatan
dicapai meski parameter yang dilatih hanya **¼** milik AlexNet, karena fitur ImageNet
DenseNet yang beku jauh lebih kaya daripada fitur AlexNet beku. Kelemahan yang
tersisa sama polanya di kedua model — Mi dan kerancuan Sc/Cb — tetapi dengan margin
yang jauh lebih kecil di DenseNet.

---

## 7. Diagnosis & rekomendasi

**Diagnosis**

1. **Baseline kuat dengan biaya minim.** Hanya head 7.175 parameter yang dilatih,
   namun macro-F1 test 0,6798 — lonjakan +0,18 atas AlexNet. Backbone ImageNet beku
   sudah menyediakan fitur yang sangat diskriminatif untuk tekstur awan.
2. **Bottleneck = recall kelas minor, bukan overfitting.** `val_loss` ≈ `train_loss`
   di akhir (tak ada overfitting), tetapi macro recall (0,6586) jauh di bawah macro
   precision (0,7589). Model konservatif: kelas kecil seperti **Mi** (recall 0,28)
   sering dilewatkan dan jatuh ke **Cb**.
3. **Kerancuan terstruktur Sc↔Cb.** Dua kelas dominan (support 1.790 & 2.761) saling
   tertukar masif (619 + 500) — kemiripan visual, bukan sekadar imbalance.
4. **Selisih val→test (0,7786 → 0,6798).** Val memakai subset seimbang 1.851, test
   memakai distribusi resmi GCD 9.000 yang timpang; angka test adalah yang dilaporkan.

**Rekomendasi (untuk eksperimen lanjutan)**

- **Fine-tuning bertahap**: buka sebagian blok akhir DenseNet (`denseblock4` +
  `norm5`) dengan lr kecil — target utama menaikkan recall Mi & memisah Sc/Cb.
- **Aktifkan augmentasi** (sudah ada di `notebooks/03_augmentation.ipynb`) untuk
  menambah variasi kelas minor.
- **Tangani imbalance** pada loss (class weights / focal loss) karena test timpang,
  meski train sudah diseimbangkan.
- Bandingkan terhadap **ResNet-18 exp_008** saat selesai untuk memilih backbone lanjutan.

---

## 8. Reproduksibilitas

- Metrik & angka harus seluruhnya berasal dari artefak `experiments/exp_005_densenet121/`
  dan `checkpoints/exp_005_densenet121/best.pt` (seed 42).
- Evaluasi ulang test dari checkpoint:
  `python scripts/train.py --config configs/experiments/exp_005_densenet121.yaml`
  (atau jalankan Sel 6 di [`notebooks/05_densenet121_training.ipynb`](../../../notebooks/05_densenet121_training.ipynb)).
- Protokol split, kontrak keluaran, dan aturan eksperimen:
  [`TRAINING_GUIDE.md`](../TRAINING_GUIDE.md); rencana lengkap: [`EXPERIMENT_PLAN.md`](../EXPERIMENT_PLAN.md).
