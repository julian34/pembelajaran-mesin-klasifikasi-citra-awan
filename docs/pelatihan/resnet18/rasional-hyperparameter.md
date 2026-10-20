# Rasional Kombinasi ResNet-18 + Hyperparameter — Baseline exp_008

Dokumen ini menjelaskan **mengapa** arsitektur **ResNet-18** dipadukan dengan **set hyperparameter baseline** pada `exp_008_resnet18`. Hyperparameter dasarnya **sama persis** dengan AlexNet (exp_006) dan DenseNet-121 (exp_005) — itu disengaja (kontrol eksperimen). Maka dokumen ini **tidak mengulang** seluruh rasional per-nilai, melainkan merujuk rasional bersama dan menyorot **yang khas ResNet**.

Tanggal: 18 Oktober 2026. Framework: PyTorch / torchvision.
Terkait: [`BASELINE_RESNET18.md`](BASELINE_RESNET18.md), [rasional bersama (AlexNet)](../alexnet/rasional-hyperparameter.md), [`PERSIAPAN_PELATIHAN.md`](../PERSIAPAN_PELATIHAN.md), [`EXPERIMENT_PLAN.md`](../EXPERIMENT_PLAN.md), [`ARSITEKTUR_BASELINE.md`](../ARSITEKTUR_BASELINE.md).
Sumber nilai: [`configs/base.yaml`](../../../configs/base.yaml) + override [`configs/experiments/exp_008_resnet18.yaml`](../../../configs/experiments/exp_008_resnet18.yaml).

---

## 1. Konfigurasi efektif yang sedang dipakai

`exp_008_resnet18.yaml` hanya menuliskan nilai yang **berbeda** dari `base.yaml` (`imbalance: none`), sisanya **diwarisi**:

| Kelompok | Parameter | Nilai efektif | Asal |
|---|---|---|---|
| Model | architecture | `resnet18` | exp_008 |
| Model | pretrained | `imagenet` (`ResNet18_Weights.IMAGENET1K_V1`) | exp_008 |
| Model | finetune | `feature_extraction` (backbone beku, latih head) | exp_008 |
| Model | num_classes | 7 | base/exp_008 |
| Model | head | `fc`: `Linear(512→7)` (trainable 3.591) | arsitektur |
| Model | regularisasi internal | BatchNorm tiap blok (tanpa dropout FC) | arsitektur |
| Data | input / normalisasi | 512×512 RGB · /255 → [0,1] | base |
| Data | train / val / test | combined_balanced 21.000 / combined 1.851 / 9.000 | exp_008 |
| Training | epochs / batch | 200 / 32 (dipakai **8** pada 512×512) | base / runtime |
| Training | optimizer / lr / wd | AdamW / 1e-4 / 0.0 | base |
| Training | scheduler / imbalance | `null` / `none` | base / exp_008 |
| Checkpoint | monitor | `val_macro_f1` (max, save_best_only) | base |
| Repro | seed / num_workers | 42 / 0 | base |

Rasional lengkap tiap nilai (lr 1e-4, AdamW, wd 0, epochs 200, checkpoint macro-F1, imbalance none, normalisasi [0,1]) **identik** dengan baseline lain → lihat [rasional bersama di AlexNet §4](../alexnet/rasional-hyperparameter.md). Di bawah hanya yang **berbeda karena arsitektur ResNet**.

---

## 2. Mengapa ResNet-18 — peran & karakter arsitekturnya

ResNet-18 dipilih sebagai **pembanding ringan ber-skip-connection** dalam rangkaian baseline (exp_005 DenseNet-121, exp_006 AlexNet, exp_007 DenseNet-264, exp_008 ResNet-18, exp_009 ResNet-50). Karakternya menentukan beberapa pertimbangan khas:

1. **Residual learning (`y = F(x) + x`).** Skip connection membuat jaringan dalam mudah dilatih; bahkan saat backbone beku, fitur pra-latih yang mengalir ke head sudah "bersih" dan dalam. Ini pembeda utama dari AlexNet (dangkal) dan pelengkap DenseNet (concat, bukan sum).
2. **Varian ResNet paling ringan (BasicBlock, bukan Bottleneck).** ~11,2 juta parameter total — di antara AlexNet (57 juta) dan tetap lebih sederhana dari ResNet-50. Cocok sebagai titik referensi cepat.
3. **BatchNorm, bukan Dropout, sebagai regularisasi internal.** Tak ada dropout FC bawaan (beda dari AlexNet) — implikasinya di §3.
4. **Head paling kecil di antara baseline.** `fc` hanya `Linear(512→7)` = **3.591** parameter trainable (vs 7.175 DenseNet, 28.679 AlexNet) — model yang "paling sedikit belajar" pada `feature_extraction`.

---

## 3. Yang khas ResNet pada kombinasi ini

### 3.1 Regularisasi: BatchNorm beku menggantikan peran Dropout
AlexNet mengandalkan Dropout(0.5) bawaan sebagai peredam overfitting "gratis". ResNet-18 **tidak punya** itu; penggantinya ada tiga lapis: **(a)** `feature_extraction` yang hanya melatih 3.591 parameter head — kapasitas belajar sangat kecil → overfitting sulit terjadi; **(b)** **BatchNorm** di setiap blok yang, pada mode `feature_extraction`, **beku** memakai statistik ImageNet (running mean/var tidak diperbarui) sehingga fitur yang masuk ke head stabil & ter-skala; **(c)** train sudah seimbang. Jadi walau tanpa dropout, baseline ResNet-18 tetap terlindungi dari overfitting tanpa menambah hyperparameter baru.

### 3.2 Catatan bila berpindah ke `finetune: full`
Jika nanti backbone di-*unfreeze*, **BatchNorm ikut aktif** (running stats diperbarui) — pada batch kecil (8) statistik BN bisa berisik. Bila itu dilakukan, lr tetap kecil (1e-4) dan pertimbangkan membekukan BN atau menaikkan batch efektif. Untuk baseline sekarang (`feature_extraction`) isu ini tidak muncul.

### 3.3 Memori & batch
ResNet-18 **lebih ringan** dari DenseNet-121 pada 512×512 (tak ada concat peta fitur yang menumpuk memori). Batch 8 sangat aman dan sebetulnya bisa dinaikkan; **tetap dipakai 8** demi perbandingan setara lintas baseline (prinsip kontrol eksperimen). Batch efektif wajib dicatat di `notes.md` run.

---

## 4. Mengapa kombinasi ini koheren

| Pasangan | Hubungan saling-menguatkan |
|---|---|
| skip connection + feature_extraction | fitur dalam pra-latih mengalir utuh ke head meski backbone beku |
| BatchNorm beku + head kecil (3.591) + train seimbang | tiga lapis anti-overfitting tanpa dropout & tanpa variabel baru |
| ImageNet + lr 1e-4 + AdamW | lr kecil + optimizer adaptif menjaga fitur pra-latih tak rusak |
| epochs 200 + checkpoint by val-macro-F1 | boleh latih lama; model akhir = epoch val-terbaik |
| semua nilai = base.yaml bersama | hanya **arsitektur** yang berbeda → perbandingan exp_005…009 adil |

---

## 5. Keterbatasan & apa yang dipantau

- **Mismatch normalisasi** ([0,1] vs mean/std ImageNet) — sama seperti baseline lain; diterima demi keseragaman, sebagian diserap adaptasi head + BatchNorm beku.
- **feature_extraction** membatasi adaptasi domain; bila macro-F1 val stagnan, pertimbangkan `finetune: full` dengan catatan BN di §3.2.
- **Head sangat kecil (3.591 param)** → ResNet-18 adalah uji "seberapa jauh fitur ImageNet beku + classifier linear tipis" bisa memisahkan 7 kelas awan; pantau apakah ia *underfit* dibanding DenseNet.
- **Batch kecil (8)** & **I/O-bound** pada disk USB — identik baseline lain; pantau kurva loss, jalankan via runner terminal.
- Angka baseline ini **lantai pembanding**, bukan target performa (`EXPERIMENT_PLAN.md §1`).

---

## 6. Referensi

- He, Zhang, Ren, Sun (2016), *Deep Residual Learning for Image Recognition* (ResNet). https://arxiv.org/abs/1512.03385
- Ioffe & Szegedy (2015), *Batch Normalization*. https://arxiv.org/abs/1502.03167
- Loshchilov & Hutter (2019), *Decoupled Weight Decay Regularization* (AdamW). https://arxiv.org/abs/1711.05101
- torchvision ResNet-18: https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
- Dokumen internal: [`BASELINE_RESNET18.md`](BASELINE_RESNET18.md), [rasional bersama (AlexNet)](../alexnet/rasional-hyperparameter.md), [`EXPERIMENT_PLAN.md`](../EXPERIMENT_PLAN.md), [`PENYETARAAN_SAMPEL.md`](../../dataset/PENYETARAAN_SAMPEL.md).
