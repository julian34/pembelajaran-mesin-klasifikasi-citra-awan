# Rasional Kombinasi AlexNet + Hyperparameter — Baseline exp_006

Dokumen ini menjelaskan **mengapa** arsitektur **AlexNet** dipadukan dengan **set hyperparameter yang sedang dipakai** pada pelatihan baseline `exp_006_alexnet`, bukan sekadar *apa* nilainya. Fokusnya adalah alasan tiap pilihan dan — yang terpenting — **mengapa pilihan-pilihan itu saling cocok sebagai satu kombinasi**.

Tanggal: 16 Oktober 2026. Framework: PyTorch / torchvision.
Terkait: [`BASELINE_ALEXNET.md`](BASELINE_ALEXNET.md), [`PERSIAPAN_PELATIHAN.md`](../PERSIAPAN_PELATIHAN.md), [`EXPERIMENT_PLAN.md`](../EXPERIMENT_PLAN.md), [`PENYETARAAN_SAMPEL.md`](../../dataset/PENYETARAAN_SAMPEL.md), [`ARSITEKTUR_BASELINE.md`](../ARSITEKTUR_BASELINE.md).
Sumber nilai: [`configs/base.yaml`](../../../configs/base.yaml) + override [`configs/experiments/exp_006_alexnet.yaml`](../../../configs/experiments/exp_006_alexnet.yaml).

> **Catatan konsistensi.** Nilai hyperparameter di dokumen ini diambil dari `configs/base.yaml` **versi sekarang** (commit `30bc319`/`eaca319`: epochs **200**, optimizer **adamw**, lr **1e-4**). Angka lama "epochs 50 / adam / lr 0.001" yang masih tertulis di `BASELINE_ALEXNET.md §5` dan `PERSIAPAN_PELATIHAN.md §4` adalah **nilai awal yang sudah disalip** — yang berlaku pada pelatihan adalah nilai di bawah.

---

## 1. Konfigurasi efektif yang sedang dipakai

`exp_006_alexnet.yaml` hanya menuliskan nilai yang **berbeda** dari `base.yaml` (`imbalance: none`), sisanya **diwarisi**. Tabel berikut adalah hasil *merge* efektif yang benar-benar dipakai saat `scripts/train.py` dijalankan untuk AlexNet:

| Kelompok | Parameter | Nilai efektif | Asal |
|---|---|---|---|
| Model | architecture | `alexnet` | exp_006 |
| Model | pretrained | `imagenet` (`AlexNet_Weights.IMAGENET1K_V1`) | exp_006 |
| Model | finetune | `feature_extraction` (backbone beku, latih head) | exp_006 |
| Model | num_classes | 7 | base/exp_006 |
| Model | Dropout | 0.5 pada FC-6 & FC-7 (bawaan AlexNet) | arsitektur |
| Data | input size | 512 × 512 RGB | base |
| Data | normalisasi | /255 → [0, 1] (tanpa mean/std) | base |
| Data | train | `combined_balanced` 3.000/kelas (21.000) | exp_006 |
| Data | val / test | combined apa adanya (1.851 / 9.000) | exp_006 |
| Training | epochs | **200** | base |
| Training | batch_size | **32** (dipakai **8** pada 512×512, GPU 8 GB) | base / runtime |
| Training | optimizer | **AdamW** | base |
| Training | learning_rate | **1e-4** (0.0001) | base |
| Training | weight_decay | **0.0** | base |
| Training | scheduler | `null` (lr konstan) | base |
| Training | imbalance | `none` | exp_006 |
| Training | num_workers | 0 | base |
| Checkpoint | monitor | `val_macro_f1` (mode max, save_best_only) | base |
| Eval | metrik utama | macro-F1 (+ balanced acc, accuracy, per-kelas, confusion) | base |
| Repro | seed | 42 | base |

---

## 2. Mengapa AlexNet dulu — peran & karakter arsitekturnya

AlexNet dipilih **bukan** karena diharapkan menjadi model terbaik, melainkan karena perannya sebagai **baseline ringan** dalam rangkaian eksperimen (exp_005 DenseNet-121, exp_006 AlexNet, exp_007 DenseNet-264, exp_008 ResNet-18, exp_009 ResNet-50). Karakter arsitektur ini **menentukan** hyperparameter yang masuk akal:

1. **Dangkal & klasik (8 lapisan berbobot).** 5 konvolusi + 3 fully-connected (Krizhevsky dkk., 2012). Kapasitasnya kecil dibanding DenseNet/ResNet, sehingga menjadi **lantai pembanding** (*lower-bound*) yang jujur pada data & metrik yang sama.
2. **Didominasi parameter di FC-6/FC-7 (2× 4.096 neuron).** Mayoritas ±58 juta parameter AlexNet ada di dua lapisan FC inilah — titik paling rawan **overfitting**. Inilah alasan Dropout(0.5) menjadi bawaan desain, dan alasan strategi `feature_extraction` + `weight_decay` dipertimbangkan hati-hati (lihat §4).
3. **Ringan secara komputasi.** Lebih murah daripada DenseNet-264, sehingga **batch & resolusi besar lebih mungkin muat** di GPU 8 GB — relevan untuk keputusan batch size di §4.6.
4. **Tersedia bobot pra-latih ImageNet di torchvision.** Memungkinkan **transfer learning** langsung, yang menjadi pangkal seluruh pilihan hyperparameter (lr kecil, feature_extraction, dsb.).

> Varian yang benar-benar dipakai adalah **AlexNet torchvision single-stream** (64/192 channel, tanpa LRN), bukan rancangan dua-GPU asli 2012. Detail di [`BASELINE_ALEXNET.md §2.1`](BASELINE_ALEXNET.md). Head 1000→7 identik untuk kedua varian.

---

## 3. Prinsip yang menyatukan kombinasi

Seluruh hyperparameter mengalir dari **tiga keputusan dasar** yang harus konsisten satu sama lain:

- **(A) Transfer learning, bukan latih dari nol.** → lr kecil, optimizer adaptif, feature_extraction.
- **(B) Train sudah diseimbangkan (21.000, 3.000/kelas).** → `imbalance: none` (tanpa koreksi ganda), metrik tetap macro-F1 karena val/test **tidak** seimbang.
- **(C) Perbandingan eksperimen harus terkontrol.** → hyperparameter dasar dibuat **sama** lintas baseline (exp_005…009), scheduler & class-weight dimatikan agar hanya arsitektur yang berubah.

Inti "mengapa kombinasi" (bukan sekadar "mengapa tiap nilai"): ketiga prinsip ini dipilih supaya tidak saling bertabrakan.

---

## 4. Rasional per-hyperparameter (detail)

### 4.1 Transfer learning + bobot ImageNet
**Mengapa:** dataset train (21.000) relatif kecil untuk melatih CNN dari nol tanpa overfit berat, terlebih ±50% isinya citra augmentasi untuk kelas minoritas. Bobot ImageNet memberi fitur visual tingkat rendah–menengah (tepi, tekstur, gradien) yang **transferable** ke tekstur awan. Konsekuensinya: kita **tidak** perlu lr besar atau ratusan epoch untuk "membentuk" filter dari acak — ini yang membenarkan lr kecil & feature_extraction di bawah.

### 4.2 `finetune: feature_extraction` (backbone beku)
**Mengapa dipadu AlexNet:** karena beban parameter AlexNet menumpuk di FC, membekukan seluruh konvolusi + FC pra-latih dan hanya melatih **head `Linear(4096→7)`** sangat mengurangi jumlah parameter yang belajar → **risiko overfitting turun drastis** pada train yang sebagian besar augmentasi. Ini juga membuat baseline **murah & cepat konvergen**, cocok sebagai titik referensi. Trade-off: fitur tidak beradaptasi ke domain awan; bila macro-F1 val mentok, opsi `full` fine-tuning tersedia sebagai langkah lanjut (dengan lr yang tetap kecil, lihat 4.4).

> Dropout(0.5) bawaan tetap bekerja di *forward pass* mode latih meski bobot FC beku, sehingga fitur yang mengalir ke head tetap teregularisasi — penguat argumen anti-overfitting (lihat [`BASELINE_ALEXNET.md §3.1`](BASELINE_ALEXNET.md)).

### 4.3 Optimizer **AdamW**
**Mengapa:** AdamW = Adam dengan **decoupled weight decay**. Dipilih karena (a) *adaptif per-parameter* → konvergensi stabil & cepat pada fine-tuning tanpa tuning lr yang rewel, pas untuk baseline; (b) bila nanti `weight_decay` dinaikkan, AdamW menerapkannya dengan benar (terpisah dari update adaptif), berbeda dari Adam klasik yang mencampur L2 ke dalam momen. Jadi AdamW adalah **pilihan yang tahan-masa-depan**: sekarang `weight_decay=0` berperilaku seperti Adam, tapi siap untuk regularisasi tanpa ganti optimizer.

### 4.4 Learning rate **1e-4** (konstan)
**Mengapa segini, bukan 1e-3:** pada transfer learning, lr terlalu besar bisa "merusak" fitur pra-latih dan membuat head berosilasi. 1e-4 adalah nilai lazim-aman untuk fine-tuning/feature-extraction dengan Adam/AdamW — cukup cepat tapi tidak destruktif. Ini **selaras** dengan rencana di `EXPERIMENT_PLAN.md §2` yang mematok lr 1e-4 untuk baseline, sehingga AlexNet sebanding dengan baseline lain. (Nilai awal 1e-3 di dok lama sengaja diturunkan ke 1e-4.)

### 4.5 Weight decay **0.0**
**Mengapa nol:** demi **kontrol eksperimen** (prinsip C). Baseline sengaja dijaga minim regularisasi eksplisit agar perbandingan antar-arsitektur tidak tercampur efek weight decay; regularisasi untuk AlexNet sudah "cukup" dari dua sumber lain — **Dropout(0.5) bawaan** dan **feature_extraction**. Bila terlihat overfitting (gap train-vs-val macro-F1 melebar), weight_decay adalah tuas pertama yang dinaikkan — dan AdamW (4.3) sudah siap menanganinya dengan benar.

### 4.6 Batch size **32** (efektif **8** pada 512×512)
**Mengapa ada dua angka:** `base.yaml` menetapkan 32 sebagai default bersama; namun input **512×512 ≈ 5,2× piksel** dibanding 224×224, sehingga pada GPU **RTX 4060 Laptop 8 GB** batch 32 berisiko OOM. Karena AlexNet ringan ia **bisa** menampung batch lebih besar daripada DenseNet, tetapi notebook/runner memakai **batch 8** sebagai nilai aman & seragam lintas baseline. Batch yang lebih besar umumnya memberi estimasi gradien lebih stabil; di sini stabilitas sudah dibantu lr kecil + optimizer adaptif, jadi batch kecil dapat diterima. **Batch efektif yang dipakai wajib dicatat** di `notes.md` run (lihat `PERSIAPAN_PELATIHAN.md §5`).

### 4.7 Epochs **200**, tanpa scheduler, tanpa early stopping
**Mengapa 200 & lr konstan:** anggaran epoch dinaikkan dari 50→200 untuk memberi ruang konvergensi penuh karena **hanya head yang belajar** (feature_extraction) dan lr kecil → penurunan loss lebih lambat per-epoch. **Tanpa scheduler** menjaga satu-faktor perbandingan (prinsip C) dan menghindari variabel tambahan antar-baseline. Pengaman overfitting bukan early stopping melainkan **pemilihan checkpoint by `val_macro_f1`** (4.9): meski dilatih 200 epoch, yang disimpan adalah bobot pada epoch val-terbaik, sehingga epoch berlebih tidak merusak model akhir.

### 4.8 `imbalance: none`
**Mengapa dimatikan padahal dataset awan tak seimbang:** train `combined_balanced` **sudah diseimbangkan** di hulu (3.000/kelas via penyetaraan GCD+CCSN). Menambah `class_weights` di atas train yang sudah seimbang = **koreksi ganda** yang justru bias ke kelas tertentu (lihat [`PENYETARAAN_SAMPEL.md §2.3`](../../dataset/PENYETARAAN_SAMPEL.md)). Maka override `exp_006` menonaktifkannya. **Penting:** val/test tetap distribusi nyata (tidak diseimbangkan) → itulah sebabnya **metrik utama tetap macro-F1 / balanced accuracy**, bukan accuracy mentah (4.10/4.9).

### 4.9 Checkpoint monitor **`val_macro_f1`** (max)
**Mengapa:** pemilihan model harus pada data **bukan-test** dan metrik yang adil untuk kelas minoritas. macro-F1 memberi bobot sama tiap kelas sehingga kelas langka (mis. Mi) tidak tenggelam. `save_best_only` + monitor ini membuat 200 epoch aman (lihat 4.7). Test resmi **tidak pernah** dipakai memilih checkpoint (`EXPERIMENT_PLAN.md §6`).

### 4.10 Normalisasi **/255 → [0,1]** & input **512×512**
**Mengapa (dan catatannya):** semua sumber sudah 512×512 RGB sehingga dipakai apa adanya untuk mempertahankan detail tekstur awan (tanpa downscale ke 224). Normalisasi dipilih **/255 saja** (tanpa standardisasi mean/std ImageNet) sebagai keputusan praproses eksplisit proyek, seragam lintas baseline. **Trade-off yang disadari:** bobot ImageNet aslinya dilatih dengan mean/std ImageNet, jadi [0,1] sedikit menggeser distribusi input dari kondisi pra-latih — risiko ini diterima demi keseragaman & kesederhanaan, dan sebagian diserap oleh adaptasi head. Detail: [`PERSIAPAN_PELATIHAN.md §1`](../PERSIAPAN_PELATIHAN.md).

### 4.11 `num_workers: 0`, seed 42
Reproduksibilitas & kemudahan debugging di Windows/Jupyter (traceback lebih mudah dibaca; `DataLoader` worker >0 sering tidak stabil di notebook Windows). Seed 42 dikunci lintas eksperimen. Konsekuensi: pelatihan **I/O-bound** pada disk USB eksternal (baca JPEG serial) — bukan GPU-bound; untuk mempercepat, naikkan `num_workers` via runner terminal (lihat `BASELINE_ALEXNET.md §5.1`).

---

## 5. Mengapa kombinasi ini koheren (bukan sekadar daftar nilai)

| Pasangan | Hubungan saling-menguatkan |
|---|---|
| ImageNet + lr 1e-4 | lr kecil menjaga fitur pra-latih tak rusak → transfer learning efektif |
| feature_extraction + Dropout(0.5) + wd 0.0 | tiga lapis anti-overfitting "gratis" tanpa menambah variabel eksperimen |
| AdamW + wd 0.0 (sekarang) | berperilaku seperti Adam kini, tapi siap wd benar bila dinaikkan |
| epochs 200 + checkpoint by val-macro-F1 | boleh latih lama tanpa early stopping karena model akhir = epoch val-terbaik |
| train seimbang + imbalance none + metrik macro-F1 | hindari koreksi ganda; keadilan antar-kelas tetap diukur di val/test nyata |
| batch 8 + lr kecil + AdamW | batch kecil (akibat 512×512 di 8 GB) tetap stabil berkat optimizer adaptif + lr kecil |
| semua nilai = base.yaml bersama | hanya **arsitektur** yang berbeda antar baseline → perbandingan exp_005…009 adil |

Inti: tiap pilihan menutup risiko pilihan lain. Batch kecil yang "lemah" ditebus lr kecil + AdamW; epoch banyak yang rawan overfit ditebus checkpoint val-terbaik + Dropout + feature_extraction; train yang diseimbangkan ditebus metrik yang mengukur keadilan di distribusi nyata.

---

## 6. Keterbatasan & apa yang dipantau

- **Mismatch normalisasi** (ImageNet mean/std vs [0,1]) dapat sedikit menurunkan manfaat bobot pra-latih — diterima demi keseragaman.
- **feature_extraction** membatasi adaptasi domain; bila macro-F1 val stagnan, pertimbangkan `finetune: full` dengan lr tetap kecil.
- **Train ±50% augmentasi** untuk kelas minoritas (mis. Mi ~10×) → risiko model menghafal pola augmentasi. **Pantau gap train-vs-val macro-F1** sebagai indikator; tuas pertama: naikkan `weight_decay` (AdamW siap).
- **Batch kecil (8)** menambah varians gradien; dipantau lewat kurva loss (`training_history.csv`).
- **I/O-bound** pada disk USB eksternal (riwayat *intermittent read error*) → run panjang berisiko crash baca transien; jalankan lewat runner terminal dan pantau.
- Angka baseline ini adalah **lantai pembanding**, bukan target performa; tidak ada akurasi yang dijanjikan (`EXPERIMENT_PLAN.md §1`).

---

## 7. Referensi

- Krizhevsky, Sutskever, Hinton (2012), *ImageNet Classification with Deep CNNs* (AlexNet). https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks
- Krizhevsky (2014), *One weird trick for parallelizing CNNs* (varian single-stream torchvision). https://arxiv.org/abs/1404.5997
- Loshchilov & Hutter (2019), *Decoupled Weight Decay Regularization* (AdamW). https://arxiv.org/abs/1711.05101
- torchvision AlexNet: https://pytorch.org/vision/stable/models/generated/torchvision.models.alexnet.html
- Dokumen internal: [`BASELINE_ALEXNET.md`](BASELINE_ALEXNET.md), [`PERSIAPAN_PELATIHAN.md`](../PERSIAPAN_PELATIHAN.md), [`EXPERIMENT_PLAN.md`](../EXPERIMENT_PLAN.md), [`PENYETARAAN_SAMPEL.md`](../../dataset/PENYETARAAN_SAMPEL.md).
