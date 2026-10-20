# Experiment Plan

Tanggal: 12 Oktober 2026 (Asia/Jakarta).
Status: rencana, belum dieksekusi. EDA, penanganan kualitas data, pembentukan split, dan penggabungan dataset (GCD + CCSN_v2) selesai; kode training belum ditulis. Arsitektur baseline: DenseNet-264 (lihat [ARSITEKTUR_BASELINE.md](ARSITEKTUR_BASELINE.md)).

## 1. Research Objective

Membangun dan mengevaluasi klasifikasi single-label citra awan GCD dengan 7 kelas dan input RGB 224 x 224. Ukur pengaruh augmentasi, dilated convolution, dan ASPP melalui eksperimen terkontrol yang masing-masing dibandingkan langsung terhadap baseline. Hasil utama adalah kualitas klasifikasi per kelas dan macro F1, disertai accuracy serta biaya model.

Menurut [audit dataset](../dataset/DATASET_DOCUMENTATION.md), GCD tersedia di `data/raw/GCD/`: 19.000 citra JPEG RGB 512 x 512, dengan train resmi 10.000 dan test resmi 9.000, tanpa validation. Rencana hyperparameter di bawah adalah pilihan awal yang eksplisit, bukan hasil tuning atau bukti performa. Tidak ada target accuracy numerik yang dijanjikan dari hasil paper pada dataset lain.

Prasyarat sebelum tahap implementasi/training:

| No | Prasyarat | Status |
|---:|---|---|
| 1 | Lokasi GCD dan manifest train/test resmi diketahui | Terpenuhi (`reports/tables/gcd_raw_manifest.csv`) |
| 2 | Distribusi 7 kelas, mode, resolusi, dan integritas diaudit | Terpenuhi |
| 3 | Pemetaan label 7 kelas ditetapkan | Terpenuhi (DATASET_DOCUMENTATION bagian 2) |
| 4 | Penanganan 156 citra yang identik di train dan test diputuskan | **Terpenuhi** — 156 identik + 32 near-dup dibuang dari train (DATASET_QUALITY_SUMMARY §2.1) |
| 5 | Pembentukan validation dari train resmi diputuskan | **Terpenuhi** — train 8.340 / val 1.471 (85/15, DATASET_SPLIT.md) |
| 6 | Environment training terpasang (PyTorch dengan CUDA) | **Belum** (bagian 7) |
| 7 | `configs/*.yaml` diselaraskan dengan dokumen ini | **Belum** (bagian 7) |
| 8 | Sumber unduhan dan izin pemakaian GCD dicatat | **Belum** |

Test tidak digunakan untuk memilih arsitektur, augmentasi, atau hyperparameter.

## 2. Baseline Experiment

Nama: `exp_001_baseline`.

| Parameter | Rencana awal |
|---|---|
| Dataset | GCD di `data/raw/GCD/`, 19.000 citra, sesuai `reports/tables/gcd_raw_manifest.csv` |
| Split | Test resmi (9.000) tetap utuh; train dan validation dibentuk dari train resmi (10.000) menurut bagian 2.1 |
| Input size | 224 x 224 RGB, tensor `[B, 3, 224, 224]` |
| Jumlah kelas | 7; output logits `[B, 7]` |
| Arsitektur | DenseNet-264 (`block_config (6,12,64,48)`) dengan global average pooling dan classifier `Linear(in_features, 7)`; `in_features ≈ 2.688` (`model.classifier.in_features`). Lihat [ARSITEKTUR_BASELINE.md](ARSITEKTUR_BASELINE.md) |
| Inisialisasi | Backbone pra-latih ImageNet via `timm` (`densenet264d`); classifier baru dengan seed tetap. torchvision tidak menyediakan bobot DenseNet-264 — **fallback** DenseNet-201 torchvision bila bobot 264 tak tersedia (ARSITEKTUR_BASELINE.md Bagian 4) |
| Strategi | Fine-tuning seluruh layer sejak epoch pertama; tidak ada fase freeze terpisah (opsi feature-extraction di ARSITEKTUR_BASELINE.md) |
| Optimizer | Adam, betas=(0.9, 0.999), eps=1e-8 |
| Learning rate | 0.0001 untuk seluruh parameter trainable |
| Weight decay | 0.0001 |
| Batch size | 32; kelayakan memori diuji sebelum eksperimen penuh |
| Jumlah epoch | 50 untuk semua eksperimen; tidak memakai early stopping pada perbandingan awal |
| Scheduler | Tidak ada; learning rate konstan |
| Loss function | CrossEntropyLoss tanpa class weights atau label smoothing |
| Random seed awal | 42 |
| Seed konfirmasi | 42, 43, 44; split tidak berubah antar seed |
| Model selection metric | Validation macro F1 tertinggi |
| Tie-break | Validation loss lebih rendah; jika masih sama, pilih epoch lebih awal |
| Augmentasi | Tidak ada transform acak |
| Presisi | FP32, AMP tidak diaktifkan pada rencana awal |
| Sampling | Shuffle training biasa; tanpa weighted sampling/resampling |
| DataLoader awal | num_workers=0 untuk mempermudah reproduksibilitas di Windows; train shuffle, val/test tidak |

Pilihan DenseNet-264 memberi baseline transfer learning (koneksi padat antarlapisan, efisien parameter) dan memungkinkan ablation arsitektur dengan backbone sama; rinciannya di [ARSITEKTUR_BASELINE.md](ARSITEKTUR_BASELINE.md). **Ketersediaan bobot:** DenseNet-264 pra-latih tersedia lewat `timm` (`densenet264d`) — torchvision hanya menyediakan DenseNet-121/161/169/201, jadi fallback-nya DenseNet-201. Dokumentasi DenseNet: [paper DenseNet](https://arxiv.org/abs/1608.06993).

Preprocessing bersama: orientasi EXIF konsisten, RGB, resize penuh bilinear dengan antialias ke 224 x 224, tensor float pada rentang awal [0,1], lalu normalisasi mean `[0.485, 0.456, 0.406]` dan std `[0.229, 0.224, 0.225]`. Nilai ini berasal dari preprocessing pretrained, bukan statistik test. Resize penuh sengaja berbeda dari resize/crop bawaan bobot untuk mempertahankan bidang awan; dicatat sebagai keputusan eksperimen. Data asli tetap utuh dan transform dilakukan di memori pada tahap training mendatang.

Preprocessing di atas sesuai hasil audit: semua citra sudah RGB dan persegi, sehingga resize penuh tidak mengubah rasio aspek, dan tidak ada tag orientasi EXIF yang perlu ditangani.

Catat juga majority-class baseline sebagai pemeriksaan kewajaran: kelas mayoritas ditentukan dari train saja dan dinilai pada validation. Ini bukan pengganti baseline CNN. Kelas mayoritas di train resmi adalah Cb (30,03%); nilainya pada validation dihitung setelah validation dibentuk.

### 2.1 Pembentukan validation (sudah dilaksanakan)

**Terlaksana (85/15, bukan 80/20 pada rancangan awal di bawah).** Validation dibentuk dari train bersih: **train 8.340 / validation 1.471 / test 9.000**, stratified per kelas, seed 42, grup near-duplicate dijaga satu sisi; manifest di `reports/tables/gcd_raw_split_{train,val,test}_manifest.csv`. Rincian: [DATASET_SPLIT.md](../dataset/DATASET_SPLIT.md). Rancangan dan estimasi di bawah dipertahankan sebagai catatan usulan awal.

Alasan: GCD resmi hanya menyediakan train dan test. Pemilihan checkpoint dan hyperparameter memerlukan data yang bukan test, sehingga validation harus diambil dari train resmi.

Rancangan:

- Sumber: hanya train resmi (10.000 citra). Test resmi tidak disentuh.
- Proporsi: 80% train dan 20% validation, terstratifikasi per kelas.
- Seed pembagian: 42, tetap untuk semua eksperimen dan semua seed pelatihan.
- Citra yang identik di dalam train (1 pasang Mixed) ditempatkan di bagian yang sama.
- Pembagian disimpan sebagai manifest (daftar path per bagian) di `data/processed/`, tanpa menyalin atau memindahkan citra di `data/raw/`.

Perkiraan jumlah citra, sebelum pengecualian pada bagian 2.2 (angka pasti dapat berbeda satu citra per kelas karena pembulatan):

| Class | Train | Validation | Test resmi |
|---|---:|---:|---:|
| Cu | 620 | 155 | 750 |
| Ac | 580 | 145 | 750 |
| Ci | 922 | 231 | 753 |
| Cl | 1.720 | 430 | 1.589 |
| Sc | 1.477 | 369 | 1.790 |
| Cb | 2.402 | 601 | 2.761 |
| Mi | 278 | 70 | 607 |
| **Total** | **7.999** | **2.001** | **9.000** |

Keterbatasan: validation kelas Mixed hanya sekitar 70 citra, sehingga F1 kelas ini, dan karenanya validation macro F1, peka terhadap beberapa citra saja. Pengulangan tiga seed pelatihan dipakai untuk menilai seberapa stabil urutan peringkat eksperimen.

### 2.2 Penanganan citra identik train-test (sudah dilaksanakan)

**Terlaksana.** 156 citra train identik dengan test dikeluarkan dari train/validation (ditambah 32 near-duplicate lintas split terkonfirmasi lewat tinjauan visual); test resmi tetap 9.000. Rincian: [DATASET_QUALITY_SUMMARY.md](../dataset/DATASET_QUALITY_SUMMARY.md) §2.1–§2.2. Uraian di bawah adalah usulan awal.

Audit menemukan 156 citra test resmi yang juga ada di train resmi dengan label yang sama (155 clear sky, 1 mixed).

Usulan: keluarkan 156 salinan di sisi **train** dari train dan validation melalui manifest. Test resmi tetap 9.000 citra, sehingga angka test tetap sebanding dengan penelitian lain yang memakai test resmi. Tidak ada berkas yang dihapus. Daftar citra yang dikeluarkan disimpan bersama manifest.

Alternatifnya adalah membiarkan dataset apa adanya dan menyebut kebocoran itu sebagai keterbatasan. Pilihan ini lebih sederhana tetapi membuat recall clear sky pada test sedikit terlalu optimistis.

Keputusan ini harus diambil sebelum validation dibentuk dan berlaku sama untuk keempat eksperimen. 29 pasangan kandidat citra mirip antar split perlu ditinjau secara visual lebih dulu sebelum diperlakukan sebagai duplikat.

## 3. Planned Experiments

| Experiment | Main Change | Dataset | Purpose |
|---|---|---|---|
| exp_001_baseline | Baseline architecture: DenseNet-264 pretrained (ImageNet), classifier 7 kelas | GCD, manifest yang sama | Reference performance |
| exp_002_augmentation | Data augmentation pada train saja | GCD, manifest yang sama | Measure augmentation effect |
| exp_003_dilated | Dilated convolution pada stage terakhir baseline | GCD, manifest yang sama | Measure receptive field effect |
| exp_004_aspp | ASPP antara backbone baseline dan global average pooling | GCD, manifest yang sama | Measure multi-scale feature effect |

Ketiganya dibandingkan langsung dengan `exp_001_baseline`, bukan diterapkan secara kumulatif. Exp_003 tidak memakai augmentasi exp_002; exp_004 tidak memakai perubahan backbone exp_003. Seluruh eksperimen dimulai dari bobot pretrained yang sama, bukan checkpoint pemenang eksperimen sebelumnya.

**exp_002_augmentation:** tambahan awal hanya random horizontal flip p=0.5 pada citra train sebelum tensor/normalisasi. Tidak menambahkan random crop, vertical flip, rotasi, atau color jitter pada eksperimen ini agar efek awal mudah ditafsirkan. Tetapkan transform sebelum membuka hasil test. Transform validation/test tetap sama dengan baseline. Jika audit train menunjukkan orientasi/instrumen membuat flip tidak sesuai, revisi rencana dahulu sebelum eksperimen dimulai.

**exp_003_dilated:** ubah hanya convolution 3 x 3 pada **dense block terakhir** DenseNet (denseblock4) menjadi dilation=2 dan padding=2. Pertahankan growth rate, jumlah channel, transition, jalur koneksi padat, classifier, serta resolusi output stage. Muat bobot pretrained dengan bentuk tensor yang sama; catat perubahan perilaku spasialnya. Tidak menambahkan ASPP. DenseNet memakai transition (BatchNorm→1x1 conv→average pooling) antar blok, sehingga penyesuaian dilation harus menjaga ukuran feature map dan koneksi padat tetap konsisten; wajib uji shape dan konkatenasi. [Paper DenseNet](https://arxiv.org/abs/1608.06993).

**exp_004_aspp:** pertahankan backbone baseline tanpa dilasi tambahan. Tambahkan modul ASPP adaptasi klasifikasi pada **feature map akhir DenseNet** (2.688 channel untuk DenseNet-264, resolusi 7 x 7) sebelum global average pooling. Usulan cabang: 1 x 1, tiga convolution 3 x 3 dengan dilation 1/2/3 dan padding sesuai, serta global average pooling yang diproyeksikan lalu di-upsample. Tiap cabang menghasilkan 256 channel dengan ReLU; concatenation 5 x 256 = 1.280 channel diproyeksikan melalui 1 x 1 kembali ke dimensi fitur backbone (`in_features`), kemudian GAP dan classifier baseline. Cabang global tidak memakai BatchNorm pada tensor 1 x 1; usulan awal semua cabang cukup convolution dan ReLU, tanpa dropout tambahan. Parameter baru diinisialisasi Kaiming, bias nol, dengan seed tercatat. Rate kecil dipilih sebagai hipotesis pada feature map 7 x 7; jangan langsung menyalin rate segmentasi beresolusi besar. Ini adaptasi usulan, bukan reproduksi persis DeepLab. [Paper ASPP/DeepLabv3](https://arxiv.org/abs/1706.05587).

ASPP menambah kapasitas dan komputasi; kenaikan metrik tidak dapat diatribusikan semata-mata kepada skala fitur. Laporkan parameter, FLOPs bila alat tersedia, latensi pada perangkat dan batch yang sama, serta memori puncak. Pembanding yang disamakan jumlah parameternya dapat menjadi eksperimen lanjutan terpisah, bukan perubahan tersembunyi dalam empat eksperimen ini.

## 4. Controlled Variables

- Dataset, manifest/hash split, official test, dan label mapping identik.
- Input 224 x 224 RGB, orientasi, interpolasi, antialias, dan normalisasi identik.
- Backbone pretrained yang sama; state awal backbone dan classifier sama untuk seed yang sama. Modul baru memakai aliran RNG tersendiri agar tidak mengubah inisialisasi bagian yang dibandingkan.
- Seed, urutan sampling training, optimizer, learning rate, weight decay, batch size, loss, epoch budget, dan kebijakan BatchNorm sama.
- Tidak ada class weights/resampling/early stopping atau scheduler yang hanya muncul pada satu eksperimen.
- Validation/test memakai eval mode, tanpa gradien dan tanpa transform acak; label target tidak diubah.
- Definisi metrik, tie-break checkpoint, perangkat, presisi, dan prosedur test identik.
- Satu faktor utama berubah sesuai tabel. Jika batch size 32 tidak muat pada salah satu varian, tetapkan batch size bersama yang layak sebelum run formal; revisi konfigurasi dan ulangi pembanding yang terdampak.
- Perubahan rencana harus dicatat sebelum eksekusi. Tidak mengubah hyperparameter berdasarkan hasil test.

## 5. Evaluation Metrics

Metrik dihitung atas seluruh prediksi pada split, bukan rata-rata F1 per batch:

- Accuracy keseluruhan.
- Precision dan recall per kelas serta macro average.
- Macro F1-score: rata-rata aritmetika F1 dari **ketujuh kelas**, bukan F1 dari macro precision/recall.
- Per-class F1-score dan support; weighted F1 boleh sebagai tambahan, tidak menggantikan macro F1.
- Confusion matrix 7 x 7: baris label sebenarnya, kolom prediksi, urutan sesuai class mapping. Simpan counts dan normalisasi per baris beserta label sumbu.
- Cross-entropy loss untuk monitoring dan tie-break.

Gunakan daftar label eksplisit 0..6. Jika precision/recall/F1 tidak terdefinisi, gunakan zero_division=0 dan laporkan kondisi/support-nya. Kelas tanpa sampel validation/test harus ditandai sebagai keterbatasan evaluasi, bukan dihapus dari macro average diam-diam. Confusion matrix dengan baris support=0 ditandai agar tidak ditafsirkan sebagai recall terukur.

Analisis kesalahan selama pengembangan memakai validation. Laporkan rata-rata dan simpangan baku antar seed 42/43/44 untuk setiap eksperimen; perbandingan menggunakan seed berpasangan. Tidak memilih seed terbaik berdasarkan test. Klaim peningkatan harus mempertimbangkan variasi seed, trade-off per kelas, dan biaya komputasi; tidak otomatis menyebut selisih kecil sebagai peningkatan signifikan.

## 6. Model Selection

1. Setiap epoch memantau train dan validation saja. Test loader tidak digunakan dalam loop training.
2. Simpan checkpoint dengan validation macro F1 tertinggi; tie-break validation loss lebih rendah, lalu epoch lebih awal. Metrik nonfinite membatalkan pemilihan checkpoint pada epoch tersebut.
3. Gunakan aturan yang sama untuk setiap seed. Pilih varian utama melalui mean validation macro F1 tiga seed; tie-break mean validation loss, kemudian model dengan parameter lebih sedikit.
4. Kunci model, konfigurasi, preprocessing, seed pelaporan, dan daftar eksperimen yang dilaporkan sebelum evaluasi test.
5. Muat `best.pt` tiap run yang telah ditetapkan dan evaluasi official test sekali pada tahap akhir. Laporkan semua seed yang ditetapkan, tanpa memilih yang paling tinggi pada test.
6. Jangan memperbaiki model/augmentasi berdasarkan hasil test lalu menyebut hasil ulang sebagai evaluasi held-out yang independen.

Checkpoint menyimpan state_dict, optimizer state, epoch, validation metrics, config lengkap, mapping kelas, identitas bobot pretrained, preprocessing, manifest/hash dataset, dan RNG state. Best checkpoint terpisah dari last/resume checkpoint. Tidak ada checkpoint dibuat pada tahap dokumentasi ini.

## 7. Reproducibility

| Komponen | Fakta saat audit / keputusan |
|---|---|
| Seed training | Rencana awal 42; konfirmasi 42, 43, 44 |
| Seed split | 42; split **sudah dibuat** — GCD (8.340/1.471/9.000) dan gabungan (10.483/1.851/9.000), serta train seimbang `combined_balanced` (21.000) |
| Sistem operasi | Windows 11 Home Single Language 10.0.26200 |
| Python proyek | 3.11.9, dari `.venv/Scripts/python.exe` (terverifikasi berjalan; audit dataset dijalankan dengan interpreter ini) |
| PyTorch | **Terpasang** `2.1.1+cu121` di `.venv` (build CUDA; `requirements.txt` mematok 2.1.1) |
| torchvision | **Terpasang** `0.16.1+cu121` |
| GPU terdeteksi | NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB, dari nvidia-smi |
| Driver NVIDIA | 610.78 dari nvidia-smi |
| CUDA | `torch.version.cuda = 12.1`; `torch.cuda.is_available() = True` (terverifikasi dari notebook pelatihan) |
| PyTorch CUDA/cuDNN | Aktif — pelatihan berjalan di GPU (RTX 4060 Laptop, cu121) |
| Dependency terpasang | torch 2.1.1+cu121; torchvision 0.16.1+cu121; numpy 1.26.2; pandas 2.1.3; matplotlib 3.8.2; Pillow 10.1.0; opencv-python 4.8.1.78; scikit-learn 1.3.2; pyyaml (dipakai `load_config`); pkbar (progress bar) |
| Konfigurasi eksperimen | YAML `exp_005…009` sudah terisi (bukan placeholder); nilai efektif via merge `base.yaml` + override |

Catatan pemasangan PyTorch: di Windows, `pip install torch==2.1.1` dari indeks bawaan memasang build CPU. Agar GPU terpakai, PyTorch harus dipasang dari indeks build CUDA (untuk 2.1.1 tersedia cu118 dan cu121) dan `requirements.txt` perlu mencatat hal itu. Setelah terpasang, catat `torch.__version__`, `torch.version.cuda`, dan versi cuDNN di sini.

Pada tahap implementasi, kunci environment yang kompatibel dan rekam versi aktual Python/PyTorch/torchvision/CUDA/cuDNN, OS, GPU, driver, dependency lengkap, dan checksum bobot pretrained. Versi aktual **sudah tercatat** di tabel §7: PyTorch 2.1.1+cu121, torchvision 0.16.1+cu121, CUDA 12.1, cuDNN 8801.

Atur RNG Python, NumPy, PyTorch CPU/CUDA, DataLoader generator, worker, dan augmentasi. Gunakan deterministic algorithms yang didukung, nonaktifkan cuDNN benchmark dan TF32 pada perbandingan awal, serta catat konfigurasi backend yang diperlukan. Ketidaktersediaan operasi deterministik harus dilaporkan, bukan diabaikan diam-diam. Reproduksibilitas lintas versi/perangkat tidak dijamin hanya oleh seed. [Panduan PyTorch 2.1](https://docs.pytorch.org/docs/2.1/notes/randomness.html).

Simpan manifest data dan checksum sebelum training, hash konfigurasi, snapshot/hash source code, dan status perubahan lokal. Catat commit ID bila repositori tersedia; audit ini tidak melakukan commit Git. Untuk beberapa seed, generator sampling terpisah dari inisialisasi modul memastikan penambahan ASPP tidak tanpa sengaja mengubah urutan data.

Perbedaan yang perlu diselaraskan pada tahap berikutnya:

- `configs/base.yaml`: split 70/15/15 **sudah diganti** menjadi `data.split.val_fraction_of_train: 0.15` (rancangan bagian 2.1 terlaksana); `train_dir`/`val_dir`/`test_dir` kini **berisi** split GCD final (8.340/1.471/9.000, via `scripts/materialize_split.py`), bukan folder kosong; `preprocess.normalize`, `training.imbalance`, dan `evaluation` sudah diisi dari hasil EDA. `class_names` ([Cu…Mi]) tetap perlu dipetakan ke nama folder (`1_cumulus`…`7_mixed`); learning rate 0.001, weight decay 0, num_workers 2 masih nilai awal (rencana 0.0001/0.0001/0).
- **Arsitektur baseline:** ditetapkan **DenseNet-264** (transfer learning ImageNet), selaras dengan [`ARSITEKTUR_BASELINE.md`](ARSITEKTUR_BASELINE.md). Sebelum eksperimen: isi `model.architecture` (`densenet264` via `timm`, atau `densenet201` sebagai fallback) dan tambahkan `timm` ke `requirements.txt`.
- **Dataset kedua:** CCSN_v2 telah digabung ke skema 7 kelas GCD ([`PENGGABUNGAN_DATASET.md`](../dataset/PENGGABUNGAN_DATASET.md)); eksperimen dapat memakai set **GCD-only** (`gcd_raw_split_*`) atau **gabungan** (`combined_*`).
- `configs/experiments/*.yaml`: architecture masih null dan rincian modifikasi belum ditulis.
- `docs/TRAINING_GUIDE.md`: memakai nama `history.csv`/`test_metrics.json`. Untuk keluaran baru gunakan kontrak bagian 8, yaitu `training_history.csv` dan `metrics.json` dengan pemisahan split; dokumen panduan lama belum diedit pada tahap ini.
- `docs/TRAINING_GUIDE.md` bagian 4 menyebut pembagian bawaan 70/15/15; yang berlaku adalah split resmi GCD + validasi 0,15 dari train (bagian 2.1, **sudah dibuat**).
- Baseline tanpa class weights dipertahankan untuk perbandingan satu-faktor, meskipun `configs/base.yaml` kini menyediakan `imbalance: class_weights` dengan `gcd_raw_class_weights.csv` / `combined_class_weights.csv`. Rasio ketidakseimbangan train 8,63; eksperimen bobot kelas memakai infrastruktur itu.

## 8. Experiment Output

Kontrak keluaran berikut untuk tahap eksekusi mendatang; tidak membuat hasil atau angka dummy sekarang.

```text
experiments/
└── exp_xxx_name/
    ├── config.yaml
    ├── metrics.json
    ├── training_history.csv
    ├── classification_report.txt
    ├── confusion_matrix.png
    └── notes.md

checkpoints/
└── exp_xxx_name/
    └── best.pt

logs/
└── exp_xxx_name/
    └── training.log
```

`config.yaml` berisi konfigurasi efektif lengkap, bukan hanya override. `metrics.json` memisahkan validation/test dan mencantumkan status test `not_evaluated` sebelum evaluasi akhir. `training_history.csv` hanya memuat train/validation per epoch, tanpa metrik test. `classification_report.txt` dan `confusion_matrix.png` merujuk validation dari best checkpoint dan menyebut split secara eksplisit; laporan akhir test disimpan sebagai `test_classification_report.txt` dan `test_confusion_matrix.png` agar tidak menimpa bukti validation. Simpan counts confusion matrix juga dalam CSV/JSON dan versi normalisasi terpisah. `notes.md` berisi provenance data, environment, perubahan utama, status run, keterbatasan, dan keputusan.

Untuk pengulangan tiga seed, simpan struktur per-run yang sama di `experiments/<nama>/seed_42/`, `seed_43/`, dan `seed_44/`; checkpoint dan log mengikuti subfolder seed yang sama. Folder eksperimen induk menyimpan rangkuman antar seed. Metrik per-run tidak ditimpa oleh seed lain.

Verifikasi yang direncanakan sebelum run formal: kesesuaian class mapping lintas split, disjointness manifest/hash, tensor input/output, transform evaluasi deterministik, seluruh label dalam rentang 0..6, perhitungan metrik pada contoh kecil dengan jawaban diketahui, dan pemuatan ulang checkpoint yang menghasilkan prediksi identik dalam environment yang sama. **Kode pelatihan sudah ditulis** (`src/datasets/`, `src/models/baseline.py`, `src/training/trainer.py`, `src/evaluation/metrics.py`, `scripts/train.py`, notebook `04`–`06`); gunakan `python scripts/train.py --config … --dry-run` untuk validasi pra-run. Pelatihan penuh dipantau sendiri oleh pengguna.
