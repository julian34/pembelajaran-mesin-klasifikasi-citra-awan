# Referensi Eksternal (Non-Jurnal) — Persiapan

Dokumen ini mengumpulkan **referensi dari luar jurnal** (tautan web, dokumentasi resmi, repositori, standar) yang akan dimasukkan ke laporan/jurnal CitraAwan. Jurnal ilmiah (PDF) ada di folder `jurnal/` dan dikelola terpisah; dokumen ini khusus untuk sumber berbasis web.

**Cara pakai.**
- Tambahkan entri pada tabel kategori yang sesuai.
- Isi **Tanggal akses** saat tautan benar-benar dibuka/diverifikasi (format `YYYY-MM-DD`), karena konten web bisa berubah.
- Kolom **Status**: `terverifikasi` (tautan sudah dibuka dan benar) atau `perlu dicek` (masih placeholder / belum dibuka).
- Tautan yang ditandai `<isi URL>` **belum diverifikasi** — isi sendiri setelah membuka sumbernya; jangan dikutip sebelum dicek.

> Catatan: tautan dokumentasi resmi pustaka di bawah berstatus awal `perlu dicek` — buka sekali untuk memastikan versi/halaman sebelum masuk daftar pustaka final.

---

## 1. Sumber dataset

| No | Judul / Deskripsi | URL | Tanggal akses | Status | Catatan |
|---:|---|---|---|---|---|
| 1 | TJNU Ground-based Cloud Dataset (GCD) — repositori resmi | https://github.com/shuangliutjnu/TJNU-Ground-based-Cloud-Dataset | | perlu dicek | Sumber dataset proyek (19.000 citra, 7 kelas); sudah disebut di `DATASET_DOCUMENTATION.md` |
| 2 | Artikel sumber GCD (Li dkk., *Sensors* 2023) — versi daring | `<isi URL / DOI>` | | perlu dicek | Verifikasi DOI/URL MDPI; catatan: jumlah per kelas di artikel tidak cocok dengan data lokal |

---

## 2. Pustaka & alat (software)

| No | Pustaka | URL dokumentasi | Tanggal akses | Status | Catatan |
|---:|---|---|---|---|---|
| 1 | PyTorch | https://pytorch.org/docs/stable/ | | perlu dicek | Framework training |
| 2 | torchvision — transforms | https://pytorch.org/vision/stable/transforms.html | | perlu dicek | Resize + Normalize pada pipeline praproses |
| 3 | scikit-learn — `compute_class_weight` | https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html | | perlu dicek | Rujukan skema bobot kelas inverse-frequency |
| 4 | scikit-learn — `train_test_split` (stratify) | https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html | | perlu dicek | Rujukan pembagian stratified |
| 5 | scikit-learn — `f1_score` (macro) | https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html | | perlu dicek | Metrik evaluasi utama (macro-F1) |
| 6 | Pillow (PIL) | https://pillow.readthedocs.io/en/stable/ | | perlu dicek | Dekode/baca citra |
| 7 | NumPy | https://numpy.org/doc/stable/ | | perlu dicek | Komputasi numerik |
| 8 | pandas | https://pandas.pydata.org/docs/ | | perlu dicek | Manifest & tabel |
| 9 | Matplotlib | https://matplotlib.org/stable/ | | perlu dicek | Visualisasi EDA |

---

## 3. Metode & konsep (artikel/dokumentasi teknik)

| No | Topik | Judul / Sumber | URL | Tanggal akses | Status | Catatan |
|---:|---|---|---|---|---|---|
| 1 | Perceptual hashing (dHash) | "Kind of Like That" — Hacker Factor (Neal Krawetz) | `<isi URL>` | | perlu dicek | Dasar deteksi near-duplicate (dHash 16×16, jarak Hamming) |
| 2 | Jarak Hamming | Definisi/rujukan Hamming distance | `<isi URL>` | | perlu dicek | Ukuran kemiripan dHash |
| 3 | Data splitting & data leakage (ML) | Khin Yadanar Lin, "Lessons From My ML Journey: Data Splitting and Data Leakage" — Towards Data Science (25 Feb 2024) | https://towardsdatascience.com/two-rookie-mistakes-i-made-in-machine-learning-improper-data-splitting-and-data-leakage-3e33a99560ea/ | 2026-10-13 | terverifikasi | Justifikasi menjaga test terpisah/utuh & menghapus kebocoran lintas split (§2.1, §2.4) |
| 4 | Imbalanced classification | Artikel/rujukan tepercaya | `<isi URL>` | | perlu dicek | Justifikasi bobot kelas & macro-F1 |
| 5 | SHA-256 | Spesifikasi/standar (mis. NIST FIPS 180-4) | `<isi URL>` | | perlu dicek | Identitas berkas/piksel identik |

---

## 4. Standar, metrik, & lain-lain

| No | Topik | Judul / Sumber | URL | Tanggal akses | Status | Catatan |
|---:|---|---|---|---|---|---|
| 1 | Klasifikasi awan (genera) | Rujukan meteorologi (mis. WMO International Cloud Atlas) | `<isi URL>` | | perlu dicek | Definisi kelas awan |
| 2 | | | | | | |

---

## Template entri (salin untuk menambah)

```
| <no> | <judul/deskripsi singkat> | <URL atau `<isi URL>`> | <YYYY-MM-DD> | <terverifikasi/perlu dicek> | <relevansi dengan proyek> |
```

## Cara mengutip (saran)

Untuk tautan web, cantumkan minimal: **penulis/organisasi, judul halaman, URL, dan tanggal akses**. Contoh gaya ringkas:

> Nama/Organisasi. *Judul halaman*. URL (diakses YYYY-MM-DD).

Sesuaikan dengan gaya sitasi jurnal tujuan (mis. IEEE, APA) saat penyusunan daftar pustaka final.
