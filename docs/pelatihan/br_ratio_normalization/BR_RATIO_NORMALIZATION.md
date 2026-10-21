# HYTA — *Hybrid Thresholding Algorithm* (deteksi/segmentasi awan–langit)

> **Status: REFAKTOR (2026-10-20).** Dokumen ini menggantikan writeup "Normalisasi Rasio B/R
> saja" dengan **algoritma HYTA penuh**. Normalisasi Rasio B/R kini menjadi **Tahap 1** di
> dalam HYTA; **Tahap 2 — Prosedur Thresholding Hibrida** adalah "tahap yang terlewat" yang
> sebelumnya belum ditulis. Sumber: Li, Lu & Yang (2011), lihat §7.
>
> HYTA **model-agnostic**: keluarannya adalah **peta biner awan/langit**, bukan citra untuk
> masukan CNN. Cara memakai peta ini di pipeline klasifikasi 7-kelas masih **TBD** (§5).

---

## 1. Ringkasan

HYTA adalah algoritma **deteksi awan** pada citra langit **RGB berbasis permukaan**. Dua tahap:

1. **Tahap 1 — Normalisasi Rasio B/R** → citra rasio ternormalisasi `λ_N` (§2).
2. **Tahap 2 — Prosedur Thresholding Hibrida** → klasifikasi tiap piksel **AWAN** vs **LANGIT**
   secara adaptif: *fixed thresholding* untuk citra **unimodal**, *Minimum Cross Entropy* (MCE)
   untuk citra **bimodal** (§3).

**Keluaran:** peta biner `citra_hasil` seukuran citra masukan (tiap piksel ∈ {AWAN, LANGIT}).

**Intuisi B/R:** langit cerah kaya biru (`B ≫ R` → `λ_N` tinggi/positif); awan putih-abu lebih
netral (`B ≈ R` → `λ_N` rendah/≈0). Karena itu **piksel AWAN = `λ_N` di bawah ambang**.

---

## 2. Tahap 1 — Normalisasi Rasio B/R

$$\lambda_N = \frac{B - R}{B + R + \varepsilon}, \qquad \varepsilon = 10^{-6}$$

- `B`/`R` = kanal biru/merah citra RGB; kanal hijau `G` **tidak** dipakai pada `λ_N`.
- `ε = 1e-6` hanya pengaman pembagian nol (saat `B + R = 0`).
- Keluaran `λ_N` berdimensi `1 × H × W`, rentang teoretis `(−1, 1)`.

---

## 3. Tahap 2 — Prosedur Thresholding Hibrida  *(tahap yang terlewat)*

Deteksi **bimodalitas** histogram `λ_N` via **standar deviasi** `std_dev`, lalu pilih metode:

| Kondisi | Klasifikasi citra | Metode ambang | Ambang |
|---|---|---|---|
| `std_dev > Ts` | **bimodal** | *Minimum Cross Entropy* (MCE) | `t` optimal (dihitung MCE) |
| `std_dev ≤ Ts` | **unimodal** | *Fixed thresholding* | `Tf` tetap |

Parameter (dari makalah, untuk citra langit permukaan):

| Simbol | Nilai | Makna |
|---|---:|---|
| `Ts` | `0.03` | ambang standar deviasi pemisah unimodal/bimodal |
| `Tf` | `0.250` | ambang tetap (hasil estimasi statistik) untuk kasus unimodal |
| `t`  | — | ambang optimal MCE (dihitung) untuk kasus bimodal |

Aturan klasifikasi piksel (identik untuk kedua cabang):

```
JIKA λ_N[piksel] < ambang  →  AWAN
SELAIN ITU                 →  LANGIT
```

---

## 4. Pseudocode (dari sumber — dibersihkan)

```text
FUNGSI jalankan_algoritma_HYTA(citra_RGB):
    # TAHAP 1: NORMALISASI RASIO B/R  -> citra rasio ternormalisasi lambda_N
    citra_lambda_N = citra_kosong(ukuran(citra_RGB))
    UNTUK SETIAP piksel (r, g, b) DALAM citra_RGB:
        penjumlah = b + r
        JIKA penjumlah == 0: penjumlah = 1e-6
        citra_lambda_N[piksel] = (b - r) / penjumlah        # persamaan (2)

    # TAHAP 2: PROSEDUR THRESHOLDING HIBRIDA  -> peta biner AWAN/LANGIT
    citra_hasil = citra_kosong(ukuran(citra_RGB))
    std_dev = standar_deviasi(citra_lambda_N)
    Ts = 0.03

    JIKA std_dev > Ts:                                       # bimodal
        t = hitung_threshold_MCE(citra_lambda_N)             # Minimum Cross Entropy
        ambang = t
    SELAIN ITU:                                              # unimodal
        ambang = 0.250                                       # Tf (fixed)

    UNTUK SETIAP piksel DALAM citra_lambda_N:
        JIKA citra_lambda_N[piksel] < ambang:
            citra_hasil[piksel] = AWAN
        SELAIN ITU:
            citra_hasil[piksel] = LANGIT

    KEMBALIKAN citra_hasil
```

---

## 5. Pertanyaan terbuka / integrasi (selesaikan saat implementasi)

- **Rincian MCE belum ditulis.** `hitung_threshold_MCE(λ_N)` (Minimum Cross Entropy) perlu
  didefinisikan: diskritisasi `λ_N` (binning histogram), rentang pencarian ambang, kriteria
  cross-entropy minimum. Ini satu-satunya sub-rutin non-trivial.
- **Pemakaian keluaran di pipeline 7-kelas (kunci).** HYTA menghasilkan **mask biner**, bukan
  citra ternormalisasi. Opsi integrasi: (a) **fitur/mask tambahan** (kanal ke-4 atau masukan
  cabang), (b) **masking** area awan sebelum ekstraksi fitur, (c) **baseline segmentasi**
  terpisah (bukan masukan klasifikasi). **Belum diputuskan.**
- **Kanal hijau `G` diabaikan** pada `λ_N` — konsisten dengan HYTA; catat sebagai asumsi.
- **Validasi parameter** `Ts=0.03`, `Tf=0.250` untuk dataset CitraAwan (nilai asli dari citra
  langit permukaan; mungkin perlu dikalibrasi ulang).
- **Lokasi kode:** rencana sebagai modul prapemrosesan/segmentasi (mis.
  `src/datasets/` atau `src/preprocessing/`) — **belum ada**.

---

## 6. Artefak & berkas terkait

- **Analisis HYTA (hasil 7 sampel):** [`ANALISIS_HYTA.md`](ANALISIS_HYTA.md)
- Artefak HYTA: [`../../../experiments/exp_hyta_segmentation/`](../../../experiments/exp_hyta_segmentation/) (`hyta_samples_preview.png`, `hyta_masks.pt`, `hyta_per_sample_stats.csv`, `hyta_config.json`)
- Analisis Tahap 1 (B/R, lama): [`../densenet121_freeze/tahap1_prapemrosesan/DRAFT_ANALISIS_TAHAP1.md`](../densenet121_freeze/tahap1_prapemrosesan/DRAFT_ANALISIS_TAHAP1.md)
- Notebook: [`notebooks/05b_densenet121_freeze_tahap1_prapemrosesan.ipynb`](../../../notebooks/05b_densenet121_freeze_tahap1_prapemrosesan.ipynb)
- Keluaran notebook: [`experiments/exp_005b_densenet121_freeze/tahap1_prapemrosesan/`](../../../experiments/exp_005b_densenet121_freeze/tahap1_prapemrosesan/)

> Catatan: berkas B/R di atas baru mencakup **Tahap 1**; analisis/notebook perlu diperluas ke
> **Tahap 2 (thresholding + evaluasi segmentasi)**. Folder masih bernama *densenet121_freeze*/
> *exp_005b* — pemindahan ke lokasi netral (mis. `hyta/`) opsional.

---

## 7. Referensi

- **Li, Q., Lu, W., & Yang, J. (2011).** *A Hybrid Thresholding Algorithm for Cloud Detection
  on Ground-Based Color Images.* Journal of Atmospheric and Oceanic Technology, 28(10),
  1286–1296. doi:10.1175/JTECH-D-11-00009.1.
  Teks sumber lokal: [`analisis/teks/atot-jtech-d-11-00009_1.txt`](../../../analisis/teks/atot-jtech-d-11-00009_1.txt).
