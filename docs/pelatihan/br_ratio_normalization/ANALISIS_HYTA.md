# Analisis HYTA — Segmentasi Awan/Langit (pratinjau 7 sampel)

> **Status: hasil nyata (skala pratinjau).** Angka di bawah dari eksekusi HYTA atas **7 sampel
> (1 per kelas)** dari `combined_balanced/train`, dijalankan 2026-10-20. **Bukan** evaluasi
> seluruh dataset — tujuannya memeriksa perilaku algoritma, bukan mengukur akurasi final.

Terkait:
- Algoritma: [`BR_RATIO_NORMALIZATION.md`](BR_RATIO_NORMALIZATION.md) (HYTA: Tahap 1 B/R + Tahap 2 thresholding)
- Notebook: [`notebooks/05c_hyta_segmentasi_awan.ipynb`](../../../notebooks/05c_hyta_segmentasi_awan.ipynb)
- Artefak: [`experiments/exp_hyta_segmentation/`](../../../experiments/exp_hyta_segmentation/)
  (`hyta_samples_preview.png`, `hyta_masks.pt`, `hyta_per_sample_stats.csv`, `hyta_config.json`)

---

## 1. Tujuan analisis

Memeriksa perilaku HYTA pada 7 kelas awan **sebelum** memutuskan integrasinya ke klasifikasi:

1. Apakah pemisah bimodal/unimodal (`std(λ_N) > Ts`) memilah tipe citra secara masuk akal.
2. Apakah ambang (`t` MCE untuk bimodal, `Tf` tetap untuk unimodal) memberi mask awan yang wajar.
3. Apakah keluaran HYTA **diskriminatif** antar-kelas (relevan bila dipakai sebagai fitur).

---

## 2. Setelan yang dianalisis

| Parameter | Nilai |
|---|---|
| Sumber | `data/combined_balanced/train` (manifes `combined_balanced_train_manifest.csv`) |
| Sampel | 7 (sampel pertama tiap kelas 0..6) |
| Tahap 1 | `λ_N = (B − R) / (B + R + ε)`, `ε = 1e-6` |
| Tahap 2 | `std(λ_N) > Ts` ⇒ MCE; selain itu fixed `Tf` |
| `Ts` / `Tf` | `0.03` / `0.250` |
| MCE | histogram 256-bin (Li & Lee), ambang pada skala `λ_N` |
| Ukuran | 512 × 512, RGB (0=R, 1=G, 2=B) |

---

## 3. Hasil per-sampel

Gambar: [`hyta_samples_preview.png`](../../../experiments/exp_hyta_segmentation/hyta_samples_preview.png)
(kolom: `asli (RGB)` | `λ_N` | `mask AWAN (putih)` | `overlay`).

| Label | Kelas | Mode | Ambang | std(λ_N) | Fraksi awan | Rentang λ_N |
|---:|---|---|---:|---:|---:|---|
| 0 | Cu (cumulus) | **bimodal → MCE** | 0,342 | 0,214 | 0,388 | [0,056; 0,773] |
| 1 | Ac (altocumulus) | **bimodal → MCE** | 0,539 | 0,226 | 0,514 | [0,170; 1,000] |
| 2 | Ci (cirrus) | unimodal → fixed | 0,250 | 0,022 | 1,000 | [0,064; 0,163] |
| 3 | Cl (clearsky) | unimodal → fixed | 0,250 | 0,021 | 0,006 | [0,241; 0,358] |
| 4 | Sc (stratocumulus) | unimodal → fixed | 0,250 | 0,011 | 1,000 | [−0,048; 0,004] |
| 5 | Cb (cumulonimbus) | unimodal → fixed | 0,250 | 0,017 | 1,000 | [−0,048; 0,061] |
| 6 | Mi (mixed) | unimodal → fixed | 0,250 | 0,028 | 1,000 | [0,023; 0,248] |

Ringkas: **bimodal 2 / unimodal 5**.

---

## 4. Interpretasi

**(a) Pemisah bimodal/unimodal bekerja sesuai harapan.** Hanya **Cu** dan **Ac** — citra dengan
awan *dan* langit biru yang berbeda jelas — melewati `Ts=0.03` (std 0,21–0,23) lalu memakai MCE.
Lima kelas lain berstd rendah (0,01–0,028): citra **relatif homogen** (overcast pada Ci/Sc/Cb/Mi,
atau langit cerah merata pada Cl) sehingga diperlakukan unimodal. Pemisahan ini konsisten dengan
asumsi HYTA.

**(b) Ambang MCE masuk akal pada kasus bimodal.** Cu → fraksi awan 0,39 (gumpalan awan di antara
langit), Ac → 0,51 (tutupan lebih rapat). Keduanya tidak jenuh, menandakan MCE memisahkan dua
modus pada histogram `λ_N`.

**(c) `Tf=0.250` mendominasi 5/7 kelas dan menghasilkan keluaran jenuh.** Pada mode unimodal, mask
menjadi **seluruhnya awan** (Ci/Sc/Cb/Mi = 1,000) atau **nyaris tanpa awan** (Cl = 0,006). Untuk
citra overcast/cerah ini wajar secara fisis, **tetapi** berarti HYTA **tidak memberi gradasi** di
kelas-kelas itu — pada praktiknya ia bertindak sebagai penanda "tipe citra", bukan peta awan halus.

**(d) Kalibrasi `Tf` patut ditinjau untuk CitraAwan.** Rentang `λ_N` **Cl = [0,241; 0,358]** berada
**tepat mengangkangi** `Tf=0.250`; hanya 0,6% piksel yang jatuh di bawahnya. Ambang dari makalah
(citra langit permukaan lain) **rawan** pada dataset ini: pergeseran kecil `Tf` dapat mengubah Cl
secara drastis. Perlu kalibrasi `Tf` (dan mungkin `Ts`) pada distribusi `λ_N` dataset.

**(e) `λ_N` negatif pada Sc/Cb** (hingga −0,048) menunjukkan `B < R` di sebagian piksel (awan
kelabu/gelap badai), konsisten dengan klasifikasi awan.

**(f) Fraksi awan TIDAK diskriminatif antar-kelas.** Empat kelas (Ci, Sc, Cb, Mi) semuanya 1,000 →
**skalar fraksi awan tak bisa membedakannya**. Implikasi untuk integrasi ke klasifikasi 7-kelas:
keluaran HYTA lebih berguna sebagai **fitur spasial** (mask/peta `λ_N` sebagai kanal/masukan cabang)
daripada sebagai **ringkasan skalar**.

---

## 5. Keterbatasan

- **Hanya 7 citra (1/kelas)** — cukup untuk memeriksa perilaku, **tidak** untuk kesimpulan statistik.
  Satu citra per kelas bisa tidak mewakili variasi intra-kelas.
- **Ts/Tf = nilai makalah**, belum dikalibrasi untuk CitraAwan (lihat §4d).
- **MCE berbasis histogram 256-bin** pada rentang data per-citra; resolusi ambang ≈ lebar bin.
- Tidak ada **ground-truth segmentasi**, sehingga mask dinilai secara kualitatif, bukan terhadap label piksel.

---

## 6. Rekomendasi / langkah lanjut

1. **Kalibrasi `Tf` (dan `Ts`)** memakai distribusi `λ_N` lintas banyak citra per kelas
   (mis. histogram agregat train) — terutama agar batas Cl stabil.
2. **Perluas evaluasi** ke puluhan–ratusan citra per kelas; laporkan sebaran fraksi awan & mode.
3. **Putuskan integrasi** (lihat [`BR_RATIO_NORMALIZATION.md`](BR_RATIO_NORMALIZATION.md) §5):
   mask/`λ_N` sebagai **kanal fitur** tampak lebih menjanjikan daripada fraksi skalar (§4f).
4. **Verifikasi MCE** dengan `skimage.filters.threshold_li` sebagai pembanding.
5. Bila dipromosikan, pindahkan fungsi HYTA ke `src/` dan catat parameter terkalibrasi di config.
