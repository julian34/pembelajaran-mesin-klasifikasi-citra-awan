# Penyetaraan dan Penanganan Ketidakseimbangan Sampel (Sample Imbalance)

Tanggal: 14 Oktober 2026 (Asia/Jakarta). Terkait: [`AUGMENTASI_DATA.md`](AUGMENTASI_DATA.md), [`DATASET_SPLIT.md`](DATASET_SPLIT.md), [`DATASET_QUALITY_SUMMARY.md`](DATASET_QUALITY_SUMMARY.md) §2.3.

Dokumen ini merinci strategi **menyetarakan jumlah sampel menjadi 3.000 gambar per kelas** (seperti pada praktik GCD/CCSN): kelas dengan **>3.000** gambar **dikurangi (trimming)**, kelas dengan **<3.000** gambar **diperbanyak dengan augmentasi**. Augmentasi memakai transformasi **geometri spasial** (translasi, flipping, rotasi) dan **warna** (color, contrast, brightness enhancement).

> **Status.** Rencana/metodologi — **belum dijalankan**. Augmentasi diterapkan di tahap training (butuh `torch`/`torchvision`, belum terpasang). Angka rencana di bawah dihitung dari manifest split aktual.

---

## 1. Ringkasan

| Butir | Nilai |
|---|---|
| Target | **3.000 gambar per kelas** |
| Kelas > 3.000 | **Trimming** (subsample acak berseed) |
| Kelas < 3.000 | **Augmentasi** sampai 3.000 |
| Diterapkan pada | **Hanya split train** |
| Val / test | **Tidak disentuh** (tidak di-trim, tidak diaugmentasi) |
| Teknik augmentasi | translasi, flip horizontal, rotasi, color, contrast, brightness |
| Hasil train (7 kelas) | 7 × 3.000 = **21.000** gambar |

---

## 2. Prinsip penting (wajib dipatuhi)

1. **Hanya train yang diseragamkan.** Validation dan test **tetap apa adanya** (distribusi nyata). Menyetarakan/augmentasi val atau test akan merusak validitas evaluasi dan membuat angka tidak sebanding dengan penelitian lain.
2. **Tanpa kebocoran.** Semua salinan augmentasi berasal dari citra **train** dan tetap di train. Trimming hanya membuang citra dari train (lewat manifest), bukan menghapus berkas `data/raw/`.
3. **Pilih satu strategi imbalance, jangan dobel.** Penyetaraan 3.000/kelas adalah **alternatif** dari pembobotan kelas (`training.imbalance: class_weights` di `configs/base.yaml`). Bila memakai resampling ke 3.000, set bobot kelas **seragam** (`imbalance: none`) agar kelas tidak dikoreksi dua kali.
4. **Determinisme.** Trimming dan pemilihan citra sumber augmentasi memakai **seed tetap (42)** agar dapat direproduksi.
5. **Evaluasi tetap jujur.** Karena val/test tetap timpang, metrik utama tetap **macro-F1 + per kelas** (bukan accuracy), sama seperti `evaluation` di config.

---

## 3. Teknik augmentasi

Enam teknik (detail parameter & contoh visual di [`AUGMENTASI_DATA.md`](AUGMENTASI_DATA.md)):

| Jenis | Teknik | Parameter usulan |
|---|---|---|
| Geometri | Translasi | ±10% lebar/tinggi |
| Geometri | Flip horizontal | peluang 0,5 (flip vertikal tidak dipakai — orientasi langit) |
| Geometri | Rotasi | ±15° |
| Warna | Color enhancement (saturasi) | faktor 0,8–1,5 |
| Warna | Contrast enhancement | faktor 0,8–1,4 |
| Warna | Brightness enhancement | faktor 0,8–1,3 |

Tiap citra augmentasi memakai kombinasi acak transformasi di atas; label kelas tidak berubah.

---

## 4. Rencana per kelas (target 3.000)

### 4.1 Train gabungan (GCD + CCSN) — basis utama

Dari `reports/tables/combined_train_manifest.csv` (train 10.483):

| Kelas | Train saat ini | Tindakan | Faktor | Target |
|---|---:|---|---:|---:|
| 1_cumulus (Cu) | 809 | augmentasi +2.191 | 3,71× | 3.000 |
| 2_altocumulus (Ac) | 1.027 | augmentasi +1.973 | 2,92× | 3.000 |
| 3_cirrus (Ci) | 1.511 | augmentasi +1.489 | 1,99× | 3.000 |
| 4_clearsky (Cl) | 1.669 | augmentasi +1.331 | 1,80× | 3.000 |
| 5_stratocumulus (Sc) | 2.187 | augmentasi +813 | 1,37× | 3.000 |
| 6_cumulonimbus (Cb) | 2.986 | augmentasi +14 | 1,00× | 3.000 |
| 7_mixed (Mi) | 294 | augmentasi +2.706 | 10,20× | 3.000 |
| **Total** | **10.483** | semua augmentasi, 0 trimming | | **21.000** |

### 4.2 Train GCD-only

Dari `reports/tables/gcd_raw_split_train_manifest.csv` (train 8.340):

| Kelas | Train saat ini | Tindakan | Faktor | Target |
|---|---:|---|---:|---:|
| 1_cumulus | 659 | augmentasi +2.341 | 4,55× | 3.000 |
| 2_altocumulus | 617 | augmentasi +2.383 | 4,86× | 3.000 |
| 3_cirrus | 980 | augmentasi +2.020 | 3,06× | 3.000 |
| 4_clearsky | 1.669 | augmentasi +1.331 | 1,80× | 3.000 |
| 5_stratocumulus | 1.568 | augmentasi +1.432 | 1,91× | 3.000 |
| 6_cumulonimbus | 2.553 | augmentasi +447 | 1,18× | 3.000 |
| 7_mixed | 294 | augmentasi +2.706 | 10,20× | 3.000 |
| **Total** | **8.340** | semua augmentasi, 0 trimming | | **21.000** |

### 4.3 Analisis

- **Trimming tidak aktif pada train kita.** Di kedua split, **semua 7 kelas < 3.000**, jadi tidak ada kelas yang dikurangi; semuanya diaugmentasi naik. Aturan trimming tetap ditulis untuk kelengkapan dan berlaku bila 3.000 diterapkan ke basis lain (mis. keseluruhan GCD yang punya Cb 5.764, Cl 3.739, Sc 3.636 > 3.000).
- **Kelas minoritas butuh augmentasi berat.** `7_mixed` harus diperbanyak ~10,2× (294→3.000), `1_cumulus`/`2_altocumulus` 3–5×. Risiko: variasi terbatas → potensi overfitting pada pola augmentasi. Pertimbangkan augmentasi lebih beragam untuk kelas ini, atau target per-kelas lebih rendah (mis. 2.000) sebagai studi sensitivitas.
- **`6_cumulonimbus` nyaris tidak berubah** (2.986→3.000, +14).

---

## 5. Implementasi (dua opsi)

**Opsi A — Pra-bangkitkan ke disk (gaya paper).**
Hasilkan citra augmentasi hingga 3.000/kelas dan simpan, mis. ke `data/combined_balanced/train/<kelas>/` (asli + augmentasi). Kelas >3.000 (bila ada) di-subsample acak. Deterministik dan mudah diaudit, tetapi menambah berkas (±21.000 train) dan menyimpan hasil augmentasi tetap.

**Opsi B — On-the-fly saat training (disarankan).**
Pakai sampling berimbang (mis. `WeightedRandomSampler` dengan bobot per kelas) + augmentasi acak tiap epoch pada train. Efek seimbang mendekati 3.000/kelas tanpa menulis berkas, dan variasi augmentasi berbeda tiap epoch (lebih kaya, risiko overfitting lebih kecil). Jumlah "efektif" per epoch disetel lewat jumlah sampel sampler.

Keduanya hanya menyentuh train; val/test memakai transform evaluasi (resize + normalisasi) tanpa augmentasi.

---

## 6. Konfigurasi

Penyetaraan ini **alternatif** dari `training.imbalance: class_weights`. Usulan penambahan di `configs/base.yaml` (pilih salah satu strategi):

```yaml
training:
  imbalance: class_weights        # class_weights | balanced_resample | weighted_sampler | none
  # Bila memakai penyetaraan 3.000/kelas:
  # imbalance: balanced_resample
  balanced_resample:
    target_per_class: 3000        # kelas > target di-trim, < target diaugmentasi
    apply_to: [train]             # tidak pernah val/test
    seed: 42
    augment: [translate, hflip, rotate, color, contrast, brightness]
```

Jika `imbalance: balanced_resample` dipakai, **jangan** sekaligus memakai `class_weights` (koreksi ganda).

---

## 7. Catatan & batasan

- Val/test tetap timpang (distribusi nyata); laporkan macro-F1 + metrik per kelas.
- Augmentasi warna jangan terlalu ekstrem agar karakteristik kelas (mis. biru polos `4_clearsky`) tetap terjaga.
- Target 3.000 adalah pilihan desain; nilai lain (mis. 2.000) dapat diuji sebagai sensitivitas, terutama karena `7_mixed` memerlukan pengali besar.
- Perbedaan dari paper sumber: paper dapat menyetarakan seluruh data; di proyek ini penyetaraan **hanya train** karena test resmi GCD dipertahankan utuh ([`DATASET_SPLIT.md`](DATASET_SPLIT.md)).
- Belum dijalankan: perlu `torch`/`torchvision` dan kode training; augmentasi diterapkan di `src/datasets/` saat pemuatan.

---

## 8. Rujukan

- Teknik & contoh augmentasi: [`AUGMENTASI_DATA.md`](AUGMENTASI_DATA.md)
- Split & jumlah per kelas: [`DATASET_SPLIT.md`](DATASET_SPLIT.md)
- Penanganan imbalance (bobot kelas): [`DATASET_QUALITY_SUMMARY.md`](DATASET_QUALITY_SUMMARY.md) §2.3
- Manifest train: `reports/tables/combined_train_manifest.csv`, `gcd_raw_split_train_manifest.csv`
