# Augmentasi Data — Dataset GCD

Tanggal: 13 Oktober 2026. Terkait: [`ARSITEKTUR_BASELINE.md`](ARSITEKTUR_BASELINE.md), [`DATASET_SPLIT.md`](DATASET_SPLIT.md), [`TRAINING_GUIDE.md`](TRAINING_GUIDE.md).

Dokumen ini mendokumentasikan strategi augmentasi data untuk memperbanyak jumlah dan kompleksitas sampel train, mengikuti teknik yang dipakai pada literatur GCD: **transformasi geometri spasial** (translasi, flipping, rotasi) dan **transformasi warna** (color enhancement, contrast enhancement, brightness enhancement).

> **Status.** Augmentasi **belum diaktifkan** pada pelatihan (`configs/base.yaml` `augmentation.enabled: false`) — belum ada training yang dijalankan. "Hasil" pada dokumen ini berupa **contoh visual** penerapan augmentasi pada citra sampel (bukan hasil metrik pelatihan).

---

## 1. Ringkasan

| Butir | Nilai |
|---|---|
| Tujuan | Memperbanyak variasi sampel train → mengurangi overfitting, meningkatkan generalisasi |
| Teknik geometri | Translasi, flip horizontal, rotasi |
| Teknik warna | Color (saturasi), kontras, brightness |
| Diterapkan pada | **Hanya split train** |
| Tidak diterapkan pada | **Validation dan test** (dievaluasi apa adanya) |
| Cara penerapan | **On-the-fly** saat pemuatan batch (di memori); citra tidak ditulis ulang |
| Label | Tidak berubah oleh augmentasi (transformasi menjaga kelas) |

---

## 2. Teknik dan parameter

Parameter di bawah adalah usulan awal (dapat disetel); rentang `[a, b]` berarti faktor diambil acak dalam rentang itu tiap kali citra dimuat.

| Jenis | Teknik | Parameter usulan | Catatan domain |
|---|---|---|---|
| Geometri | Translasi | ±10% lebar/tinggi | Menggeser posisi awan |
| Geometri | Flip horizontal | peluang 0,5 | Aman untuk citra langit |
| Geometri | Rotasi | ±15° | Sudut kecil agar tetap realistis |
| Warna | Color (saturasi) | faktor 0,8–1,5 | Variasi kejenuhan warna langit |
| Warna | Kontras | faktor 0,8–1,4 | Variasi ketegasan tepi awan |
| Warna | Brightness | faktor 0,8–1,3 | Variasi kondisi pencahayaan |

**Catatan khusus citra awan berbasis darat:**
- **Flip vertikal tidak disarankan** — citra langit punya orientasi atas–bawah alami (langit di atas), membalik vertikal menghasilkan citra tak realistis. Paper menyebut "flipping" secara umum; di sini dipakai **horizontal** saja.
- **Area kosong** akibat translasi/rotasi perlu diisi (*fill*): `reflect` atau `edge` lebih natural daripada hitam (`0`). Contoh figur memakai hitam agar efek terlihat jelas.
- Augmentasi warna sebaiknya tidak terlalu ekstrem agar karakteristik kelas (mis. biru polos `4_clearsky`) tetap terjaga.

---

## 3. Contoh hasil (visual)

Figur: `reports/figures/augmentation_examples.png` — tiga kelas contoh (`1_cumulus`, `6_cumulonimbus`, `7_mixed`), kolom kiri citra asli lalu enam augmentasi.

| Kolom | Augmentasi | Yang terlihat |
|---|---|---|
| 1 | Asli | Citra sumber 512×512 |
| 2 | Translasi 10% | Citra bergeser; sudut kosong (di contoh diisi hitam) |
| 3 | Flip horizontal | Cerminan kiri–kanan |
| 4 | Rotasi 15° | Citra miring; sudut kosong terisi |
| 5 | Color ×1,5 | Warna lebih jenuh |
| 6 | Kontras ×1,4 | Tepi awan lebih tegas |
| 7 | Brightness ×1,3 | Citra lebih terang |

Contoh ini dibuat dengan Pillow (`ImageOps`, `ImageEnhance`) sebagai ilustrasi; implementasi pelatihan memakai pustaka augmentasi (Bagian 6).

---

## 4. Prinsip penerapan

1. **Hanya train.** Augmentasi memperkaya data pelatihan; validation dan test **tidak** diaugmentasi agar evaluasi mencerminkan data asli.
2. **On-the-fly.** Diterapkan acak saat setiap epoch memuat batch, bukan disimpan sebagai berkas baru — hemat ruang dan menambah variasi antar-epoch.
3. **Menjaga label.** Translasi/flip/rotasi/perubahan warna tidak mengubah jenis awan, jadi label tetap.
4. **Berdampingan dengan penanganan imbalance.** Bobot kelas (`training.imbalance`) tetap dipakai; augmentasi bukan pengganti penyeimbangan kelas. Opsional: augmentasi lebih agresif untuk kelas minoritas (mis. `7_mixed`) dapat dieksplorasi sebagai eksperimen lanjutan.
5. **Reprodusibilitas.** Setel seed agar transformasi acak dapat direproduksi saat dibutuhkan.

---

## 5. Usulan konfigurasi (`configs/base.yaml`)

Blok `augmentation` saat ini `enabled: false`. Usulan perluasan:

```yaml
augmentation:
  enabled: false          # set true untuk melatih dengan augmentasi
  apply_to: [train]       # tidak pernah val/test
  seed: 42
  geometric:
    translate: 0.10       # ±10% lebar/tinggi
    horizontal_flip: 0.5  # peluang
    rotation_deg: 15      # ±derajat
    fill: reflect         # reflect | edge | 0
  color:
    color: [0.8, 1.5]     # saturasi
    contrast: [0.8, 1.4]
    brightness: [0.8, 1.3]
```

Urutan dalam pipeline praproses: augmentasi (train) → resize 224 → normalisasi (lihat `preprocess`).

---

## 6. Catatan implementasi

Dua pustaka umum (keduanya di-apply hanya pada train):

- **torchvision.transforms** — `RandomAffine(degrees=15, translate=(0.1,0.1))`, `RandomHorizontalFlip(0.5)`, `ColorJitter(brightness=.., contrast=.., saturation=..)`.
- **albumentations** (sudah di `requirements.txt`) — `ShiftScaleRotate`, `HorizontalFlip`, `RandomBrightnessContrast`, `HueSaturationValue`.

`torchvision`/`torch` tercantum di `requirements.txt` tetapi **belum terpasang** di environment ini; pasang sebelum menjalankan pelatihan.

---

## 7. Langkah berikutnya

1. Aktifkan `augmentation.enabled: true` dan setel parameter pada eksperimen yang relevan.
2. Terapkan augmentasi di Dataset/DataLoader train (`src/datasets/`), pastikan val/test tanpa augmentasi.
3. Bandingkan baseline **tanpa** vs **dengan** augmentasi (macro-F1 di validation) sebagai studi ablasi, catat di `experiments/` sesuai `EXPERIMENT_PLAN.md`.
