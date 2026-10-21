# exp_005b_densenet121_freeze — artefak Tahap 1 (Prapemrosesan B/R)

Tanggal dibuat: 2026-10-20. Status: **folder artefak** (belum terisi; notebook belum dijalankan).

Folder ini mencatat **artefak** Tahap 1 pipeline `05b_densenet121_Freeze_weight`
(Normalisasi Rasio B/R). Sesuai konvensi proyek, **artefak keluaran disimpan di
`experiments/`**, sedangkan notebook ada di `notebooks/` dan dokumen/analisis di `docs/`.

## Sumber
- Notebook: [`notebooks/05b_densenet121_freeze_tahap1_prapemrosesan.ipynb`](../../../notebooks/05b_densenet121_freeze_tahap1_prapemrosesan.ipynb)
- Rancangan B/R: `docs/pelatihan/br_ratio_normalization/BR_RATIO_NORMALIZATION.md` (draft DenseNet dihapus 2026-10-20)
- Analisis: `docs/pelatihan/densenet121_freeze/tahap1_prapemrosesan/DRAFT_ANALISIS_TAHAP1.md`

## Artefak yang akan dibuat notebook (saat dijalankan)
| Berkas | Isi |
|---|---|
| `tahap1_samples.pt` | 5 sampel: `X_norm` [5,3,512,512], `lambda_N`, `labels`, `class_names`, `paths`, meta |
| `tahap1_samples_preview.png` | Grid 5 sampel: asli \| λ_N \| X_norm |
| `tahap1_config.json` | Parameter Tahap 1 (formula, eps, reconstruct_mode, remap) |

Belum ada berkas di sini sampai notebook dijalankan.
