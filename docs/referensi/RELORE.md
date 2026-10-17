# Relore untuk CitraAwan

Relore 0.3.18 terpasang pada `.venv-relore`, terpisah dari lingkungan pelatihan.
Revisi sumber dikunci dalam `requirements-relore.txt`. Database berada di
`.relore/index.db`; database, clone, log, dan lingkungan ini diabaikan Git.

## Pemakaian lokal sekarang

Jalankan dari root proyek dengan PowerShell:

```powershell
.\scripts\relore.ps1 --version
.\scripts\relore.ps1 defs scripts/audit_dataset.py
.\scripts\relore.ps1 map src --limit 10
```

Pembacaan struktur Python ini tidak memerlukan server atau GitHub. Gunakan path
folder kode seperti `src` atau `scripts` untuk membatasi cakupan pemetaan.

## Server lokal

Database SQLite sudah diinisialisasi. Jalankan di terminal terpisah:

```powershell
.\scripts\relored.ps1 serve --host 127.0.0.1 --port 8766 --allow-sqlite
```

Di terminal kerja:

```powershell
.\scripts\relore.ps1 status
```

Server berjalan selama terminal pertama terbuka; Ctrl+C menghentikannya.
SQLite dipakai untuk pengembangan lokal. Deployment produksi memerlukan PostgreSQL.
`configs/relore.json` mengarahkan client ke localhost, bukan layanan internal Hugging Face.

## Hubungkan riwayat GitHub

Repository sudah dikonfigurasi sebagai `julian34/pembelajaran-mesin-klasifikasi-citra-awan`. Clone khusus Relore sudah tersedia di `.relore/clones/`. Folder kerja utama belum menjadi repository Git. Indeks diskusi belum terisi karena token GitHub belum tersedia. Relore tidak otomatis mengubah dokumen, notebook,
atau percakapan Codex menjadi memori keputusan.

1. Repository pada `configs/relore.json` sudah diisi sesuai URL yang diberikan.
2. Sediakan `GITHUB_TOKEN` di lingkungan terminal pengindeksan. Gunakan akses baca
   issue dan pull request; jangan menyimpan token dalam file proyek atau chat.
3. Jalankan perintah berikut:

```powershell
.\scripts\relored.ps1 backfill --repo julian34/pembelajaran-mesin-klasifikasi-citra-awan
.\scripts\relored.ps1 clone --repo julian34/pembelajaran-mesin-klasifikasi-citra-awan
# Untuk repository milik organisasi:
.\scripts\relored.ps1 authority --repo julian34/pembelajaran-mesin-klasifikasi-citra-awan
# Opsional: pembaruan berulang, Ctrl+C untuk berhenti.
.\scripts\relored.ps1 poll --repo julian34/pembelajaran-mesin-klasifikasi-citra-awan --interval 5m
```

Repository privat juga memerlukan kredensial Git yang sesuai untuk clone.
Setelah indeks terisi dan server berjalan:

```powershell
.\scripts\relore.ps1 search 'dataset split'
.\scripts\relore.ps1 search --symbol image_stats
```

Nomor issue dapat diperiksa dengan `inflight` dan `thread`; `why PATH:LINE`
menghubungkan baris pada clone GitHub dengan diskusi perubahannya. Clone server
bukan salinan perubahan lokal yang belum dikirim. Hasil kosong pada indeks yang
belum diisi bukan bukti tidak adanya diskusi.

## Instalasi ulang

Python >= 3.10 dan Git diperlukan. Pilih interpreter secara eksplisit jika perlu:

```powershell
.\scripts\setup-relore.ps1 -Python 'C:\Users\jardm\AppData\Local\Programs\Python\Python313\python.exe'
```

Skrip membuat lingkungan khusus, memasang paket, menjalankan migrasi, dan memeriksa
dependensi. `.venv` pelatihan lama tidak diubah oleh pemasangan ini.

## Rujukan

- [Sumber resmi Relore](https://github.com/huggingface/relore)
- [Operasional](https://github.com/huggingface/relore/blob/main/docs/operations.md)
- [Perintah CLI](https://github.com/huggingface/relore/blob/main/docs/cli.md)



## Hasil pemeriksaan pemasangan (14 Oktober 2026)

- CLI 0.3.18, pembacaan definisi/peta Python, dan pemeriksaan dependensi berhasil.
- Migrasi SQLite 1-7 selesai; koneksi client ke server localhost berhasil diuji.
- Clone GitHub berhasil pada commit `b9e5db31b09c`; repository saat diperiksa hanya
  berisi `README.md`. Kode CitraAwan lokal belum tersedia dalam clone tersebut.
- API GitHub mengembalikan daftar issue dan pull request kosong.
- Backfill memerlukan `GITHUB_TOKEN` dan belum dijalankan sampai selesai karena
  token belum tersedia. Tidak ada riwayat diskusi yang diindeks.
- Server uji sudah dihentikan. Gunakan perintah server di atas saat diperlukan.
