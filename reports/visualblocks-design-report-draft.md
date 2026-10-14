# Laporan Desain Visual Blocks

Draf isi laporan · 14 Oktober 2026

## Ringkasan eksekutif

Visual Blocks menyediakan lingkungan penyusunan pipeline ML berbasis graf. Rekomendasi laporan ini adalah mengevaluasinya sebagai sarana prototipe dan komunikasi alur ML, dengan uji reproduksibilitas sebelum diadopsi lebih luas. Rekomendasi tersebut merupakan penilaian desain, bukan hasil pengujian aplikasi.

README menjelaskan tiga komponen: editor graf berbasis Angular, pustaka node ML, dan runtime yang menentukan urutan eksekusi dari graf JSON. Integrasi Colab memungkinkan fungsi Python digunakan dari antarmuka visual. Proyek menyatakan bahwa produk ini bersifat eksperimental, tidak didukung resmi oleh Google, dan tidak menjamin pemeliharaan jangka panjang. [README Visual Blocks](https://github.com/google/visualblocks#readme)

## Temuan utama

### Kontrak node menjadi pusat interoperabilitas

Dokumentasi custom node memisahkan metadata `NodeSpec` dari implementasi custom element. Koneksi memerlukan kecocokan nama tipe, tetapi bentuk data tidak divalidasi otomatis saat runtime. Perubahan graf memicu eksekusi ulang; pengembang perlu menangani caching dan deduplikasi. Node harus mengirim event `outputs` agar eksekusi downstream dapat berlanjut. Mekanisme pelaporan error dan node Logger telah didokumentasikan. [Panduan custom nodes](https://github.com/google/visualblocks/blob/main/custom_nodes/README.md)

Implikasi desainnya adalah kompatibilitas koneksi belum cukup untuk memastikan kompatibilitas isi data. Usulan prioritas adalah pemeriksaan bentuk, nilai, serta ukuran input pada batas setiap node khusus. Pesan kegagalan sebaiknya menunjuk node, input yang salah, dan tindakan pemulihan.

### Berbagi graf perlu disertai pengelolaan dependensi

Pipeline dapat diekspor sebagai JSON dan diimpor untuk dikembangkan kembali. API key dan gambar yang diunggah lokal tidak ikut diekspor. Alur kontribusi komunitas juga meminta metadata dan gambar pratinjau. [Panduan pipeline](https://github.com/google/visualblocks/blob/main/pipelines/README.md)

URL custom node tersimpan dalam proyek dan dimuat kembali saat proyek dibuka. URL lokal tidak dapat digunakan dari komputer lain. [Panduan custom nodes](https://github.com/google/visualblocks/blob/main/custom_nodes/README.md)

Implikasinya, keberhasilan ekspor bukan bukti bahwa penerima dapat menjalankan eksperimen. Usulan desain adalah pemeriksaan dependensi ketika proyek dibuka, disertai daftar aset yang perlu disediakan, versi node, dan status koneksi. Kredensial tetap dipisahkan dari berkas berbagi.

### Integrasi Python memerlukan urutan operasi yang jelas

Alur Colab mencakup instalasi paket, definisi atau registrasi fungsi inferensi, pembuatan server, dan pemanggilan `server.display()`. README mengingatkan bahwa sel tampilan pipeline tersimpan perlu dijalankan manual setelah sel lain selesai, sehingga penggunaan Run all memiliki batasan. [README Visual Blocks](https://github.com/google/visualblocks#visual-blocks-in-google-colaboratory)

Usulan desain adalah panduan awal yang menampilkan kesiapan server dan fungsi terdaftar sebelum pengguna menjalankan graf. Tutorial perlu memiliki hasil yang dapat dikenali pada setiap langkah.

## Implikasi untuk rancangan produk

Pemisahan editor dan runtime membuka kemungkinan antarmuka authoring yang berbeda dari antarmuka penggunaan hasil. Namun, kelayakan embedding, distribusi mandiri, dan kompatibilitas versi perlu diuji pada kebutuhan aplikasi konkret sebelum dijadikan keputusan arsitektur.

Status eksperimental sebaiknya diterjemahkan menjadi batas adopsi: simpan kontrak pemrosesan dan artefak eksperimen di luar ketergantungan antarmuka tertentu. Dengan demikian, hasil prototipe tetap dapat diperiksa ketika lingkungan visual berubah.

## Rekomendasi

| Prioritas | Usulan | Bukti penerimaan yang perlu dikumpulkan |
| --- | --- | --- |
| P0 | Buat satu pipeline gambar sederhana dengan input, pemrosesan, dan keluaran yang jelas | Pengguna baru dapat menjalankan contoh dan mengenali hasil yang benar |
| P0 | Tambahkan validasi data pada node khusus | Input tidak sesuai menghasilkan pesan terarah dan tidak menggantung eksekusi |
| P1 | Sertakan manifest dependensi dan petunjuk pemulihan | Proyek hasil ekspor dapat dijalankan di sesi bersih dengan aset yang tercatat |
| P1 | Kendalikan pekerjaan mahal akibat perubahan graf | Jumlah eksekusi, latensi, dan permintaan eksternal tercatat selama pengeditan |
| P2 | Uji navigasi keyboard serta penyampaian status dan error | Pengguna dapat menyelesaikan alur utama tanpa mengandalkan warna saja |

Prioritas dan bukti penerimaan di atas adalah usulan evaluasi. Laporan ini tidak menetapkan angka kinerja atau mengklaim temuan aksesibilitas yang belum diuji.

## Lampiran

### Skenario evaluasi

1. Jalankan contoh dari awal pada sesi bersih dan catat langkah yang memerlukan bantuan.
2. Hubungkan node dengan tipe yang cocok tetapi bentuk data yang salah; periksa perilaku pemulihan.
3. Ubah parameter berulang kali dan catat apakah pekerjaan yang sama dijalankan kembali.
4. Ekspor lalu buka proyek di lingkungan lain; identifikasi aset dan dependensi yang hilang.
5. Putuskan dependensi eksternal secara terkendali dan periksa apakah status kegagalan dapat dipahami.

### Dasar penilaian

Penelaahan menggunakan README repositori, panduan custom node, dan panduan pipeline yang ditautkan pada setiap temuan. Penilaian ini berbasis dokumentasi; belum mencakup uji aplikasi langsung, benchmark, wawancara pengguna, atau audit keamanan. Tidak ada perubahan pipeline atau model CitraAwan yang dilakukan.
