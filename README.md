## External Test Dataset

Dataset uji eksternal digunakan untuk menguji model CNN dan ResNet18 di luar dataset pelatihan.  

**Catatan penting:** Nama folder atau file di dataset uji eksternal dapat berubah sewaktu-waktu.  
Jika mengalami error atau file tidak ditemukan saat menjalankan skrip, periksa kembali nama folder dan file, lalu sesuaikan di skrip atau konfigurasi berikut:

- Folder default lama: `external_test_data/`
- Folder baru (perubahan terbaru): `"dataset analisa statistik/"`
- Contoh sub-folder: `Cloudy/`, `Rainy/`, `Foggy/`, `Sunny/`, `Snowy/`
- Pastikan nama file mengikuti format: `nama_kelas nomor.jpg` (misal: `cloudy 1.jpg`)

Jika dataset diperbarui atau diganti namanya, silakan sesuaikan path di skrip Python:

```python
PATH_DATASET = "dataset analisa statistik/"

