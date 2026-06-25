# Models

Tempatkan file hasil training di folder ini:

- `svm_model.joblib`
- `scaler.joblib` atau `scaler_proposed.joblib`
- `pca.joblib` atau `pca_proposed.joblib`
- `kmeans_vocab.joblib`
- `metadata.json`
- `hasil_perbandingan.json`

`kmeans_vocab.joblib` adalah vocabulary visual word untuk Bag-of-Features ORB.
`hasil_perbandingan.json` dipakai oleh menu Peneliti untuk mengisi Dashboard,
Perbandingan Metode, Detail Metode, `per_class`, dan `confusion_matrix`.
Jika file ini belum ada, aplikasi tetap berjalan memakai data contoh lokal dan
menampilkan keterangan bahwa evaluasi asli belum dimuat.

Validasi dari folder `backend`:

```powershell
.\.venv\Scripts\python.exe validate_model_export.py
.\.venv\Scripts\python.exe validate_method_results.py
```
