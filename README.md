# Backend Flask

Backend ini menyediakan endpoint prediksi untuk aplikasi Android.

## Menjalankan server

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Server berjalan di:

```text
http://127.0.0.1:5000
```

Android emulator mengakses komputer host melalui:

```text
http://10.0.2.2:5000
```

## Endpoint

```text
POST /predict
Content-Type: multipart/form-data
Field: image
```

Contoh curl:

```bash
curl -X POST http://127.0.0.1:5000/predict -F "image=@sample.jpg"
```

## File model

Untuk program Colab versi kamu, tempatkan file hasil export training di folder `models`:

```text
models/
  svm_model.joblib
  scaler_proposed.joblib atau scaler.joblib
  pca_proposed.joblib atau pca.joblib
  kmeans_vocab.joblib
  hasil_perbandingan.json
```

`hasil_perbandingan.json` bersifat opsional untuk data menu peneliti.

Tambahkan kode berikut di akhir notebook Colab setelah variabel `model_main`,
`scaler_proposed`, `pca_proposed`, `kmeans`, dan `all_results` terbentuk:

```python
import json
import joblib
import os
import shutil
from google.colab import files

export_dir = "/content/chili_export_model"
os.makedirs(export_dir, exist_ok=True)

joblib.dump(model_main, f"{export_dir}/svm_model.joblib")
joblib.dump(scaler_proposed, f"{export_dir}/scaler.joblib")
joblib.dump(pca_proposed, f"{export_dir}/pca.joblib")
joblib.dump(kmeans, f"{export_dir}/kmeans_vocab.joblib")

with open(f"{export_dir}/hasil_perbandingan.json", "w", encoding="utf-8") as file:
    json.dump(all_results, file, indent=2)

shutil.make_archive("/content/chili_model_export", "zip", export_dir)
files.download("/content/chili_model_export.zip")
```

Setelah ZIP diunduh, ekstrak isinya dan salin seluruh file `.joblib` serta
`hasil_perbandingan.json` ke folder `backend/models`.

Untuk memastikan model export sudah cocok dengan pipeline Flask, jalankan:

```powershell
.\.venv\Scripts\python.exe validate_model_export.py
```

Validasi ini memeriksa dimensi fitur `3328`, scaler, PCA, kelas SVM, dan
jumlah cluster KMeans. Jika validasi berhasil, endpoint `/predict` sudah
memakai model export, bukan heuristic fallback.

Untuk memastikan file evaluasi peneliti sudah sesuai format aplikasi, jalankan:

```powershell
.\.venv\Scripts\python.exe validate_method_results.py
```

Validator ini memeriksa tiga metode `BoF`, `BoF+HSV`, `BoF+HSV+SPM`,
metrik `accuracy`, `precision_macro`, `recall_macro`, `f1_macro`,
`per_class`, `confusion_matrix`, waktu training/prediksi, PCA, dan dimensi
fitur. Jika file `hasil_perbandingan.json` belum ada, menu Peneliti akan
tetap berjalan memakai data contoh lokal dan menampilkan keterangan sumber
data di aplikasi.

Pipeline Flask sudah disamakan dengan program Colab:

- Resize gambar: `128 x 128`
- ORB: `nfeatures=500`
- BoF vocabulary: `128 visual word`
- HSV global: `16 x 4 x 8`
- SPM: level `1x1`, `2x2`, `4x4` dengan histogram HSV patch `8 x 4 x 4`
- Urutan fitur: `BoF + HSV + SPM`
- Transformasi: `StandardScaler -> PCA -> SVM`

Jika model belum ada, server memakai heuristic sederhana berbasis HSV agar alur Android tetap bisa diuji.
