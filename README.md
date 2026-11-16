# Prediksi Persetujuan Pengajuan Pinjaman (Loan Approval Prediction) Menggunakan Algoritma Random Forest

Tugas Akhir TA-06 — Pembelajaran Mesin

Aplikasi web untuk memprediksi apakah pengajuan pinjaman **disetujui (Approved)** atau
**ditolak (Rejected)** berdasarkan profil pemohon, menggunakan **Random Forest Classifier**.

| Hasil | Nilai |
|---|---|
| Konfigurasi terbaik | `n_estimators = 200`, `max_depth = 20` |
| Akurasi data uji | **98,24%** |
| F1-macro | 0,9813 |
| Fitur paling berpengaruh | Skor Kredit / CIBIL (±80%) |

---

## 1. Struktur Folder

```
loan-approval-rf/
├── data/
│   └── loan_approval_dataset.csv        # dataset
├── notebooks/
│   └── TA06_Random_Forest_Loan_Approval.ipynb   # eksperimen lengkap + analisis (untuk dinilai)
├── src/
│   ├── __init__.py
│   └── preprocess.py                    # aturan cleaning & encoding (dipakai train.py & app.py)
├── model/                               # DIBUAT OTOMATIS oleh train.py
│   ├── model.joblib                     # model Random Forest terbaik
│   └── metadata.json                    # hasil eksperimen & metrik (ditampilkan di web)
├── static/
│   ├── css/style.css                    # tampilan web
│   └── img/                             # grafik hasil eksperimen (dibuat oleh train.py)
├── templates/
│   ├── base.html                        # kerangka halaman
│   ├── index.html                       # halaman form prediksi
│   └── eksperimen.html                  # halaman hasil eksperimen
├── reports/figures/                     # grafik dari notebook (untuk laporan PDF)
├── api/index.py                         # pintu masuk untuk deploy Vercel
├── app.py                               # aplikasi web Flask
├── train.py                             # pipeline training: Skenario A, B, FI, evaluasi
├── requirements.txt                     # library untuk menjalankan web
├── requirements-dev.txt                 # library tambahan untuk training & notebook
├── vercel.json / .vercelignore          # konfigurasi deploy
└── README.md
```

**Alur kerja singkat:** `train.py` melatih model → menyimpan `model/` + grafik → `app.py` memuat model → web siap dipakai.

---

## 2. Persiapan (cukup sekali)

1. Install **Python 3.10–3.13** dari python.org. Saat instalasi, centang **"Add Python to PATH"**.
2. Install **Visual Studio Code**, lalu pasang extension berikut:
   - **Python** (Microsoft)
   - **Jupyter** (Microsoft)
3. Ekstrak folder proyek, lalu di VS Code pilih **File → Open Folder… → `loan-approval-rf`**.

---

## 3. Menjalankan Proyek di VS Code (Windows / PowerShell)

Buka terminal: **Terminal → New Terminal**. Jalankan perintah di bawah satu per satu.

### Langkah 1 — Buat virtual environment
```powershell
python -m venv venv
```

### Langkah 2 — Aktifkan virtual environment
```powershell
.\venv\Scripts\Activate.ps1
```
Jika berhasil, muncul tulisan `(venv)` di awal baris terminal.

> **Muncul error "running scripts is disabled on this system"?** Jalankan perintah berikut sekali, lalu ulangi Langkah 2:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```

Pengguna macOS/Linux memakai `source venv/bin/activate`.

### Langkah 3 — Pilih interpreter di VS Code
Tekan `Ctrl+Shift+P` → ketik **Python: Select Interpreter** → pilih yang berlabel **`venv`**.

### Langkah 4 — Install library
```powershell
pip install -r requirements-dev.txt
```

### Langkah 5 — Latih model
```powershell
python train.py
```
Proses ini memakan sekitar 20–60 detik. Terminal akan menampilkan hasil Skenario A, Skenario B,
Feature Importance, dan Classification Report. Setelah selesai, folder `model/` dan
`static/img/` akan terisi.

### Langkah 6 — Jalankan aplikasi web
```powershell
python app.py
```
Buka browser ke **http://127.0.0.1:5000**. Hentikan server dengan `Ctrl+C`.

Halaman yang tersedia:
- **Prediksi** — isi form, atau klik *Contoh disetujui* / *Contoh ditolak* untuk demo cepat.
- **Hasil Eksperimen** — grafik dan tabel Skenario A, B, Feature Importance, dan Confusion Matrix.

### Langkah 7 — Jalankan notebook
1. Buka `notebooks/TA06_Random_Forest_Loan_Approval.ipynb`.
2. Klik **Select Kernel** (kanan atas) → **Python Environments** → pilih **venv**.
3. Klik **Run All**. Grafik juga tersimpan ke `reports/figures/` untuk laporan.

---

## 4. Penjelasan Kode (untuk presentasi/video)

| File | Isi utama |
|---|---|
| `src/preprocess.py` | Menghapus spasi di nama kolom & nilai teks, membuang `loan_id`, mengganti aset negatif dengan 0, Label Encoding (`Graduate`=1, `Yes`=1, `Approved`=1) |
| `train.py` | Split 80:20 (stratified) → loop Skenario A `[10, 50, 100, 200, 500]` + CV 5-fold → loop Skenario B `[None, 3, 5, 10, 20]` → model terbaik → feature importance → evaluasi → simpan |
| `app.py` | Memuat model sekali saat start, memvalidasi input (wajib isi, angka, rentang), memprediksi + probabilitas, memberi peringatan jika input di luar rentang data latih |
| `notebooks/*.ipynb` | Versi bertahap dari `train.py` beserta **analisis tertulis** setiap eksperimen |

**Aturan pemilihan model terbaik:**
- Skenario A: akurasi testing tertinggi. Jika seri, pilih jumlah pohon paling sedikit (lebih ringan).
- Skenario B: akurasi testing tertinggi. Jika seri, pilih pohon yang lebih dangkal (lebih sederhana).

**Endpoint API (opsional, untuk pengujian):**
```powershell
curl.exe -X POST http://127.0.0.1:5000/api/predict -H "Content-Type: application/json" `
  -d '{\"no_of_dependents\":2,\"education\":\"Graduate\",\"self_employed\":\"No\",\"income_annum\":9600000,\"loan_amount\":29900000,\"loan_term\":12,\"cibil_score\":778,\"residential_assets_value\":2400000,\"commercial_assets_value\":17600000,\"luxury_assets_value\":22700000,\"bank_asset_value\":8000000}'
```

---

## 5. Upload ke GitHub

```powershell
git init
git add .
git commit -m "TA-06 Loan Approval Prediction - Random Forest"
git branch -M main
git remote add origin https://github.com/USERNAME/loan-approval-rf.git
git push -u origin main
```
Pastikan folder `model/` **ikut ter-upload**, karena Vercel membutuhkan `model.joblib`.

---

## 6. Deploy ke Vercel

1. **Kunci versi scikit-learn.** Model `.joblib` harus dibuka dengan versi scikit-learn yang
   sama dengan saat training. Cek versinya:
   ```powershell
   pip show scikit-learn
   ```
   Lalu di `requirements.txt` ganti baris `scikit-learn>=1.4` menjadi versi tersebut,
   misalnya `scikit-learn==1.6.1`. Commit dan push ulang.
2. Login ke **vercel.com** dengan akun GitHub → **Add New… → Project** → pilih repo `loan-approval-rf`.
3. Framework Preset: **Other**. Biarkan pengaturan lain default, lalu klik **Deploy**.
4. Setelah selesai, Vercel memberi URL seperti `https://loan-approval-rf.vercel.app`.

Cara kerjanya: `vercel.json` mengarahkan semua URL ke `api/index.py`, yang mengimpor `app` dari
`app.py`. File `.vercelignore` membuang dataset & notebook agar ukuran deploy kecil.

> Jika build gagal karena ukuran paket terlalu besar (scikit-learn + pandas cukup besar),
> alternatifnya adalah **Render.com** (Web Service, Start Command: `gunicorn app:app`, tambahkan
> `gunicorn` ke `requirements.txt`) atau **PythonAnywhere**.

---

## 7. Troubleshooting

| Masalah | Solusi |
|---|---|
| `FileNotFoundError: Model belum ada` | Jalankan `python train.py` dulu |
| `ModuleNotFoundError: No module named 'src'` | Jalankan perintah dari folder **root proyek** (tempat `app.py` berada), bukan dari dalam subfolder |
| `ModuleNotFoundError: sklearn/flask` | venv belum aktif atau interpreter VS Code salah → ulangi Langkah 2–4 |
| Notebook: `FileNotFoundError ../data/...` | Notebook harus dibuka dari folder `notebooks/` (sudah benar jika memakai struktur ini) |
| `InconsistentVersionWarning` saat load model | Versi scikit-learn berbeda dengan saat training → jalankan ulang `python train.py` |
| Angka hasil sedikit berbeda dari README | Wajar jika versi library berbeda; `random_state=42` sudah dikunci |

---

## 8. Pemetaan ke Ketentuan TA-06

| Ketentuan | Lokasi |
|---|---|
| Preprocessing (encoding, split 80/20, cek imbalance) | Notebook bagian 2, `src/preprocess.py` |
| Skenario A + line chart | Notebook bagian 3, `static/img/skenario_a.png` |
| Skenario B + analisis overfitting | Notebook bagian 4, `static/img/skenario_b.png` |
| Feature importance top-10 + interpretasi | Notebook bagian 5 |
| Evaluasi lengkap + kesimpulan | Notebook bagian 6–7 |
| Deployment berbasis web | `app.py`, `templates/`, Vercel |
| Laporan PDF | Gunakan template jurnal; grafik dari `reports/figures/` |
| Video demo 3–5 menit | Lihat saran alur di bawah |

### Saran alur video demo (±4 menit)
1. **(0:00–0:30)** Masalah: bank perlu keputusan kredit yang cepat dan konsisten; dataset 4.269 pemohon.
2. **(0:30–1:30)** Skenario A: tunjukkan grafik. Akurasi mendatar setelah ±100 pohon, sehingga menambah pohon tidak signifikan.
3. **(1:30–2:30)** Skenario B: mulai depth 10 akurasi training mencapai 1,0, tetapi testing tetap ±0,98. Jelaskan mengapa *bagging* menjaga generalisasi.
4. **(2:30–3:15)** Feature importance: CIBIL ±80%, lalu wawasan bisnisnya.
5. **(3:15–4:00)** Demo web: contoh disetujui, contoh ditolak, lalu ubah skor CIBIL dan tunjukkan probabilitas berubah.
