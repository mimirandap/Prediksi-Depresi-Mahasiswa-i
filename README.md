# 🧠 Prediksi Depresi pada Mahasiswa menggunakan Random Forest

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange?logo=scikitlearn)
![Status](https://img.shields.io/badge/Status-Completed-green)
![Dataset](https://img.shields.io/badge/Dataset-502%20data-lightblue)

> Proyek Machine Learning untuk memprediksi apakah seorang mahasiswa mengalami depresi berdasarkan faktor akademik, gaya hidup, dan riwayat kesehatan mental menggunakan algoritma **Random Forest Classifier**.

---

## 📌 Deskripsi Proyek

Depresi pada mahasiswa merupakan salah satu isu kesehatan mental yang semakin meningkat. Tekanan akademik, kurang tidur, kebiasaan makan yang buruk, dan stres finansial menjadi faktor utama pemicunya.

Proyek ini membangun model Machine Learning untuk mengklasifikasikan apakah seorang mahasiswa **mengalami depresi (Yes) atau tidak (No)** berdasarkan data survei dari 502 mahasiswa.

---

## 📁 Struktur Repository

```
📦 prediksi-depresi-mahasiswa/
├── 📓 prediksi_depresi_mahasiswa.ipynb     # Source code utama
├── 📊 Depression Student Dataset.csv        # Dataset (502 data)
├── 📄 README.md                             # Dokumentasi ini
├── 🎞️ slide_presentasi.pdf                 # Slide presentasi
├── 🎥 link_video_demo.txt                  # Link video demo
└── 📂 hasil_evaluasi/
    ├── distribusi_kelas.png
    ├── gender_vs_depresi.png
    ├── analisis_fitur_kategorikal.png
    ├── analisis_fitur_numerik.png
    ├── heatmap_korelasi.png
    ├── confusion_matrix_roc.png
    ├── feature_importance.png
    └── cross_validation.png
```

---

## 📊 Dataset

| Info | Detail |
|------|--------|
| **Nama** | Depression Student Dataset |
| **Sumber** | [Kaggle — ikynahidwin](https://www.kaggle.com/datasets/ikynahidwin/depression-student-dataset) |
| **Jumlah Data** | 502 baris |
| **Jumlah Fitur** | 10 fitur + 1 target |
| **Target** | `Depression` (Yes = Depresi, No = Tidak Depresi) |

### Deskripsi Kolom

| Kolom | Tipe | Deskripsi |
|-------|------|-----------|
| `Gender` | Kategorikal | Jenis kelamin (Male / Female) |
| `Age` | Numerik | Usia mahasiswa |
| `Academic Pressure` | Numerik | Tingkat tekanan akademik (skala 1–5) |
| `Study Satisfaction` | Numerik | Tingkat kepuasan belajar (skala 1–5) |
| `Sleep Duration` | Kategorikal | Durasi tidur per hari |
| `Dietary Habits` | Kategorikal | Kebiasaan makan (Healthy / Moderate / Unhealthy) |
| `Have you ever had suicidal thoughts ?` | Kategorikal | Riwayat pikiran bunuh diri (Yes / No) |
| `Study Hours` | Numerik | Jam belajar per hari |
| `Financial Stress` | Numerik | Tingkat stres finansial (skala 1–5) |
| `Family History of Mental Illness` | Kategorikal | Riwayat keluarga dengan penyakit mental (Yes / No) |
| `Depression` ⭐ | Kategorikal | **TARGET** — Apakah mahasiswa mengalami depresi |

---

## ⚙️ Metodologi

```
Dataset (502 data)
        ↓
Exploratory Data Analysis (EDA)
  ├── Distribusi target
  ├── Analisis fitur kategorikal vs depresi
  └── Analisis fitur numerik vs depresi
        ↓
Preprocessing
  ├── Cek & handle missing values
  ├── Hapus data duplikat
  ├── Label Encoding (kategorikal → numerik)
  └── StandardScaler (normalisasi)
        ↓
Split Data
  ├── Training set : 80% (401 data)
  └── Testing set  : 20% (101 data)
        ↓
Training — Random Forest Classifier
  ├── n_estimators = 100 pohon
  └── max_depth    = 10
        ↓
Evaluasi
  ├── Accuracy Score
  ├── Classification Report (Precision, Recall, F1-Score)
  ├── Confusion Matrix
  ├── ROC-AUC Score & Curve
  └── 5-Fold Cross Validation
        ↓
Demo Prediksi Input Manual
```

---

## 📈 Hasil Evaluasi Model

| Metrik | Nilai |
|--------|-------|
| **Accuracy** | (lihat output notebook) |
| **ROC-AUC Score** | (lihat output notebook) |
| **Cross-Validation Score** | (lihat output notebook) |

### Confusion Matrix & ROC Curve
![Confusion Matrix & ROC](hasil_evaluasi/confusion_matrix_roc.png)

### Feature Importance
![Feature Importance](hasil_evaluasi/feature_importance.png)

### Distribusi Kelas
![Distribusi](hasil_evaluasi/distribusi_kelas.png)

### Heatmap Korelasi
![Heatmap](hasil_evaluasi/heatmap_korelasi.png)

---

## 🚀 Cara Menjalankan

### Menggunakan Google Colab (Direkomendasikan)

1. Buka [Google Colab](https://colab.research.google.com/)
2. Klik **File → Upload notebook** → pilih `prediksi_depresi_mahasiswa.ipynb`
3. Upload dataset: klik ikon folder di sidebar kiri → Upload → pilih `Depression Student Dataset.csv`
4. Jalankan semua cell: **Runtime → Run all** (atau `Ctrl+F9`)

### Menggunakan Jupyter Notebook (Lokal)

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook prediksi_depresi_mahasiswa.ipynb
```

---

## 🎮 Demo Prediksi Manual

Pada Cell 7 di notebook, kamu bisa mengubah nilai-nilai berikut untuk memprediksi mahasiswa baru:

```python
sample = {
    'Gender'                                : 1,   # 0=Female, 1=Male
    'Age'                                   : 21,
    'Academic Pressure'                     : 4,   # skala 1-5
    'Study Satisfaction'                    : 2,   # skala 1-5
    'Sleep Duration'                        : 2,   # 0=5-6h, 1=7-8h, 2=<5h, 3=>8h
    'Dietary Habits'                        : 2,   # 0=Healthy, 1=Moderate, 2=Unhealthy
    'Have you ever had suicidal thoughts ?' : 1,   # 0=No, 1=Yes
    'Study Hours'                           : 9,
    'Financial Stress'                      : 4,   # skala 1-5
    'Family History of Mental Illness'      : 1,   # 0=No, 1=Yes
}
```

Output yang dihasilkan:
```
==============================================
         🎯 HASIL PREDIKSI DEMO
==============================================
   Status Prediksi           : 😢 DEPRESI (Yes)
   Probabilitas Tidak Depresi: 23.00%
   Probabilitas Depresi      : 77.00%
==============================================
```

---

## 🛠️ Teknologi yang Digunakan

| Library | Kegunaan |
|---------|----------|
| Python 3.10 | Bahasa pemrograman |
| Pandas | Manipulasi & analisis data |
| NumPy | Komputasi numerik |
| Scikit-learn | Model ML, preprocessing & evaluasi |
| Matplotlib | Visualisasi data |
| Seaborn | Visualisasi statistik |

---

## 🎥 Video Demo

📹 Link video demo (3–5 menit): **[Tambahkan link YouTube/Drive di sini]**

Isi video:
1. Penjelasan singkat proyek dan dataset
2. Menjalankan notebook di Google Colab
3. Penjelasan hasil evaluasi model
4. Demo prediksi input manual

---

## 📚 Referensi

- [Kaggle — Student Depression Dataset](https://www.kaggle.com/datasets/ikynahidwin/depression-student-dataset)
- [Scikit-learn — Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [WHO — Mental Health](https://www.who.int/health-topics/mental-health)

---

## 👤 Informasi Pengumpulan

| | |
|--|--|
| **Nama** | Miranda Ayudia Putri |
| **NIM** | 235060301111044|
| **Kelas** | C |
| **Departemen** | Teknik Elektro — Universitas Brawijaya |
| **Mata Kuliah** | Kecerdasan Buatan |
| **Dosen** | Dr. Raden Arief Setyawan, ST., MT |
