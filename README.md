# 🌟 Proyek Klasifikasi Gambar: Intel Natural Scene Classification
### 🎓 Submission Proyek Akhir Belajar Pengembangan Machine Learning - Dicoding Indonesia

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mhmmdragilpy/image-clasification/blob/main/notebook.ipynb)
[![Rating](https://img.shields.io/badge/Target_Rating-5_Stars_⭐⭐⭐⭐⭐-brightgreen.svg)](#)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](#)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](#)

---

## 👨‍💻 Profil Pengembang
- **Nama Lengkap:** Muhammad Ragil
- **Email:** mhmmdragilpy
- **ID Dicoding:** mhmmdragilpy
- **Tautan Langsung Google Colab:** [Buka Notebook di Colab](https://colab.research.google.com/github/mhmmdragilpy/image-clasification/blob/main/notebook.ipynb)

---

## 🎯 Ringkasan Proyek
Repositori ini berisi implementasi *end-to-end* sistem klasifikasi citra pemandangan alam berskala besar (**Intel Image Dataset**) menggunakan arsitektur **Convolutional Neural Network (CNN)** dengan target akurasi $\ge 95\%$ dan konversi model ke 3 format multi-platform (*SavedModel*, *TF-Lite*, *TensorFlow.js*).

### 📊 Karakteristik Dataset:
- **Total Citra Asli:** 24.335 citra (17.034 citra berlabel + 7.301 citra prediksi)
- **Jumlah Kelas:** 6 Kategori (`buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`)
- **Variasi Resolusi Asli:** 37 variasi dimensi citra unik (terbukti multi-resolution / tanpa preprocessing awal)
- **Partisi Data:** 80% Train, 10% Validation, 10% Test (Stratified Split)

---

## 🏆 Checklist Kriteria Bintang 5 (⭐⭐⭐⭐⭐):
- [x] **Kriteria Wajib 1:** Dataset bebas & memiliki minimal 1.000 gambar (**17.034 gambar berlabel**)
- [x] **Kriteria Wajib 2:** Tidak menggunakan dataset terlarang RPS/X-Ray
- [x] **Kriteria Wajib 3:** Dataset dibagi menjadi Train, Validation, dan Test set
- [x] **Kriteria Wajib 4:** Model Sequential dengan Conv2D dan Pooling Layer
- [x] **Kriteria Wajib 5:** Akurasi pada train & test minimal 85% (**Target $\ge 95\%$**)
- [x] **Kriteria Wajib 6:** Visualisasi plot akurasi dan loss (Training vs Validation)
- [x] **Kriteria Wajib 7:** Menyimpan model dalam format **SavedModel**, **TF-Lite**, dan **TFJS**
- [x] **Saran 1:** Implementasi Callback (`EarlyStopping`, `ReduceLROnPlateau`, `TargetAccuracyCallback`)
- [x] **Saran 2:** Gambar pada dataset asli memiliki resolusi tidak seragam (37 variasi dimensi)
- [x] **Saran 3:** Dataset yang digunakan berisi minimal 10.000 gambar (**17.034 gambar**)
- [x] **Saran 4:** Akurasi pada training set dan testing set minimal 95%
- [x] **Saran 5:** Memiliki 3 buah kelas atau lebih (**6 Kelas**)
- [x] **Saran 6:** Melakukan inference menggunakan model TF-Lite + bukti visual output

---

## 📁 Struktur Direktori
```
image-clasification/
├── .gitignore                   # Konfigurasi pengabaian dataset lokal
├── README.md                    # Dokumentasi utama proyek
├── Instruksi_Submission.md      # Master Blueprint & Panduan Lengkap Bintang 5
├── LEARNING_GUIDE.md            # Panduan Belajar, Arsitektur Mermaid & Q&A Sidang
├── notebook.ipynb               # Master Jupyter Notebook (Colab Ready)
└── submission/                  # Folder artefak siap submit
    ├── saved_model/             # SavedModel TensorFlow format
    ├── tflite/                  # model.tflite & label.txt
    ├── tfjs_model/              # model.json & binary shards
    ├── requirements.txt         # Dependensi terisolasi (pipreqs)
    └── README.md
```

---

## 🚀 Cara Menjalankan (1-Click Google Colab)
1. Klik badge **[Open In Colab](https://colab.research.google.com/github/mhmmdragilpy/image-clasification/blob/main/notebook.ipynb)** di bagian atas.
2. Pastikan Runtime menggunakan **T4 GPU** (`Runtime > Change runtime type > T4 GPU`).
3. Tekan `Ctrl + F9` (**Run all**). Seluruh proses dari unduh dataset hingga pembuatan `submission.zip` akan selesai secara otomatis.
