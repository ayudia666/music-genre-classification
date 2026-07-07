# 🎵 Music Genre Classification

**Identifikasi Genre Musik Berdasarkan Karakteristik Spektral Audio Menggunakan GTZAN Music Genre Dataset**

Aplikasi web berbasis **Streamlit** yang mengklasifikasikan genre musik dari sebuah file audio menggunakan model **Support Vector Machine (SVM)** yang dilatih pada fitur-fitur spektral audio hasil ekstraksi manual dengan **Librosa**.

> Final Project — Mata Kuliah Audio Processing

---

## 📖 Deskripsi Project

Project ini mengimplementasikan pipeline end-to-end untuk klasifikasi genre musik:

1. Ekstraksi fitur spektral dari audio menggunakan Librosa (bukan file `features_3_sec.csv`/`features_30_sec.csv` bawaan GTZAN).
2. Segmentasi audio menjadi potongan berdurasi 3 detik untuk memperkaya jumlah data latih.
3. Feature selection menggunakan `SelectKBest` (ANOVA F-test) untuk memilih fitur paling relevan.
4. Klasifikasi menggunakan **SVM kernel RBF**, dituning dengan `GridSearchCV`.
5. Prediksi akhir sebuah lagu ditentukan melalui **majority voting** atas prediksi seluruh segmennya.
6. Seluruh pipeline dibungkus dalam antarmuka web modern menggunakan **Streamlit**.

Model saat ini mencapai **akurasi ±91.29%** pada data uji.

---

## 🎼 Dataset

- **[GTZAN Music Genre Dataset](http://marsyas.info/downloads/datasets.html)**
- 10 genre musik: *Blues, Classical, Country, Disco, Hiphop, Jazz, Metal, Pop, Reggae, Rock*
- Setiap file audio berformat `.wav`, berdurasi ±30 detik
- Fitur diekstrak langsung dari sinyal audio, **tidak menggunakan** file CSV fitur bawaan dataset

---

## 🧠 Feature Extraction

Setiap segmen audio (3 detik) diekstrak menjadi fitur berikut, masing-masing diringkas menjadi **mean** dan **standard deviation**:

| Fitur                     | Deskripsi Singkat                                              |
|----------------------------|------------------------------------------------------------------|
| MFCC (20 koefisien)         | Bentuk spectral envelope, menangkap karakteristik timbre         |
| Delta MFCC                 | Turunan orde-1 MFCC, menangkap dinamika perubahan spektral       |
| Delta-Delta MFCC            | Turunan orde-2 MFCC, menangkap percepatan perubahan spektral      |
| Spectral Centroid           | "Pusat massa" spektrum frekuensi                                 |
| Spectral Bandwidth           | Lebar rentang frekuensi di sekitar centroid                      |
| Spectral Roll-off           | Frekuensi batas 85% energi spektral                              |
| Spectral Contrast           | Perbedaan energi puncak vs. lembah tiap sub-band frekuensi        |
| Zero Crossing Rate (ZCR)    | Frekuensi perubahan tanda sinyal                                  |
| Chroma STFT                 | Distribusi energi pada 12 kelas pitch                             |
| RMS Energy                  | Energi/loudness rata-rata sinyal                                  |
| Tempo                       | Estimasi kecepatan ketukan (beat) sinyal                          |

---

## 🤖 Model

- **Algoritma:** Support Vector Machine (kernel RBF)
- **Hyperparameter tuning:** `GridSearchCV` (parameter `C` dan `gamma`)
- **Validasi:** `StratifiedKFold` 5-fold cross-validation, `scoring='f1_macro'`
- **Feature Selection:** `SelectKBest` dengan ANOVA F-test
- **Preprocessing:** `LabelEncoder` (label genre) + `StandardScaler` (standardisasi fitur)
- **Akurasi pada data uji:** ±91.29%

Seluruh artefak model (`svm_model.pkl`, `scaler.pkl`, `label_encoder.pkl`, `selected_features.pkl`) dihasilkan dari notebook training terpisah dan **tidak dilatih ulang** oleh aplikasi ini — aplikasi hanya melakukan *inference*.

---

## 🔄 Pipeline Inference

```
Upload Audio
     ↓
Segmentasi Audio (3 detik per segmen)
     ↓
Feature Extraction (MFCC, Delta, Delta², Spectral, Chroma, RMS, Tempo)
     ↓
Feature Selection (SelectKBest — transform saja)
     ↓
Standardization (StandardScaler — transform saja)
     ↓
SVM Prediction (per segmen)
     ↓
Majority Voting (seluruh segmen)
     ↓
Genre Akhir
```

---

## 📂 Struktur Folder

```
music_genre_classification/
│
├── app.py                      # Layout Streamlit & orkestrasi UI (tanpa logika ML)
│
├── models/                     # Artefak model hasil training (letakkan file .pkl Anda di sini)
│   ├── svm_model.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   └── selected_features.pkl
│
├── utils/                      # Seluruh logika machine learning (modular)
│   ├── __init__.py
│   ├── model_loader.py         # load_model()
│   ├── feature_extraction.py   # segment_audio(), extract_features()
│   ├── preprocessing.py        # preprocess_features()
│   ├── prediction.py           # predict_segments(), majority_voting(), predict_audio()
│   └── visualization.py        # plot_waveform(), plot_melspectrogram(), plot_vote_bar_chart()
│
├── assets/
│   ├── style.css                # Custom CSS untuk tampilan modern
│   ├── logo.png                 # (opsional) logo aplikasi — tambahkan sendiri
│   └── banner.png                # (opsional) banner aplikasi — tambahkan sendiri
│
├── requirements.txt
└── README.md
```

> **Catatan:** Folder `assets/` belum menyertakan `logo.png` dan `banner.png` karena bersifat opsional/branding. Aplikasi akan tetap berjalan normal tanpa file tersebut (menampilkan ikon 🎵 sebagai fallback) — tambahkan file gambar Anda sendiri di folder ini jika ingin menampilkan logo kustom.

---

## ⚙️ Cara Instalasi

1. **Clone / salin folder project ini**, lalu masuk ke direktorinya:
   ```bash
   cd music_genre_classification
   ```

2. **Buat virtual environment** (disarankan):
   ```bash
   python -m venv venv
   source venv/bin/activate      # Linux/Mac
   venv\Scripts\activate         # Windows
   ```

3. **Install seluruh dependency:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Letakkan file model** hasil training Anda ke dalam folder `models/`:
   ```
   models/svm_model.pkl
   models/scaler.pkl
   models/label_encoder.pkl
   models/selected_features.pkl
   ```

5. **(Untuk dukungan file MP3)** pastikan `ffmpeg` sudah terinstal di sistem Anda, karena Librosa/audioread membutuhkannya untuk decoding MP3:
   ```bash
   # Ubuntu/Debian
   sudo apt install ffmpeg

   # MacOS (Homebrew)
   brew install ffmpeg
   ```

---

## ▶️ Cara Menjalankan

```bash
streamlit run app.py
```

Aplikasi akan terbuka otomatis di browser pada alamat `http://localhost:8501`.

---

## 🖼️ Screenshot UI

> Tambahkan screenshot aplikasi Anda di sini setelah dijalankan, misalnya:
>
> `assets/screenshot_hero.png` — Tampilan Hero Section
> `assets/screenshot_prediction.png` — Tampilan Hasil Prediksi
>
> ```markdown
> ![Hero Section](assets/screenshot_hero.png)
> ![Prediction Result](assets/screenshot_prediction.png)
> ```

---

## 🛠️ Kualitas Kode

- **Docstring** pada setiap fungsi (format NumPy-style)
- **Type hints** pada seluruh parameter dan return value
- **Modular programming** — logika ML dipisah total dari layout UI
- **Reusable functions** — setiap fungsi di `utils/` dapat dipanggil ulang tanpa modifikasi (misalnya untuk batch prediction atau integrasi ke API/Flask)
- **Clean code** — penamaan variabel dan fungsi deskriptif, komentar pada tiap tahapan penting

---

## 📌 Catatan Penting

- Aplikasi ini **hanya melakukan inference**, tidak pernah melakukan training ulang model.
- Seluruh preprocessing (segmentasi, ekstraksi fitur, feature selection, scaling) **identik** dengan pipeline yang digunakan saat training — objek `selector` dan `scaler` hanya dipanggil dengan `transform()`, tidak pernah `fit()`.
- Jika muncul pesan error saat memuat model, pastikan keempat file `.pkl` sudah diletakkan dengan benar di folder `models/`.

---

## 📄 Lisensi

Project ini dibuat untuk keperluan akademik (Final Project mata kuliah PPDM) dan bebas digunakan/dikembangkan lebih lanjut untuk keperluan pembelajaran maupun portofolio.