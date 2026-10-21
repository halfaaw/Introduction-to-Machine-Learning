# Studi Kasus Sesi 4: Pengantar Machine Learning & Prediksi Klasifikasi (CRISP-DM & Supervised Learning)

## 1. Latar Belakang
PT Maju Bersama (perusahaan fiktif) ingin memprediksi status kelayakan persetujuan pinjaman (loan approval) atau klasifikasi pelanggan berdasarkan riwayat kredit dan fitur-fitur transaksi. Mengingat volume data operasional yang terus bertambah, perusahaan memerlukan alur analitik prediktif berbasis data (*data-driven thinking*) untuk menggantikan aturan manual (*rule-based thinking*). 

Keterkaitan dengan seminar: Proyek ini terinspirasi dari materi *"Introduction to Machine Learning"* oleh Dr. Taniza binti Tajuddin (UiTM Malaysia). Seminar tersebut menekankan pergeseran dari *rule-based* ke *data-driven*, proses *data preprocessing* (penanganan *missing values*, normalisasi, dan *train-test split* 80:20), pemilihan algoritma seperti *Supervised Learning* (Classification & Regression), serta pemahaman batasan AI (bahwa model bergantung penuh pada kualitas data input: *"garbage in, garbage out"*).

## 2. Tujuan
- Memvalidasi dan membersihkan data historis (*data cleaning* dan *missing value handling*) sebelum masuk ke pemodelan.
- Membagi dataset menjadi data latih (*training set*) dan data uji (*testing set*) dengan proporsi standar (misal: 80:20).
- Melatih model Machine Learning (menggunakan algoritma klasifikasi seperti *Decision Tree* atau *Logistic Regression*).
- Mengevaluasi performa model menggunakan metrik akurasi dan *Confusion Matrix*.
- Menuliskan batasan analisis bahwa model Machine Learning belajar dari data historis dan membutuhkan pengawasan manusia (*human oversight*), bukan sistem mutlak yang bebas dari bias.

## 3. Data
File `data/loan_data.csv` berisi data sintetis untuk keperluan simulasi pelatihan, bukan data transaksi finansial perusahaan sungguhan. Setiap baris mewakili satu catatan pengajuan pinjaman pelanggan.

| Kolom | Arti |
|---|---|
| `loan_id` | ID unik untuk setiap pengajuan pinjaman |
| `income` | Pendapatan tahunan pemohon |
| `credit_history` | Riwayat skor kredit (1 = Baik, 0 = Buruk) |
| `employment_status` | Status pekerjaan (1 = Tetap/Formal, 0 = Lainnya) |
| `loan_status` | Status persetujuan aktual (Approved / Rejected) sebagai target label |

Konsep yang digunakan:
- *Data Preprocessing*: Menghilangkan kolom ID (`loan_id`) untuk menghindari *overfitting* atau kebocoran data (*data leakage*).
- *Train-Test Split*: Memisahkan data untuk generalisasi model dan validasi independen.
- *Evaluation Metrics*: Mengukur akurasi prediksi dengan rumus $(	ext{Prediksi Benar} / 	ext{Total Data}) 	imes 100\%$.

## 4. Struktur Proyek
studi-kasus-sesi-4-machine-learning/
├── data/
│   └── loan_data.csv
├── src/
│   └── ml_prediction.py
├── tests/
│   └── test_ml.py
├── outputs/
│   └── metrics_report.txt  # Laporan evaluasi akurasi dan confusion matrix
├── README.md
└── requirements.txt
