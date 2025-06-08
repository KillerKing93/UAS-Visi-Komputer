# **Sistem Deteksi Alat Pelindung Diri (APD) di Area Konstruksi menggunakan YOLOv11**

Sistem ini adalah sebuah solusi berbasis Visi Komputer yang dirancang untuk mendeteksi penggunaan Alat Pelindung Diri (APD) seperti helm (_hardhat_) dan rompi keselamatan (_safety vest_) pada pekerja di lingkungan konstruksi secara _real-time_ dari rekaman video. Proyek ini bertujuan untuk meningkatkan kepatuhan terhadap protokol keselamatan kerja, mengurangi risiko kecelakaan, dan menyediakan analisis data untuk audit keselamatan.

Model deteksi objek yang digunakan adalah **YOLOv11**, yang telah dilatih khusus (_fine-tuned_) pada dataset publik yang relevan. Sistem ini diimplementasikan dalam bentuk dasbor web interaktif yang dibangun menggunakan **Gradio**, memungkinkan analisis video dari berbagai sumber seperti YouTube dan Google Drive.

## **Tim Pengembang**

| Nama                | NPM       |
| :------------------ | :-------- |
| Ayu Anggraini       | G1A022007 |
| Anissa Shanniyah A. | G1A022044 |
| Alif Nurhidayat     | G1A022073 |

## **Tautan & Demo Proyek**

- **Repositori GitHub**: [KillerKing93/UAS-Visi-Komputer](https://github.com/KillerKing93/UAS-Visi-Komputer)
- **Repositori Aplikasi (Hugging Face)**: [Lihat Kode Aplikasi yang Di-deploy](https://huggingface.co/spaces/KillerKing93/UAS-VisiKomputer-ConstructionWorker)
- **Demo Aplikasi (Hugging Face)**: [Coba Demo Langsung](https://killerKing93-UAS-VisiKomputer-ConstructionWorker.hf.space)
- **Video Demo (YouTube)**: [Tonton Video Demo](https://youtu.be/4KZygNTIRJw)
- **Presentasi Proyek (Canva)**: [Lihat Presentasi](https://www.canva.com/design/DAGmdjBp-g8/e-WFxyfP_Z15-a0fOmzktg/edit?utm_content=DAGmdjBp-g8&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)
- **Eksport Bobot Model dan Metadata:** [Yolo V11 Exports](https://drive.google.com/drive/folders/1ZgIKv7K4RTLR4VsM2PULomU4B46b70jt)
- **Bobot Model & Gambar Evaluasi:**: [Google Drive](https://drive.google.com/drive/folders/17R3B7BB8y0sVvjztTU5xI17a-m2GFBtN)

## **1\. Pemilihan Metode dan Justifikasi Teknis (Kesesuaian 20%)**

### **Metode Deteksi Objek: YOLOv11**

Untuk tugas deteksi objek secara _real-time_, **YOLOv11 (You Only Look Once versi 11\)** dipilih sebagai model utama.

**Justifikasi Pemilihan:**

1. **Kecepatan dan Efisiensi**: Arsitektur YOLO dikenal memiliki kecepatan inferensi yang sangat tinggi, menjadikannya pilihan ideal untuk analisis video _real-time_. Arsitekturnya yang _single-shot_ memproses seluruh gambar sekali jalan untuk efisiensi maksimal.
2. **Akurasi Tinggi**: Meskipun cepat, YOLOv11 tetap menawarkan akurasi yang sangat kompetitif (diukur dengan mAP) dan mampu mendeteksi objek dengan berbagai ukuran, termasuk objek kecil seperti helm di kejauhan.
3. **Kemudahan _Fine-Tuning_**: _Framework_ Ultralytics menyediakan alur kerja yang sangat sederhana untuk melakukan _transfer learning_ atau _fine-tuning_ pada dataset kustom, yang secara signifikan mempercepat siklus pengembangan.
4. **Model _Pre-trained_**: Kami menggunakan model yolo11n.pt yang telah dilatih pada dataset COCO, yang mencakup kelas 'Person'. Ini memberikan fondasi yang kuat untuk model kami dalam mengenali pekerja sebelum mendeteksi APD mereka.

## **2\. Inovasi dan Kelengkapan Rancangan Sistem (Rancangan 25%)**

### **Strategi Pengumpulan dan Anotasi Data**

- **Sumber Data**: Kami menggunakan dataset publik **"Construction Site Safety"** dari **Roboflow Universe**. Dataset ini berisi 7.639 gambar dengan 39.267 anotasi yang mencakup 10 kelas relevan, termasuk Hardhat, Safety Vest, Person, dan kelas negatif seperti NO-Hardhat.
- **Manajemen Data**: Dataset diunduh dan dikelola langsung melalui API Roboflow di dalam lingkungan Google Colab, memastikan proses yang _reproducible_ dan efisien.
- **Labeling**: Data yang digunakan telah dianotasi dengan format YOLO, yaitu _bounding box_ dengan koordinat \[x_center, y_center, width, height\] yang dinormalisasi.

### **Arsitektur dan Alur Kerja Sistem**

Sistem ini dirancang dengan dua komponen utama: **Notebook Pelatihan Model** dan **Aplikasi Dasbor Gradio**.

1. **Pelatihan (Notebook UAS_Visi_Komputer.ipynb)**:
   - **Setup Lingkungan**: Kloning repositori Ultralytics dan instalasi dependensi.
   - **Akuisisi Data**: Mengunduh dataset dari Roboflow.
   - **_Fine-Tuning_**: Melatih model yolo11n.pt pada dataset kustom selama 100 epoch.
   - **Validasi**: Model divalidasi secara berkala pada _validation set_ untuk memonitor performa.
   - **Ekspor Artefak**: Menyimpan bobot model terbaik (best.pt) dan semua grafik hasil pelatihan (misalnya, _confusion matrix_) ke Google Drive.
2. **Inferensi (Aplikasi app.py)**:
   - **Antarmuka Gradio**: Menyediakan UI berbasis web yang ramah pengguna.
   - **Input**: Pengguna memasukkan URL video dari YouTube atau Google Drive.
   - **Backend**: Video diunduh, kemudian setiap _frame_ diproses oleh model YOLOv11 yang telah dimuat.
   - **Output**: Menghasilkan video baru dengan _bounding box_ dan label deteksi, beserta ringkasan analisis (jumlah objek, kepatuhan APD, dll.).
   - **Fitur Tambahan**: Sistem login dengan manajemen pengguna (admin/operator) dan panel untuk memperbarui model secara dinamis.

## **3\. Kualitas Alur Kerja dan Logika Sistem (Kualitas 20%)**

Logika sistem pada app.py dirancang untuk menjadi modular dan tangguh:

- **Manajemen File**: Setelah model telah dilatih dan diupload ke dashboard, konfigurasi model dan data pengguna disimpan dalam file JSON terpisah (config/), memungkinkan pembaruan tanpa mengubah kode utama.
- **Fungsi Inti**: Proses pengunduhan video, pemrosesan _frame-by-frame_, dan pembuatan visualisasi dipisahkan menjadi fungsi-fungsi yang jelas.
- **Error Handling**: Sistem menangani URL yang tidak valid dan kegagalan saat mengunduh video atau memuat model, dengan memberikan umpan balik yang jelas kepada pengguna.
- **Real-time Simulation**: Meskipun pemrosesan dilakukan pasca-unggah, alur kerja _frame-by-frame_ mensimulasikan bagaimana sistem akan bekerja pada _stream_ video _real-time_.

## **4\. Realisme Data dan Metode Evaluasi (Evaluasi 15%)**

### **Realisme Data**

Dataset dari Roboflow dipilih karena merepresentasikan kondisi nyata di lapangan, dengan berbagai skenario pencahayaan, sudut pandang kamera, dan oklusi (objek yang terhalang sebagian).

### **Metrik Evaluasi Performa**

Performa model dievaluasi menggunakan metrik standar dalam deteksi objek: [Google Drive](https://drive.google.com/drive/folders/17R3B7BB8y0sVvjztTU5xI17a-m2GFBtN)

- **Precision & Recall**: Mengukur seberapa akurat prediksi model dan seberapa baik model menemukan semua objek yang relevan.
- **mAP (mean Average Precision)**: Metrik utama untuk mengevaluasi performa model deteksi objek. Kami mencapai:
  - **mAP50-95**: **0.469** (rata-rata AP pada IoU dari 0.5 hingga 0.95).
  - **mAP50**: **0.769** (AP pada IoU 0.5), menunjukkan performa yang sangat baik dalam deteksi umum.
- **Confusion Matrix**: Digunakan untuk menganalisis kesalahan klasifikasi antar kelas (misalnya, apakah model sering keliru antara Hardhat dan NO-Hardhat).

## **5\. Kreativitas dan Pengembangan Lanjutan (Pengembangan 20%)**

### **Inovasi dan Kreativitas Saat Ini**

- **Dasbor Interaktif**: Penggunaan Gradio menciptakan antarmuka yang memungkinkan pengguna non-teknis untuk berinteraksi langsung dengan model AI.
- **Manajemen Sistem**: Fitur login dan manajemen model menunjukkan pemikiran ke arah produk yang siap digunakan dalam skala kecil.
- **Analisis Kuantitatif**: Sistem tidak hanya mendeteksi, tetapi juga menghitung dan menyajikan ringkasan kepatuhan, memberikan nilai tambah untuk audit keselamatan.

### **Saran Pengembangan Lanjutan**

- **Integrasi CCTV Real-time**: Mengadaptasi sistem untuk memproses _stream_ video langsung dari kamera IP menggunakan protokol seperti RTSP.
- **Sistem Notifikasi**: Mengintegrasikan peringatan otomatis (misalnya, melalui email, Telegram, atau SMS) ketika terdeteksi pelanggaran keselamatan (misalnya, pekerja tanpa helm).
- **Optimasi Performa**: Menggunakan teknik seperti _quantization_ (misalnya dengan TensorRT) untuk mengoptimalkan model agar dapat berjalan lebih cepat pada perangkat _edge computing_ seperti NVIDIA Jetson.
- **Analisis Dasbor yang Lebih Mendalam**: Membangun dasbor analitik (misalnya dengan Power BI atau Streamlit) untuk melacak tren kepatuhan dari waktu ke waktu di berbagai lokasi proyek.
