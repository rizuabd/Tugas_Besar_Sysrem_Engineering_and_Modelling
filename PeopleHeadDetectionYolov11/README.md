# Pemodelan dan Simulasi Sistem Penghitung Penumpang Otomatis Pada Bus Rapid Transit (BRT) Berbasis YOLOv11 Sebagai Cyber-Physical System

Transportasi umum seperti Bus Rapid Transit (BRT) memiliki peran penting dalam mendukung mobilitas masyarakat perkotaan. Namun, masalah kelebihan kapasitas penumpang masih sering terjadi, terutama pada jam sibuk yang berpotensi membahayakan keselamatan. Untuk mengatasi hal tersebut, diperlukan sistem penghitung penumpang yang akurat dan efisien. Penelitian ini mengusulkan perancangan sistem penghitung penumpang otomatis berbasis algoritma object detection YOLO (You Only Look Once), yang mampu bekerja secara real-time dan dapat diimplementasikan pada perangkat edge. Sistem dirancang sebagai bagian dari cyber-physical system dan dimodelkan menggunakan pendekatan discrete-event system untuk menggambarkan aliran masuk dan keluar penumpang pad BRT. Implementasi dilakukan dalam bentuk simulasi berbasis Python, dengan integrasi model YOLO untuk deteksi objek serta logika penghitungan berdasarkan pergerakan penumpang dalam Region of Interest. Hasil dari penelitian ini merupakan model YOLOv11 yang bisa melakukan deteksi dan sistem yang bisa menghitung keluar dan masuknya penumpang pada BRT, untuk evaluasi model YOLOv11 didapatkan mAP@0.5 sebesar 91,8%, Precision 91,4%, Recall 86,6%, dan F1-score 88,9%. Untuk penghitungan keluar dan masuk penumpang pada early real-world test didapatkan akurasi penghitungan sebesar 100% dan inference time pada interval 15-45ms untuk setiap skenario yang diujikan 


untuk melakukan inference dengan model yang sudah ada, jalankan file inference_engine_with_cuda.py (bisa dijalankan pada edge device/local)
jika ingin melakukan training, jalankan file Head_Detection_YoloV11_70_20_10.ipynb (disarankan untuk dijalankan pada google collab atau device dengan GPU high-performance)

link gdrive dataset dan model: https://drive.google.com/drive/folders/1kdYZ9s8q08ZBYkNf_uGVkgC0E9zLtyJn?usp=sharing

pada gdrive folder "merged_dataset" berisikan data image yang digunakan, folder "model" berisikan file .yaml dari yolov11 (otomatis akan memilih yolo11n), dan untuk folder "runs" berisikan file model yang sudah di-trained dengan extension .pt (karena menggunakan pytorch) dan juga terdapat file result dari hasil evaluasi (bukan real-world).


![image](https://imgur.com/a/hcaBhT9)
