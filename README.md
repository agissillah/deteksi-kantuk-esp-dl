# Deteksi Kantuk Operator Alat Berat — CNN pada ESP32 dengan ESP-DL

Sistem peringatan kantuk yang mengklasifikasikan kondisi mata (terbuka / tertutup) menggunakan
**Convolutional Neural Network (CNN)** buatan sendiri. Inferensi model dijalankan **di ESP32**
menggunakan framework **ESP-DL** dari Espressif. Jika mata tertutup terus-menerus ≥ 1,5 detik,
buzzer berbunyi dan LCD menampilkan peringatan.

## Alur sistem

```
Webcam laptop ─► OpenCV + YuNet ─► potong area mata 32×32 ─► (USB serial) ─►
ESP32: model CNN INT8 (ESP-DL) ─► OPEN / CLOSED ─► hitung durasi ─► buzzer + LCD
```

| Bagian | Teknologi | Berjalan di |
|---|---|---|
| Ambil gambar, deteksi wajah, potong mata | OpenCV, YuNet | laptop |
| Training model CNN | PyTorch | laptop |
| Kuantisasi INT8 → `.espdl` | esp-ppq | laptop |
| **Klasifikasi OPEN / CLOSED** | **ESP-DL** | **ESP32** |
| Durasi, buzzer, LCD | ESP-IDF | ESP32 |

## Hasil

| | |
|---|---|
| Dataset | 1.000 citra mata (500 OPEN, 500 CLOSED), dikumpulkan sendiri |
| Model | CNN 3 lapis konvolusi, 6.010 parameter, 13 KB setelah INT8 |
| Akurasi data uji (FLOAT32) | 96,00% |
| Akurasi setelah kuantisasi INT8 | 95,33% |
| Akurasi uji langsung di ESP32 | 93,8% |
| Waktu inferensi di ESP32 | 16 ms per mata |

## Perangkat keras

| Komponen | Pin ESP32 |
|---|---|
| Buzzer pasif (+) | GPIO 25 |
| Buzzer pasif (−) | GND |
| LCD 16×2 I2C — SDA | GPIO 21 |
| LCD 16×2 I2C — SCL | GPIO 22 |
| LCD 16×2 I2C — VCC / GND | VIN (5V) / GND |

Board: ESP32-D0WD-V3 (ESP32 klasik, tanpa SIMD). Buzzer di-drive PWM 4000 Hz.

## Struktur folder

```
collect_dataset.py      ambil dataset dari webcam (tekan O / C)
bersihkan_dataset.py    pindah/hapus gambar yang salah label
cek_dataset.py          periksa jumlah & contoh dataset
cari_salah_label.py     cari label salah otomatis (cross-validation)
model.py                arsitektur CNN (EyeNet)
dataset_loader.py       pembaca dataset + augmentasi
train.py                training → eyenet.pth
ekspor_onnx.py          eyenet.pth → eyenet.onnx + data kalibrasi
siapkan_data_uji.py     simpan data uji untuk cek kuantisasi
kuantisasi.py           eyenet.onnx → eyenet.espdl (INT8), jalankan di env_quant
uji_webcam.py           uji model di laptop lewat webcam
uji_esp32.py            uji akurasi & kecepatan model di ESP32
deteksi_kantuk.py       PROGRAM UTAMA: webcam → ESP32 → buzzer & LCD
dataset/OPEN, CLOSED    dataset citra mata 64×64 grayscale
models/                 detektor wajah YuNet (ONNX)
firmware/               project ESP-IDF untuk ESP32
patches/                patch wajib untuk ESP-DL (lihat catatan)
```

## Menjalankan sistem

```powershell
pip install opencv-python numpy torch onnx onnxruntime pyserial matplotlib
python deteksi_kantuk.py          # default COM6, atau: python deteksi_kantuk.py COM7
```

## Melatih ulang dari awal

```powershell
python train.py
python ekspor_onnx.py
python siapkan_data_uji.py
.\env_quant\Scripts\python.exe kuantisasi.py
copy eyenet.espdl firmware\main\models\eyenet.espdl
```

Lalu build dan flash firmware (dari terminal ESP-IDF v5.3):

```powershell
cd firmware
idf.py set-target esp32
idf.py build
idf.py -p COM6 flash      # tahan tombol BOOT saat proses flash dimulai
```

## Catatan penting

1. **esp-ppq butuh Python 3.8–3.12.** Buat environment terpisah:
   `python3.11 -m venv env_quant` lalu `env_quant\Scripts\pip install "esp_ppq[cpu]"`.
2. **Patch ESP-DL wajib untuk ESP32 klasik.** Tanpa patch, firmware crash
   (`LoadStoreError`) karena buffer model dialokasikan di IRAM. Terapkan dari folder repo ESP-DL:
   `git apply <path>/patches/esp-dl-malloc-8bit.patch`
3. **Lokasi folder.** `firmware/main/idf_component.yml` mengambil ESP-DL dari `../../../esp-dl`,
   jadi project ini diletakkan di dalam clone ESP-DL (`esp-dl/drowsiness/`). Kalau dipindah,
   sesuaikan `override_path` tersebut.
4. Ketik huruf `T` lewat serial untuk menguji nada buzzer (2500–5000 Hz).
