"""
kuantisasi.py
-------------
Mengubah eyenet.onnx (FLOAT32) menjadi eyenet.espdl (INT8) memakai esp-ppq.

Kenapa perlu dikuantisasi?
  ESP32 tidak punya unit floating point yang cepat untuk perhitungan
  neural network. Bobot model diubah dari FLOAT32 (4 byte) menjadi
  INT8 (1 byte), sehingga model 4x lebih kecil dan jauh lebih cepat.

Kenapa perlu data kalibrasi?
  Untuk mengubah FLOAT32 ke INT8, esp-ppq perlu tahu rentang nilai yang
  wajar di tiap lapisan. Rentang itu diukur dengan menjalankan model
  memakai contoh gambar asli dari dataset (file kalibrasi.npy).

PENTING: jalankan dengan Python dari env_quant, BUKAN python biasa,
karena esp-ppq hanya mendukung Python 3.8 - 3.12:

    .\\env_quant\\Scripts\\python.exe kuantisasi.py
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from esp_ppq.api import espdl_quantize_onnx

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PATH_ONNX  = os.path.join(BASE_DIR, "eyenet.onnx")
PATH_ESPDL = os.path.join(BASE_DIR, "eyenet.espdl")
PATH_KALIB = os.path.join(BASE_DIR, "kalibrasi.npy")
PATH_UJI   = os.path.join(BASE_DIR, "data_uji.npz")

# ESP32-D0WD-V3 adalah ESP32 biasa (bukan S3/P4) yang tidak punya
# instruksi SIMD, jadi targetnya "c" = implementasi C umum.
TARGET      = "c"
QUANT_TYPE  = "w8a8"      # bobot 8 bit, aktivasi 8 bit
INPUT_SHAPE = [1, 1, 32, 32]
BATCH       = 32
DEVICE      = "cpu"


def collate_fn(batch):
    """DataLoader mengembalikan Tuple(x,), kuantisasi hanya butuh x."""
    return batch[0].to(DEVICE)


def main():
    for path in (PATH_ONNX, PATH_KALIB):
        if not os.path.exists(path):
            print("[ERROR] Tidak ditemukan:", path)
            print("        Jalankan dulu:  python ekspor_onnx.py")
            return

    data_kalibrasi = torch.from_numpy(np.load(PATH_KALIB))
    print("[INFO] Data kalibrasi :", tuple(data_kalibrasi.shape))
    print("[INFO] Target         :", TARGET, "(ESP32 biasa, tanpa SIMD)")
    print("[INFO] Skema          :", QUANT_TYPE)
    print("")

    # shuffle HARUS False, karena dataset dilewati berkali-kali saat
    # menghitung galat kuantisasi
    pemuat = DataLoader(TensorDataset(data_kalibrasi),
                        batch_size=BATCH, shuffle=False)

    graf_int8 = espdl_quantize_onnx(
        onnx_import_file=PATH_ONNX,
        espdl_export_file=PATH_ESPDL,
        calib_dataloader=pemuat,
        calib_steps=len(pemuat),
        input_shape=INPUT_SHAPE,
        inputs=None,
        target=TARGET,
        quant_type=QUANT_TYPE,
        collate_fn=collate_fn,
        device=DEVICE,
        error_report=True,       # tampilkan galat tiap lapisan
        skip_export=False,
        export_test_values=True, # simpan contoh masukan/keluaran untuk diuji di ESP32
        verbose=1,
    )

    print("")
    if os.path.exists(PATH_ESPDL):
        print("[OK] Model ESP-DL tersimpan :", PATH_ESPDL)
        print("     ukuran                 : %.1f KB"
              % (os.path.getsize(PATH_ESPDL) / 1024))
    else:
        print("[ERROR] File .espdl tidak terbentuk.")
        return

    bandingkan_akurasi(graf_int8)


def bandingkan_akurasi(graf_int8):
    """
    Membandingkan akurasi model FLOAT32 (ONNX) dengan model INT8 hasil
    kuantisasi, memakai data uji yang tidak pernah dipakai saat training.
    """
    if not os.path.exists(PATH_UJI):
        print("")
        print("[INFO] data_uji.npz tidak ada, perbandingan akurasi dilewati.")
        print("       Jalankan dulu:  python siapkan_data_uji.py")
        return

    import onnxruntime as ort
    from esp_ppq.executor.torch import TorchExecutor

    data = np.load(PATH_UJI)
    x_uji, y_uji = data["x"], data["y"]

    # model FLOAT32
    sesi = ort.InferenceSession(PATH_ONNX, providers=["CPUExecutionProvider"])
    tebak_fp32 = np.array([sesi.run(None, {"input": x_uji[i:i + 1]})[0].argmax()
                           for i in range(len(y_uji))])

    # model INT8 (perhitungannya disimulasikan persis seperti di ESP32)
    eksekutor = TorchExecutor(graph=graf_int8, device=DEVICE)
    tebak_int8 = np.array([
        int(eksekutor.forward(torch.from_numpy(x_uji[i:i + 1]))[0].argmax())
        for i in range(len(y_uji))])

    akurasi_fp32 = (tebak_fp32 == y_uji).mean()
    akurasi_int8 = (tebak_int8 == y_uji).mean()

    print("")
    print("=" * 54)
    print(" AKURASI SEBELUM DAN SESUDAH KUANTISASI (%d gambar uji)" % len(y_uji))
    print("=" * 54)
    print(" FLOAT32 (asli)      : %.2f%%" % (100 * akurasi_fp32))
    print(" INT8    (ESP-DL)    : %.2f%%" % (100 * akurasi_int8))
    print(" Selisih             : %+.2f%%" % (100 * (akurasi_int8 - akurasi_fp32)))
    print(" Keputusan berbeda   : %d gambar" % (tebak_fp32 != tebak_int8).sum())

    matriks = np.zeros((2, 2), dtype=int)
    for asli, tebak in zip(y_uji, tebak_int8):
        matriks[asli, tebak] += 1
    print("")
    print(" Matriks kebingungan INT8 (baris = asli, kolom = tebakan):")
    print("              CLOSED      OPEN")
    print("   CLOSED   %8d  %8d" % tuple(matriks[0]))
    print("   OPEN     %8d  %8d" % tuple(matriks[1]))
    print("=" * 54)

    if akurasi_fp32 - akurasi_int8 > 0.03:
        print(" [PERINGATAN] Akurasi turun lebih dari 3%.")
        print("              Coba ubah QUANT_TYPE menjadi 'w8a16'.")


if __name__ == "__main__":
    main()
