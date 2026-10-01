"""
ekspor_onnx.py
--------------
Mengubah model hasil training (eyenet.pth) menjadi file ONNX.

ONNX adalah format perantara. Alurnya:
    eyenet.pth  ->  eyenet.onnx  ->  (esp-ppq, kuantisasi INT8)  ->  eyenet.espdl

Selain itu, program ini juga menyimpan data kalibrasi:
    kalibrasi.npy  -> berisi beberapa ratus gambar dari data latih.
Data ini dipakai esp-ppq untuk mengukur rentang nilai tiap lapisan,
supaya konversi dari FLOAT32 ke INT8 tidak merusak akurasi.

Cara pakai:
    python ekspor_onnx.py
"""

import os
import numpy as np
import torch

from model import EyeNet, IMG_SIZE
from dataset_loader import muat_semua, siapkan

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PATH_MODEL = os.path.join(BASE_DIR, "eyenet.pth")
PATH_ONNX  = os.path.join(BASE_DIR, "eyenet.onnx")
PATH_KALIB = os.path.join(BASE_DIR, "kalibrasi.npy")

JUMLAH_KALIBRASI = 512      # jumlah gambar untuk kalibrasi kuantisasi


def main():
    if not os.path.exists(PATH_MODEL):
        print("[ERROR] eyenet.pth tidak ditemukan. Jalankan train.py dulu.")
        return

    model = EyeNet()
    model.load_state_dict(torch.load(PATH_MODEL, map_location="cpu"))
    model.eval()

    # ---- 1. ekspor ke ONNX ----
    contoh = torch.zeros(1, 1, IMG_SIZE, IMG_SIZE)
    torch.onnx.export(
        model,
        contoh,
        PATH_ONNX,
        input_names=["input"],
        output_names=["output"],
        opset_version=13,           # versi yang didukung esp-ppq
        dynamo=False,               # pakai eksportir lama yang lebih stabil
    )
    print("[OK] ONNX tersimpan :", PATH_ONNX)
    print("     ukuran         : %.1f KB" % (os.path.getsize(PATH_ONNX) / 1024))

    # ---- 2. siapkan data kalibrasi ----
    gambar, label, _nama = muat_semua()
    rng = np.random.default_rng(42)
    indeks = rng.choice(len(gambar), size=min(JUMLAH_KALIBRASI, len(gambar)),
                        replace=False)

    batch = torch.stack([siapkan(gambar[i]) for i in indeks]).numpy()
    np.save(PATH_KALIB, batch)
    print("[OK] Data kalibrasi :", PATH_KALIB)
    print("     bentuk         :", batch.shape)

    # ---- 3. periksa hasil ONNX sama dengan PyTorch ----
    try:
        import onnxruntime as ort

        sesi = ort.InferenceSession(PATH_ONNX, providers=["CPUExecutionProvider"])
        uji = batch[:64]

        with torch.no_grad():
            hasil_torch = model(torch.from_numpy(uji)).numpy()

        # model ONNX dibuat untuk 1 gambar sekali jalan (seperti di ESP32),
        # jadi pengecekan dilakukan satu per satu
        hasil_onnx = np.concatenate(
            [sesi.run(None, {"input": uji[i:i + 1]})[0] for i in range(len(uji))])

        selisih = np.abs(hasil_torch - hasil_onnx).max()
        sama = (hasil_torch.argmax(1) == hasil_onnx.argmax(1)).mean()
        print("")
        print("[CEK] Selisih terbesar PyTorch vs ONNX : %.2e" % selisih)
        print("[CEK] Kesamaan keputusan               : %.1f%%" % (100 * sama))
    except ImportError:
        print("")
        print("[INFO] onnxruntime belum terpasang, pemeriksaan dilewati.")


if __name__ == "__main__":
    main()
