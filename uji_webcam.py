"""
uji_webcam.py
-------------
Menguji model hasil training secara LANGSUNG lewat webcam laptop.

Ini pengujian jujur: model belum pernah melihat wajah Anda
dalam kondisi saat ini (posisi, cahaya, dan waktu yang berbeda).

Alur di sini sama persis dengan sistem akhir nanti, kecuali belum ada ESP32:
  Webcam -> potong mata (kode yang SAMA dengan saat ambil dataset)
         -> model -> OPEN / CLOSED -> hitung durasi mata tertutup
         -> tampilkan peringatan KANTUK

Belum ada ESP32 dan buzzer pada tahap ini.

Tombol:
  Q = keluar
"""

import os
import cv2
import numpy as np
import torch

from model import EyeNet, KELAS
from dataset_loader import siapkan
import collect_dataset as cd          # memakai ulang kode deteksi & pemotongan mata

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PATH_MODEL = os.path.join(BASE_DIR, "eyenet.pth")

# Mata dianggap tertutup kalau peluang CLOSED di atas ini
AMBANG_TERTUTUP = 0.5

# Berapa detik mata harus tertutup terus-menerus sampai dianggap MENGANTUK
DURASI_KANTUK = 1.5


def main():
    if not os.path.exists(PATH_MODEL):
        print("[ERROR] eyenet.pth tidak ditemukan. Jalankan train.py dulu.")
        return

    model = EyeNet()
    model.load_state_dict(torch.load(PATH_MODEL, map_location="cpu"))
    model.eval()

    kamera = cv2.VideoCapture(cd.CAMERA_INDEX, cv2.CAP_DSHOW)
    if not kamera.isOpened():
        print("[ERROR] Webcam tidak bisa dibuka.")
        return

    kamera.set(cv2.CAP_PROP_FRAME_WIDTH, cd.LEBAR_KAMERA)
    kamera.set(cv2.CAP_PROP_FRAME_HEIGHT, cd.TINGGI_KAMERA)
    detektor = cv2.FaceDetectorYN.create(
        cd.YUNET_PATH, "", (cd.LEBAR_KAMERA, cd.TINGGI_KAMERA), cd.FACE_SCORE_MIN)

    print("=" * 58)
    print(" UJI MODEL LEWAT WEBCAM")
    print("=" * 58)
    print(" Coba: mata terbuka, terpejam, menyipit, menoleh, cahaya redup.")
    print(" Tekan Q untuk keluar.")
    print("=" * 58)

    mulai_tertutup = None       # kapan mata mulai terpejam
    fps_t = cv2.getTickCount()
    fps = 0.0

    while True:
        ok, frame = kamera.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tinggi, lebar = frame.shape[:2]

        detektor.setInputSize((lebar, tinggi))
        _, hasil = detektor.detect(cd.terangkan_untuk_deteksi(frame))

        label, peluang_tutup = "TIDAK ADA WAJAH", None

        if hasil is not None and len(hasil) > 0:
            wajah = max(hasil, key=lambda w: w[2] * w[3])
            x, y, w, h = [int(v) for v in wajah[:4]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            daftar_mata, pesan = cd.ambil_area_mata(gray, wajah)

            if daftar_mata:
                # kedua mata diklasifikasi, lalu hasilnya dirata-rata
                batch = torch.stack([siapkan(g) for g, _t, _u in daftar_mata])
                with torch.no_grad():
                    peluang = torch.softmax(model(batch), dim=1).numpy()

                peluang_tutup = float(peluang[:, KELAS.index("CLOSED")].mean())
                label = "CLOSED" if peluang_tutup > AMBANG_TERTUTUP else "OPEN"

                for (_g, (cx, cy), ukuran), p in zip(daftar_mata, peluang):
                    p_tutup = p[KELAS.index("CLOSED")]
                    warna = (0, 0, 255) if p_tutup > AMBANG_TERTUTUP else (0, 255, 0)
                    s = int(ukuran / 2)
                    cv2.rectangle(frame, (int(cx) - s, int(cy) - s),
                                  (int(cx) + s, int(cy) + s), warna, 2)
            else:
                label = pesan

        # ---- hitung durasi mata tertutup ----
        sekarang = cv2.getTickCount() / cv2.getTickFrequency()
        if label == "CLOSED":
            if mulai_tertutup is None:
                mulai_tertutup = sekarang
            lama_tertutup = sekarang - mulai_tertutup
        else:
            mulai_tertutup = None
            lama_tertutup = 0.0

        mengantuk = lama_tertutup >= DURASI_KANTUK

        # ---- tampilan ----
        warna_label = (0, 255, 0) if label == "OPEN" else (0, 0, 255)
        teks = label if peluang_tutup is None else "%s  (%.0f%%)" % (
            label, 100 * (peluang_tutup if label == "CLOSED" else 1 - peluang_tutup))

        cv2.putText(frame, teks, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, warna_label, 2)
        cv2.putText(frame, "tertutup: %.1f detik" % lama_tertutup, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        fps_sekarang = cv2.getTickFrequency() / max(1, (cv2.getTickCount() - fps_t))
        fps = 0.9 * fps + 0.1 * fps_sekarang
        fps_t = cv2.getTickCount()
        cv2.putText(frame, "%.0f FPS" % fps, (lebar - 90, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        if mengantuk:
            cv2.rectangle(frame, (0, 0), (lebar - 1, tinggi - 1), (0, 0, 255), 8)
            cv2.putText(frame, "!!! KANTUK TERDETEKSI !!!", (30, tinggi // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        cv2.imshow("Uji Model - tekan Q untuk keluar", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
            break

    kamera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
