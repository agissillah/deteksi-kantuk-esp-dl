"""
deteksi_kantuk.py
-----------------
PROGRAM UTAMA sistem deteksi kantuk.

Alur lengkap:
    Webcam laptop
      -> OpenCV
      -> deteksi wajah + titik mata (YuNet)
      -> potong area mata (kode yang SAMA dengan saat mengambil dataset)
      -> kirim ke ESP32 lewat serial
      -> ESP32 menjalankan model ESP-DL (INT8) dan menentukan OPEN / CLOSED
      -> ESP32 menghitung durasi mata tertutup
      -> ESP32 membunyikan buzzer di GPIO 25 kalau melewati batas
      -> hasilnya dikirim balik ke laptop untuk ditampilkan

Yang memutuskan dan membunyikan buzzer adalah ESP32, bukan laptop.
Laptop hanya bertugas menyediakan gambar mata.

Cara pakai:
    python deteksi_kantuk.py              (pakai COM6)
    python deteksi_kantuk.py COM7         (port lain)

Tombol:
    Q = keluar
"""

import sys
import time
import cv2
import numpy as np
import serial

import collect_dataset as cd      # memakai ulang kode deteksi & pemotongan mata

PORT   = sys.argv[1] if len(sys.argv) > 1 else "COM6"
BAUD   = 921600
HEADER = bytes([0xA5, 0x5A])
UKURAN = 32                       # ukuran gambar yang diminta firmware


def buka_esp32(nama_port):
    """Membuka serial ke ESP32 dan me-reset board supaya mulai dari awal."""
    port = serial.Serial(nama_port, BAUD, timeout=0.5)

    # Di Windows, pyserial mengaktifkan DTR/RTS saat membuka port dan itu
    # menahan ESP32 dalam kondisi reset. Kedua sinyal dimatikan dulu.
    port.setDTR(False)
    port.setRTS(True)             # EN low  -> reset
    time.sleep(0.15)
    port.setRTS(False)            # EN high -> jalan lagi
    time.sleep(1.5)               # tunggu ESP32 selesai booting
    port.reset_input_buffer()
    return port


def tanya_esp32(port, daftar_gambar):
    """
    Mengirim 1-2 potongan mata ke ESP32, lalu membaca hasilnya.
    Mengembalikan (label, peluang_tertutup, lama_ms, buzzer, inference_us).
    """
    data = b"".join(g.tobytes() for g in daftar_gambar)
    port.reset_input_buffer()
    port.write(HEADER + bytes([len(daftar_gambar)]) + data)
    port.flush()

    batas = time.time() + 1.0
    while time.time() < batas:
        baris = port.readline().decode("utf-8", errors="ignore").strip()
        if baris.startswith("RES "):
            b = baris.split()
            return b[1], float(b[2]), int(b[3]), int(b[4]), int(b[5])
    return None, None, None, None, None


def main():
    print("=" * 58)
    print(" SISTEM DETEKSI KANTUK  (webcam -> ESP32 -> buzzer)")
    print("=" * 58)

    try:
        port = buka_esp32(PORT)
    except serial.SerialException as e:
        print("[ERROR] Tidak bisa membuka", PORT, ":", e)
        print("        Pastikan ESP32 tercolok dan port-nya benar.")
        return
    print(" ESP32 terhubung di", PORT)

    kamera = cv2.VideoCapture(cd.CAMERA_INDEX, cv2.CAP_DSHOW)
    if not kamera.isOpened():
        print("[ERROR] Webcam tidak bisa dibuka.")
        port.close()
        return

    kamera.set(cv2.CAP_PROP_FRAME_WIDTH, cd.LEBAR_KAMERA)
    kamera.set(cv2.CAP_PROP_FRAME_HEIGHT, cd.TINGGI_KAMERA)
    detektor = cv2.FaceDetectorYN.create(
        cd.YUNET_PATH, "", (cd.LEBAR_KAMERA, cd.TINGGI_KAMERA), cd.FACE_SCORE_MIN)

    print(" Webcam siap. Tekan Q untuk keluar.")
    print("=" * 58)
    print("")

    fps = 0.0
    fps_t = cv2.getTickCount()

    while True:
        ok, frame = kamera.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tinggi, lebar = frame.shape[:2]

        detektor.setInputSize((lebar, tinggi))
        _, hasil = detektor.detect(cd.terangkan_untuk_deteksi(frame))

        label = "TIDAK ADA WAJAH"
        peluang = lama_ms = buzzer = inference_us = None

        if hasil is not None and len(hasil) > 0:
            wajah = max(hasil, key=lambda w: w[2] * w[3])
            x, y, w, h = [int(v) for v in wajah[:4]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            daftar_mata, pesan = cd.ambil_area_mata(gray, wajah)

            if daftar_mata:
                # perkecil ke 32x32 seperti yang diminta firmware
                gambar_kirim = [
                    cv2.resize(g, (UKURAN, UKURAN), interpolation=cv2.INTER_AREA)
                    for g, _titik, _ukuran in daftar_mata
                ]
                label, peluang, lama_ms, buzzer, inference_us = tanya_esp32(
                    port, gambar_kirim)

                if label is None:
                    label = "ESP32 TIDAK MEMBALAS"
                else:
                    warna_mata = (0, 0, 255) if label == "CLOSED" else (0, 255, 0)
                    for _g, (cx, cy), ukuran in daftar_mata:
                        s = int(ukuran / 2)
                        cv2.rectangle(frame, (int(cx) - s, int(cy) - s),
                                      (int(cx) + s, int(cy) + s), warna_mata, 2)
            else:
                label = pesan

        # ---------- tampilan ----------
        warna = (0, 255, 0) if label == "OPEN" else (0, 0, 255)
        teks = label if peluang is None else "%s (%.0f%%)" % (
            label, 100 * (peluang if label == "CLOSED" else 1 - peluang))

        cv2.putText(frame, teks, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, warna, 2)

        if lama_ms is not None:
            cv2.putText(frame, "tertutup: %.1f detik" % (lama_ms / 1000.0), (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, "inference ESP32: %.0f ms" % (inference_us / 1000.0),
                        (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        fps_sekarang = cv2.getTickFrequency() / max(1, cv2.getTickCount() - fps_t)
        fps = 0.9 * fps + 0.1 * fps_sekarang
        fps_t = cv2.getTickCount()
        cv2.putText(frame, "%.1f FPS" % fps, (lebar - 100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        if buzzer:
            cv2.rectangle(frame, (0, 0), (lebar - 1, tinggi - 1), (0, 0, 255), 10)
            cv2.putText(frame, "!!! BUZZER MENYALA !!!", (40, tinggi // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)

        cv2.imshow("Deteksi Kantuk (ESP32 + ESP-DL) - tekan Q untuk keluar", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
            break

    kamera.release()
    cv2.destroyAllWindows()
    port.close()
    print("Selesai.")


if __name__ == "__main__":
    main()
