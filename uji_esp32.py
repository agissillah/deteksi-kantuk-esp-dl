"""
uji_esp32.py
------------
Menguji firmware ESP32: mengirim gambar mata dari dataset lewat serial,
lalu membandingkan jawaban ESP32 dengan label yang sebenarnya.

Yang diukur:
  - apakah hasil inference di ESP32 sama dengan di laptop
  - berapa lama satu inference di ESP32 (mikrodetik)

Cara pakai:
    python uji_esp32.py            (pakai COM6)
    python uji_esp32.py COM7       (port lain)
"""

import os
import sys
import time
import cv2
import numpy as np
import serial

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

PORT  = sys.argv[1] if len(sys.argv) > 1 else "COM6"
BAUD  = 921600
HEADER = bytes([0xA5, 0x5A])

UKURAN = 32                 # ukuran gambar yang diminta firmware
JUMLAH_UJI = 40             # berapa gambar per kelas yang dikirim


def ambil_gambar(kelas, jumlah):
    """Mengambil beberapa gambar dari folder kelas, diperkecil ke 32x32."""
    folder = os.path.join(DATASET_DIR, kelas)
    berkas = sorted(f for f in os.listdir(folder) if f.endswith(".png"))
    indeks = np.linspace(0, len(berkas) - 1, jumlah).astype(int)

    hasil = []
    for i in indeks:
        g = cv2.imread(os.path.join(folder, berkas[i]), cv2.IMREAD_GRAYSCALE)
        g = cv2.resize(g, (UKURAN, UKURAN), interpolation=cv2.INTER_AREA)
        hasil.append((berkas[i], g))
    return hasil


def kirim_dan_terima(port, gambar):
    """Mengirim satu gambar mata ke ESP32, lalu membaca balasannya."""
    port.reset_input_buffer()
    port.write(HEADER + bytes([1]) + gambar.tobytes())
    port.flush()

    batas = time.time() + 3.0
    while time.time() < batas:
        baris = port.readline().decode("utf-8", errors="ignore").strip()
        if baris.startswith("RES "):
            bagian = baris.split()
            # RES <label> <peluang> <lama_ms> <buzzer> <inference_us>
            return bagian[1], float(bagian[2]), int(bagian[5])
    return None, None, None


def main():
    print("=" * 58)
    print(" UJI FIRMWARE ESP32 (ESP-DL)")
    print("=" * 58)
    print(" Port :", PORT, "| baud:", BAUD)

    try:
        port = serial.Serial(PORT, BAUD, timeout=0.5)
    except serial.SerialException as e:
        print("[ERROR] Tidak bisa membuka", PORT, ":", e)
        return

    # PENTING: di Windows, pyserial mengaktifkan DTR/RTS saat membuka port,
    # dan itu membuat ESP32 tertahan dalam kondisi reset. Jadi kedua sinyal
    # dimatikan dulu, sekalian dipakai untuk me-reset board dengan rapi.
    port.setDTR(False)
    port.setRTS(True)          # EN low  -> reset
    time.sleep(0.15)
    port.setRTS(False)         # EN high -> jalan lagi
    time.sleep(1.5)            # tunggu ESP32 selesai booting
    port.reset_input_buffer()

    benar = 0
    total = 0
    waktu_inference = []
    salah = []

    for kelas in ("OPEN", "CLOSED"):
        for nama, gambar in ambil_gambar(kelas, JUMLAH_UJI):
            label, peluang, lama_us = kirim_dan_terima(port, gambar)

            if label is None:
                print("[GAGAL] Tidak ada balasan untuk", nama)
                continue

            total += 1
            waktu_inference.append(lama_us)
            if label == kelas:
                benar += 1
            else:
                salah.append((nama, kelas, label, peluang))

        print(" %-7s: %d gambar terkirim" % (kelas, JUMLAH_UJI))

    port.close()

    print("")
    print("=" * 58)
    if total == 0:
        print(" [GAGAL] ESP32 tidak membalas sama sekali.")
        print(" Periksa: apakah port benar, dan firmware sudah di-flash?")
        print("=" * 58)
        return

    print(" HASIL")
    print("=" * 58)
    print(" Gambar diuji       : %d" % total)
    print(" Jawaban benar      : %d (%.1f%%)" % (benar, 100 * benar / total))
    print(" Waktu inference    : rata-rata %.1f ms  (min %.1f, maks %.1f)"
          % (np.mean(waktu_inference) / 1000,
             np.min(waktu_inference) / 1000,
             np.max(waktu_inference) / 1000))
    print(" Perkiraan kecepatan: %.1f gambar per detik"
          % (1e6 / np.mean(waktu_inference)))

    if salah:
        print("")
        print(" Yang salah (maksimal 10 ditampilkan):")
        for nama, asli, tebak, peluang in salah[:10]:
            print("   %-20s asli=%-7s tebakan=%-7s p_closed=%.3f"
                  % (nama, asli, tebak, peluang))
    print("=" * 58)


if __name__ == "__main__":
    main()
