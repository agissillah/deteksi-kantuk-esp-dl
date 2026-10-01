"""
collect_dataset.py
------------------
Program pengambilan dataset mata (OPEN / CLOSED) dari webcam laptop.

Alur:
  Webcam -> OpenCV -> deteksi wajah + titik mata (YuNet, bawaan OpenCV)
         -> potong area mata -> simpan ke dataset/OPEN atau dataset/CLOSED
            saat tombol ditekan

Kenapa YuNet, bukan Haar Cascade?
  Haar Cascade sering gagal saat kepala menoleh dan saat cahaya redup.
  YuNet (cv2.FaceDetectorYN) adalah detektor wajah deep learning kecil yang
  sudah ada di OpenCV. Selain kotak wajah, YuNet juga memberi posisi titik
  kedua mata, sehingga potongan mata lebih tepat.

Tahap ini HANYA pengumpulan dataset. Belum ada training, belum ada ESP32.

Tombol:
  O = simpan gambar mata TERBUKA  (dataset/OPEN)
  C = simpan gambar mata TERTUTUP (dataset/CLOSED)
  U = hapus gambar terakhir yang baru saja disimpan (undo)
  Q = keluar
"""

import os
import math
import time
import cv2

# ============================================================
# 1. PENGATURAN
# ============================================================

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
OPEN_DIR    = os.path.join(DATASET_DIR, "OPEN")
CLOSED_DIR  = os.path.join(DATASET_DIR, "CLOSED")

YUNET_PATH = os.path.join(BASE_DIR, "models", "face_detection_yunet_2023mar.onnx")

CAMERA_INDEX = 0        # 0 = webcam bawaan laptop
IMG_SIZE     = 64       # gambar mata disimpan 64x64 piksel (grayscale)

# Pengaturan deteksi wajah
FACE_SCORE_MIN = 0.5    # makin kecil -> makin mudah terdeteksi (tapi bisa salah deteksi)

# Resolusi kamera. Makin tinggi, wajah yang jauh tetap punya cukup piksel,
# sehingga masih bisa dideteksi. Konsekuensinya deteksi jadi sedikit lebih berat.
LEBAR_KAMERA  = 1280
TINGGI_KAMERA = 720

# Ukuran potongan mata = EYE_CROP_SCALE x jarak antar kedua mata.
# Makin besar -> area potong makin luas (ikut alis/pipi).
EYE_CROP_SCALE = 0.8

# Kalau jarak antar mata kurang dari ini (piksel), wajah dianggap terlalu jauh
# dan gambar tidak disimpan karena potongan matanya akan buram.
EYE_DIST_MIN = 45

# --- Pengaturan mode rekam otomatis ---
# Jeda antar penyimpanan saat merekam (detik).
# 0.25 detik = 4 kali simpan per detik = 8 gambar per detik (mata kiri + kanan).
JEDA_REKAM = 0.25

# Hitung mundur sebelum perekaman dimulai (detik), untuk bersiap-siap.
HITUNG_MUNDUR = 2.0

# Target jumlah gambar untuk TIAP kelas.
# Perekaman berhenti sendiri begitu jumlah gambar di folder kelas itu
# sudah mencapai target, jadi tombol cukup ditekan sekali.
TARGET_PER_KELAS = 1000

# Cahaya redup: frame dibuat lebih terang/kontras KHUSUS untuk deteksi wajah.
# Gambar mata yang disimpan tetap diambil dari frame asli.
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


# ============================================================
# 2. FUNGSI BANTU
# ============================================================

def buat_folder():
    """Membuat folder dataset/OPEN dan dataset/CLOSED jika belum ada."""
    os.makedirs(OPEN_DIR, exist_ok=True)
    os.makedirs(CLOSED_DIR, exist_ok=True)


def hitung_gambar(folder):
    """Menghitung jumlah file .png yang sudah ada di dalam folder."""
    return len([f for f in os.listdir(folder) if f.lower().endswith(".png")])


def nomor_berikutnya(folder, prefix):
    """
    Mencari nomor urut berikutnya supaya penyimpanan bisa dilanjutkan
    tanpa menimpa gambar lama (misalnya saat program dijalankan ulang).
    """
    nomor = 0
    for nama in os.listdir(folder):
        if nama.startswith(prefix) and nama.lower().endswith(".png"):
            angka = nama[len(prefix):-4]
            if angka.isdigit():
                nomor = max(nomor, int(angka))
    return nomor + 1


def terangkan_untuk_deteksi(frame):
    """
    Meratakan terang-gelap frame (CLAHE pada kanal kecerahan) supaya wajah
    tetap terdeteksi saat cahaya redup. Hanya dipakai untuk DETEKSI.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def potong_mata(gray, titik_mata, sudut, ukuran_potong):
    """
    Memotong satu mata berbentuk persegi dengan titik mata di tengahnya.
    Potongan diputar sesuai kemiringan kepala supaya mata selalu mendatar,
    lalu langsung diperkecil menjadi IMG_SIZE x IMG_SIZE.
    """
    cx, cy = titik_mata
    skala = IMG_SIZE / ukuran_potong

    M = cv2.getRotationMatrix2D((cx, cy), sudut, skala)
    # geser supaya titik mata berada di tengah gambar hasil
    M[0, 2] += IMG_SIZE / 2 - cx
    M[1, 2] += IMG_SIZE / 2 - cy

    return cv2.warpAffine(gray, M, (IMG_SIZE, IMG_SIZE),
                          flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)


def ambil_area_mata(gray, wajah):
    """
    Mengambil dua potongan mata dari hasil deteksi YuNet.
    Mengembalikan (daftar_mata, pesan):
      daftar_mata = list berisi (gambar_mata, titik_mata, ukuran_potong)
      pesan       = alasan jika mata tidak bisa diambil
    """
    # format satu baris hasil YuNet:
    # [x, y, w, h, mata1_x, mata1_y, mata2_x, mata2_y, hidung.., mulut.., skor]
    mata1 = (float(wajah[4]), float(wajah[5]))
    mata2 = (float(wajah[6]), float(wajah[7]))

    # urutkan dari kiri ke kanan layar, supaya urutan simpan selalu sama
    kiri, kanan = sorted((mata1, mata2), key=lambda p: p[0])

    dx = kanan[0] - kiri[0]
    dy = kanan[1] - kiri[1]
    jarak_mata = math.hypot(dx, dy)

    if jarak_mata < EYE_DIST_MIN:
        return [], "TERLALU JAUH DARI KAMERA"

    sudut = math.degrees(math.atan2(dy, dx))   # kemiringan kepala
    ukuran_potong = EYE_CROP_SCALE * jarak_mata

    daftar_mata = []
    for titik in (kiri, kanan):
        gambar_mata = potong_mata(gray, titik, sudut, ukuran_potong)
        daftar_mata.append((gambar_mata, titik, ukuran_potong))

    return daftar_mata, ""


def simpan_gambar(daftar_mata, folder, prefix):
    """Menyimpan semua potongan mata ke folder tujuan. Mengembalikan daftar path."""
    tersimpan = []
    nomor = nomor_berikutnya(folder, prefix)

    for gambar_mata, _titik, _ukuran in daftar_mata:
        nama_file = "%s%04d.png" % (prefix, nomor)
        path = os.path.join(folder, nama_file)
        cv2.imwrite(path, gambar_mata)
        tersimpan.append(path)
        nomor += 1

    return tersimpan


def cetak_instruksi():
    print("=" * 58)
    print(" PENGAMBILAN DATASET MATA  (OPEN / CLOSED)")
    print("=" * 58)
    print(" Tombol pada jendela kamera:")
    print("   O      ->  mulai / berhenti merekam mata TERBUKA  (dataset/OPEN)")
    print("   C      ->  mulai / berhenti merekam mata TERTUTUP (dataset/CLOSED)")
    print("   SPASI  ->  berhenti merekam")
    print("   U      ->  hapus semua gambar dari rekaman terakhir")
    print("   Q      ->  keluar")
    print("")
    print(" Cara pakai:")
    print("   1. Tekan O (atau C) satu kali, lalu ada hitung mundur %d detik." % HITUNG_MUNDUR)
    print("   2. Gambar tersimpan otomatis %d kali per detik selama merekam." % round(1 / JEDA_REKAM))
    print("   3. GERAKKAN KEPALA PERLAHAN selama merekam supaya gambarnya bervariasi.")
    print("   4. Tekan tombol yang sama (atau SPASI) untuk berhenti.")
    print("")
    print(" Catatan:")
    print("   - Perekaman berhenti SENDIRI setelah kelas itu mencapai %d gambar."
          % TARGET_PER_KELAS)
    print("   - Kalau dihentikan di tengah jalan, tekan tombolnya lagi untuk")
    print("     melanjutkan. Hitungannya memakai isi folder, jadi tidak mengulang.")
    print("   - Gambar hanya disimpan saat status di layar 'WAJAH OK'.")
    print("   - Klik dulu jendela kamera supaya tombol terbaca.")
    print("=" * 58)
    print("")


# ============================================================
# 3. PROGRAM UTAMA
# ============================================================

def main():
    buat_folder()
    cetak_instruksi()

    if not os.path.exists(YUNET_PATH):
        print("[ERROR] File model YuNet tidak ditemukan:", YUNET_PATH)
        return

    kamera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)   # CAP_DSHOW = Windows
    if not kamera.isOpened():
        print("[ERROR] Webcam tidak bisa dibuka. Coba ganti CAMERA_INDEX jadi 1.")
        return

    kamera.set(cv2.CAP_PROP_FRAME_WIDTH, LEBAR_KAMERA)
    kamera.set(cv2.CAP_PROP_FRAME_HEIGHT, TINGGI_KAMERA)

    # ukuran input detektor disesuaikan otomatis dengan ukuran frame di dalam loop
    detektor_wajah = cv2.FaceDetectorYN.create(
        YUNET_PATH, "", (LEBAR_KAMERA, TINGGI_KAMERA), FACE_SCORE_MIN)

    jumlah_open   = hitung_gambar(OPEN_DIR)
    jumlah_closed = hitung_gambar(CLOSED_DIR)
    print("[INFO] Dataset awal -> OPEN: %d | CLOSED: %d" % (jumlah_open, jumlah_closed))
    print("")

    terakhir_disimpan = []   # semua gambar dari rekaman terakhir (untuk undo)
    mode        = None       # None, "OPEN", atau "CLOSED"
    waktu_mulai = 0.0        # kapan perekaman benar-benar dimulai (setelah hitung mundur)
    waktu_simpan_terakhir = 0.0
    jumlah_rekam = 0         # jumlah gambar pada rekaman yang sedang berjalan

    def mulai_rekam(mode_baru):
        """Menyiapkan perekaman baru. Mengembalikan (mode, waktu_mulai, jumlah)."""
        print("[REKAM ] Bersiap merekam %s ..." % mode_baru)
        return mode_baru, time.time() + HITUNG_MUNDUR, 0

    while True:
        ok, frame = kamera.read()
        if not ok:
            print("[ERROR] Gagal membaca frame dari webcam.")
            break

        frame = cv2.flip(frame, 1)                       # efek cermin, lebih natural
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        tinggi, lebar = frame.shape[:2]
        detektor_wajah.setInputSize((lebar, tinggi))
        _, hasil = detektor_wajah.detect(terangkan_untuk_deteksi(frame))

        daftar_mata = []
        pesan = "WAJAH TIDAK TERDETEKSI"

        if hasil is not None and len(hasil) > 0:
            # kalau ada beberapa wajah, ambil yang paling besar (paling dekat kamera)
            wajah = max(hasil, key=lambda w: w[2] * w[3])
            x, y, w, h = [int(v) for v in wajah[:4]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            daftar_mata, pesan = ambil_area_mata(gray, wajah)
            for _gambar, (cx, cy), ukuran in daftar_mata:
                setengah = int(ukuran / 2)
                cx, cy = int(cx), int(cy)
                cv2.rectangle(frame, (cx - setengah, cy - setengah),
                              (cx + setengah, cy + setengah), (255, 200, 0), 2)
                cv2.circle(frame, (cx, cy), 2, (0, 0, 255), -1)

        # ---- perekaman otomatis ----
        sekarang = time.time()
        sisa_mundur = waktu_mulai - sekarang

        if mode is not None and sisa_mundur <= 0:
            if daftar_mata and (sekarang - waktu_simpan_terakhir) >= JEDA_REKAM:
                folder = OPEN_DIR if mode == "OPEN" else CLOSED_DIR
                prefix = "open_" if mode == "OPEN" else "closed_"

                terakhir_disimpan += simpan_gambar(daftar_mata, folder, prefix)
                jumlah_rekam += len(daftar_mata)
                waktu_simpan_terakhir = sekarang

                jumlah_open   = hitung_gambar(OPEN_DIR)
                jumlah_closed = hitung_gambar(CLOSED_DIR)
                jumlah_kelas  = jumlah_open if mode == "OPEN" else jumlah_closed
                print("[%-6s] %d / %d   (rekaman ini: %d gambar)"
                      % (mode, jumlah_kelas, TARGET_PER_KELAS, jumlah_rekam))

                if jumlah_kelas >= TARGET_PER_KELAS:
                    print("")
                    print("[SELESAI] Target kelas %s tercapai: %d gambar."
                          % (mode, jumlah_kelas))
                    print("")
                    mode = None

        # ---- teks informasi di layar ----
        status = "WAJAH OK" if daftar_mata else pesan
        warna  = (0, 255, 0) if daftar_mata else (0, 0, 255)

        if mode is not None:
            if sisa_mundur > 0:
                teks_rekam = "SIAP-SIAP %s ... %.0f" % (mode, sisa_mundur + 1)
                warna_rekam = (0, 255, 255)
            else:
                jumlah_kelas = jumlah_open if mode == "OPEN" else jumlah_closed
                sisa = max(0, TARGET_PER_KELAS - jumlah_kelas)
                teks_rekam = "MEREKAM %s : %d / %d  (sisa ~%.0f detik)" % (
                    mode, jumlah_kelas, TARGET_PER_KELAS, sisa * JEDA_REKAM / 2)
                warna_rekam = (0, 0, 255)
                cv2.circle(frame, (frame.shape[1] - 30, 30), 12, (0, 0, 255), -1)

            cv2.putText(frame, teks_rekam, (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, warna_rekam, 2)
            cv2.putText(frame, "GERAKKAN KEPALA PERLAHAN", (10, 135),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        cv2.putText(frame, status, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, warna, 2)
        cv2.putText(frame, "OPEN  : %d" % jumlah_open, (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "CLOSED: %d" % jumlah_closed, (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "[O] rekam open  [C] rekam closed  [SPASI] stop  [U] undo  [Q] keluar",
                    (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        cv2.imshow("Pengambilan Dataset Mata", frame)

        # ---- baca tombol ----
        tombol = cv2.waitKey(1) & 0xFF

        if tombol in (ord("q"), ord("Q"), 27):           # 27 = ESC
            break

        elif tombol in (ord("o"), ord("O"), ord("c"), ord("C")):
            mode_baru = "OPEN" if tombol in (ord("o"), ord("O")) else "CLOSED"

            sudah = jumlah_open if mode_baru == "OPEN" else jumlah_closed

            if mode == mode_baru:                    # tombol sama -> berhenti merekam
                print("[REKAM ] Berhenti. Total rekaman ini: %d gambar." % jumlah_rekam)
                mode = None
            elif sudah >= TARGET_PER_KELAS:
                print("[REKAM ] Kelas %s sudah mencapai target (%d gambar)."
                      % (mode_baru, sudah))
                print("         Ubah TARGET_PER_KELAS kalau mau menambah lagi.")
            else:
                mode, waktu_mulai, jumlah_rekam = mulai_rekam(mode_baru)
                terakhir_disimpan = []               # rekaman baru, undo di-reset

        elif tombol == ord(" "):
            if mode is not None:
                print("[REKAM ] Berhenti. Total rekaman ini: %d gambar." % jumlah_rekam)
                mode = None

        elif tombol in (ord("u"), ord("U")):
            if mode is not None:
                print("[UNDO  ] Hentikan perekaman dulu (SPASI) sebelum undo.")
            elif terakhir_disimpan:
                for path in terakhir_disimpan:
                    if os.path.exists(path):
                        os.remove(path)
                print("[UNDO  ] %d gambar dari rekaman terakhir dihapus."
                      % len(terakhir_disimpan))
                terakhir_disimpan = []
                jumlah_open   = hitung_gambar(OPEN_DIR)
                jumlah_closed = hitung_gambar(CLOSED_DIR)
            else:
                print("[UNDO  ] Tidak ada gambar untuk dihapus.")

    kamera.release()
    cv2.destroyAllWindows()

    print("")
    print("=" * 58)
    print(" SELESAI")
    print("   dataset/OPEN   : %d gambar" % hitung_gambar(OPEN_DIR))
    print("   dataset/CLOSED : %d gambar" % hitung_gambar(CLOSED_DIR))
    print("=" * 58)


if __name__ == "__main__":
    main()
