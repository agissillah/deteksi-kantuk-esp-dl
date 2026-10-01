"""
bersihkan_dataset.py
--------------------
Program untuk MEMBERSIHKAN dataset: memindahkan gambar yang salah label.

Contoh kasus: saat merekam OPEN, Anda berkedip, sehingga ada gambar mata
tertutup yang ikut tersimpan di folder OPEN. Program ini menampilkan semua
gambar dalam bentuk grid, lalu Anda tinggal mengklik gambar yang salah.

Cara pakai:
    python bersihkan_dataset.py OPEN
    python bersihkan_dataset.py CLOSED

Tombol:
  KLIK KIRI  = tandai / batal tandai gambar yang salah label
  ENTER      = terapkan (gambar bertanda dipindah ke kelas seberangnya)
  D          = hapus gambar bertanda (bukan dipindah)
  N / SPASI  = halaman berikutnya
  P          = halaman sebelumnya
  A          = tandai semua di halaman ini
  X          = batalkan semua tanda di halaman ini
  Q          = keluar
"""

import os
import sys
import cv2
import numpy as np

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

PREFIX   = {"OPEN": "open_", "CLOSED": "closed_"}
SEBERANG = {"OPEN": "CLOSED", "CLOSED": "OPEN"}

THUMB  = 96     # ukuran satu gambar di grid (piksel)
KOLOM  = 10
BARIS  = 5
HEADER = 30     # tinggi baris judul di atas grid

PER_HALAMAN = KOLOM * BARIS

# dipakai bersama antara fungsi utama dan callback mouse
ditandai = set()
berkas_halaman = []


def daftar_berkas(kelas):
    folder = os.path.join(DATASET_DIR, kelas)
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(".png"))


def nomor_berikutnya(folder, prefix):
    """Mencari nomor urut berikutnya agar tidak menimpa file yang sudah ada."""
    nomor = 0
    for nama in os.listdir(folder):
        if nama.startswith(prefix) and nama.lower().endswith(".png"):
            angka = nama[len(prefix):-4]
            if angka.isdigit():
                nomor = max(nomor, int(angka))
    return nomor + 1


def gambar_grid(kelas, halaman, total_halaman):
    """Menyusun satu halaman grid beserta tanda pada gambar yang dipilih."""
    folder = os.path.join(DATASET_DIR, kelas)
    kanvas = np.zeros((HEADER + BARIS * THUMB, KOLOM * THUMB, 3), dtype=np.uint8)

    judul = "%s  |  halaman %d/%d  |  ditandai: %d  |  ENTER=pindah  D=hapus  N/P=halaman  Q=keluar" % (
        kelas, halaman + 1, total_halaman, len(ditandai))
    cv2.putText(kanvas, judul, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    for i, nama in enumerate(berkas_halaman):
        b, k = divmod(i, KOLOM)
        y, x = HEADER + b * THUMB, k * THUMB

        gambar = cv2.imread(os.path.join(folder, nama), cv2.IMREAD_GRAYSCALE)
        if gambar is None:
            continue
        petak = cv2.cvtColor(cv2.resize(gambar, (THUMB, THUMB)), cv2.COLOR_GRAY2BGR)

        if nama in ditandai:
            # gambar bertanda: diberi kotak merah dan tanda silang
            cv2.rectangle(petak, (1, 1), (THUMB - 2, THUMB - 2), (0, 0, 255), 3)
            cv2.line(petak, (8, 8), (THUMB - 8, THUMB - 8), (0, 0, 255), 2)
            cv2.line(petak, (THUMB - 8, 8), (8, THUMB - 8), (0, 0, 255), 2)

        cv2.putText(petak, nama[-8:-4], (3, THUMB - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
        kanvas[y:y + THUMB, x:x + THUMB] = petak

    return kanvas


def klik_mouse(event, mx, my, flags, param):
    """Klik kiri pada sebuah gambar untuk menandai / membatalkan tanda."""
    if event != cv2.EVENT_LBUTTONDOWN or my < HEADER:
        return

    kolom = mx // THUMB
    baris = (my - HEADER) // THUMB
    indeks = baris * KOLOM + kolom

    if 0 <= indeks < len(berkas_halaman):
        nama = berkas_halaman[indeks]
        if nama in ditandai:
            ditandai.remove(nama)
        else:
            ditandai.add(nama)


def pindahkan(kelas, nama_berkas):
    """Memindahkan gambar bertanda ke folder kelas seberangnya."""
    asal_dir   = os.path.join(DATASET_DIR, kelas)
    tujuan     = SEBERANG[kelas]
    tujuan_dir = os.path.join(DATASET_DIR, tujuan)
    prefix     = PREFIX[tujuan]

    nomor = nomor_berikutnya(tujuan_dir, prefix)
    for nama in sorted(nama_berkas):
        baru = "%s%04d.png" % (prefix, nomor)
        os.rename(os.path.join(asal_dir, nama), os.path.join(tujuan_dir, baru))
        print("   %s/%s  ->  %s/%s" % (kelas, nama, tujuan, baru))
        nomor += 1

    print("[PINDAH] %d gambar dipindahkan ke %s." % (len(nama_berkas), tujuan))


def hapus(kelas, nama_berkas):
    """Menghapus gambar bertanda."""
    folder = os.path.join(DATASET_DIR, kelas)
    for nama in nama_berkas:
        os.remove(os.path.join(folder, nama))
    print("[HAPUS ] %d gambar dihapus dari %s." % (len(nama_berkas), kelas))


def main():
    global berkas_halaman

    if len(sys.argv) < 2 or sys.argv[1].upper() not in PREFIX:
        print("Cara pakai:  python bersihkan_dataset.py OPEN")
        print("        atau python bersihkan_dataset.py CLOSED")
        return

    kelas = sys.argv[1].upper()

    print("=" * 62)
    print(" MEMBERSIHKAN DATASET:", kelas)
    print("=" * 62)
    print(" KLIK KIRI  = tandai gambar yang SALAH label")
    print(" ENTER      = pindahkan gambar bertanda ke %s" % SEBERANG[kelas])
    print(" D          = hapus gambar bertanda")
    print(" N / SPASI  = halaman berikutnya      P = halaman sebelumnya")
    print(" A          = tandai semua            X = batalkan semua tanda")
    print(" Q          = keluar")
    print("=" * 62)
    print("")

    judul_jendela = "Bersihkan Dataset"
    cv2.namedWindow(judul_jendela)
    cv2.setMouseCallback(judul_jendela, klik_mouse)

    halaman = 0
    while True:
        semua = daftar_berkas(kelas)
        if not semua:
            print("[INFO] Folder %s kosong." % kelas)
            break

        total_halaman = (len(semua) + PER_HALAMAN - 1) // PER_HALAMAN
        halaman = max(0, min(halaman, total_halaman - 1))
        berkas_halaman = semua[halaman * PER_HALAMAN:(halaman + 1) * PER_HALAMAN]

        cv2.imshow(judul_jendela, gambar_grid(kelas, halaman, total_halaman))
        tombol = cv2.waitKey(30) & 0xFF

        if tombol in (ord("q"), ord("Q"), 27):
            break

        elif tombol in (ord("n"), ord("N"), ord(" ")):
            halaman += 1
            ditandai.clear()

        elif tombol in (ord("p"), ord("P")):
            halaman -= 1
            ditandai.clear()

        elif tombol in (ord("a"), ord("A")):
            ditandai.update(berkas_halaman)

        elif tombol in (ord("x"), ord("X")):
            ditandai.clear()

        elif tombol == 13:                     # ENTER
            if ditandai:
                pindahkan(kelas, list(ditandai))
                ditandai.clear()
            else:
                print("[INFO] Belum ada gambar yang ditandai.")

        elif tombol in (ord("d"), ord("D")):
            if ditandai:
                hapus(kelas, list(ditandai))
                ditandai.clear()
            else:
                print("[INFO] Belum ada gambar yang ditandai.")

    cv2.destroyAllWindows()

    print("")
    print("=" * 62)
    for k in ("OPEN", "CLOSED"):
        print("   dataset/%-7s: %d gambar" % (k, len(daftar_berkas(k))))
    print("=" * 62)


if __name__ == "__main__":
    main()
