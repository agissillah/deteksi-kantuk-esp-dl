"""
cek_dataset.py
--------------
Program bantu untuk MEMERIKSA dataset yang sudah dikumpulkan.

Yang diperiksa:
  1. Jumlah gambar di tiap kelas (OPEN / CLOSED)
  2. Keseimbangan jumlah antar kelas
  3. Ukuran gambar (harus 64x64, grayscale)
  4. File rusak / tidak terbaca
  5. Menampilkan contoh gambar dalam bentuk grid

Tombol saat grid tampil:  tekan tombol apa saja untuk menutup.
"""

import os
import cv2
import numpy as np

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
KELAS       = ["OPEN", "CLOSED"]

IMG_SIZE   = 64
GRID_KOLOM = 10
GRID_BARIS = 3


def periksa_kelas(nama_kelas):
    """Memeriksa satu folder kelas. Mengembalikan (jumlah_valid, daftar_masalah)."""
    folder = os.path.join(DATASET_DIR, nama_kelas)

    if not os.path.isdir(folder):
        return 0, ["Folder %s tidak ada." % folder]

    berkas   = sorted(f for f in os.listdir(folder) if f.lower().endswith(".png"))
    masalah  = []
    valid    = 0

    for nama in berkas:
        path = os.path.join(folder, nama)
        gambar = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if gambar is None:
            masalah.append("%s -> tidak bisa dibaca (file rusak)" % nama)
            continue

        if gambar.ndim != 2:
            masalah.append("%s -> bukan grayscale (%d channel)" % (nama, gambar.shape[2]))
            continue

        if gambar.shape != (IMG_SIZE, IMG_SIZE):
            masalah.append("%s -> ukuran %dx%d, seharusnya %dx%d"
                           % (nama, gambar.shape[1], gambar.shape[0], IMG_SIZE, IMG_SIZE))
            continue

        valid += 1

    return valid, masalah


def buat_grid(nama_kelas):
    """Menyusun beberapa contoh gambar jadi satu grid untuk dilihat sekilas."""
    folder = os.path.join(DATASET_DIR, nama_kelas)
    if not os.path.isdir(folder):
        return None

    berkas = sorted(f for f in os.listdir(folder) if f.lower().endswith(".png"))
    if not berkas:
        return None

    # ambil contoh yang tersebar merata, bukan hanya gambar pertama
    jumlah_contoh = min(len(berkas), GRID_KOLOM * GRID_BARIS)
    indeks = np.linspace(0, len(berkas) - 1, jumlah_contoh).astype(int)

    petak = []
    for i in indeks:
        gambar = cv2.imread(os.path.join(folder, berkas[i]), cv2.IMREAD_GRAYSCALE)
        if gambar is None:
            gambar = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        petak.append(cv2.resize(gambar, (IMG_SIZE, IMG_SIZE)))

    # lengkapi sisa petak dengan kotak hitam supaya grid rapi
    while len(petak) < GRID_KOLOM * GRID_BARIS:
        petak.append(np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8))

    baris = [np.hstack(petak[b * GRID_KOLOM:(b + 1) * GRID_KOLOM])
             for b in range(GRID_BARIS)]
    return np.vstack(baris)


def main():
    print("=" * 58)
    print(" PEMERIKSAAN DATASET")
    print("=" * 58)

    jumlah = {}
    total_masalah = 0

    for nama_kelas in KELAS:
        valid, masalah = periksa_kelas(nama_kelas)
        jumlah[nama_kelas] = valid
        total_masalah += len(masalah)

        print("")
        print(" Kelas %s" % nama_kelas)
        print("   gambar valid : %d" % valid)
        print("   bermasalah   : %d" % len(masalah))
        for baris in masalah[:10]:
            print("      - %s" % baris)
        if len(masalah) > 10:
            print("      ... dan %d masalah lain" % (len(masalah) - 10))

    print("")
    print("-" * 58)
    total = sum(jumlah.values())
    print(" TOTAL gambar valid : %d" % total)

    # cek keseimbangan antar kelas
    if total > 0 and all(jumlah[k] > 0 for k in KELAS):
        kecil = min(jumlah.values())
        besar = max(jumlah.values())
        rasio = besar / kecil
        print(" Rasio kelas        : %.2f : 1" % rasio)
        if rasio > 1.5:
            print(" [PERINGATAN] Jumlah kelas timpang. Tambah gambar untuk kelas")
            print("              yang lebih sedikit supaya model tidak berat sebelah.")
        else:
            print(" [OK] Jumlah kedua kelas cukup seimbang.")
    else:
        print(" [PERINGATAN] Ada kelas yang masih kosong.")

    if total_masalah == 0:
        print(" [OK] Semua gambar berukuran %dx%d grayscale." % (IMG_SIZE, IMG_SIZE))

    print("-" * 58)
    print("")

    # tampilkan contoh gambar tiap kelas
    for nama_kelas in KELAS:
        grid = buat_grid(nama_kelas)
        if grid is None:
            print("[INFO] Kelas %s kosong, grid tidak ditampilkan." % nama_kelas)
            continue

        grid_besar = cv2.resize(grid, None, fx=2, fy=2,
                                interpolation=cv2.INTER_NEAREST)
        judul = "Contoh dataset: %s (tekan tombol apa saja untuk lanjut)" % nama_kelas
        cv2.imshow(judul, grid_besar)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
