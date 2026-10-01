"""
siapkan_data_uji.py
-------------------
Menyimpan DATA UJI (15% dataset yang tidak pernah dipakai saat training)
ke dalam file data_uji.npz.

Kenapa perlu file terpisah?
  Pemeriksaan kuantisasi dijalankan di environment lain (env_quant,
  Python 3.11) yang tidak punya OpenCV. Jadi datanya disiapkan di sini
  memakai Python utama, lalu tinggal dibaca di sana.

Pembagian data memakai SEED yang sama dengan train.py, sehingga data uji
di sini benar-benar sama dengan data uji saat training.

Cara pakai:
    python siapkan_data_uji.py
"""

import os
import numpy as np
import torch

from dataset_loader import muat_semua, siapkan
from train import bagi_data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_UJI = os.path.join(BASE_DIR, "data_uji.npz")


def main():
    gambar, label, _nama = muat_semua()
    _i_latih, _i_val, i_uji = bagi_data(label)

    x = torch.stack([siapkan(gambar[i]) for i in i_uji]).numpy()
    y = label[i_uji].astype(np.int64)

    np.savez(PATH_UJI, x=x, y=y)

    print("[OK] Data uji tersimpan:", PATH_UJI)
    print("     jumlah gambar     :", len(y))
    print("     bentuk            :", x.shape)
    print("     CLOSED / OPEN     : %d / %d" % ((y == 0).sum(), (y == 1).sum()))


if __name__ == "__main__":
    main()
