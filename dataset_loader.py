"""
dataset_loader.py
-----------------
Pembaca dataset mata untuk training.

Seluruh gambar dimuat sekali ke memori (2000 gambar 64x64 = sekitar 8 MB),
jadi training berjalan cepat tanpa membaca ulang file dari disk.

Augmentasi (hanya saat training) dipakai supaya model tidak menghafal:
  - kecerahan & kontras diubah acak  -> tahan terhadap perubahan cahaya
  - digeser sedikit                  -> tahan kalau potongan mata agak meleset
  - diputar sedikit                  -> tahan kalau kepala miring
  - dicerminkan kiri-kanan           -> mata kiri dan kanan dianggap sama
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from model import IMG_SIZE, KELAS

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")


def muat_semua():
    """
    Membaca seluruh dataset.
    Mengembalikan (gambar, label, nama_berkas):
      gambar      : array (N, 64, 64) uint8
      label       : array (N,) int    -> 0 = CLOSED, 1 = OPEN
      nama_berkas : list (N,) berisi "OPEN/open_0001.png"
    """
    gambar, label, nama_berkas = [], [], []

    for indeks_kelas, nama_kelas in enumerate(KELAS):
        folder = os.path.join(DATASET_DIR, nama_kelas)
        for nama in sorted(os.listdir(folder)):
            if not nama.lower().endswith(".png"):
                continue
            g = cv2.imread(os.path.join(folder, nama), cv2.IMREAD_GRAYSCALE)
            if g is None:
                continue
            gambar.append(g)
            label.append(indeks_kelas)
            nama_berkas.append("%s/%s" % (nama_kelas, nama))

    return np.array(gambar), np.array(label), nama_berkas


def augmentasi(g):
    """Mengubah satu gambar secara acak (dipakai saat training saja)."""
    # kecerahan dan kontras
    alpha = np.random.uniform(0.7, 1.3)     # kontras
    beta  = np.random.uniform(-40, 40)      # kecerahan
    g = cv2.convertScaleAbs(g, alpha=alpha, beta=beta)

    # putar sedikit + geser sedikit
    sudut = np.random.uniform(-12, 12)
    geser_x = np.random.uniform(-0.08, 0.08) * g.shape[1]
    geser_y = np.random.uniform(-0.08, 0.08) * g.shape[0]
    M = cv2.getRotationMatrix2D((g.shape[1] / 2, g.shape[0] / 2), sudut, 1.0)
    M[0, 2] += geser_x
    M[1, 2] += geser_y
    g = cv2.warpAffine(g, M, (g.shape[1], g.shape[0]), borderMode=cv2.BORDER_REPLICATE)

    # cermin kiri-kanan
    if np.random.rand() < 0.5:
        g = cv2.flip(g, 1)

    return g


def siapkan(g):
    """
    Mengubah gambar 64x64 uint8 menjadi tensor 1x32x32 float.
    Langkah ini HARUS sama persis saat training dan saat dipakai nanti.
    """
    g = cv2.resize(g, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    x = g.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5                      # jadi rentang -1 .. 1
    return torch.from_numpy(x).unsqueeze(0)


class DatasetMata(Dataset):
    def __init__(self, gambar, label, latih=False):
        self.gambar = gambar
        self.label  = label
        self.latih  = latih

    def __len__(self):
        return len(self.label)

    def __getitem__(self, i):
        g = self.gambar[i]
        if self.latih:
            g = augmentasi(g)
        return siapkan(g), int(self.label[i])


if __name__ == "__main__":
    gambar, label, nama = muat_semua()
    print("Total gambar :", len(gambar))
    for i, k in enumerate(KELAS):
        print("  %-7s: %d" % (k, (label == i).sum()))
    print("Bentuk satu tensor :", siapkan(gambar[0]).shape)
