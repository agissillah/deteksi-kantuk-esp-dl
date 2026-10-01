"""
model.py
--------
Definisi model CNN untuk klasifikasi mata OPEN / CLOSED.

Model sengaja dibuat KECIL supaya muat dan cepat di ESP32-D0WD-V3
(RAM 520 KB, 240 MHz, tanpa akselerator SIMD).

Rancangan:
  input 1 x 32 x 32 (grayscale)
    -> Conv 3x3 stride 2,  8 filter  -> 16x16
    -> Conv 3x3 stride 2, 16 filter  ->  8x8
    -> Conv 3x3 stride 2, 32 filter  ->  4x4
    -> rata-rata global               -> 32 angka
    -> Linear                         -> 2 kelas (CLOSED, OPEN)

Semua operasi (Conv2d, BatchNorm, ReLU, AvgPool, Linear) didukung ESP-DL.
BatchNorm nanti digabung ke Conv saat ekspor, jadi tidak membebani ESP32.
"""

import torch
import torch.nn as nn

IMG_SIZE = 32                      # ukuran input model (gambar dataset 64x64 diperkecil)
KELAS    = ["CLOSED", "OPEN"]      # urutan ini menentukan indeks keluaran: 0=CLOSED, 1=OPEN


def blok(masuk, keluar):
    """Satu blok: Conv 3x3 stride 2 -> BatchNorm -> ReLU."""
    return nn.Sequential(
        nn.Conv2d(masuk, keluar, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(keluar),
        nn.ReLU(inplace=True),
    )


class EyeNet(nn.Module):
    def __init__(self, jumlah_kelas=2):
        super().__init__()
        self.fitur = nn.Sequential(
            blok(1, 8),        # 32x32 -> 16x16
            blok(8, 16),       # 16x16 -> 8x8
            blok(16, 32),      # 8x8   -> 4x4
        )
        self.rata_rata = nn.AvgPool2d(4)          # 4x4 -> 1x1
        self.klasifikasi = nn.Linear(32, jumlah_kelas)

    def forward(self, x):
        x = self.fitur(x)
        x = self.rata_rata(x)
        x = torch.flatten(x, 1)
        return self.klasifikasi(x)


if __name__ == "__main__":
    model = EyeNet()
    jumlah = sum(p.numel() for p in model.parameters())
    print("Jumlah parameter :", jumlah)
    print("Perkiraan ukuran INT8 : %.1f KB" % (jumlah / 1024))
    print("Bentuk keluaran :", model(torch.zeros(1, 1, IMG_SIZE, IMG_SIZE)).shape)
