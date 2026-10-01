"""
train.py
--------
Melatih model CNN untuk klasifikasi mata OPEN / CLOSED.

Dataset dibagi tiga:
  - 70% data latih  (untuk melatih)
  - 15% data validasi (untuk memilih model terbaik selama training)
  - 15% data uji    (TIDAK pernah dilihat saat training, untuk nilai akhir)

Hasil akhir:
  eyenet.pth        -> bobot model terbaik
  grafik_training.png -> grafik akurasi & loss tiap epoch

Cara pakai:
    python train.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import EyeNet, KELAS, IMG_SIZE
from dataset_loader import muat_semua, DatasetMata

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EPOCH = 60
BATCH = 64
LR    = 2e-3
SEED  = 42


def bagi_data(label):
    """Membagi indeks jadi latih/validasi/uji, seimbang untuk tiap kelas."""
    rng = np.random.default_rng(SEED)
    latih, validasi, uji = [], [], []

    for kelas in np.unique(label):
        indeks = np.where(label == kelas)[0]
        rng.shuffle(indeks)
        n = len(indeks)
        n_latih = int(0.70 * n)
        n_val   = int(0.15 * n)
        latih    += list(indeks[:n_latih])
        validasi += list(indeks[n_latih:n_latih + n_val])
        uji      += list(indeks[n_latih + n_val:])

    return np.array(latih), np.array(validasi), np.array(uji)


def evaluasi(model, pemuat, perangkat):
    """Menghitung akurasi dan matriks kebingungan."""
    model.eval()
    benar = total = 0
    matriks = np.zeros((len(KELAS), len(KELAS)), dtype=int)

    with torch.no_grad():
        for x, y in pemuat:
            x, y = x.to(perangkat), y.to(perangkat)
            tebakan = model(x).argmax(dim=1)
            benar += (tebakan == y).sum().item()
            total += y.numel()
            for asli, tebak in zip(y.cpu().numpy(), tebakan.cpu().numpy()):
                matriks[asli, tebak] += 1

    return benar / total, matriks


def main():
    torch.manual_seed(SEED)
    perangkat = "cuda" if torch.cuda.is_available() else "cpu"

    gambar, label, _nama = muat_semua()
    i_latih, i_val, i_uji = bagi_data(label)

    print("=" * 58)
    print(" TRAINING MODEL MATA (OPEN / CLOSED)")
    print("=" * 58)
    print(" Perangkat   :", perangkat)
    print(" Total data  : %d gambar" % len(label))
    for i, k in enumerate(KELAS):
        print("   %-7s   : %d" % (k, (label == i).sum()))
    print(" Data latih  : %d" % len(i_latih))
    print(" Data validasi: %d" % len(i_val))
    print(" Data uji    : %d" % len(i_uji))
    print("=" * 58)
    print("")

    pemuat_latih = DataLoader(DatasetMata(gambar[i_latih], label[i_latih], latih=True),
                              batch_size=BATCH, shuffle=True)
    pemuat_val   = DataLoader(DatasetMata(gambar[i_val], label[i_val]),
                              batch_size=BATCH)
    pemuat_uji   = DataLoader(DatasetMata(gambar[i_uji], label[i_uji]),
                              batch_size=BATCH)

    model = EyeNet().to(perangkat)
    print("Jumlah parameter:", sum(p.numel() for p in model.parameters()))
    print("")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    penjadwal = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH)
    kriteria  = nn.CrossEntropyLoss()

    riwayat = {"loss": [], "akurasi_val": []}
    akurasi_terbaik = 0.0
    path_model = os.path.join(BASE_DIR, "eyenet.pth")

    for epoch in range(1, EPOCH + 1):
        model.train()
        total_loss = 0.0
        for x, y in pemuat_latih:
            x, y = x.to(perangkat), y.to(perangkat)
            optimizer.zero_grad()
            rugi = kriteria(model(x), y)
            rugi.backward()
            optimizer.step()
            total_loss += rugi.item() * y.numel()

        penjadwal.step()
        loss_rata = total_loss / len(i_latih)
        akurasi_val, _ = evaluasi(model, pemuat_val, perangkat)

        riwayat["loss"].append(loss_rata)
        riwayat["akurasi_val"].append(akurasi_val)

        tanda = ""
        if akurasi_val > akurasi_terbaik:
            akurasi_terbaik = akurasi_val
            torch.save(model.state_dict(), path_model)
            tanda = "  <- disimpan"

        if epoch % 5 == 0 or epoch == 1 or tanda:
            print("Epoch %2d/%d  loss=%.4f  akurasi validasi=%.1f%%%s"
                  % (epoch, EPOCH, loss_rata, 100 * akurasi_val, tanda))

    # ---- nilai akhir memakai data uji ----
    model.load_state_dict(torch.load(path_model))
    akurasi_uji, matriks = evaluasi(model, pemuat_uji, perangkat)

    print("")
    print("=" * 58)
    print(" HASIL AKHIR")
    print("=" * 58)
    print(" Akurasi validasi terbaik : %.1f%%" % (100 * akurasi_terbaik))
    print(" Akurasi DATA UJI         : %.1f%%" % (100 * akurasi_uji))
    print("")
    print(" Matriks kebingungan (baris = label asli, kolom = tebakan):")
    print("           %s" % "  ".join("%8s" % k for k in KELAS))
    for i, k in enumerate(KELAS):
        print("   %-7s %s" % (k, "  ".join("%8d" % v for v in matriks[i])))
    print("")
    print(" Model tersimpan:", path_model)
    print("=" * 58)

    # ---- grafik ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (kiri, kanan) = plt.subplots(1, 2, figsize=(10, 4))
        kiri.plot(riwayat["loss"]);         kiri.set_title("Loss data latih")
        kiri.set_xlabel("epoch")
        kanan.plot([100 * a for a in riwayat["akurasi_val"]])
        kanan.set_title("Akurasi data validasi (%)"); kanan.set_xlabel("epoch")
        fig.tight_layout()
        fig.savefig(os.path.join(BASE_DIR, "grafik_training.png"), dpi=110)
        print(" Grafik tersimpan: grafik_training.png")
    except ImportError:
        print(" (matplotlib belum terpasang, grafik dilewati)")


if __name__ == "__main__":
    main()
