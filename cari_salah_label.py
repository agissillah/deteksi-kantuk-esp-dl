"""
cari_salah_label.py
-------------------
Mencari gambar yang kemungkinan SALAH LABEL, tanpa perlu memeriksa
seluruh dataset satu per satu.

Caranya (teknik ini namanya cross-validation / confident learning):
  1. Dataset dibagi 5 bagian.
  2. Model dilatih dengan 4 bagian, lalu menebak 1 bagian yang belum pernah
     dilihatnya. Diulang 5 kali sampai semua gambar pernah ditebak.
  3. Gambar yang ditebak model BERBEDA dari labelnya, dengan keyakinan
     tinggi, ditandai sebagai calon salah label.

Hasilnya disimpan sebagai:
  - laporan_salah_label.txt  (daftar nama berkas)
  - laporan_salah_label.png  (grid gambar untuk diperiksa mata sendiri)

Program ini TIDAK memindahkan atau menghapus apa pun.

Cara pakai:
    python cari_salah_label.py
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import EyeNet, KELAS
from dataset_loader import muat_semua, DatasetMata

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

JUMLAH_BAGIAN = 5        # dataset dibagi berapa bagian
EPOCH         = 12       # berapa kali model melihat seluruh data latih
BATCH         = 64
AMBANG        = 0.90     # tebakan dianggap "yakin" kalau peluangnya di atas ini


def latih_sekali(gambar, label, indeks_latih, indeks_uji, perangkat):
    """Melatih model dengan data latih, lalu menebak data uji."""
    model = EyeNet().to(perangkat)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    kriteria  = nn.CrossEntropyLoss()

    pemuat_latih = DataLoader(
        DatasetMata(gambar[indeks_latih], label[indeks_latih], latih=True),
        batch_size=BATCH, shuffle=True)
    pemuat_uji = DataLoader(
        DatasetMata(gambar[indeks_uji], label[indeks_uji], latih=False),
        batch_size=BATCH, shuffle=False)

    model.train()
    for _ in range(EPOCH):
        for x, y in pemuat_latih:
            x, y = x.to(perangkat), y.to(perangkat)
            optimizer.zero_grad()
            rugi = kriteria(model(x), y)
            rugi.backward()
            optimizer.step()

    model.eval()
    peluang = []
    with torch.no_grad():
        for x, _ in pemuat_uji:
            keluaran = model(x.to(perangkat))
            peluang.append(torch.softmax(keluaran, dim=1).cpu().numpy())

    return np.concatenate(peluang)


def main():
    perangkat = "cuda" if torch.cuda.is_available() else "cpu"
    gambar, label, nama_berkas = muat_semua()
    print("Total gambar:", len(gambar), "| perangkat:", perangkat)
    for i, k in enumerate(KELAS):
        print("   %-7s: %d" % (k, (label == i).sum()))
    print("")

    # bagi data secara acak tapi tetap seimbang antar kelas
    rng = np.random.default_rng(42)
    bagian = np.zeros(len(label), dtype=int)
    for kelas in np.unique(label):
        indeks = np.where(label == kelas)[0]
        rng.shuffle(indeks)
        bagian[indeks] = np.arange(len(indeks)) % JUMLAH_BAGIAN

    semua_peluang = np.zeros((len(label), len(KELAS)), dtype=np.float32)

    for b in range(JUMLAH_BAGIAN):
        indeks_uji   = np.where(bagian == b)[0]
        indeks_latih = np.where(bagian != b)[0]
        print("[%d/%d] melatih dengan %d gambar, menebak %d gambar ..."
              % (b + 1, JUMLAH_BAGIAN, len(indeks_latih), len(indeks_uji)))
        semua_peluang[indeks_uji] = latih_sekali(
            gambar, label, indeks_latih, indeks_uji, perangkat)

    tebakan = semua_peluang.argmax(axis=1)
    akurasi = (tebakan == label).mean()
    print("")
    print("Akurasi model terhadap label saat ini: %.1f%%" % (100 * akurasi))

    # calon salah label: tebakan berbeda DAN model yakin
    peluang_tebakan = semua_peluang.max(axis=1)
    curiga = np.where((tebakan != label) & (peluang_tebakan > AMBANG))[0]
    curiga = curiga[np.argsort(-peluang_tebakan[curiga])]     # paling yakin di atas

    print("Calon salah label: %d gambar (%.1f%% dari dataset)"
          % (len(curiga), 100 * len(curiga) / len(label)))
    print("")

    # ---- simpan laporan teks ----
    path_teks = os.path.join(BASE_DIR, "laporan_salah_label.txt")
    with open(path_teks, "w", encoding="utf-8") as f:
        f.write("Daftar gambar yang kemungkinan salah label\n")
        f.write("(diurutkan dari yang paling meyakinkan)\n\n")
        f.write("%-28s %-8s %-8s %s\n" % ("berkas", "label", "tebakan", "keyakinan"))
        for i in curiga:
            f.write("%-28s %-8s %-8s %.3f\n"
                    % (nama_berkas[i], KELAS[label[i]], KELAS[tebakan[i]],
                       peluang_tebakan[i]))
    print("Laporan teks  :", path_teks)

    # ---- simpan grid gambar ----
    if len(curiga) > 0:
        KOLOM, THUMB = 10, 96
        petak = []
        for i in curiga[:200]:                      # tampilkan maksimal 200
            g = cv2.imread(os.path.join(BASE_DIR, "dataset", nama_berkas[i]),
                           cv2.IMREAD_GRAYSCALE)
            g = cv2.cvtColor(cv2.resize(g, (THUMB, THUMB)), cv2.COLOR_GRAY2BGR)
            cv2.rectangle(g, (0, 0), (THUMB, 14), (0, 0, 0), -1)
            cv2.putText(g, nama_berkas[i].split("_")[-1][:-4], (2, 11),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
            petak.append(g)
        while len(petak) % KOLOM:
            petak.append(np.zeros((THUMB, THUMB, 3), np.uint8))
        grid = np.vstack([np.hstack(petak[i:i + KOLOM])
                          for i in range(0, len(petak), KOLOM)])
        path_gambar = os.path.join(BASE_DIR, "laporan_salah_label.png")
        cv2.imwrite(path_gambar, grid)
        print("Laporan gambar:", path_gambar)

    print("")
    print("Langkah berikutnya: buka laporan gambar, lalu perbaiki yang memang")
    print("salah lewat  python bersihkan_dataset.py OPEN  (atau CLOSED).")


if __name__ == "__main__":
    main()
