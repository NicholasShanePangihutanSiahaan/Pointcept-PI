# Konfirmasi multi-seed `drop_path=0.1`

## Tujuan

Mengonfirmasi apakah peningkatan validation allAcc pada seed 42 dari 78.71%
(`drop_path=0.3`) menjadi 79.21% (`drop_path=0.1`) konsisten pada seed lain.
Konfirmasi diperlukan karena kenaikannya hanya 0.50 poin dan macro F1 seed 42
menurun dari 62.48% menjadi 61.03%.

## Desain eksperimen

- Bandingkan `drop_path=0.1` dan kontrol `drop_path=0.3` pada seed 42, 123,
  dan 501.
- Parameter lain identik: Z-order, LR utama 5e-4, LR block 5e-5, patch size
  1024, batch size 4, 1024 titik, dan 100 epoch.
- Gunakan split tetap `train10_train`/`train10_val`.
- Official test tidak digunakan untuk seleksi.

## Menjalankan dua run yang belum tersedia

```bash
conda activate pointcept-torch2.5.0-cu12.4
cd ~/Pointcept
sh scripts/run_modelnet_dp01_multiseed.sh
```

Setelah selesai:

```bash
python tools/summarize_modelnet_dp01_multiseed.py
```

Pilih `drop_path=0.1` hanya jika peningkatan allAcc cukup konsisten dan metrik
pendukung tidak memburuk secara berarti. Jika hasil campuran atau selisih rerata
sangat kecil, pertahankan `0.3` sebagai pilihan yang lebih konservatif.
