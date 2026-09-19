# Protocol: PTv3 drop-path ablation

## Pertanyaan eksperimen

Apakah pengurangan stochastic-depth regularization (`drop_path`) meningkatkan
generalisasi konfigurasi terbaik Z-order + learning rate 5e-4?

`drop_path` adalah probabilitas menjatuhkan residual branch selama training.
Nilai lebih besar memberi regularisasi lebih kuat. Nilai ini tidak mengurangi
arsitektur atau jumlah parameter saat inferensi.

## Variabel terkontrol

- Model: PT-v3m1 classifier, Z-order, initial order shuffle dinonaktifkan.
- Learning rate utama/block: 5e-4/5e-5.
- Patch size: 1024; seed: 42.
- Split tetap: 798 train dan 202 validation; 1024 titik per objek.
- Batch size 4, workers 2, epoch 100, augmentasi dan loss tetap.
- Official test tidak digunakan dalam pemilihan hyperparameter.

## Kelompok eksperimen

| Run | drop_path | Status awal |
|---|---:|---|
| `ordering-z-lr5e4-dp01-s42` | 0.1 | baru |
| `ordering-z-lr5e4-dp02-s42` | 0.2 | baru |
| `ordering-z-lr5e4-s42` | 0.3 | kontrol selesai |

## Menjalankan dan merangkum

```bash
conda activate pointcept-torch2.5.0-cu12.4
cd ~/Pointcept
sh scripts/run_modelnet_drop_path_ablation.sh
python tools/summarize_modelnet_drop_path_ablation.py
```

Dua run baru diperkirakan membutuhkan total sekitar 1,5 jam pada RTX 3060
Laptop. Pilih kandidat berdasarkan validation `allAcc`, dengan `mAcc` dan macro
F1 sebagai metrik pendukung. Ini adalah screening satu seed. Jika nilai selain
0.3 menang secara berarti, konfirmasi kandidat tersebut pada seed 123 dan 501
sebelum mengganti konfigurasi final.
