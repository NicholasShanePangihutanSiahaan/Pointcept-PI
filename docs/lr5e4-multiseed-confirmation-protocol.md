# Konfirmasi Multi-seed Learning Rate 5e-4

## Tujuan

Memastikan peningkatan Z-order dengan learning rate utama `5e-4` tidak hanya
terjadi pada seed 42. Kandidat dibandingkan secara berpasangan terhadap kontrol
Z-order learning rate `1e-3` pada seed 42, 123, dan 501.

## Variabel terkontrol

- Dataset train/validation: `train10_train` (798 objek) dan `train10_val`
  (202 objek), hasil stratified split yang sama.
- Arsitektur: PT-v3m1 encoder, patch size 1024, Z-order, tanpa initial-order
  shuffle.
- Epoch: 100; batch size: 4; jumlah titik per objek: 1024.
- Evaluasi pemilihan model hanya memakai validation set. Official test set tidak
  digunakan dalam tahap tuning ini.
- Satu-satunya perbedaan dalam setiap pasangan adalah learning rate. Antar-run
  dalam satu setting hanya berbeda pada seed.

## Setting

| Setting | Main LR | Block LR | Seed |
|---|---:|---:|---|
| Kontrol | 1e-3 | 1e-4 | 42, 123, 501 |
| Kandidat | 5e-4 | 5e-5 | 42, 123, 501 |

## Menjalankan eksperimen yang belum ada

```bash
conda activate pointcept-torch2.5.0-cu12.4
sh scripts/run_modelnet_lr5e4_multiseed.sh
```

Runner akan menjalankan kandidat seed 123 lalu 501. Seed 42 dan seluruh kontrol
sudah tersedia dari eksperimen sebelumnya. Perkiraan waktu dua run sekitar 1,5
jam pada RTX 3060 Laptop berdasarkan durasi run sebelumnya.

## Melihat hasil

```bash
python tools/summarize_modelnet_lr5e4_multiseed.py
```

Kandidat diterima jika rerata validation allAcc meningkat dan hasilnya tidak
bergantung pada satu seed saja. Laporkan rerata dan sample standard deviation
dari ketiga seed untuk allAcc, mAcc, dan macro F1.
