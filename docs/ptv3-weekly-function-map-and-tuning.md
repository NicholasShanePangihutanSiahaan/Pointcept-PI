# Pemetaan fungsi PTv3 dan rencana tuning mingguan

## Ruang lingkup

Tugas: memahami alur klasifikasi ModelNet40 dan menguji pengaruh satu
hyperparameter, yaitu `enc_patch_size`. Ini adalah hyperparameter tuning,
bukan fine-tuning dari pretrained checkpoint atau modifikasi rumus QKV.

Dataset tetap: 798 train (`train10_train`), 202 validation (`train10_val`),
dan 2.468 official test. Seed training 42, input 1.024 titik per objek,
batch size 4, epoch 100 dan eval_epoch 100. Jangan membuat ulang split
selama perbandingan eksperimen.

## Alur pemanggilan

```text
scripts/train.sh
  -> tools/train.py: main()
  -> engines/defaults.py: default_config_parser()
  -> engines/launch.py: launch()
  -> tools/train.py: main_worker()
  -> engines/train.py: Trainer.__init__()
       -> build_model(), build_train_loader(), build_val_loader()
       -> build_optimizer(), build_scheduler(), build_scaler()
  -> Trainer.train()
       -> Dataset.__getitem__() -> transforms -> batch
       -> Trainer.run_step()
            -> DefaultClassifier.forward()
                 -> PointTransformerV3.forward()
                 -> pooling fitur objek -> classification head -> loss
            -> backward -> optimizer step -> scheduler step
       -> hooks setelah epoch: validation dan checkpoint
  -> exp/modelnet40/<nama-run>/
```

`scripts/train.sh` menyalin kode ke folder `exp/.../code/`. Training memakai
snapshot itu. Perubahan source setelah run dimulai tidak mengubah run aktif.

## Peta fungsi utama

Path di bawah relatif terhadap root repository. `forward()` berarti
perhitungan ketika data melewati sebuah modul PyTorch.

| File / fungsi | Input | Proses / fungsi | Output / efek |
|---|---|---|---|
| `tools/train.py: main()` | Argumen CLI | Membaca config dan meluncurkan worker GPU | Proses training |
| `tools/train.py: main_worker(cfg)` | Config | Setup environment run dan membangun trainer | Memanggil `trainer.train()` |
| `engines/defaults.py: default_config_parser()` | Path config dan overrides | Menggabungkan config, menentukan seed jika kosong, menetapkan train.loop | Config run dan salinannya |
| `engines/defaults.py: default_setup()` | Config | Menetapkan batch/worker per GPU dan seed | Environment eksperimen |
| `datasets/modelnet.py: get_data_list()` | Nama split | Membaca `modelnet40_<split>.txt` | Daftar nama objek |
| `ModelNetDataset.__init__()` | Root, split, num_points, transform | Mempersiapkan daftar dan membaca/membuat record cache | Dataset siap dipakai |
| `ModelNetDataset.get_data(idx)` | Indeks objek | Membaca cache atau XYZ+normal, sampling titik dan label | Dictionary coord, normal, category |
| `ModelNetDataset.__len__()` | Dataset | Mengalikan jumlah objek dengan loop | Panjang dataset |
| `ModelNetDataset.__getitem__(idx)` | Indeks | Memilih jalur train/validation atau test | Sampel yang sudah disiapkan |
| `prepare_train_data(idx)` | Indeks | Mengambil data dan menerapkan transform | Tensor fitur dan label |
| `prepare_test_data(idx)` | Indeks | Membuat beberapa versi augmentasi test | Voting list dan kategori |
| `datasets/transform.py: NormalizeCoord` | Koordinat | Normalisasi bentuk | Koordinat ternormalisasi |
| `RandomScale`, `RandomShift` | Koordinat | Augmentasi ukuran dan translasi | Contoh training beragam |
| `GridSample` | Koordinat dan fitur | Sampling berbasis grid, membentuk grid_coord | Point cloud tersampling |
| `ToTensor`, `Collect` | Dictionary data | Konversi tensor dan memilih field/fitur | Input model |
| `models/default.py: DefaultClassifier.forward()` | Dictionary batch | Backbone, agregasi fitur per objek, head dan loss sesuai mode | Loss saat training / logits saat inferensi |
| `models/utils/structure.py: Point.serialization()` | Grid coordinates, batch, order | Encode dan sort titik | Serialized code, order, inverse |
| `models/utils/serialization/default.py: encode()` | Grid coordinates dan nama order | Memilih Z atau Hilbert, termasuk pertukaran sumbu untuk trans | Kode spasial integer |
| `point_transformer_v3m1_base.py: Embedding.forward()` | Point dengan fitur awal | Sparse convolution embedding | Point dengan fitur channel awal |
| `PointTransformerV3.forward()` | Dictionary point cloud | Serialization, sparsify, embedding dan encoder; decoder jika diperlukan | Point dengan fitur backbone |
| `Block.forward()` | Point features | CPE, attention, MLP, normalisasi dan residual | Fitur hasil satu transformer block |
| `SerializedAttention.get_padding_and_inverse()` | Offset titik setiap objek | Membentuk patch/padding dan pemetaan balik | pad, unpad, cu_seqlens |
| `SerializedAttention.forward()` | Fitur dan serialized order | Proyeksi QKV, attention dalam patch, proyeksi output | Fitur hasil attention |
| `MLP.forward()` | Fitur per titik | Linear -> GELU -> linear dan dropout | Fitur dengan dimensi output |
| `SerializedPooling.forward()` | Point resolusi saat ini | Pengelompokan hierarkis dan agregasi fitur | Point cloud dengan lebih sedikit titik |
| `SerializedUnpooling.forward()` | Fitur decoder dan parent | Upsampling dengan skip feature | Resolusi lebih tinggi; tidak digunakan saat enc_mode=True |
| `engines/train.py: build_model()` | model config | Membentuk model dan menempatkannya di GPU | Model trainable |
| `build_train_loader()`, `build_val_loader()` | data config | Membentuk dataset, sampler, DataLoader | Batch iterator |
| `build_optimizer()`, `build_scheduler()` | Config dan parameter model | Menyusun optimizer serta jadwal LR | Komponen update bobot |
| `Trainer.train()` | Trainer siap | Iterasi epoch/batch dan pemanggilan hooks | Training lengkap |
| `Trainer.run_step()` | Batch | Forward, backward, update optimizer dan scheduler | Bobot baru dan loss |
| `hooks/evaluator.py: ClsEvaluator` | Model dan validation loader | Prediksi serta agregasi confusion statistics | Validation allAcc, mAcc, F1 dan metrik lain |
| `hooks/misc.py: CheckpointSaver` | Epoch dan metrik validation | Menyimpan model terakhir/terbaik | model_last.pth, model_best.pth |
| `hooks/misc.py: PreciseEvaluator.after_train()` | Model/checkpoint dan test config | Testing otomatis setelah training | Official test jika data.test menunjuk test |
| `engines/test.py: ClsVotingTester.test()` | Checkpoint dan test dataset | Evaluasi berulang dengan augmentasi | Metrik voting test |

Semua path `engines/`, `datasets/`, dan `models/` dalam tabel berada di
`pointcept/`. Sampler FPS pada `get_data()` memilih titik dalam satu objek;
program `tools/create_modelnet_subset.py` memilih objek antar-dataset.
Keduanya merupakan sampling pada level yang berbeda.

## Hyperparameter yang dipelajari minggu ini

`enc_patch_size`: batas jumlah titik dalam satu kelompok serialized attention.
Nilai kecil memberi neighborhood lebih lokal; nilai besar mencakup lebih
banyak titik. Attention dibatasi patch, bukan selalu seluruh scene.

Dengan Flash Attention, nilai ini menjadi batas panjang patch; pada stage
yang jumlah titiknya lebih sedikit, perubahan patch size dapat memiliki
pengaruh kecil. Karena input 1.024 titik dan pooling mengurangi titik,
pengaruh paling jelas diperkirakan terjadi pada stage awal. Ini hipotesis,
bukan jaminan bahwa patch kecil selalu lebih cepat atau lebih akurat.

## Eksperimen terkendali

| ID | Config | Patch size semua encoder stage | Nama run |
|---|---|---|---|
| B0 | Baseline split yang sudah selesai | 1024 | baseline-10pct-split |
| P128 | cls-ptv3-v1m1-0-laptop-patch128 | 128 | tuning-patch128-s42 |
| P256 | cls-ptv3-v1m1-0-laptop-patch256 | 256 | tuning-patch256-s42 |
| P512 | cls-ptv3-v1m1-0-laptop-patch512 | 512 | tuning-patch512-s42 |

Ketiga config tuning mewarisi `cls-ptv3-v1m1-0-laptop-tuning.py`, yang
menghilangkan hook `PreciseEvaluator`. Validation tetap berjalan setiap
epoch; official test tidak dijalankan otomatis. Model tidak berubah karena
penghilangan hook tersebut. Baseline lama dan checkpointnya tetap utuh.

Jalankan satu per satu, jangan bersamaan pada RTX 3060 6 GB:

```bash
conda activate pointcept-torch2.5.0-cu12.4
sh scripts/train.sh -g 1 -d modelnet40 -c cls-ptv3-v1m1-0-laptop-patch128 -n tuning-patch128-s42
sh scripts/train.sh -g 1 -d modelnet40 -c cls-ptv3-v1m1-0-laptop-patch256 -n tuning-patch256-s42
sh scripts/train.sh -g 1 -d modelnet40 -c cls-ptv3-v1m1-0-laptop-patch512 -n tuning-patch512-s42
```

Setiap config hanya mengubah patch size, dengan initialization dari awal
(`weight=None`, `resume=False`). Jangan memakai checkpoint baseline untuk
run ini: itu akan mencampur efek hyperparameter dengan efek initialization.

Cari hasil di `exp/modelnet40/<nama-run>/train.log`:

```bash
rg 'Best validation|Val result' exp/modelnet40/tuning-patch128-s42/train.log
```

## Tabel hasil yang diisi

| Run | Best val allAcc | mAcc pada epoch terbaik | Best epoch | Durasi training | Catatan |
|---|---|---|---|---|---|
| Baseline 1024 | 75,25% | 60,81% | 73 | sekitar 47 menit 36 detik | Seed 42 |
| Patch 128 | Belum dijalankan | — | — | — | — |
| Patch 256 | Belum dijalankan | — | — | — | — |
| Patch 512 | Belum dijalankan | — | — | — | — |

Durasi baseline dihitung dari Start Training 22:28:24 sampai validation
epoch terakhir 23:16:00, tidak termasuk persiapan data dan voting test.
Bandingkan durasi menggunakan batas yang sama untuk semua run.

Pilih konfigurasi berdasarkan validation, bukan angka official test.
Validation hanya 202 objek: satu prediksi benar tambahan mengubah allAcc
sekitar 0,495 poin persentase. Perbedaan kecil pada satu seed belum membuktikan
peningkatan; ulangi baseline/model terbaik dengan beberapa seed kemudian.

Log voting menampilkan hasil terbaik dari beberapa evaluasi. Untuk laporan
final, tetapkan aturan agregasi terlebih dahulu dan terapkan identik pada
semua model; hindari memilih voting berdasarkan skor test tertinggi saja.

## Target laporan minggu ini

1. Peta fungsi utama beserta input, output dan hubungan pemanggilan.
2. Penjelasan QKV, serialization, patch attention dan hierarchical pooling.
3. Tiga hasil tuning patch size dengan seluruh kondisi lain tetap.
4. Tabel/kurva validation loss dan allAcc, serta durasi training.
5. Kesimpulan sementara: nilai patch mana yang paling menjanjikan dan alasan
   geometris/komputasinya; jangan mengklaim peningkatan bila belum teruji.

Jadwal: hari 1-2 memahami peta fungsi, hari 3 menjalankan P128, hari 4 P256,
hari 5 P512, hari 6 analisis tabel/kurva, hari 7 menulis laporan.
