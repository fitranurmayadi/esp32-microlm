# Kibo: Micro Language Model (Micro-LM) & Hybrid AI di ESP32-S3 (Edisi Bahasa Indonesia)

[![SoC: ESP32-S3](https://img.shields.io/badge/SoC-ESP32--S3-red.svg)](https://www.espressif.com/en/products/socs/esp32-s3)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](../LICENSE)
[![Platform: Arduino Core 3.x](https://img.shields.io/badge/Platform-Arduino%20Core%203.x-green.svg)](https://github.com/espressif/arduino-esp32)
[![Quantization: INT8](https://img.shields.io/badge/Kuantisasi-INT8-orange.svg)]()
[![PSRAM: 8MB Octal](https://img.shields.io/badge/PSRAM-8MB%20Octal%20OPI-purple.svg)]()

[English Global Edition](../README.md) | [Dokumen Arsitektur & Perjalanan Teknis](JOURNEY_ID.md) | [Profil Benchmark Ilmiah](../BENCHMARK.md)

Kibo Edisi Bahasa Indonesia adalah paket mandiri dari model bahasa generatif mikro (Micro-LM) dan mesin inferensi berbasis C++ bare-metal yang berjalan secara lokal di mikrokontroler ESP32-S3.

Seluruh proses inferensi dieksekusi 100% on-device tanpa ketergantungan eksternal. Pada **v2.0 / v3.0**, engine ini memanfaatkan **komputasi paralel Dual-Core FreeRTOS** (Core 0 + Core 1 @ 240MHz) dan **kernel unrolling 8-way** untuk menghasilkan kecepatan generasi riil **14.5–15.6 token/detik** pada hardware fisik (~65ms total per token: 30ms streaming bus Octal PSRAM + 35ms komputasi dual-core CPU).

---

## Demonstrasi Nyata di Hardware ESP32-S3

![Kibo ESP32-S3 Live Demo](../testing_video/kibo_terminal_cropped.gif)

> **Video Demonstrasi Lengkap**: [`../testing_video/ujicoba_kibo.mp4`](../testing_video/ujicoba_kibo.mp4) (Sesi live UART terminal 1080p @ 60 FPS langsung di ESP32-S3 DevKit N16R8).

---

## Fitur Arsitektur

* **Engine Paralel Dual-Core FreeRTOS (v2.0)**: Membagi beban Multi-Head Attention dan proyeksi MLP 768-dim secara simetris di Core 0 (PRO_CPU) dan Core 1 (APP_CPU) @ 240MHz.
* **Vektorisasi 4-Way 32-Bit Unrolled**: Membaca bobot dalam chunk 32-bit terkemas dengan unrolling FPU untuk menghilangkan jeda pipeline.
* **Hierarki Memori Ketat**: Buffer aktivasi panas terkunci di SRAM internal cepat (240 MB/s), bobot INT8 dan KV-cache 128 token di Octal PSRAM (80 MB/s).
* **Eksekusi Hybrid AI**: Menggabungkan kemampuan generasi autoregresif teks (persona dan tag kendali emosi) dengan parser deterministik (kalkulator instan, telemetri status, dan hook aktuator robot).
* **Dukungan Multi-Board**: Telah tervalidasi di ESP32-S3 DevKit N16R8, Arduino Nano ESP32, dan Seeed Studio XIAO ESP32-S3.

---

## Tabel Versi Rilis & Roadmap

| Versi | Framework | Arsitektur Engine & Paralelisme | Kecepatan Riil Hardware | Batas Teoretis (Hanya Komputasi) | Git Branch & Release Tag |
| :--- | :---: | :--- | :---: | :---: | :--- |
| **v1.0** | Arduino Core 3.x | Single-Core 240MHz Baseline | **~12.3 tok/s** | ~14.1 tok/s | [`release/v1.0`](https://github.com/fitranurmayadi/esp32-microlm/tree/release/v1.0) / `v1.0` |
| **v2.0** | Arduino Core 3.x | Dual-Core FreeRTOS + 8-Way Unrolled | **14.2 – 15.6 tok/s** | ~18.5 tok/s | [`release/v2.0`](https://github.com/fitranurmayadi/esp32-microlm/tree/release/v2.0) / `v2.0` |
| **v3.0** | ESP-IDF Native v5.x | Native FreeRTOS + Dual-Core Task Pinning | **14.5 – 15.8 tok/s** | ~22.4 tok/s | [`release/v3.0`](https://github.com/fitranurmayadi/esp32-microlm/tree/release/v3.0) / `v3.0` |
| **v4.0 (Terbaru)** | ESP-IDF Native v5.x | **Gemma 3n Per-Layer Embeddings (PLE) Hybrid** | **⚡ 14.2 – 15.8 tok/s** | ~22.4 tok/s | [`release/v4.0`](https://github.com/fitranurmayadi/esp32-microlm/tree/release/v4.0) / `v4.0` |

---

## 🔬 Perbandingan Ilmiah vs Proyek Open-Source Lain

| Metrik / Parameter | **DaveBben (`esp32-llm`)** | **slvDev (`esp32-ai`)** | **Kibo Micro-LM v4.0 (Karya Kita)** |
| :--- | :---: | :---: | :---: |
| **Core Penalaran Aktif** | 260K parameter (tinyllamas) | 559K parameter (Gemma 3n PLE) | **1.84M Parameter Dense** |
| **Kapasitas Memori Tersimpan** | 260K parameter (1.04 MB FP32) | 28.9M (25.2M Tabel Flash) | **10M–25M Flash PLE + 1.84M PSRAM** |
| **Format Bobot** | FP32 (Float 4-byte) | INT4 (4-bit PTQ group 32) | **INT8 W8A32 Simetris** |
| **Kecepatan Riil di Silicon** | **19.13 tok/s** *(pada 260K)* | **9.5 – 9.88 tok/s** *(pada 559K)* | **⚡ 14.2 – 15.8 tok/s** *(pada 1.84M)* |
| **Throughput Komputasi Total** | $4.97\text{ Juta ops/detik}$ | $5.31\text{ Juta ops/detik}$ | $\mathbf{28.7\text{ Juta ops/detik}}$ 🚀 |
| **Efisiensi vs DaveBben** | 1.0x (Baseline model kecil) | 1.07x | **5.77x Lebih Padat Komputasi** |
| **Tool Calling Deterministik** | ❌ Tidak Ada | ❌ Tidak Ada | **✅ Aritmatika Instan (<0.1ms) + Robotik** |

---

## Struktur Direktori Edisi Bahasa Indonesia

```
indonesian_edition/
├── kibo_esp32_id/               # Firmware C++ ESP32 Edisi Bahasa Indonesia
│   ├── kibo_esp32_id.ino        # Setup dan serial interface loop
│   ├── kibo_inference.h         # Definisi struktur tensor dan model
│   ├── kibo_inference.cpp       # MatMul INT8, Attention & Tool Calling Bahasa Indonesia
│   ├── kibo_model_data.S        # Embedded model assembly (.rodata)
│   ├── kibo_model_int8.bin      # Bobot model INT8 (1.79 MB)
│   ├── kibo_vocab.h             # Kosakata tokenizer Bahasa Indonesia
│   └── serial_chat_id.py        # CLI Terminal Chat Bahasa Indonesia
├── kibo_microlm_id/             # Pipeline Pelatihan & Kuantisasi Python
│   ├── train_id.py              # Pelatihan model PyTorch
│   ├── export_int8_id.py        # Ekspor binary INT8 dan header C++
│   ├── kibo_dataset_id.txt      # Dataset dialog Bahasa Indonesia (~170+ dialog)
│   ├── kibo_vocab.json          # Tabel lookup kosakata
│   └── kibo_model_int8.bin      # Model INT8 hasil ekspor
├── flash_devkit_id.sh           # Skrip flash untuk ESP32-S3 DevKit N16R8
├── flash_nano_id.sh             # Skrip flash untuk Arduino Nano ESP32
├── flash_xiao_id.sh             # Skrip flash untuk Seeed Studio XIAO ESP32-S3
├── chat_kibo_id.sh              # Skrip live terminal chat Bahasa Indonesia
├── README_ID.md                 # Dokumentasi Bahasa Indonesia
└── JOURNEY_ID.md                # Catatan Teknis Bahasa Indonesia
```

---

## Panduan Menjalankan

### 1. Flash ke Board Target

* **Seeed Studio XIAO ESP32-S3**:
  ```bash
  ./indonesian_edition/flash_xiao_id.sh
  ```
* **ESP32-S3 DevKit N16R8**:
  ```bash
  ./indonesian_edition/flash_devkit_id.sh
  ```
* **Arduino Nano ESP32**:
  ```bash
  ./indonesian_edition/flash_nano_id.sh
  ```

### 2. Membuka Terminal Chat
```bash
./indonesian_edition/chat_kibo_id.sh
```

---

## Kustomisasi Persona & Dataset Sendiri

Seluruh alur pelatihan dan ekspor bersifat mandiri (*self-contained*), sehingga siapa pun dapat melatih persona, nama robot, bahasa, atau karakter percakapan sendiri dalam 3 langkah mudah:

### 1. Edit Dataset Percakapan (`kibo_microlm/kibo_dataset.txt`)
Tambahkan dialog percakapan Anda menggunakan format terstruktur dan tag emosi:
```text
User: halo
Kibo: [HAPPY] Halo sahabatku! Senang bertemu denganmu! <EOS>

User: siapa pembuatmu?
Kibo: [NEUTRAL] Aku dibuat menggunakan Micro-LM langsung di ESP32-S3! <EOS>
```
*Tag emosi (`[HAPPY]`, `[NEUTRAL]`, `[ANGRY]`, `[THINKING]`) juga dapat dimanfaatkan sebagai pemicu perangkat keras untuk menampilkan animasi mata di layar LCD SPI.*

### 2. Latih Ulang Model
Jalankan pelatihan model 1.84M Causal Transformer pada dataset baru (hanya butuh ~2 menit di CPU/GPU biasa):
```bash
python3 kibo_microlm/train.py
```

### 3. Ekspor & Flash ke ESP32
Kuantisasi bobot baru ke INT8 dan unggah langsung ke board Anda:
```bash
python3 kibo_microlm/export_int8_bin.py
./flash_devkit_s3.sh
```

---

## Contoh Interaksi

```text
kibo-mcu [1.84M params | int8 | 8mb psram]
terhubung di /dev/ttyUSB0 (115200 baud)

User: halo kibo
Kibo: [HAPPY] Halo sahabatku! Ada yang bisa Kibo bantu? Ayo main bareng!

User: siapa kamu?
Kibo: [NEUTRAL] Aku Kibo! Robot mini desktop pintar dan sahabat terbaikmu!

User: berapa 100 kali 100?
[calc: 100 * 100 = 10000]
Kibo: [CALC: 100 * 100] [HAPPY] Hasil dari 100 * 100 adalah 10000! Kibo pintar berhitung kan!

User: dadah kibo
Kibo: [NEUTRAL] Sampai jumpa sahabatku! Tetap semangat dan jaga kesehatan ya!
```

---

## Status Proyek & Milestone Pengembangan

### Milestone v1.0 (Fondasi Awal) — ✅ Selesai
* [x] **Engine Bare-Metal**: Forward pass Causal Transformer 1.84M murni C++ tanpa dependensi framework ML.
* [x] **Efisiensi Memori**: Kuantisasi bobot INT8 (W8A32) di Octal PSRAM (ukuran file 1.79 MB).
* [x] **Manajemen KV-Cache**: Buffer konteks 128 token dinamis di PSRAM dengan mekanisme fallback SRAM.
* [x] **Kecepatan Real-Time**: Terukur stabil di ~12.3 token/detik pada perangkat keras ESP32-S3 (240MHz).
* [x] **Arsitektur Hybrid**: Eksekusi aritmatika deterministik instan (<0.1 ms) untuk mencegah halusinasi angka.
* [x] **Dukungan Multi-Board**: Tervalidasi pada DevKit N16R8, Arduino Nano ESP32, dan Seeed XIAO S3.

### Milestone v2.0 (Dual-Core Berkecepatan Tinggi & Hierarki Memori) — ✅ Selesai
* [x] **Paralelisme Dual-Core FreeRTOS**: Pembagian beban komputasi matriks MLP dan Multi-Head Attention di Core 0 (PRO_CPU) dan Core 1 (APP_CPU).
* [x] **Vektorisasi Unrolling 4-Way 32-Bit**: Pembacaan bobot terkemas 32-bit dengan 4 register FPU akumulasi independen.
* [x] **Hierarki Memori Ketat**: Buffer aktivasi panas terkunci di SRAM internal (240 MB/s), bobot dan KV-cache di Octal PSRAM (80 MB/s).
* [x] **Telemetri Hardware & Hook Aksi**: Diagnostik RTOS on-chip (`status`), perintah tampilan ekspresi mata, dan aktuasi servo.
* [x] **Suite Benchmark Ilmiah**: Pengukuran empiris lengkap terdokumentasi di [`BENCHMARK.md`](../BENCHMARK.md).

---

## 🔖 Versi Rilis & Strategi Percabangan Git (Branching)

| Branch / Tag | Framework | Engine Inferensi | Kecepatan Riil Hardware | Batas Teoretis Siklus ALU | Deskripsi |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **`main`** | Universal | Dual-Core 8-Way FPU + Port ESP-IDF | **⚡ 14.5 – 15.6 tok/s** | ~22 tok/s | Rilis default produksi dengan alat matematika bilingual & telemetri |
| **[`release/v3.0`](https://github.com/fitranurmayadi/esp32-microlm/tree/release/v3.0)** (Tag `v3.0`) | ESP-IDF v5.x Native | Dual-Core Paralel + Coprocessor CP0 | **⚡ 14.5 – 15.6 tok/s** | ~22 tok/s | Rilis milestone v3.0 native ESP-IDF dengan dukungan CP0 SIMD |
| **[`release/v2.0`](https://github.com/fitranurmayadi/esp32-microlm/tree/release/v2.0)** (Tag `v2.0`) | Arduino Core 3.x | Dual-Core FreeRTOS + 8-Way FPU | **⚡ 14.2 – 15.6 tok/s** | ~18 tok/s | Rilis milestone v2.0 dual-core 240MHz paralel |
| **[`release/v1.0`](https://github.com/fitranurmayadi/esp32-microlm/tree/release/v1.0)** (Tag `v1.0`) | Arduino Core 3.x | Single-Core Skalar W8A32 | **~12.3 tok/s** | ~14 tok/s | Rilis milestone v1.0 fondasi awal single-core |

> **Catatan Fisika Latensi (1.84M Parameter)**: Setiap 1 token yang digenerasikan membaca 1.79 MB data bobot model melewati bus Octal PSRAM SPI 80MHz (transit memori ~30 ms) ditambah komputasi dual-core CPU (~35 ms), menghasilkan batas throughput fisik terukur ~65 ms per token (14.5–15.6 token/detik).


---

## Lisensi

Didistribusikan di bawah lisensi MIT. Lihat file [`LICENSE`](../LICENSE) untuk detail lengkap.

