# Kibo: Micro Language Model (Micro-LM) & Hybrid AI di ESP32-S3 (Edisi Bahasa Indonesia)

[![SoC: ESP32-S3](https://img.shields.io/badge/SoC-ESP32--S3-red.svg)](https://www.espressif.com/en/products/socs/esp32-s3)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](../LICENSE)
[![Platform: Arduino Core 3.x](https://img.shields.io/badge/Platform-Arduino%20Core%203.x-green.svg)](https://github.com/espressif/arduino-esp32)
[![Quantization: INT8](https://img.shields.io/badge/Kuantisasi-INT8-orange.svg)]()
[![PSRAM: 8MB Octal](https://img.shields.io/badge/PSRAM-8MB%20Octal%20OPI-purple.svg)]()

[English Global Edition](../README.md) | [Dokumen Arsitektur & Perjalanan Teknis](JOURNEY_ID.md)

Kibo Edisi Bahasa Indonesia adalah paket mandiri dari model bahasa generatif mikro (Micro-LM) dan mesin inferensi berbasis C++ bare-metal yang berjalan secara lokal di mikrokontroler ESP32-S3.

Seluruh proses inferensi dieksekusi 100% on-device tanpa jaringan internet atau server eksternal. Perkalian matriks INT8, dynamic KV-cache, dan evaluasi kalkulator hardware dijalankan langsung pada prosesor Xtensa LX7 dual-core dengan kecepatan **~12–13 token/detik** (~80 ms per token).

---

## Demonstrasi Nyata di Hardware ESP32-S3

![Kibo ESP32-S3 Live Demo](../testing_video/kibo_demo.gif)

> **Video Demonstrasi Lengkap**: [`../testing_video/ujicoba_kibo.mp4`](../testing_video/ujicoba_kibo.mp4) (Sesi live UART terminal 1080p @ 60 FPS langsung di ESP32-S3 DevKit N16R8).

---

## Fitur Arsitektur

* **Transformer C++ Bare-Metal**: Implementasi mandiri 4-Layer Causal Transformer (Self-Attention, LayerNorm, GeLU, MatMul INT8, dan KV-Cache) tanpa runtime framework berat seperti TFLite Micro atau ONNX.
* **Eksekusi Hybrid AI**: Menggabungkan kemampuan generasi autoregresif teks (persona dan tag kendali emosi) dengan kalkulator hardware ALU deterministik (<0.1 ms).
* **Alokasi Octal PSRAM**: Bobot model (1.79 MB) dan KV-cache 128-token dialokasikan di 8MB Octal PSRAM 80MHz (`AP_3v3`), menghindari latensi pembacaan SPI Flash.
* **Dukungan Multi-Board**: Telah tervalidasi di ESP32-S3 DevKit N16R8, Arduino Nano ESP32, dan Seeed Studio XIAO ESP32-S3.

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

## Lisensi

Didistribusikan di bawah lisensi MIT. Lihat file [`LICENSE`](../LICENSE) untuk detail lengkap.
