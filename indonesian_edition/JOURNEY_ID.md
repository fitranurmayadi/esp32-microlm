# Dokumen Desain & Arsitektur Kibo (Edisi Bahasa Indonesia)

Dokumentasi teknis perancangan, kuantisasi, dan eksekusi Micro Language Model (Micro-LM) serta mesin inferensi hybrid pada mikrokontroler ESP32-S3.

---

## 1. Tujuan Sistem & Batasan Perangkat Keras

Tujuan proyek ini adalah menjalankan model bahasa generatif mandiri pada mikrokontroler tanpa ketergantungan pada jaringan internet atau API eksternal.

### Batasan Target Hardware:
* **Prosesor**: Espressif ESP32-S3 (Dual-core Xtensa LX7 @ 240MHz, single-precision FP32 FPU).
* **RAM**: 512 KB internal SRAM + 8 MB Octal OPI PSRAM (80MHz).
* **Penyimpanan**: 8 MB / 16 MB SPI Flash.
* **Target Latensi**: Kecepatan generasi interaktif (>10 token/detik).
* **Batasan Memori**: Alokasi bobot model + aktivasi + KV-cache harus <4 MB total PSRAM.

---

## 2. Perbandingan Teknis dengan Proyek Serupa

1. **`slvDev/esp32-ai`**:
   * *Pendekatan*: Model 28.9M parameter yang dibaca secara sekuensial dari SPI Flash (*Per-Layer Embeddings*) untuk penyelesaian cerita anak (*TinyStories*, ~9.5 tok/dtk).
   * *Perbedaan Desain*: Kibo dirancang untuk percakapan interaktif dua arah dengan pemanggilan tool deterministik. Bobot model (1.79 MB) dan KV-cache dialokasikan penuh di 8MB Octal PSRAM 80MHz untuk meminimalkan latensi pembacaan Flash (~12–13 tok/dtk).

2. **`mailtopk/esp32_ai`**:
   * *Pendekatan*: Klasifikasi teks statis menggunakan runtime TensorFlow Lite Micro.
   * *Perbedaan Desain*: Kibo adalah Transformer decoder autoregresif yang diimplementasikan dalam C++ bare-metal tanpa overhead framework runtime.

---

## 3. Fondasi Teknis & Referensi

| Komponen | Asal / Referensi | Peran dalam Kibo |
| :--- | :--- | :--- |
| **Causal Decoder Transformer** | Vaswani et al. (2017) / nanoGPT (Karpathy) | Generasi token autoregresif dan multi-head self-attention. |
| **Kuantisasi Simetris INT8** | Dettmers et al. (2022) / Standar TensorRT | Mengompresi bobot dari 7.02 MB (FP32) menjadi 1.79 MB tanpa degradasi akurasi kata. |
| **Key-Value (KV) Caching** | vLLM / llama.cpp | Menyimpan cache proyeksi key dan value masa lalu agar komputasi tetap $O(1)$ per langkah generasi. |
| **Deterministic Tool Dispatch** | Toolformer (Schick et al., 2023) | Mengintersepsi pertanyaan aritmatika untuk dieksekusi langsung oleh hardware ALU dalam <0.1 ms dengan presisi 100%. |

---

## 4. Analisis Presisi Kuantisasi (1.84M Parameter)

Pengujian empiris dilakukan pada prosesor Xtensa LX7 ESP32-S3:

| Format | Ukuran Bobot | Rasio Kompresi | Logit Cosine Sim | Kelayakan Hardware |
| :--- | :---: | :---: | :---: | :--- |
| **FP32** | 7.02 MB | Baseline (0.0%) | 100.00% | Beban alokasi memori tinggi (~92% PSRAM terpakai). |
| **FP16** | 3.51 MB | 50.0% | 100.00% | Membutuhkan emulasi software (tidak ada hardware FP16 di Xtensa LX7). |
| **INT8 (Terpilih)** | **1.79 MB** | **74.4%** | **100.00%** | **Optimal: Akses byte native, fidelitas semantik 100%.** |
| **INT4** | 0.88 MB | 87.2% | 98.87% | Penurunan presisi; latensi +26% lebih lambat karena overhead unpack bit. |

### Alasan Teknis Pemilihan INT8:
1. **Fidelitas**: Kuantisasi simetris per-tensor mempertahankan 100.00% kemiripan token dibanding FP32.
2. **Efisiensi Memori**: Dengan ukuran 1.79 MB, model hanya memakai ~22% dari kapasitas 8MB PSRAM.
3. **Overhead Eksekusi INT4**: Pengujian langsung pada hardware ESP32-S3 membuktikan perkalian matriks INT4 lebih lambat 26% dibanding INT8 (8.26 ms vs 6.55 ms per 147K bobot) akibat siklus CPU yang terpakai untuk operasi bit-masking (`& 0x0F`), bit-shifting (`>> 4`), dan sign extension 4-bit.

---

## 5. Tantangan Rekayasa & Solusi

### 1. Konfigurasi Flash Mode
* **Masalah**: Flashing dalam mode `QIO` memicu reset watchdog bootloader ROM pada modul tertentu (`ets_loader.c 78`).
* **Solusi**: Standarisasi seluruh skrip build dan upload ke mode `DIO`.

### 2. Penanganan Aliran Serial DTR
* **Masalah**: Board dengan chip UART bridge (CH340) ter-reset jika `DTR=True`, sementara port native USB CDC pada Nano ESP32 dan Seeed XIAO membutuhkan `DTR=True` agar buffer tidak drop.
* **Solusi**: Pengaturan adaptif pada PySerial: `DTR=False, RTS=False` untuk `/dev/ttyUSB*` dan `DTR=True, RTS=False` untuk `/dev/ttyACM*`.

### 3. Pencegahan Pemotongan Tensor Ekor (Tail Tensor)
* **Masalah**: Ukuran model yang di-hardcode memotong tensor akhir (`ln_f.bias`, `head.weight`) saat layout alignment binary bergeser.
* **Solusi**: Perhitungan ukuran dinamis menggunakan batas simbol linker: `(size_t)(kibo_embedded_model_end - kibo_embedded_model_start)`.

### 4. Akumulasi Matmul Presisi Tinggi
* **Masalah**: Akumulasi nilai floating-point mengalami pergeseran pembulatan pada perkalian dot-product yang panjang.
* **Solusi**: Akumulasi hasil kali integer dalam presisi float penuh per baris sebelum dikalikan dengan faktor skala tensor:
  $$\text{out}[r] = \left(\sum_{c=0}^{\text{cols}-1} x[c] \cdot w[r, c]\right) \cdot \text{scale} + \text{bias}[r]$$

---

## 6. Ringkasan Validasi Hardware

| Board | Flash / PSRAM | Flash Mode | Hasil Uji Nyata |
| :--- | :--- | :---: | :---: |
| **ESP32-S3 DevKit N16R8** | 16MB Flash / 8MB Octal PSRAM | `DIO` | Terverifikasi (~12.3 tok/dtk) |
| **Arduino Nano ESP32** | 16MB Flash / 8MB Octal PSRAM (NORA-W106) | `DIO` | Terverifikasi (~12.2 tok/dtk) |
| **Seeed Studio XIAO ESP32-S3** | 8MB Flash / 8MB Octal PSRAM | `DIO` | Terverifikasi (~12.4 tok/dtk) |

---

## 7. Rencana Integrasi Periferal

* **Layar**: LCD SPI bulat (GC9A01, 240x240) yang dikontrol oleh token emosi (`[HAPPY]`, `[NEUTRAL]`, `[ANGRY]`).
* **Input Audio**: Mikrofon MEMS I2S (INMP441) untuk pengenalan kata kunci offline.
* **Output Audio**: DAC/amplifier I2S (MAX98357A) untuk sintesis suara lokal.
