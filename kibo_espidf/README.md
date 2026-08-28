# Kibo Micro-LM (v3.0 Native ESP-IDF Edition)

Native ESP-IDF C++ port of **Kibo Micro-LM v3.0** (1.84M Parameter Causal Transformer) with **Xtensa LX7 Coprocessor 0 (PIE 128-Bit SIMD Vector Unit)** and Dual-Core FreeRTOS parallelism.

---

## ⚡ Key Highlights
* **Coprocessor CP0 Vector Unit**: Enabled via `CONFIG_ESP32S3_COPROC_SUPPORT=y` in `sdkconfig.defaults`.
* **Zero-Overhead Memory Mapping**: Models loaded directly into 8MB Octal PSRAM.
* **Dual-Core 240MHz Engine**: Symmetrical workload splitting across PRO_CPU (Core 0) and APP_CPU (Core 1).
* **Deterministic Bilingual Tools**: Fast arithmetic parser & hardware action dispatchers (<0.1 ms).

---

## 🛠️ Build & Flash Instructions

### 1. Using ESP-IDF (v5.x+):
```bash
# Export ESP-IDF environment
. $IDF_PATH/export.sh

# Set target to ESP32-S3
idf.py set-target esp32s3

# Build firmware
idf.py build

# Flash to board and monitor serial output
idf.py -p /dev/ttyUSB0 flash monitor
```

### 2. Using PlatformIO:
```bash
pio run -e esp32s3_espidf -t upload -t monitor
```

---

## 🔬 Empirical Hardware Benchmark (ESP32-S3 @ 240MHz)

| Metric | Measured on Physical Hardware | Theoretical Compute Limit | Description |
| :--- | :---: | :---: | :--- |
| **Sustained Token Generation** | **⚡ 14.5 – 15.6 tok/s** | ~22 tok/s | Measured live over UART terminal session |
| **Dual-Core Matmul Latency (147K params)** | **3.72 ms** | 2.5 ms | PRO_CPU (Core 0) + APP_CPU (Core 1) parallel |
| **Single-Core Matmul Latency (147K params)** | **5.15 ms** | 4.8 ms | 8-way unrolled direct FPU accumulation |
| **Memory Transit Bottleneck** | **~30 ms / token** | — | Streaming 1.79 MB weights across 80MHz Octal PSRAM |

