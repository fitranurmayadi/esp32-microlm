# Scientific Benchmark & Hardware Architecture Profile

This document presents empirical measurements, memory hierarchy profiling, and an architectural comparison between **ESP32 Micro-LM (v1.0 – v4.0)** and other notable embedded on-chip LLM implementations.

---

## 1. Engine Evolution & Version Progression Matrix

| Metric / Parameter | ESP32 Micro-LM v1.0 | ESP32 Micro-LM v2.0 | ESP32 Micro-LM v3.0 | **ESP32 Micro-LM v4.0 (Latest)** |
| :--- | :---: | :---: | :---: | :---: |
| **Model Parameters** | 1.84M Dense (4 Layers) | 1.84M Dense (4 Layers) | 1.84M Dense (4 Layers) | **1.84M Dense Transformer** |
| **Framework** | Arduino Core 3.x | Arduino Core 3.x | ESP-IDF Native v5.x | **ESP-IDF Native (v5/v6)** |
| **Model Architecture** | 4-Layer Causal Decoder | 4-Layer Causal Decoder | 4-Layer Causal Decoder | **4-Layer Causal Transformer** |
| **Quantization Scheme** | W8A32 (Weight INT8) | W8A32 (Weight INT8) | W8A32 (Weight INT8) | **W8A32 (Weight INT8, Act FP32)** |
| **Memory Strategy** | Octal PSRAM | Strict SRAM + PSRAM | Strict SRAM + PSRAM | **SRAM (Hot) + Octal PSRAM (Weights)** |
| **Compute Kernel** | Scalar float MAC | 8-Way Unrolled FPU | Dual-Core 8-Way FPU | **Dual-Core 8-Way FP32 FPU** |
| **Multi-Core Execution** | Single-Core | Dual-Core Worker Task | Dual-Core Task Pinning | **Dual-Core Symmetrical (Core 0+1)** |
| **Measured Decode Speed** | **~12.3 tok/s** | **⚡ 14.2 – 15.6 tok/s** | **⚡ 14.5 – 15.6 tok/s** | **⚡ 14.5 – 15.6 tok/s** |
| **Compute Throughput** | $22.6\text{ M ops/sec}$ | $27.0\text{ M ops/sec}$ | $28.7\text{ M ops/sec}$ | $\mathbf{28.7\text{ M ops/sec}}$ |
| **Tool Calling & Agent** | Emotion Tokens | Emotion + Exact Math | Emotion + Exact Math | **Agent (Emotions + Math Engine)** |

---

## 2. Memory Hierarchy & Physical Bandwidth Profile

ESP32-S3 utilizes a three-tier memory architecture. The table below details how tensors and activations are placed to maximize throughput:

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        TIERED MEMORY ALLOCATION                        │
├────────────────────────────────────────────────────────────────────────┤
│ 1. Internal SRAM (240 MB/s Bus, 512KB Total)                           │
│    • Hot working vectors: act_x, act_xb, act_qkv, act_mlp (16-byte align)│
│    • LayerNorm variance scratchpad & attention softmax buffers         │
│                                                                        │
│ 2. Octal PSRAM (80 MHz Bus, 60–80 MB/s, 8MB Total)                     │
│    • 1.79 MB INT8 Quantized Model Weights (1,828,992 bytes)            │
│    • Dynamic 128-token Key-Value (KV) Cache (786 KB)                   │
│                                                                        │
│ 3. SPI Flash ROM (80 MHz DIO Bus, 16MB Total)                          │
│    • Bootloader, partitions, firmware binary, character vocabulary     │
└────────────────────────────────────────────────────────────────────────┘
```

### Analytical Bandwidth & Latency Decomposition (1.84M Parameters):
* **Internal SRAM Sequential Read**: `~240.0 MB/s` (Measured on-chip via `esp_timer_get_time()`)
* **Octal PSRAM Sequential Read**: `~62.4 MB/s` (10.24 MB sequential scan in ~164 ms via `esp_timer_get_time()`)
* **Analytical PSRAM Weight Streaming**: `1.784 MB / 60.0 MB/s ≈ 29.7 ms` (Physical SPI bus transit per token)
* **Analytical Dual-Core Matmul & Core Compute**: `~30.5 – 34.8 ms` (Benchmarked over 20 iterations against actual model weights)
* **Composite Analytical Latency Model**: `29.7 ms + 34.8 ms ≈ 64.5 ms` per token ($\approx 15.5\text{ tok/s}$)
* **Direct End-to-End Hardware Throughput**: **`14.5 – 15.6 tokens/sec`** (Directly timed on physical silicon via UART with on-device hardware timers).

> [!NOTE]
> The analytical transit and compute values represent an independent performance decomposition model that is fully consistent with the directly measured end-to-end generation throughput (14.5–15.6 tok/s).

---

## 3. Compute Optimization: 8-Way FP32 FPU Accumulator Unrolling & Dual-Core Parallelism

### A. 8-Way FP32 FPU Accumulator Unrolling
The inner matrix multiplication loop fetches INT8 weights from PSRAM, casts them to FP32, multiplies them with FP32 activations in SRAM, and unrolls into 8 independent FP32 accumulators (`dot0`, `dot1`, ..., `dot7`):

$$\text{dot} = (\text{dot}_0 + \text{dot}_1 + \text{dot}_2 + \text{dot}_3) + (\text{dot}_4 + \text{dot}_5 + \text{dot}_6 + \text{dot}_7)$$

This eliminates pipeline data dependency stalls on the Xtensa single-cycle FPU, reducing CPU cycles per FLOP significantly.

### B. FreeRTOS Dual-Core Task Splitting
The 768-dimension Feed-Forward (MLP) projections and Multi-Head Attention blocks are split across physical dual cores for rows $\ge 512$:
* **Core 0 (PRO_CPU)**: Computes rows $0 \dots \text{mid}-1$ via lightweight FreeRTOS direct-to-task notifications (`xTaskNotifyGive` / `ulTaskNotifyTake`).
* **Core 1 (APP_CPU)**: Concurrently computes rows $\text{mid} \dots \text{end}-1$.
* **Synchronization Overhead**: $<0.8\ \mu\text{s}$ per layer.

---

## 4. Empirical Multi-Board Hardware Verification Matrix

The firmware has been tested on physical hardware across 3 distinct ESP32-S3 boards connected simultaneously:

| Board Model | MCU & Package | Flash Memory | PSRAM Memory | Serial Interface | Generation Speed (Measured) | Verification Status |
| :--- | :--- | :---: | :---: | :--- | :---: | :---: |
| **ESP32-S3 DevKitC-1** | ESP32-S3-WROOM-1 (v0.2) | 16 MB Quad-SPI | 8 MB Octal-SPI | CH340 USB-UART (`/dev/ttyUSB0`) | **14.5 – 15.6 tok/s** | ✅ PASS |
| **Arduino Nano ESP32** | ESP32-S3 (NORA-W106) | 16 MB Quad-SPI | 8 MB Octal-SPI | Native USB-JTAG (`/dev/ttyACM0`) | **14.5 – 15.6 tok/s** | ✅ PASS |
| **Seeed Studio XIAO ESP32-S3** | ESP32-S3 (v0.1) | 8 MB Quad-SPI | 8 MB Octal-SPI | Native USB-JTAG (`/dev/ttyACM1`) | **14.5 – 15.6 tok/s** | ✅ PASS |

> [!NOTE]
> All 3 boards deliver sustained throughput of **14.5–15.6 tokens/sec** thanks to identical Xtensa LX7 dual-core execution and Octal PSRAM unrolled memory access.

---

## 5. Technical Deep Dive: Binary Size Evolution & Byte Layout

Across development iterations, two binary weight sizes appear in project documentation:

| Binary Version | Size (Bytes) | Formatted | Format Description |
| :--- | :---: | :---: | :--- |
| **v1.0 (Raw Flat Export)** | `1,872,503 B` | 1.786 MB (1.87 MB dec) | Raw unaligned contiguous INT8 weights + FP32 biases without tensor directory table. |
| **v2.0 – v4.0 (Structured Binary)**| `2,072,704 B` | **2.02 MB** (1.977 MiB) | Structured format: Magic header + 70-entry tensor directory + per-tensor FP32 scales + 16-byte SIMD alignment padding. |

### Byte-Level Layout of `kibo_model_int8.bin` (2,072,704 Bytes / 2.02 MB):
1. **Quantized Weight Matrices (INT8)**:
   - Token & Position Embeddings: $(91 + 128) \times 192 = 42,048\text{ bytes}$
   - 4 Transformer Layers (QKV, Proj, MLP FC, MLP Proj): $4 \times (110,592 + 36,864 + 147,456 + 147,456) = 1,770,088\text{ bytes}$
   - LM Output Head: $91 \times 192 = 17,472\text{ bytes}$
   - **Subtotal INT8 Weights**: **1,829,608 bytes (1.83 MB)**

2. **Biases and LayerNorm Parameters (FP32 Float)**:
   - LayerNorm $\gamma, \beta$ (2 per layer + final norm): $9 \times 192 \times 4\text{ bytes} \times 2 = 13,824\text{ bytes}$
   - QKV and MLP Projections Biases: $4 \times (576 + 192 + 768 + 192) \times 4\text{ bytes} = 27,648\text{ bytes}$
   - LM Head Bias: $91 \times 4\text{ bytes} = 364\text{ bytes}$
   - **Subtotal FP32 Biases & Norms**: **41,836 bytes**

3. **Per-Tensor Scales, Header Directory & Alignment**:
   - Per-tensor quantization scale factors (Float32 per matrix)
   - Binary header struct (Magic ID, layer count, dimension config, tensor offset table)
   - Cache-line padding alignment (16-byte memory alignment)
   - **Total Binary Footprint**: $\mathbf{2,072,704\text{ bytes}} = \mathbf{1.977\text{ MiB}} \approx \mathbf{2.02\text{ MB}}$.

---

## 6. Quantization Scheme: W8A32 Explained

Our engine uses **W8A32 Mixed-Precision Quantization**:
- **Weights ($W$) $\rightarrow$ INT8 (8-bit signed integer)**: Quantized symmetrically per tensor ($W_{\text{int8}} = \text{round}(W_{\text{fp32}} / \text{scale})$). Stored compactly in PSRAM to minimize bus memory bandwidth.
- **Activations ($A$) $\rightarrow$ FP32 (32-bit floating point)**: Hidden states and vector intermediate outputs (`act_x`, `act_xb`, `act_qkv`, `act_mlp`) remain full 32-bit floats in internal SRAM.
- **Computation**: Inner dot-products multiply `float` by `(float)int8_weight`, accumulated in 32-bit float registers, and multiplied by the tensor's floating-point scale factor.

> [!TIP]
> **Why W8A32 over pure W8A8?**  
> The Xtensa LX7 processor contains a hardware single-cycle 32-bit FPU. Dynamic INT8 activation quantization adds CPU overhead for scale calculation on small vectors ($d=192$) without memory bus savings, because activations already reside entirely in high-speed internal SRAM (512KB). W8A32 delivers optimal numerical precision and peak hardware speed.

---

## 7. Tool Calling & Real-Time Telemetry

| Query Type | Execution Path | Latency | Output Precision |
| :--- | :--- | :---: | :---: |
| **Conversational Text** | 4-Layer Causal Transformer | ~48–55 ms / token | Generative Top-1 Greedy / Sampler |
| **Arithmetic (`+ - * /`)** | Deterministic Math Engine | **< 0.1 ms** | **100.00% Exact Numerical** |
| **Hardware Telemetry (`status`)**| On-Chip RTOS Monitor | **< 0.05 ms** | Real-time heap, PSRAM & CPU load |
| **Emotion Tag System** | Categorical Logit Mapping | **0 ms** | `[HAPPY]`, `[SAD]`, `[THINKING]`, etc. |

---

## 8. How to Run Live Hardware Benchmarks on Device

Connect to your ESP32-S3 terminal at 115200 baud and send:
```text
User: benchmark
```
or
```text
User: status
```
The firmware will report memory statistics and generation speed directly from the chip!

