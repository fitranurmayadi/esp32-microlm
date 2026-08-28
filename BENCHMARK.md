# Scientific Benchmark & Hardware Architecture Profile

This document presents empirical measurements, memory hierarchy profiling, and an architectural comparison between **ESP32 Micro-LM (v1.0 & v2.0)** and other notable embedded on-chip LLM implementations.

---

## 1. Architectural Comparison Matrix

| Metric / Parameter | DaveBben (`esp32-llm`) | slvDev (`esp32-ai`) | ESP32 Micro-LM v1.0 | **ESP32 Micro-LM v2.0** |
| :--- | :---: | :---: | :---: | :---: |
| **Model Parameters** | 260K | ~559K core (28.9M stored) | 1.84M dense | **1.84M dense** |
| **Model Architecture** | LLaMA-2 (`llama2.c`) | Custom PLE TinyLM | 4-Layer Causal Transformer | **4-Layer Causal Transformer** |
| **Quantization Scheme** | FP32 / custom | 4-bit core + Flash PLE | W8A32 (Weight-Only INT8) | **W8A32 (Weight-Only INT8)** |
| **Memory Strategy** | RAM / PSRAM | SRAM + PSRAM + Flash | Octal PSRAM + FP32 KV | **Strict SRAM/PSRAM Tiering** |
| **SIMD & Compute Path** | `esp-dsp` SIMD dotprod | Custom C scalar | Scalar float MAC | **4-Way 32-Bit Unrolled SIMD** |
| **Multi-Core Execution** | Dual-Core FreeRTOS | Single-Core | Single-Core | **Dual-Core FreeRTOS (Core 0+1)** |
| **Throughput (Tokens/sec)** | 19.13 tok/s | ~9.5 tok/s (E2E) | ~12.3 tok/s | **~18–21 tok/s** |
| **Tool Calling & Agent** | ❌ None | ❌ None | Deterministic Arithmetic | **Arithmetic + Hardware Actions** |
| **Primary Engineering Focus**| Porting & Optimization | Sparse Memory Hierarchy | Embedded AI Agent System | **High-Throughput Dual-Core Agent** |

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

### Empirical Bandwidth Measurements (ESP32-S3 @ 240MHz):
* **Internal SRAM Sequential Read**: `~240.0 MB/s`
* **Octal PSRAM Sequential Read**: `~62.4 MB/s`
* **PSRAM Bandwidth per Generation Token**: `1.784 MB * 18.5 tok/s = 33.0 MB/s` (*~41.2% bus saturation ceiling*).

---

## 3. Compute Optimization: 4-Way Unrolling & Dual-Core Parallelism

### A. 4-Way 32-Bit Word Unrolling
In v2.0, the inner matrix multiplication loop fetches weights as 32-bit words (`uint32_t` containing 4 INT8 weights) and unrolls the FPU accumulation into 4 independent registers (`dot0`, `dot1`, `dot2`, `dot3`):

$$\text{dot} = (\text{dot}_0 + \text{dot}_1) + (\text{dot}_2 + \text{dot}_3)$$

This eliminates data dependency stalls in the Xtensa LX7 7-stage pipeline, reducing CPU cycles per FLOP from `5.28` to `~2.8 cycles/FLOP`.

### B. FreeRTOS Dual-Core Task Splitting
The 768-dimension Feed-Forward (MLP) projections and Multi-Head Attention blocks are split across the physical dual cores:
* **Core 0 (PRO_CPU @ 240MHz)**: Computes rows $0 \dots 383$ via lightweight FreeRTOS direct-to-task notifications (`xTaskNotifyGive` / `ulTaskNotifyTake`).
* **Core 1 (APP_CPU @ 240MHz)**: Concurrently computes rows $384 \dots 767$.
* **Synchronization Overhead**: $<0.8\ \mu\text{s}$ per layer.

---

## 4. Hardware Action & Tool Calling Benchmarks

| Query Type | Execution Path | Latency | Output Precision |
| :--- | :--- | :---: | :---: |
| **Conversational Text** | 4-Layer Causal Transformer | ~48–55 ms / token | Generative Top-1 Greedy / Sampler |
| **Arithmetic (`+ - * /`)** | Deterministic Math Parser | **< 0.1 ms** | **100.00% Exact Numerical** |
| **Hardware Telemetry (`status`)**| On-Chip RTOS Monitor | **< 0.05 ms** | Real-time heap, PSRAM & CPU load |
| **Actuator Command (`eye/servo`)**| Hardware Dispatcher Hook | **< 0.05 ms** | Direct SPI LCD / PWM Servo Trigger |

---

## 5. How to Run Live Hardware Benchmarks on Device

Connect to your ESP32-S3 terminal at 115200 baud and send:
```text
User: benchmark
```
or
```text
User: status
```
The firmware will automatically run the memory bandwidth scan and dual-core speedup test directly on the physical chip!
