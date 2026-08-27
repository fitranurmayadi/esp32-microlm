# Kibo Architecture & Design Document

An engineering breakdown of designing, quantizing, and executing a generative Micro Language Model (Micro-LM) and hybrid inference engine on ESP32-S3 microcontrollers.

---

## 1. System Goals & Constraints

The objective was to run a self-contained conversational model on a resource-constrained microcontroller ($5–$10 SoC) without external network dependencies.

### Hardware Targets & Constraints:
* **Processor**: Espressif ESP32-S3 (Dual-core Xtensa LX7 @ 240MHz, single-precision FP32 FPU).
* **RAM**: 512 KB internal SRAM + 8 MB Octal OPI PSRAM (80MHz).
* **Storage**: 8 MB / 16 MB SPI Flash.
* **Latency Requirement**: Interactive generation speed (>10 tokens/second).
* **Footprint Constraint**: Model weights + activations + KV-cache must consume <4 MB total PSRAM.

---

## 2. Technical Influences & Differences

1. **`slvDev/esp32-ai`**:
   * *Approach*: Demonstrated a 28.9M parameter model on ESP32-S3 using Per-Layer Embeddings (PLE) streamed sequentially from SPI Flash for story completion (~9.5 tok/s).
   * *Our Design Difference*: Designed for interactive multi-turn dialogue with tool dispatch rather than open-ended story completion. Model weights (1.79 MB) and 128-token KV-cache reside entirely in 8MB Octal PSRAM at 80MHz to eliminate serial Flash read latency (~12–13 tok/s).

2. **`mailtopk/esp32_ai`**:
   * *Approach*: Static text classification using TensorFlow Lite Micro.
   * *Our Design Difference*: Kibo is an autoregressive generative decoder Transformer implemented in bare-metal C++ without runtime framework overhead.

---

## 3. Foundational References

| Component | Origin / Reference | Purpose in Kibo |
| :--- | :--- | :--- |
| **Causal Decoder Transformer** | Vaswani et al. (2017) / nanoGPT (Karpathy) | Autoregressive token generation and multi-head self-attention. |
| **Symmetric INT8 Quantization** | Dettmers et al. (2022) / TensorRT standards | Compressing weights from 7.02 MB (FP32) to 1.79 MB with zero loss in output quality. |
| **Key-Value (KV) Caching** | vLLM / llama.cpp | Caches previous key and value projection vectors to maintain $O(1)$ computation per generation step. |
| **Deterministic Tool Dispatch** | Toolformer (Schick et al., 2023) | Intercepts arithmetic queries for hardware ALU execution in <0.1 ms with 100% precision. |

---

## 4. Quantization Precision Analysis (1.84M Parameters)

We evaluated four precision formats for the Xtensa LX7 architecture:

| Format | Weight Footprint | Compression Ratio | Logit Cosine Sim | Hardware Feasibility |
| :--- | :---: | :---: | :---: | :--- |
| **FP32** | 7.02 MB | Baseline (0.0%) | 100.00% | High memory pressure (~92% PSRAM allocation). |
| **FP16** | 3.51 MB | 50.0% | 100.00% | Software emulation required; no native FP16 vector ALU on Xtensa LX7. |
| **INT8 (Selected)** | **1.79 MB** | **74.4%** | **100.00%** | **Optimal balance: Byte-aligned SIMD MAC and 100% semantic fidelity.** |
| **INT4** | 0.88 MB | 87.2% | 98.87% | Degraded precision; +26% latency overhead from bit unpacking operations. |

### Technical Rationale for INT8:
1. **Fidelity**: Symmetric per-tensor scaling preserves 100.00% output token similarity relative to FP32.
2. **Memory Efficiency**: At 1.79 MB, the model uses only ~22% of the 8MB PSRAM buffer.
3. **Execution Overhead in INT4**: Real hardware profiling on the ESP32-S3 revealed that INT4 matrix multiplication is 26% slower than INT8 (8.26 ms vs 6.55 ms per 147K weight projection) due to CPU cycles spent on bit-masking (`& 0x0F`), bit-shifting (`>> 4`), and 4-bit sign extensions.

---

## 5. Engineering Challenges & Solutions

### 1. Flash Mode Configuration
* **Problem**: Flashing in `QIO` mode caused ROM bootloader watchdog reset loops on certain modules (`ets_loader.c 78`).
* **Solution**: Standardized compile and flash flags to `DIO` mode across all targets.

### 2. PySerial Flow Control Differences
* **Problem**: DevKit boards with CH340 UART bridges reset when `DTR=True` is asserted, while native USB CDC ports on Nano ESP32 and Seeed XIAO require `DTR=True` to prevent buffer drops.
* **Solution**: Implemented adaptive port handling: `DTR=False, RTS=False` for `/dev/ttyUSB*` and `DTR=True, RTS=False` for `/dev/ttyACM*`.

### 3. Tail Tensor Truncation Prevention
* **Problem**: Hardcoding fixed model buffer sizes caused tail tensors (`ln_f.bias`, `head.weight`) to be truncated when binary alignments shifted.
* **Solution**: Implemented dynamic size calculation using linker symbol boundaries: `(size_t)(kibo_embedded_model_end - kibo_embedded_model_start)`.

### 4. High-Precision Matmul Accumulation
* **Problem**: Floating-point rounding errors drifted during long forward-pass dot products.
* **Solution**: Accumulate integer products in full float precision per row before applying the tensor scale factor:
  $$\text{out}[r] = \left(\sum_{c=0}^{\text{cols}-1} x[c] \cdot w[r, c]\right) \cdot \text{scale} + \text{bias}[r]$$

---

## 6. Hardware Validation Summary

| Board | Flash / PSRAM | Flash Mode | Result |
| :--- | :--- | :---: | :---: |
| **ESP32-S3 DevKit N16R8** | 16MB Flash / 8MB Octal PSRAM | `DIO` | Verified (~12.3 tok/s) |
| **Arduino Nano ESP32** | 16MB Flash / 8MB Octal PSRAM (NORA-W106) | `DIO` | Verified (~12.2 tok/s) |
| **Seeed Studio XIAO ESP32-S3** | 8MB Flash / 8MB Octal PSRAM | `DIO` | Verified (~12.4 tok/s) |

---

## 7. Peripheral Integration Roadmap

* **Display**: Round SPI LCD (GC9A01, 240x240) driven by emotion tokens (`[HAPPY]`, `[NEUTRAL]`, `[ANGRY]`).
* **Audio Input**: I2S MEMS microphone (INMP441) for on-device keyword detection.
* **Audio Output**: I2S DAC/amplifier (MAX98357A) for local voice synthesis.
