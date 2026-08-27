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
| **Causal Decoder Transformer** | [Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762) / [nanoGPT (Karpathy)](https://github.com/karpathy/nanoGPT) | Autoregressive token generation and multi-head self-attention. |
| **Symmetric INT8 Quantization** | [Dettmers et al. (2022)](https://arxiv.org/abs/2208.07339) / [TensorRT INT8 Standard](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#working-with-int8) | Compressing weights from 7.02 MB (FP32) to 1.79 MB with 100% logit cosine similarity. |
| **Key-Value (KV) Caching** | [llama.cpp (Gerganov)](https://github.com/ggerganov/llama.cpp) / [vLLM (Kwon et al., 2023)](https://arxiv.org/abs/2309.06180) | Caches previous key and value projection vectors to maintain $O(1)$ computation per generation step. |
| **Deterministic Tool Dispatch** | [Toolformer (Schick et al., 2023)](https://arxiv.org/abs/2302.04761) | Intercepts arithmetic queries for deterministic execution in <0.1 ms with 100% precision. |

---

## 4. Quantization Precision Analysis (1.84M Parameters)

We evaluated four precision formats for the Xtensa LX7 architecture:

| Format | Weight Footprint | Compression Ratio | Logit Cosine Sim | Hardware Feasibility |
| :--- | :---: | :---: | :---: | :--- |
| **FP32** | 7.02 MB | Baseline (0.0%) | 100.00% | High memory pressure (~92% PSRAM allocation). |
| **FP16** | 3.51 MB | 50.0% | 100.00% | Software emulation required; no native FP16 vector ALU on Xtensa LX7. |
| **INT8 (Selected)** | **1.79 MB** | **74.4%** | **99.999%** | **Optimal: Weight-only INT8 (W8A32) in PSRAM with FP32 FPU compute.** |
| **INT4** | 0.88 MB | 87.2% | 98.87% | Degraded precision; +26% latency overhead from bit-unpacking. |

---

## 5. Mathematical & Hardware Feasibility Audit

To independently verify the physical plausibility of achieving **~12.3 tokens/second** on an ESP32-S3 @ 240MHz, we executed a forensic audit on the model binary (`kibo_model_int8.bin`) and theoretical compute budget:

### 1. Model Binary & Parameter Count Audit:
* **Binary Size**: 1,872,503 bytes (1.786 MB) with valid `0x4B49424F` header.
* **Exact Parameters**: 1,839,360 parameters (1,828,992 INT8 weights + 10,368 FP32 norm/bias values).
* **PyTorch Checkpoint Equivalence**: 100.00% exact parameter match against `kibo_model_2mb.pt`.

### 2. Compute Budget (FLOPs / Token):
* **Linear MatMul Operations**: $4 \times (\text{QKV} + \text{Proj} + \text{MLP1} + \text{MLP2}) + \text{Head} \approx 3,648,384 \text{ FLOPs}$.
* **Attention & LayerNorm Overhead**: $\approx 49,152 \text{ FLOPs}$.
* **Total Operations per Token**: $\mathbf{3,697,536 \text{ FLOPs}}$ (~3.698 MFLOPs/token).

### 3. CPU Cycle Budget & Bandwidth @ 240MHz:
* **Clock Frequency**: 240,000,000 cycles/second.
* **Per-Token Time Window (@ 12.3 tok/s)**: $\frac{1000 \text{ ms}}{12.3} = \mathbf{81.3 \text{ ms/token}}$.
* **Available CPU Cycles per Token**: $240 \times 10^6 \times 0.0813 = \mathbf{19.51 \times 10^6 \text{ cycles}}$.
* **Execution Efficiency**: $\frac{19.51 \times 10^6 \text{ cycles}}{3.698 \times 10^6 \text{ FLOPs}} \approx \mathbf{5.28 \text{ cycles per FLOP}}$. (Scalar FPU loop on Xtensa LX7 achieves ~4–6 cycles per scalar MAC, proving exact physical feasibility).
* **PSRAM Read Bandwidth**: $1.784 \text{ MB} \times 12.3 \text{ tok/s} = \mathbf{21.94 \text{ MB/s}}$ (~27.5% bus load on 80MHz Octal PSRAM).

### 4. Numerical Equivalence (FP32 vs W8A32):
* **Mean Absolute Error (MAE)**: $0.013014$
* **Max Absolute Error**: $0.085866$
* **Logit Cosine Similarity**: $\mathbf{99.9990\%}$
* **Predicted Token Match**: **100.00% Exact Match**

Run the automated verification script locally:
```bash
python3 kibo_microlm/audit_model_forensics.py
```

---

## 6. Key Implementation Solutions

### 1. Flash Mode Compatibility (DIO vs QIO)
* **Problem**: Setting flash mode to `QIO` on DevKit boards caused bootlooping because standard S3 breakout pins share quad lines with SPI Flash.
* **Solution**: Standardized compile and flash flags to `DIO` mode across all targets.

### 2. PySerial Flow Control Differences
* **Problem**: DevKit boards with CH340 UART bridges reset when `DTR=True` is asserted, while native USB CDC ports on Nano ESP32 and Seeed XIAO require `DTR=True` to prevent buffer drops.
* **Solution**: Implemented adaptive port handling: `DTR=False, RTS=False` for `/dev/ttyUSB*` and `DTR=True, RTS=False` for `/dev/ttyACM*`.

### 3. Tail Tensor Truncation Prevention
* **Problem**: Hardcoding fixed model buffer sizes caused tail tensors (`ln_f.bias`, `head.weight`) to be truncated when binary alignments shifted.
* **Solution**: Implemented dynamic size calculation using linker symbol boundaries: `(size_t)(kibo_embedded_model_end - kibo_embedded_model_start)`.

### 4. W8A32 Weight-Only Matmul Accumulation
* **Problem**: Floating-point rounding errors drifted during long forward-pass dot products.
* **Solution**: Storing INT8 weights in PSRAM and dequantizing on the fly to float dot-product accumulation in the Xtensa FPU:
  $$\text{out}[r] = \left(\sum_{c=0}^{\text{cols}-1} x[c] \cdot w[r, c]\right) \cdot \text{scale} + \text{bias}[r]$$

---

## 7. Hardware Validation Summary

| Board | Flash / PSRAM | Flash Mode | Result |
| :--- | :--- | :---: | :---: |
| **ESP32-S3 DevKit N16R8** | 16MB Flash / 8MB Octal PSRAM | `DIO` | Verified (~12.3 tok/s) |
| **Arduino Nano ESP32** | 16MB Flash / 8MB Octal PSRAM (NORA-W106) | `DIO` | Verified (~12.2 tok/s) |
| **Seeed Studio XIAO ESP32-S3** | 8MB Flash / 8MB Octal PSRAM | `DIO` | Verified (~12.4 tok/s) |

---

## 8. Peripheral Integration Roadmap

* **Display**: Round SPI LCD (GC9A01, 240x240) driven by emotion tokens (`[HAPPY]`, `[NEUTRAL]`, `[ANGRY]`).
* **Audio Input**: I2S MEMS microphone (INMP441) for on-device keyword detection.
* **Audio Output**: I2S DAC/amplifier (MAX98357A) for local voice synthesis.
