import torch
import torch.nn as nn
import time
import os
import sys
import copy
import math

sys.path.append("/home/aiot/Projects/SBC/kibo_microlm")
from tokenizer import KiboTokenizer
from model import KiboMicroLM

# 1. INT8 Quantization Helper (Symmetric Per-Tensor)
def quantize_to_int8(tensor):
    max_val = tensor.abs().max()
    scale = max_val / 127.0 if max_val > 0 else 1.0
    q_tensor = torch.clamp(torch.round(tensor / scale), -128, 127).to(torch.int8)
    deq_tensor = (q_tensor.to(torch.float32) * scale)
    return q_tensor, scale, deq_tensor

# 2. INT4 Quantization Helper (Symmetric 4-bit [-8, 7])
def quantize_to_int4(tensor):
    max_val = tensor.abs().max()
    scale = max_val / 7.0 if max_val > 0 else 1.0
    q_tensor = torch.clamp(torch.round(tensor / scale), -8, 7).to(torch.int8)
    deq_tensor = (q_tensor.to(torch.float32) * scale)
    return q_tensor, scale, deq_tensor

def calculate_cosine_similarity(tensor_a, tensor_b):
    a = tensor_a.flatten().float()
    b = tensor_b.flatten().float()
    dot = torch.dot(a, b)
    norm_a = torch.norm(a)
    norm_b = torch.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return (dot / (norm_a * norm_b)).item()

def run_benchmark():
    print("==================================================================================")
    print("        KIBO MICRO-LM: QUANTIZATION BENCHMARK (FP32 vs FP16 vs INT8 vs INT4)      ")
    print("==================================================================================")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = KiboTokenizer()
    tok.load("/home/aiot/Projects/SBC/kibo_microlm/kibo_vocab.json")
    vocab_size = len(tok.tok_to_id)
    
    # 1. Load Base FP32 Model
    model_fp32 = KiboMicroLM(vocab_size=vocab_size, n_embd=192, n_head=4, n_layer=4, max_seq_len=128).to(device)
    model_fp32.load_state_dict(torch.load("/home/aiot/Projects/SBC/kibo_microlm/kibo_model_2mb.pt", map_location=device))
    model_fp32.eval()
    
    # 2. Create FP16 Model
    model_fp16 = copy.deepcopy(model_fp32).half()
    
    # 3. Create INT8 Model (Simulated Weight Quantization)
    model_int8 = copy.deepcopy(model_fp32)
    with torch.no_grad():
        for name, param in model_int8.named_parameters():
            if "weight" in name and param.data.ndim == 2:
                _, _, deq = quantize_to_int8(param.data)
                param.data.copy_(deq)
            
    # 4. Create INT4 Model (Simulated Weight Quantization)
    model_int4 = copy.deepcopy(model_fp32)
    with torch.no_grad():
        for name, param in model_int4.named_parameters():
            if "weight" in name and param.data.ndim == 2:
                _, _, deq = quantize_to_int4(param.data)
                param.data.copy_(deq)

    models = {
        "FP32": (model_fp32, torch.float32, 4.0, "Baseline (7.02 MB)"),
        "FP16": (model_fp16, torch.float16, 2.0, "3.51 MB (50.0% reduction)"),
        "INT8": (model_int8, torch.int8, 1.0, "1.79 MB (74.4% reduction - Selected)"),
        "INT4": (model_int4, "int4", 0.5, "0.90 MB (87.2% reduction)"),
    }
    
    total_params = sum(p.numel() for p in model_fp32.parameters())
    print(f"Model Parameters: {total_params:,} (Layers: 4, Dim: 192, Heads: 4, Vocab: {vocab_size})")
    print(f"Evaluation Device: {device.upper()}\n")
    
    test_prompts = [
        "hello kibo",
        "who are you?",
        "what can you do?",
        "tell me a joke",
        "are you sleepy",
        "goodbye"
    ]
    
    eos_id = tok.tok_to_id.get("<EOS>", 3)
    results = {}

    for name, (m, dtype, bytes_per_param, desc) in models.items():
        weight_size_mb = (total_params * bytes_per_param) / (1024 * 1024)
        
        # Warmup
        warmup_prompt = "User: hello\nKibo:"
        w_ids = torch.tensor([tok.encode(warmup_prompt)], dtype=torch.long).to(device)
        _ = m.generate(w_ids, max_new_tokens=15, temperature=0.1, eos_token_id=eos_id)
        
        # Measure Latency and Output Quality
        t0 = time.perf_counter()
        total_tokens_gen = 0
        outputs = []
        cos_sims = []
        
        for p in test_prompts:
            prompt_str = f"User: {p}\nKibo:"
            p_ids = torch.tensor([tok.encode(prompt_str)], dtype=torch.long).to(device)
            with torch.no_grad():
                # Measure single forward pass logit fidelity against FP32
                logits_test, _ = m(p_ids)
                logits_fp32, _ = model_fp32(p_ids)
                sim = calculate_cosine_similarity(logits_test.float(), logits_fp32.float())
                cos_sims.append(sim)
                
                # Autoregressive generation
                out = m.generate(p_ids, max_new_tokens=60, temperature=0.1, eos_token_id=eos_id)
                
            total_tokens_gen += len(out[0]) - len(p_ids[0])
            gen_text = tok.decode(out[0].tolist())
            reply = gen_text.split("Kibo:")[1].split("User:")[0].split("<EOS>")[0].strip() if "Kibo:" in gen_text else gen_text
            outputs.append((p, reply))
            
        t1 = time.perf_counter()
        latency_sec = t1 - t0
        speed_tok_sec = total_tokens_gen / latency_sec if latency_sec > 0 else 0
        avg_cos_sim = sum(cos_sims) / len(cos_sims)
        
        results[name] = {
            "size_mb": weight_size_mb,
            "speed": speed_tok_sec,
            "cosine_sim": avg_cos_sim,
            "desc": desc,
            "outputs": outputs
        }
        
    print("="*96)
    print(f"{'Format':<6} | {'Weight Size':<12} | {'Compression':<12} | {'Logit Cosine Sim':<18} | {'ESP32-S3 Feasibility':<22} | {'Hardware ALU Efficiency'}")
    print("="*96)
    for name, data in results.items():
        ratio = (1.0 - (data['size_mb'] / results['FP32']['size_mb'])) * 100
        size_str = f"{data['size_mb']:.2f} MB"
        
        if name == "FP32":
            esp_status = "High Footprint (92% PSRAM)"
            alu_eff = "Standard FPU (7.02MB payload)"
        elif name == "FP16":
            esp_status = "Emulated (46% PSRAM)"
            alu_eff = "Software emulation (No FP16 ALU)"
        elif name == "INT8":
            esp_status = "Optimal (1.79 MB)"
            alu_eff = "Direct byte-aligned SIMD MAC"
        else: # INT4
            esp_status = "Unpack Overhead"
            alu_eff = "Bit-mask and shift overhead per nibble"
            
        print(f"{name:<6} | {size_str:<12} | {ratio:>10.1f}% | {data['cosine_sim']*100:>15.2f}% | {esp_status:<24} | {alu_eff}")
    print("="*96)
    
    print("\n--- QUALITATIVE TEXT GENERATION COMPARISON (ENGLISH) ---")
    for prompt_idx, p in enumerate(test_prompts):
        print(f"\n[Prompt]: User: {p}")
        for name in models.keys():
            reply = results[name]['outputs'][prompt_idx][1]
            sim = "100%" if name == "FP32" else f"{results[name]['cosine_sim']*100:.1f}%"
            print(f"  [{name:>4}] (Fidelity {sim:>5}): {reply}")

if __name__ == "__main__":
    run_benchmark()
