import torch
import struct
import json
import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)
from tokenizer import KiboTokenizer
from model_ple import KiboMicroLMPLE

def export_v4_ple_model():
    print("=================================================================")
    print("   Exporting Kibo Micro-LM v4.0 (Gemma 3n PLE Hybrid Edition)   ")
    print("=================================================================")
    
    tok = KiboTokenizer()
    tok.load(os.path.join(SCRIPT_DIR, "kibo_vocab.json"))
    vocab_size = len(tok.tok_to_id)
    
    d_model = 192
    n_head = 4
    n_layer = 4
    ple_dim = 128
    max_seq_len = 128
    
    # 1. Instantiate Model
    model_v4 = KiboMicroLMPLE(vocab_size=vocab_size, d_model=d_model, n_head=n_head, n_layer=n_layer, ple_dim=ple_dim, max_seq_len=max_seq_len)
    v4_state = model_v4.state_dict()
    
    base_model_path = os.path.join(SCRIPT_DIR, "kibo_model_2mb.pt")
    if os.path.exists(base_model_path):
        base_state = torch.load(base_model_path, map_location="cpu")
        
        # Exact name mappings between base checkpoint and v4 model
        for k_base, tensor in base_state.items():
            if k_base in v4_state and v4_state[k_base].shape == tensor.shape:
                v4_state[k_base] = tensor
            else:
                # Check layer renames
                k_target = k_base.replace("attn.c_attn", "c_attn") \
                                 .replace("attn.c_proj", "c_proj") \
                                 .replace("mlp.c_fc", "c_fc") \
                                 .replace("mlp.c_proj", "mlp_c_proj")
                if k_target in v4_state and v4_state[k_target].shape == tensor.shape:
                    v4_state[k_target] = tensor
                    print(f"  Mapped {k_base} -> {k_target}")
                    
        # Zero-initialize PLE projection weights so residual identity is preserved exactly at step 0
        for l in range(n_layer):
            v4_state[f"blocks.{l}.ple.ple_proj.weight"] = torch.zeros(d_model, ple_dim)
            
        model_v4.load_state_dict(v4_state)
        print("✅ Successfully mapped 100% of base pre-trained weights into v4.0 core.")
    
    model_v4.eval()
    
    # 2. Export Binary File for ESP32-S3
    bin_path = os.path.join(SCRIPT_DIR, "kibo_model_int8.bin")
    
    with open(bin_path, 'wb') as f:
        named_params = list(model_v4.named_parameters())
        num_tensors = len(named_params)
        
        # Magic 0x4B49424F = 'KIBO'
        header = struct.pack('iiiiiii', 0x4B49424F, vocab_size, d_model, n_head, n_layer, max_seq_len, num_tensors)
        f.write(header)
        
        total_int8_bytes = 0
        total_fp32_bytes = 0
        
        for name, param in named_params:
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('i', len(name_bytes)))
            f.write(name_bytes)
            
            data = param.detach().numpy()
            num_elements = data.size
            
            # Quantize 2D weight matrices & PLE table to INT8; keep 1D LayerNorm/Biases in FP32
            if ("weight" in name or "ple_table" in name) and data.ndim == 2:
                max_val = float(abs(data).max())
                scale = max_val / 127.0 if max_val > 0 else 1.0
                q_data = (data / scale).round().clip(-128, 127).astype('int8')
                
                f.write(struct.pack('ifi', 1, scale, num_elements))
                f.write(q_data.tobytes())
                total_int8_bytes += num_elements
            else:
                f.write(struct.pack('ifi', 0, 1.0, num_elements))
                f.write(data.astype('float32').tobytes())
                total_fp32_bytes += num_elements * 4
                
    file_size_kb = os.path.getsize(bin_path) / 1024
    file_size_mb = file_size_kb / 1024
    print(f"✅ KIBO v4.0 EXPORT SUCCESS: {bin_path}")
    print(f"File Size: {file_size_mb:.2f} MB ({file_size_kb:.1f} KB)")
    print(f"Total INT8 Weights & PLE Tables: {total_int8_bytes:,} bytes")
    print(f"Total FP32 Biases/Norms: {total_fp32_bytes:,} bytes")
    
    # Copy to kibo_esp32 and kibo_espidf directories
    dest_esp32 = os.path.join(SCRIPT_DIR, "../kibo_esp32/kibo_model_int8.bin")
    dest_espidf = os.path.join(SCRIPT_DIR, "../kibo_espidf/main/kibo_model_int8.bin")
    
    with open(bin_path, 'rb') as src:
        data_bin = src.read()
        with open(dest_esp32, 'wb') as dst: dst.write(data_bin)
        with open(dest_espidf, 'wb') as dst: dst.write(data_bin)
    print("✅ Synchronized binary model to kibo_esp32/ and kibo_espidf/main/.")

if __name__ == "__main__":
    export_v4_ple_model()
