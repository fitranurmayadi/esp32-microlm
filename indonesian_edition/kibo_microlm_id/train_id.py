import torch
import torch.nn as nn
import os
import time
import json
import sys

from tokenizer import KiboTokenizer
from model import KiboMicroLM

def train_kibo_indonesian():
    print("==================================================")
    print("   🇮🇩 Kibo Micro-LM Pipeline Pelatihan Indonesia   ")
    print("==================================================")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"Menggunakan GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Menggunakan CPU")
        
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "kibo_dataset_id.txt")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()
        
    # 1. Tokenizer
    tok = KiboTokenizer()
    tok.build_vocab(raw_text)
    vocab_path = os.path.join(base_dir, "kibo_vocab.json")
    tok.save(vocab_path)
    vocab_size = len(tok.tok_to_id)
    pad_id = tok.tok_to_id.get("<PAD>", 0)
    eos_id = tok.tok_to_id.get("<EOS>", 3)
    
    # 2. Parsing dataset
    raw_dialogues = [d.strip() for d in raw_text.split("<EOS>") if d.strip()]
    print(f"Total Sampel Dialog Bahasa Indonesia: {len(raw_dialogues)}")
    
    max_seq_len = 128
    encoded_dialogues = []
    
    for d in raw_dialogues:
        tokens = tok.encode(d + "\n<EOS>")
        if len(tokens) <= max_seq_len:
            padded = tokens + [pad_id] * (max_seq_len - len(tokens))
            encoded_dialogues.append(padded)
            
    encoded_tensor = torch.tensor(encoded_dialogues, dtype=torch.long)
    print(f"Bentuk Tensor: {encoded_tensor.shape}")
    
    # 3. Model
    model = KiboMicroLM(
        vocab_size=vocab_size,
        n_embd=192,
        n_head=4,
        n_layer=4,
        max_seq_len=max_seq_len
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=0.01)
    
    # 4. Loop Pelatihan
    epochs = 5000
    batch_size = min(32, len(encoded_dialogues))
    
    print("\nMemulai Pelatihan...")
    start_time = time.time()
    
    for step in range(1, epochs + 1):
        ix = torch.randint(0, len(encoded_dialogues), (batch_size,))
        x_batch = encoded_tensor[ix].to(device)
        
        inputs = x_batch[:, :-1]
        targets = x_batch[:, 1:]
        
        logits, _ = model(inputs)
        loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
        
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % 500 == 0 or step == epochs:
            print(f"Langkah {step:5d}/{epochs} | Loss: {loss.item():.4f}")
            
    elapsed = time.time() - start_time
    print(f"\n🎉 Pelatihan Selesai dalam {elapsed:.2f} detik!")
    
    model_pt_path = os.path.join(base_dir, "kibo_model_id.pt")
    torch.save(model.state_dict(), model_pt_path)
    print(f"Bobot PyTorch tersimpan di: {model_pt_path}")

if __name__ == "__main__":
    train_kibo_indonesian()
