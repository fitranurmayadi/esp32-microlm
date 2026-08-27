import torch
import torch.nn as nn
import os
import time
import json
from tokenizer import KiboTokenizer
from model import KiboMicroLM

def train_kibo_model():
    print("==========================================")
    print("   Kibo Micro-LM 2MB Training Pipeline   ")
    print("==========================================")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
        
    dataset_path = "/home/aiot/Projects/SBC/kibo_microlm/kibo_dataset.txt"
    with open(dataset_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()
        
    # 1. Build Tokenizer
    tok = KiboTokenizer()
    tok.build_vocab(raw_text)
    tok.save("/home/aiot/Projects/SBC/kibo_microlm/kibo_vocab.json")
    vocab_size = len(tok.tok_to_id)
    pad_id = tok.tok_to_id.get("<PAD>", 0)
    eos_id = tok.tok_to_id.get("<EOS>", 3)
    
    # 2. Parse into clean dialogue samples
    raw_dialogues = [d.strip() for d in raw_text.split("<EOS>") if d.strip()]
    print(f"Total Unique Dialogues in Dataset: {len(raw_dialogues)}")
    
    max_seq_len = 128
    encoded_dialogues = []
    
    for d in raw_dialogues:
        tokens = tok.encode(d + "\n<EOS>")
        if len(tokens) <= max_seq_len:
            # Pad to max_seq_len
            padded = tokens + [pad_id] * (max_seq_len - len(tokens))
            encoded_dialogues.append(padded)
            
    encoded_tensor = torch.tensor(encoded_dialogues, dtype=torch.long)
    print(f"Training Tensor Shape: {encoded_tensor.shape}")
    
    # 3. Model Configuration
    model = KiboMicroLM(
        vocab_size=vocab_size,
        n_embd=192,
        n_head=4,
        n_layer=4,
        max_seq_len=max_seq_len
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=0.01)
    
    max_steps = 5000
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=1e-5)
    batch_size = 32
    num_samples = len(encoded_tensor)
    
    def get_batch():
        ix = torch.randint(num_samples, (batch_size,))
        batch = encoded_tensor[ix].to(device)
        x = batch[:, :-1]
        y = batch[:, 1:]
        return x, y
        
    print("\nStarting Training on GPU...")
    t0 = time.time()
    
    model.train()
    for step in range(1, max_steps + 1):
        xb, yb = get_batch()
        
        # Forward pass
        logits, _ = model(xb)
        # Reshape for loss with ignore_index
        loss = criterion(logits.reshape(-1, vocab_size), yb.reshape(-1))
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if step % 500 == 0 or step == max_steps:
            print(f"Step {step:4d}/{max_steps} | Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
            
    t1 = time.time()
    print(f"🎉 Training Completed in {t1 - t0:.2f} seconds!")
    
    # Save Model Weights
    save_path = "/home/aiot/Projects/SBC/kibo_microlm/kibo_model_2mb.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Saved PyTorch Model Weights to {save_path}")
    
    # Test Generation
    model.eval()
    test_cases = [
        "User: hello\nKibo:",
        "User: hello kibo\nKibo:",
        "User: hi\nKibo:",
        "User: who are you?\nKibo:",
        "User: what is your name?\nKibo:",
        "User: what can you do?\nKibo:",
        "User: tell me a joke\nKibo:",
        "User: can you laugh?\nKibo:",
        "User: are you sleepy\nKibo:",
        "User: goodbye\nKibo:"
    ]
    
    print("\n=== GENERATION TEST ON HOST ===")
    for prompt in test_cases:
        prompt_ids = torch.tensor([tok.encode(prompt)], dtype=torch.long).to(device)
        out_ids = model.generate(prompt_ids, max_new_tokens=60, temperature=0.1, eos_token_id=eos_id)
        generated = tok.decode(out_ids[0].tolist())
        kibo_reply = generated.split("Kibo:")[-1].strip()
        if "<EOS>" in kibo_reply:
            kibo_reply = kibo_reply.split("<EOS>")[0].strip()
        print(f"\n{prompt}")
        print(f"Kibo: {kibo_reply}")
        
    print("\n===============================\n")

if __name__ == "__main__":
    train_kibo_model()
