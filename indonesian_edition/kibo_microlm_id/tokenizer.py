import json
import os

class KiboTokenizer:
    def __init__(self):
        self.special_tokens = [
            "<PAD>", "<UNK>", "<BOS>", "<EOS>",
            "User:", "Kibo:",
            "[HAPPY]", "[LAUGH]", "[CRYING]", "[SLEEPY]",
            "[LOVE]", "[ANGRY]", "[DIZZY]", "[VAMPIRE]",
            "[NEUTRAL]", "[CAT]"
        ]
        self.tok_to_id = {}
        self.id_to_tok = {}
        
    def build_vocab(self, text):
        # 1. Add special tokens
        for st in self.special_tokens:
            if st not in self.tok_to_id:
                idx = len(self.tok_to_id)
                self.tok_to_id[st] = idx
                self.id_to_tok[idx] = st
                
        # 2. Add characters
        chars = sorted(list(set(text)))
        for c in chars:
            if c not in self.tok_to_id:
                idx = len(self.tok_to_id)
                self.tok_to_id[c] = idx
                self.id_to_tok[idx] = c
                
        print(f"Built Vocab Size: {len(self.tok_to_id)} tokens")
        
    def encode(self, text):
        tokens = []
        i = 0
        n = len(text)
        while i < n:
            matched = False
            for st in self.special_tokens:
                if text.startswith(st, i):
                    tokens.append(self.tok_to_id[st])
                    i += len(st)
                    matched = True
                    break
            if not matched:
                char = text[i]
                tokens.append(self.tok_to_id.get(char, self.tok_to_id["<UNK>"]))
                i += 1
        return tokens
        
    def decode(self, ids):
        res = []
        for idx in ids:
            tok = self.id_to_tok.get(idx, "")
            if tok not in ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]:
                res.append(tok)
        return "".join(res)
        
    def save(self, filepath):
        data = {
            "tok_to_id": self.tok_to_id,
            "id_to_tok": {str(k): v for k, v in self.id_to_tok.items()},
            "vocab_size": len(self.tok_to_id)
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
    def load(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.tok_to_id = data["tok_to_id"]
        self.id_to_tok = {int(k): v for k, v in data["id_to_tok"].items()}
