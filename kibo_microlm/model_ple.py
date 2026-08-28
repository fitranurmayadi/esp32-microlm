import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class KiboPLEBlock(nn.Module):
    """
    Per-Layer Embedding (PLE) Adapter inspired by Google's Gemma 3n.
    Projects layer activations, gates with per-layer flash embedding lookup, and injects via residual.
    """
    def __init__(self, d_model: int, ple_dim: int):
        super().__init__()
        self.ple_gate = nn.Linear(d_model, ple_dim, bias=False)
        self.ple_proj = nn.Linear(ple_dim, d_model, bias=False)
        self.ple_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, ple_token_vector: torch.Tensor) -> torch.Tensor:
        gated = F.gelu(self.ple_gate(x)) * ple_token_vector
        inj = self.ple_norm(self.ple_proj(gated))
        return x + inj

class KiboTransformerBlockPLE(nn.Module):
    def __init__(self, d_model: int, n_head: int, ple_dim: int):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        
        # PLE Adapter
        self.ple = KiboPLEBlock(d_model, ple_dim)
        
        # Attention
        self.ln_1 = nn.LayerNorm(d_model)
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=True)
        self.c_proj = nn.Linear(d_model, d_model, bias=True)
        
        # MLP
        self.ln_2 = nn.LayerNorm(d_model)
        self.c_fc = nn.Linear(d_model, 4 * d_model, bias=True)
        self.mlp_c_proj = nn.Linear(4 * d_model, d_model, bias=True)

    def forward(self, x: torch.Tensor, ple_token_vector: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # 1. Gemma 3n PLE Injection
        x = self.ple(x, ple_token_vector)
        
        # 2. Multi-Head Self Attention
        B, T, C = x.size()
        h_norm = self.ln_1(x)
        qkv = self.c_attn(h_norm)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        if mask is not None:
            att = att.masked_fill(mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.c_proj(out)
        
        # 3. Feed-Forward Network
        mlp_h = self.ln_2(x)
        mlp_out = self.mlp_c_proj(F.gelu(self.c_fc(mlp_h)))
        x = x + mlp_out
        return x

class KiboMicroLMPLE(nn.Module):
    """
    Kibo Micro-LM v4.0 (Gemma 3n PLE Hybrid Edition):
    - 1.84M Dense Core in Octal PSRAM (4 Layers, d_model=192, 4 Heads)
    - 10M–25M Stored Flash Memory Table via Per-Layer Embeddings (PLE)
    """
    def __init__(self, vocab_size: int = 91, d_model: int = 192, n_head: int = 4, n_layer: int = 4, ple_dim: int = 128, max_seq_len: int = 128):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layer = n_layer
        self.ple_dim = ple_dim
        self.max_seq_len = max_seq_len
        
        # Token & Positional Embeddings
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        
        # Gemma 3n Flash-Resident PLE Table [Vocab, n_layer * ple_dim]
        self.ple_table = nn.Embedding(vocab_size, n_layer * ple_dim)
        
        # Transformer Blocks
        self.blocks = nn.ModuleList([
            KiboTransformerBlockPLE(d_model, n_head, ple_dim) for _ in range(n_layer)
        ])
        
        # Output Head
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        
        tok_vec = self.tok_emb(idx)
        pos_vec = self.pos_emb(pos)
        x = tok_vec + pos_vec
        
        all_ple = self.ple_table(idx)
        mask = torch.tril(torch.ones(self.max_seq_len, self.max_seq_len, device=idx.device)).view(1, 1, self.max_seq_len, self.max_seq_len)
        
        for l, block in enumerate(self.blocks):
            ple_l = all_ple[:, :, l * self.ple_dim : (l + 1) * self.ple_dim]
            x = block(x, ple_l, mask)
            
        x = self.ln_f(x)
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss
