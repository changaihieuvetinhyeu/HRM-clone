import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

    def forward(self, x, mask=None):
        # x: (B, L, D)
        attn_output, _ = self.attn(x, x, x, attn_mask=mask)
        return attn_output

# -------------------------
# Remaining unchanged components
# -------------------------
class rms_norm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm_x = x.norm(2, dim=-1, keepdim=True)
        rms_x = norm_x / math.sqrt(x.shape[-1])
        return self.weight * x / (rms_x + self.eps)


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(dim, hidden_dim)

    def forward(self, x):
        return F.silu(self.w1(x)) * self.w2(x)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb


class CosSin(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.rotary = RotaryEmbedding(dim)

    def forward(self, seq_len):
        freqs = self.rotary(seq_len)
        return freqs.cos(), freqs.sin()


class CastedEmbedding(nn.Embedding):
    def forward(self, x):
        return super().forward(x).to(dtype=torch.float32)


class CastedLinear(nn.Linear):
    def forward(self, x):
        return super().forward(x.to(dtype=self.weight.dtype)).to(dtype=torch.float32)
