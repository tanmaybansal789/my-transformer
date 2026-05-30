import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import einops
import tiktoken

device = 'cuda' if torch.cuda.is_available() \
         else 'mps' if torch.backends.mps.is_available() \
         else 'cpu'

print(f'Using device: {device}')

class CharEncoder:
    def __init__(self, text):
        self.chars = sorted(set(text))
        self.char_to_idx = { ch: i for i, ch in enumerate(self.chars) }

    def encode(self, s):
        return [self.char_to_idx[c] for c in s]
    
    def decode(self, l):
        return ''.join(self.chars[i] for i in l)
    
    @property
    def n_vocab(self):
        return len(self.chars)

class RotaryEmbeddings(nn.Module):
    def __init__(self, d_k, base, max_len=2048):
        super().__init__()
        self.base = base
        self.max_len = max_len

        # the i'th frequency of a pair of dimensions is base^(-i/d_k)
        freqs = base ** (-torch.arange(0, d_k, 2).float() / d_k)

        # multiplier for each position (timeline)
        m = torch.arange(max_len).float()

        # make angles
        angles = torch.outer(m, freqs)
        # duplicate
        # we have d_k/2 frequencies, but we need d_k dimensions, so we duplicate each frequency for a pair of dimensions
        # this gives us
        # [f1, f2, f3, f1, f2, f3]

        angles = einops.repeat(angles, 't d -> t (r d)', r=2)

        # cache cos and sine using register_buffer, so they are saved as part of the state dict but not trained
        self.register_buffer('cos', einops.rearrange(torch.cos(angles), 't d -> 1 1 t d'))
        self.register_buffer('sin', einops.rearrange(torch.sin(angles), 't d -> 1 1 t d'))

    def forward(self, q_split, k_split):
        # q_split, k_split: (b, h, t, d_k)
        B, H, T, D = q_split.shape
        cos_slice = self.cos[:, :, :T, :] 
        sin_slice = self.sin[:, :, :T, :]

        def rotate_half(x):
            mid = D // 2
            return torch.cat([-x[..., mid:], x[..., :mid]], dim=-1)

        # we will pair q0 with qmid, q1 with qmid+1, etc. so that each pair of dimensions shares a frequency
        q_rot = (q_split * cos_slice) + (rotate_half(q_split) * sin_slice)
        k_rot = (k_split * cos_slice) + (rotate_half(k_split) * sin_slice)

        return q_rot, k_rot

class MultiHeadAttention(nn.Module):
    def __init__(self, d_embed, n_heads, rotary_embeddings):
        super().__init__()
        if d_embed % n_heads != 0:
            raise ValueError(f'{d_embed=} must be a multiple of {n_heads=}')

        self.n_heads = n_heads
        self.rot = rotary_embeddings

        # projections
        self.q_proj = nn.Linear(d_embed, d_embed)
        self.k_proj = nn.Linear(d_embed, d_embed)
        self.v_proj = nn.Linear(d_embed, d_embed)
        self.out_proj = nn.Linear(d_embed, d_embed)

    def forward(self, x):
        B, T, D = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # We have projected them, now split for multi-head attention
        q_split = einops.rearrange(q, 'b t (h d) -> b h t d', h=self.n_heads)
        k_split = einops.rearrange(k, 'b t (h d) -> b h t d', h=self.n_heads)
        v_split = einops.rearrange(v, 'b t (h d) -> b h t d', h=self.n_heads)

        q_split, k_split = self.rot(q_split, k_split)

        # we want to independently attend to each (t x d) matrix, treat all heads independently
        raw_scores = q_split @ k_split.mT 
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn_weights = F.softmax(raw_scores.masked_fill(mask, -float('inf')), dim=-1)
        attn_out = attn_weights @ v_split

        # now, recombine the heads
        attn_out = einops.rearrange(attn_out, 'b h t d -> b t (h d)')
        out = self.out_proj(attn_out)

        return out

class TransformerBlock(nn.Module):
    def __init__(self, d_embed, n_heads, d_hidden, rotary_embeddings):
        super().__init__()

        self.attn = MultiHeadAttention(d_embed, n_heads, rotary_embeddings)
        self.ln_attn = nn.LayerNorm(d_embed)
    
        self.ffn = nn.Sequential(
            nn.Linear(d_embed, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_embed)
        )
        self.ln_ffn = nn.LayerNorm(d_embed)
    
    def forward(self, x):
        x = x + self.attn(self.ln_attn(x))
        x = x + self.ffn(self.ln_ffn(x))
        return x

class Transformer(nn.Module):
    def __init__(self, n_vocab, d_embed, n_heads, d_hidden, n_blocks, base, max_len):
        super().__init__()

        self.embed = nn.Embedding(n_vocab, d_embed)
        self.rot = RotaryEmbeddings(d_embed // n_heads, base, max_len)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_embed, n_heads, d_hidden, self.rot)
            for _ in range(n_blocks)
        ])

    def forward(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        logits = x @ self.embed.weight.T
        return logits