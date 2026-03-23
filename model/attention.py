import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=128, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, S, _ = x.shape
        qkv = self.qkv_proj(x) # (B, S, d_model * 3)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / self.head_dim ** 0.5
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = (attn_weights @ v).transpose(1, 2).reshape(B, S, -1)
        return self.out_proj(attn_output)
        