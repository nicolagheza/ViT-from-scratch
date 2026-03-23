import torch.nn as nn
from model.attention import MultiHeadAttention

class TransformerBlock(nn.Module):
    def __init__(self, d_model=128, num_heads=4, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Linear(d_model * mlp_ratio, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.dropout(self.attention(self.norm1(x)))
        x = x + self.mlp(self.norm2(x))
        return x