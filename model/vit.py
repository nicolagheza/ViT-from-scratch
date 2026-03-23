import torch.nn as nn
from model.patch_embedding import PatchEmbedding
from model.transformer_block import TransformerBlock

class ViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, d_model=128, num_heads=4, num_layers=6, num_classes=10, dropout=0.1):
        super().__init__()
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, d_model)
        self.encoder = nn.Sequential(*[TransformerBlock(d_model, num_heads, dropout=dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.encoder(x)
        x = self.norm(x)
        cls_token = x[:, 0]
        return self.classifier(cls_token)