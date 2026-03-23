from dataclasses import dataclass

@dataclass
class Config:
    # Model
    img_size: int = 32
    patch_size: int = 4
    in_channels: int = 3
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 6
    num_classes: int = 10
    # Training
    batch_size: int = 64
    lr: float = 1e-3
    dropout: float = 0.1
    epochs: int = 30