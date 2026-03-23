# ViT from Scratch

A Vision Transformer (ViT) implementation built from scratch in PyTorch, trained on CIFAR-10.

This project implements the core ideas from [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) (Dosovitskiy et al., 2020) without relying on any pre-built transformer libraries — every component is written from the ground up.

## Architecture

The model follows the standard ViT pipeline:

1. **Patch Embedding** — splits 32×32 images into 4×4 patches, projects each to a 128-dim embedding via a single Conv2d operation
2. **CLS Token + Positional Encoding** — a learnable classification token is prepended to the sequence, and learned position embeddings are added
3. **Transformer Encoder** — 6 blocks of pre-norm multi-head self-attention (4 heads) with MLP expansion (4×), GELU activation, residual connections, and dropout
4. **Classification Head** — the CLS token output is passed through a linear layer to produce 10 class logits

Key implementation details:
- Fused QKV projection in the attention module
- Pre-norm (LayerNorm before attention/MLP) rather than post-norm
- AdamW optimizer with cosine annealing learning rate schedule
- Data augmentation with random crops and horizontal flips

## Project Structure

```
vit-from-scratch/
├── model/
│   ├── __init__.py
│   ├── patch_embedding.py    # Patch projection, CLS token, positional encoding
│   ├── attention.py          # Multi-head self-attention with fused QKV
│   ├── transformer_block.py  # Pre-norm transformer encoder block
│   └── vit.py                # Full ViT assembly
├── data.py                   # CIFAR-10 dataloaders with augmentation
├── train.py                  # Training loop with LR scheduling
├── evaluate.py               # Test set evaluation
├── config.py                 # All hyperparameters in one place
└── README.md
```

## Setup

```bash
# Clone the repository
git clone https://github.com/nicolagheza/ViT-from-scratch.git
cd ViT-from-scratch

# Install dependencies (using uv)
uv sync

# Or with pip
pip install torch torchvision
```

## Usage

### Training

```bash
uv run train.py
```

Trains the ViT on CIFAR-10 for the number of epochs specified in `config.py`. Automatically detects and uses Apple Silicon MPS acceleration when available. Saves a checkpoint to `checkpoint.pth` after training.

### Evaluation

```bash
uv run evaluate.py
```

Loads the saved checkpoint and evaluates accuracy on the CIFAR-10 test set.

## Configuration

All hyperparameters are centralized in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `img_size` | 32 | Input image resolution |
| `patch_size` | 4 | Size of each image patch |
| `in_channels` | 3 | Number of input channels (RGB) |
| `d_model` | 128 | Transformer embedding dimension |
| `num_heads` | 4 | Number of attention heads |
| `num_layers` | 6 | Number of transformer encoder blocks |
| `num_classes` | 10 | Number of output classes |
| `batch_size` | 64 | Training batch size |
| `lr` | 1e-3 | Initial learning rate |
| `epochs` | 30 | Number of training epochs |
| `dropout` | 0.1 | Dropout rate |

## Results

| Epochs | Augmentation | Dropout | LR Schedule | Test Accuracy |
|--------|-------------|---------|-------------|---------------|
| 5 | None | None | Constant | 55.69% |
| 30 | RandomCrop + HFlip | 0.1 | Cosine Annealing | 74.12% |

Note: ViTs are designed for large-scale pretraining. Achieving 74% on CIFAR-10 training from scratch on 50k images is a solid result that validates the architecture. Production ViTs pretrained on ImageNet-21k reach 99%+ on CIFAR-10 with fine-tuning.

## Learning Goals

This project was built as a learning exercise to understand:
- How images are tokenized into patch sequences
- The mechanics of multi-head self-attention (Q, K, V projections, head splitting, scaled dot-product attention)
- Transformer encoder architecture (pre-norm, residual connections, MLP expansion)
- The role of CLS tokens and learned positional embeddings
- PyTorch fundamentals (`nn.Module`, training loops, MPS acceleration)