# Transformer From Scratch

PyTorch implementation of **"Attention Is All You Need"** (Vaswani et al., 2017) built from scratch, without using any pre-built Transformer libraries.

> 📄 Paper: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

---

## Overview

This project implements the full Transformer architecture described in the original paper, including:

- Scaled Dot-Product Attention
- Multi-Head Attention
- Positional Encoding
- Encoder with residual connections and layer normalization
- Decoder with masked self-attention and cross-attention
- Training with teacher forcing and greedy decoding inference

The model is trained on a small Italian → English translation task to demonstrate the architecture end-to-end.

---

## Architecture

```
src/
├── utils.py        # Positional encoding and attention masks
├── attention.py    # Scaled dot-product and multi-head attention
├── encoder.py      # Encoder layer and full encoder stack
├── decoder.py      # Decoder layer and full decoder stack
└── transformer.py  # Full Transformer model

train.py            # Training loop and inference
```

### Model dimensions (this implementation)

| Hyperparameter | Value | Original Paper |
|---|---|---|
| d_model | 128 | 512 |
| num_heads | 4 | 8 |
| num_layers | 2 | 6 |
| d_ff | 512 | 2048 |
| Parameters | ~932K | ~65M |

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/FraViss/transformer-from-scratch.git
cd transformer-from-scratch
```

### 2. Install dependencies

```bash
uv venv
uv sync
```

### 3. Train the model

```bash
python train.py
```

Expected output after 300 epochs:

```
Using device: cpu
Model parameters: 932,496
Epoch 50/300  | Loss: 0.0010
Epoch 100/300 | Loss: 0.7081
Epoch 150/300 | Loss: 0.0832
Epoch 200/300 | Loss: 0.0001
Epoch 250/300 | Loss: 0.0017
Epoch 300/300 | Loss: 0.0001

Training complete!
Model saved to transformer.pth

-- Translations --
  il gatto dorme    -> the cat sleeps
  il cane mangia    -> the dog eats
  la ragazza corre  -> the girl runs
  il bambino piange -> the child cries
```

---

## Key Concepts

### Scaled Dot-Product Attention

The core of the Transformer. Given queries Q, keys K, and values V:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### Multi-Head Attention

Instead of computing attention once, the model runs h attention heads in parallel, each learning different relationships between tokens, then concatenates and projects the results.

### Positional Encoding

Since the Transformer processes all tokens in parallel, positional encodings are added to embeddings using sine and cosine functions of different frequencies to inject sequence order information.

---

## Requirements

- Python 3.12+
- PyTorch 2.x
- NumPy
- Matplotlib
- Jupyter

---

## Reference

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017).
*Attention Is All You Need*. NeurIPS 2017.
[https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

---

## License

MIT
