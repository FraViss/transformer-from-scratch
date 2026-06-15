import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Positional Encoding as described in 'Attention Is All You Need'.

    Adds positional information to token embeddings using sine and cosine
    functions of different frequencies.

    Args:
        d_model: Embedding dimension
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
    """

    def __init__(self, d_model: int, max_seq_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices

        pe = pe.unsqueeze(0)  # shape: (1, max_seq_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token embeddings of shape (batch_size, seq_len, d_model)
        Returns:
            Embeddings with positional encoding added
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


def create_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Creates a mask to ignore padding tokens.

    Args:
        seq: Input sequence of shape (batch_size, seq_len)
        pad_idx: Index used for padding tokens
    Returns:
        Mask of shape (batch_size, 1, 1, seq_len)
    """
    return (seq == pad_idx).unsqueeze(1).unsqueeze(2)


def create_causal_mask(seq_len: int) -> torch.Tensor:
    """
    Creates a causal (autoregressive) mask for the decoder.
    Prevents positions from attending to subsequent positions.

    Args:
        seq_len: Length of the sequence
    Returns:
        Upper triangular mask of shape (seq_len, seq_len)
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return mask
