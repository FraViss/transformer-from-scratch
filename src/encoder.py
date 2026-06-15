import torch
import torch.nn as nn
from typing import Optional
from .attention import MultiHeadAttention


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network as described in 'Attention Is All You Need'.

    FFN(x) = max(0, xW1 + b1)W2 + b2

    Args:
        d_model: Embedding dimension
        d_ff: Inner layer dimension (typically 4 * d_model)
        dropout: Dropout probability
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderLayer(nn.Module):
    """
    Single Encoder Layer as described in 'Attention Is All You Need'.

    Each layer has two sub-layers:
    1. Multi-Head Self-Attention
    2. Position-wise Feed-Forward Network

    Each sub-layer has a residual connection and layer normalization:
    output = LayerNorm(x + sublayer(x))

    Args:
        d_model: Embedding dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward inner dimension
        dropout: Dropout probability
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional padding mask
        Returns:
            output: of shape (batch_size, seq_len, d_model)
            attn_weights: attention weights for visualization
        """
        # Sub-layer 1: Multi-Head Self-Attention + residual + norm
        attn_output, attn_weights = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Sub-layer 2: Feed-Forward + residual + norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x, attn_weights


class Encoder(nn.Module):
    """
    Full Encoder as described in 'Attention Is All You Need'.

    Stacks N identical EncoderLayers on top of each other.

    Args:
        vocab_size: Size of the input vocabulary
        d_model: Embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of stacked encoder layers (N)
        d_ff: Feed-forward inner dimension
        dropout: Dropout probability
        max_seq_len: Maximum sequence length
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        dropout: float = 0.1,
        max_seq_len: int = 5000,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model

        # Positional encoding is imported inline to avoid circular imports
        from .utils import PositionalEncoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, list]:
        """
        Args:
            x: Input token indices of shape (batch_size, seq_len)
            mask: Optional padding mask
        Returns:
            output: Encoded representation of shape (batch_size, seq_len, d_model)
            attn_weights_all: Attention weights from all layers for visualization
        """
        # Token embedding + positional encoding
        x = self.embedding(x) * (self.d_model ** 0.5)
        x = self.positional_encoding(x)

        # Pass through N encoder layers
        attn_weights_all = []
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            attn_weights_all.append(attn_weights)

        return self.norm(x), attn_weights_all
