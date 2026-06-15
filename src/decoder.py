import torch
import torch.nn as nn
from typing import Optional
from .attention import MultiHeadAttention
from .encoder import FeedForward


class DecoderLayer(nn.Module):
    """
    Single Decoder Layer as described in 'Attention Is All You Need'.

    Each layer has three sub-layers:
    1. Masked Multi-Head Self-Attention (on target sequence)
    2. Multi-Head Cross-Attention (on encoder output)
    3. Position-wise Feed-Forward Network

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

        # Sub-layer 1: Masked Self-Attention (looks at target sequence so far)
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)

        # Sub-layer 2: Cross-Attention (looks at encoder output)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)

        # Sub-layer 3: Feed-Forward
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Target sequence of shape (batch_size, tgt_seq_len, d_model)
            encoder_output: Encoder output of shape (batch_size, src_seq_len, d_model)
            src_mask: Padding mask for source sequence
            tgt_mask: Causal mask for target sequence
        Returns:
            output: of shape (batch_size, tgt_seq_len, d_model)
            self_attn_weights: self-attention weights for visualization
            cross_attn_weights: cross-attention weights for visualization
        """
        # Sub-layer 1: Masked Self-Attention
        # Query, Key, Value all come from the target sequence
        self_attn_output, self_attn_weights = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))

        # Sub-layer 2: Cross-Attention
        # Query comes from decoder, Key and Value come from encoder output
        cross_attn_output, cross_attn_weights = self.cross_attention(
            x, encoder_output, encoder_output, src_mask
        )
        x = self.norm2(x + self.dropout(cross_attn_output))

        # Sub-layer 3: Feed-Forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x, self_attn_weights, cross_attn_weights


class Decoder(nn.Module):
    """
    Full Decoder as described in 'Attention Is All You Need'.

    Stacks N identical DecoderLayers on top of each other.

    Args:
        vocab_size: Size of the output vocabulary
        d_model: Embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of stacked decoder layers (N)
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
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model

        from .utils import PositionalEncoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, list, list]:
        """
        Args:
            x: Target token indices of shape (batch_size, tgt_seq_len)
            encoder_output: Encoder output of shape (batch_size, src_seq_len, d_model)
            src_mask: Padding mask for source sequence
            tgt_mask: Causal mask for target sequence
        Returns:
            output: of shape (batch_size, tgt_seq_len, d_model)
            self_attn_weights_all: self-attention weights from all layers
            cross_attn_weights_all: cross-attention weights from all layers
        """
        # Token embedding + positional encoding
        x = self.embedding(x) * (self.d_model ** 0.5)
        x = self.positional_encoding(x)

        # Pass through N decoder layers
        self_attn_weights_all = []
        cross_attn_weights_all = []
        for layer in self.layers:
            x, self_attn_weights, cross_attn_weights = layer(
                x, encoder_output, src_mask, tgt_mask
            )
            self_attn_weights_all.append(self_attn_weights)
            cross_attn_weights_all.append(cross_attn_weights)

        return self.norm(x), self_attn_weights_all, cross_attn_weights_all
