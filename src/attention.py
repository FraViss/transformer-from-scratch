import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Dropout] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Scaled Dot-Product Attention as described in 'Attention Is All You Need'.

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

    Args:
        query: Query tensor of shape (batch_size, num_heads, seq_len, d_k)
        key:   Key tensor of shape   (batch_size, num_heads, seq_len, d_k)
        value: Value tensor of shape (batch_size, num_heads, seq_len, d_k)
        mask:  Optional mask tensor to block certain positions
        dropout: Optional dropout layer
    Returns:
        output: Attention output of shape (batch_size, num_heads, seq_len, d_k)
        attn_weights: Attention weights for visualization
    """
    d_k = query.size(-1)

    # Compute attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # Apply mask if provided (set masked positions to -inf so softmax gives 0)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    # Convert scores to probabilities
    attn_weights = F.softmax(scores, dim=-1)

    if dropout is not None:
        attn_weights = dropout(attn_weights)

    output = torch.matmul(attn_weights, value)
    return output, attn_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention as described in 'Attention Is All You Need'.

    Instead of performing a single attention function, projects queries, keys
    and values h times with different learned projections, computes attention
    in parallel, then concatenates and projects the results.

    Args:
        d_model: Total embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # dimension per head

        # Learned linear projections for Q, K, V and output
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split the last dimension into (num_heads, d_k) and transpose.

        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor of shape (batch_size, num_heads, seq_len, d_k)
        """
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: of shape (batch_size, seq_len, d_model)
            key:   of shape (batch_size, seq_len, d_model)
            value: of shape (batch_size, seq_len, d_model)
            mask:  optional mask tensor
        Returns:
            output: of shape (batch_size, seq_len, d_model)
            attn_weights: attention weights for visualization
        """
        # Linear projections
        Q = self.split_heads(self.W_q(query))
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))

        # Scaled dot-product attention on all heads in parallel
        attn_output, attn_weights = scaled_dot_product_attention(
            Q, K, V, mask=mask, dropout=self.dropout
        )

        # Concatenate heads and apply final projection
        batch_size, _, seq_len, _ = attn_output.size()
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        output = self.W_o(attn_output)

        return output, attn_weights
