import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder
from .utils import create_padding_mask, create_causal_mask


class Transformer(nn.Module):
    """
    Full Transformer model as described in 'Attention Is All You Need'
    (Vaswani et al., 2017).

    Combines an Encoder and a Decoder with a final linear projection
    to produce output vocabulary logits.

    Args:
        src_vocab_size: Size of the source vocabulary
        tgt_vocab_size: Size of the target vocabulary
        d_model: Embedding dimension (default: 512)
        num_heads: Number of attention heads (default: 8)
        num_layers: Number of encoder/decoder layers (default: 6)
        d_ff: Feed-forward inner dimension (default: 2048)
        dropout: Dropout probability (default: 0.1)
        max_seq_len: Maximum sequence length (default: 5000)
        pad_idx: Padding token index (default: 0)
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 5000,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model

        self.encoder = Encoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )

        self.decoder = Decoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )

        # Final projection: d_model → tgt_vocab_size
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        # Initialize parameters (as suggested in the paper)
        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights using Xavier uniform initialization.
        Biases are initialized to zero.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(
        self, src: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, list]:
        """
        Encode the source sequence.

        Args:
            src: Source token indices of shape (batch_size, src_seq_len)
        Returns:
            encoder_output: of shape (batch_size, src_seq_len, d_model)
            src_mask: padding mask for source sequence
            encoder_attn_weights: attention weights for visualization
        """
        src_mask = create_padding_mask(src, self.pad_idx)
        encoder_output, encoder_attn_weights = self.encoder(src, src_mask)
        return encoder_output, src_mask, encoder_attn_weights

    def decode(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, list, list]:
        """
        Decode the target sequence given encoder output.

        Args:
            tgt: Target token indices of shape (batch_size, tgt_seq_len)
            encoder_output: of shape (batch_size, src_seq_len, d_model)
            src_mask: padding mask for source sequence
        Returns:
            decoder_output: of shape (batch_size, tgt_seq_len, d_model)
            self_attn_weights: self-attention weights for visualization
            cross_attn_weights: cross-attention weights for visualization
        """
        tgt_mask = create_causal_mask(tgt.size(1)).to(tgt.device)
        decoder_output, self_attn_weights, cross_attn_weights = self.decoder(
            tgt, encoder_output, src_mask, tgt_mask
        )
        return decoder_output, self_attn_weights, cross_attn_weights

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Full forward pass through the Transformer.

        Args:
            src: Source token indices of shape (batch_size, src_seq_len)
            tgt: Target token indices of shape (batch_size, tgt_seq_len)
        Returns:
            logits: of shape (batch_size, tgt_seq_len, tgt_vocab_size)
            attn_weights: dictionary with all attention weights for visualization
        """
        # Encode source sequence
        encoder_output, src_mask, encoder_attn_weights = self.encode(src)

        # Decode target sequence
        decoder_output, self_attn_weights, cross_attn_weights = self.decode(
            tgt, encoder_output, src_mask
        )

        # Project to vocabulary size
        logits = self.output_projection(decoder_output)

        attn_weights = {
            "encoder": encoder_attn_weights,
            "decoder_self": self_attn_weights,
            "decoder_cross": cross_attn_weights,
        }

        return logits, attn_weights

    def count_parameters(self) -> int:
        """Returns the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
