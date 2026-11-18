"""Decoder-only Transformer layers with rotary position embeddings."""

from __future__ import annotations

import math

import torch
from torch import nn

if __package__:
    from .config import ModelConfig
else:
    from config import ModelConfig


class SingleHeadCausalAttention(nn.Module):
    """Apply rotary positions and causal attention within one head."""

    def __init__(
        self, d_model: int, d_h: int, RoPE_base: int, max_seq_len: int
    ) -> None:
        """Create projections and cache position rotations up to max_seq_len."""
        super().__init__()
        self.d_h = d_h
        self.W_Q = nn.Linear(d_model, d_h, bias=False)
        self.W_K = nn.Linear(d_model, d_h, bias=False)
        self.W_V = nn.Linear(d_model, d_h, bias=False)

        # Precompute RoPE rotation factors for all supported positions.
        if d_h <= 0 or d_h % 2 != 0:
            raise ValueError("RoPE requires a positive, even head dimension.")
        thetas = RoPE_base ** (-torch.arange(0, self.d_h, 2).float() / self.d_h)
        positions = torch.arange(max_seq_len)
        freqs = torch.outer(positions, thetas)
        self.register_buffer(
            "freqs_im",
            torch.polar(torch.ones_like(freqs), freqs),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Map features [batch, time, d_model] to [batch, time, d_h]."""
        Q = self.W_Q(X)
        K = self.W_K(X)
        V = self.W_V(X)
        # Interpret each adjacent pair of features as one complex number.
        Q_complex = torch.view_as_complex(Q.float().reshape(*Q.shape[:-1], -1, 2))
        K_complex = torch.view_as_complex(K.float().reshape(*K.shape[:-1], -1, 2))
        # Select rotation factors for the actual sequence length.
        seq_len = Q.shape[-2]
        if seq_len > self.freqs_im.shape[0]:
            raise ValueError("Sequence length exceeds the RoPE cache.")
        freqs = self.freqs_im[:seq_len]
        # Rotate queries and keys, then restore real-valued features.
        Q_out = torch.view_as_real(Q_complex * freqs).flatten(-2)
        K_out = torch.view_as_real(K_complex * freqs).flatten(-2)
        # Mask future keys before normalizing attention scores.
        S = Q_out @ K_out.transpose(-1, -2) / math.sqrt(self.d_h)
        M = torch.triu(-torch.inf * torch.ones_like(S), diagonal=1)
        A = torch.softmax(S + M, dim=-1)
        return A @ V


class MultiHeadAttention(nn.Module):
    """Concatenate independent attention heads and project to model width."""

    def __init__(
        self, d_model: int, num_heads: int, RoPE_base: int, max_seq_len: int,
        head_dim: int | None = None,
    ) -> None:
        """Project concatenated heads back to residual-stream width."""
        super().__init__()
        if num_heads <= 0 or (head_dim is None and d_model % num_heads != 0):
            raise ValueError("d_model must be divisible by a positive num_heads.")
        head_dim = d_model // num_heads if head_dim is None else head_dim
        self.heads = nn.ModuleList(
            [
                SingleHeadCausalAttention(
                    d_model, head_dim, RoPE_base, max_seq_len
                )
                for _ in range(num_heads)
            ]
        )
        self.W_O = nn.Linear(num_heads * head_dim, d_model, bias=False)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Preserve the input shape [batch, time, d_model]."""
        C = torch.concat([head(X) for head in self.heads], dim=-1)
        return self.W_O(C)


class TransformerBlock(nn.Module):
    """Combine attention and feed-forward sublayers with residual connections."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_hidden: int,
        RoPE_base: int,
        max_seq_len: int,
        head_dim: int | None = None,
    ) -> None:
        """Create the existing shared-normalization pre-LN block."""
        super().__init__()
        self.MHA = MultiHeadAttention(d_model, num_heads, RoPE_base, max_seq_len, head_dim)
        self.FFN = nn.Sequential(
            nn.Linear(d_model, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_model)
        )
        self.LN = nn.LayerNorm(d_model)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Apply both residual sublayers without changing the feature shape."""
        # Keep the shared LayerNorm to preserve the trained architecture.
        H = X + self.MHA(self.LN(X))
        Y = H + self.FFN(self.LN(H))
        return Y


class Transformer(nn.Module):
    """Predict next-token logits with a causal Transformer backbone."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_hidden: int,
        num_blocks: int,
        RoPE_base: int,
        max_seq_len: int,
        head_dim: int | None = None,
    ) -> None:
        """Create embeddings, Transformer blocks, and the vocabulary projection."""
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(d_model, num_heads, d_hidden, RoPE_base, max_seq_len, head_dim)
                for _ in range(num_blocks)
            ]
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Map long token IDs [batch, time] to logits [batch, time, vocab]."""
        if X.ndim != 2 or X.shape[1] == 0:
            raise ValueError("Token IDs must have shape (batch, nonempty sequence).")
        if X.dtype != torch.long:
            raise TypeError("Token IDs must use torch.long.")
        E = self.embedding(X)
        Y = self.blocks(E)
        return self.lm_head(Y)


def build_model(config: ModelConfig) -> Transformer:
    """Construct a model while retaining legacy constructor parameter names."""
    return Transformer(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        num_heads=config.num_heads,
        d_hidden=config.d_hidden,
        num_blocks=config.num_blocks,
        RoPE_base=config.rope_base,
        max_seq_len=config.max_seq_len,
        head_dim=config.head_dim,
    )
