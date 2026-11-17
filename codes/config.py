"""Configuration values and validation for model construction and training."""

import math
from dataclasses import dataclass, fields


@dataclass
class ModelConfig:
    """Store the dimensions required to reconstruct the language model."""

    vocab_size: int
    d_model: int = 256
    d_hidden: int = 1024
    num_heads: int = 4
    num_blocks: int = 4
    rope_base: int = 10000
    max_seq_len: int = 256

    def __post_init__(self) -> None:
        """Reject nonpositive dimensions and incompatible attention heads."""
        for field in fields(self):
            name = field.name
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer.")
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        # RoPE rotates adjacent pairs of features within each head.
        head_dim = self.d_model // self.num_heads
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even.")


@dataclass
class TrainConfig:
    """Store training settings independently of the model architecture."""

    seed: int = 2026
    epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 0.001
    log_every: int = 250
    device: str = "cuda"

    def __post_init__(self) -> None:
        """Validate counts, the random seed, learning rate, and device name."""
        for name in ("epochs", "batch_size", "log_every"):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} must be an integer.")
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        if type(self.seed) is not int:
            raise TypeError("seed must be an integer.")
        if not 0 <= self.seed <= 2**64 - 1:
            raise ValueError("seed must be between 0 and 2**64 - 1.")
        if isinstance(self.learning_rate, bool) or not isinstance(
            self.learning_rate, (int, float)
        ):
            raise TypeError("learning_rate must be a number.")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive.")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a nonempty string.")
