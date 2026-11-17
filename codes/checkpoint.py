"""Save training state and restore current or legacy model checkpoints."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

if __package__:
    from .config import ModelConfig, TrainConfig
    from .model import Transformer, build_model
else:
    from config import ModelConfig, TrainConfig
    from model import Transformer, build_model


def tokenizer_hash(path: str | Path) -> str:
    """Return a SHA-256 fingerprint of the exact tokenizer file contents."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None,
    model_config: ModelConfig,
    train_config: TrainConfig,
    epoch: int,
    global_step: int,
    val_loss: float,
    tokenizer_path: str | Path,
) -> None:
    """Write model state, optional optimizer state, configs, and run metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format_version": 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "epoch": epoch,
        "global_step": global_step,
        "val_loss": val_loss,
        "tokenizer_sha256": tokenizer_hash(tokenizer_path),
    }
    # Replace the destination only after the complete checkpoint is written.
    temporary_path = path.with_name(path.name + ".tmp")
    try:
        torch.save(checkpoint, temporary_path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
    tokenizer_path: str | Path | None = None,
) -> tuple[Transformer, dict[str, Any]]:
    """Restore an eval-mode model and metadata, optionally checking tokenizer identity.

    Legacy checkpoints lack a tokenizer fingerprint, so their identity cannot
    be verified. Optimizer state is returned in metadata but is not applied.
    """
    device = torch.device(device)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("format_version", 0) not in (0, 1):
        raise ValueError("Unsupported checkpoint format version.")
    if "model_config" in checkpoint:
        config = ModelConfig(**checkpoint["model_config"])
    else:
        # Preserve compatibility with checkpoints from the original script.
        config = ModelConfig(
            vocab_size=checkpoint["vocab_size"],
            d_model=checkpoint["d_model"],
            d_hidden=checkpoint["d_hidden"],
            num_heads=checkpoint["num_heads"],
            num_blocks=checkpoint["num_blocks"],
            rope_base=checkpoint["RoPE_base"],
            max_seq_len=checkpoint["max_seq_len"],
        )
    expected_hash = checkpoint.get("tokenizer_sha256")
    if tokenizer_path is not None and expected_hash is not None:
        if tokenizer_hash(tokenizer_path) != expected_hash:
            raise ValueError("Tokenizer does not match the saved checkpoint.")
    state = checkpoint.get("model_state_dict", checkpoint.get("model_state"))
    if state is None:
        raise ValueError("Checkpoint has no model weights.")
    model = build_model(config).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, checkpoint
