"""Training and validation loops with token-weighted loss reporting."""

from __future__ import annotations

from collections.abc import Iterable
from time import perf_counter
from typing import TypedDict

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Optimizer


class EpochMetrics(TypedDict):
    """Summarize one epoch; tokens excludes ignored PAD targets."""

    loss: float
    tokens: int
    steps: int
    elapsed_seconds: float


def synchronize(device: torch.device) -> None:
    """Wait for queued CUDA work before reading wall-clock elapsed time."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_epoch(
    model: nn.Module,
    dataloader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    optimizer: Optimizer,
    pad_id: int,
    log_every: int = 250,
) -> EpochMetrics:
    """Train on one loader pass and return loss, token count, steps, and time."""
    model.train()
    total_loss = 0.0
    total_tokens = 0
    steps = 0
    synchronize(device)
    start_time = perf_counter()

    for step, (features, labels) in enumerate(dataloader, start=1):
        features = features.to(device)
        labels = labels.to(device)
        batch_tokens = (labels != pad_id).sum().item()
        if batch_tokens == 0:
            raise ValueError("Training batch has no valid target tokens.")

        optimizer.zero_grad()
        logits = model(features)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=pad_id,
        )
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f"Training loss is not finite at step {step}.")
        loss.backward()
        optimizer.step()

        # Weight batch means by valid targets, not by padded sequence lengths.
        batch_loss = loss.item()
        total_loss += batch_loss * batch_tokens
        total_tokens += batch_tokens
        steps = step
        if step % log_every == 0:
            synchronize(device)
            elapsed = perf_counter() - start_time
            print(
                f"step={step}, batch_loss={batch_loss:.4f}, "
                f"average_loss={total_loss / total_tokens:.4f}, elapsed={elapsed:.1f}s"
            )

    if total_tokens == 0:
        raise ValueError("Training loader has no valid target tokens.")
    synchronize(device)
    return {
        "loss": total_loss / total_tokens,
        "tokens": total_tokens,
        "steps": steps,
        "elapsed_seconds": perf_counter() - start_time,
    }


def evaluate(
    model: nn.Module,
    dataloader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    pad_id: int,
) -> float:
    """Return mean loss over valid targets, leaving the model in eval mode."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            loss_sum = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=pad_id,
                reduction="sum",
            )
            if not torch.isfinite(loss_sum).item():
                raise FloatingPointError("Validation loss is not finite.")
            total_loss += loss_sum.item()
            total_tokens += (labels != pad_id).sum().item()

    if total_tokens == 0:
        raise ValueError("Validation loader has no valid target tokens.")
    return total_loss / total_tokens
