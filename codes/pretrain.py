"""Coordinate local data loading, pretraining, evaluation, and checkpointing."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import torch

if __package__:
    from .config import ModelConfig, TrainConfig
    from .data import load_tokenizer, make_dataset, special_token_ids
    from .model import build_model
    from .training import train_epoch, evaluate
    from .checkpoint import save_checkpoint, load_checkpoint, model_config_from_checkpoint
else:
    from config import ModelConfig, TrainConfig
    from data import load_tokenizer, make_dataset, special_token_ids
    from model import build_model
    from training import train_epoch, evaluate
    from checkpoint import save_checkpoint, load_checkpoint, model_config_from_checkpoint


def main(argv: list[str] | None = None) -> Path:
    """Train for the requested additional epochs and return a new run directory."""
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="Pretrain the StorySprout language model.")
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--epochs", type=int, default=defaults.epochs,
        help="Number of additional epochs for this run.",
    )
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument(
        "--learning-rate", type=float, default=None,
        help="Override the saved optimizer learning rate; use the default for fresh runs.",
    )
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--log-every", type=int, default=defaults.log_every)
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Continue from a training checkpoint without replacing the original run.",
    )
    args = parser.parse_args(argv)
    train_config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=(
            args.learning_rate if args.learning_rate is not None else defaults.learning_rate
        ),
        device=args.device,
        log_every=args.log_every,
    )
    device = torch.device(train_config.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("This project currently supports CPU and CUDA devices.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Check PyTorch or explicitly choose --device cpu."
        )
    torch.manual_seed(train_config.seed)
    project_dir = args.project_dir.resolve()
    tokenizer = load_tokenizer(project_dir)
    tokenizer_path = project_dir / "artifacts" / "tokenizer.json"
    checkpoint_path = args.checkpoint.expanduser().resolve() if args.checkpoint else None
    start_epoch = 0
    global_step = 0
    if args.checkpoint is None:
        model_config = ModelConfig(vocab_size=tokenizer.get_vocab_size())
        model = build_model(model_config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=train_config.learning_rate)
    else:
        model, checkpoint = load_checkpoint(
            checkpoint_path, device=device, tokenizer_path=tokenizer_path
        )
        model_config = model_config_from_checkpoint(checkpoint)
        for name in ("epoch", "global_step"):
            value = checkpoint.get(name)
            if type(value) is not int or value < 0:
                raise ValueError(f"Training checkpoint must contain a nonnegative integer {name}.")
        start_epoch = checkpoint["epoch"]
        global_step = checkpoint["global_step"]
        optimizer_state = checkpoint.get(
            "optimizer_state_dict", checkpoint.get("optimizer_state")
        )
        if optimizer_state is None:
            raise ValueError("Continuing training requires saved optimizer state.")
        optimizer = torch.optim.Adam(model.parameters(), lr=train_config.learning_rate)
        optimizer.load_state_dict(optimizer_state)
        # Loading optimizer state restores its LR; explicit CLI values take precedence.
        if args.learning_rate is not None:
            for group in optimizer.param_groups:
                group["lr"] = train_config.learning_rate
        else:
            train_config = replace(train_config, learning_rate=optimizer.param_groups[0]["lr"])
        print(f"Continuing from epoch={start_epoch}, global_step={global_step}")
    print(f"learning_rate={train_config.learning_rate}")
    dataloader = make_dataset(project_dir, tokenizer, model_config, train_config)
    pad_id = special_token_ids(tokenizer)["<pad>"]
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_parameters:,}")
    initial_val_loss = evaluate(model, dataloader["validation"], device, pad_id)
    print(f"initial_val_loss={initial_val_loss:.4f}")

    # Keep separate runs from overwriting earlier experiment checkpoints.
    run_dir = project_dir / "models" / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"Run directory: {run_dir}")
    best_val_loss = float("inf")
    for epoch in range(start_epoch + 1, start_epoch + train_config.epochs + 1):
        print(f"Epoch {epoch}:")
        metrics = train_epoch(
            model, dataloader["train"], device, optimizer, pad_id,
            log_every=train_config.log_every,
        )
        global_step += metrics["steps"]
        val_loss = evaluate(model, dataloader["validation"], device, pad_id)
        print(f"train_loss={metrics['loss']:.4f}, val_loss={val_loss:.4f}\n")
        save_args = dict(
            model=model,
            optimizer=optimizer,
            model_config=model_config,
            train_config=train_config,
            epoch=epoch,
            global_step=global_step,
            val_loss=val_loss,
            tokenizer_path=tokenizer_path,
            source_checkpoint=str(checkpoint_path) if checkpoint_path is not None else None,
        )
        save_checkpoint(run_dir / f"epoch_{epoch}.pth", **save_args)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(run_dir / "best.pth", **save_args)
        with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
            record = {"epoch": epoch, "global_step": global_step, "val_loss": val_loss, **metrics}
            f.write(json.dumps(record) + "\n")
    return run_dir


if __name__ == "__main__":
    main()
