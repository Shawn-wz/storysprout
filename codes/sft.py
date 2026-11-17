"""SFT"""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import torch

if __package__:
    from .config import TrainConfig
    from .data import load_tokenizer
    from .training import train_epoch, evaluate
    from .checkpoint import (
        save_checkpoint,
        load_checkpoint,
        model_config_from_checkpoint,
    )
    from .sft_data import make_sft_dataset
else:
    from config import TrainConfig
    from data import load_tokenizer
    from training import train_epoch, evaluate
    from checkpoint import (
        save_checkpoint,
        load_checkpoint,
        model_config_from_checkpoint,
    )
    from sft_data import make_sft_dataset


def main(argv: list[str] | None = None) -> Path:
    """Train for the requested additional epochs and return a new run directory."""
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="SFT the StorySprout language model.")
    parser.add_argument(
        "--project-dir", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing train.jsonl and validation.jsonl; defaults to data/sft_pilot. Relative paths use the current working directory.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=defaults.epochs,
        help="Number of additional epochs for this run.",
    )
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate for the fresh SFT optimizer; defaults to TrainConfig.",
    )
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--log-every", type=int, default=defaults.log_every)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Initialize SFT model weights from this checkpoint with a fresh optimizer.",
    )
    args = parser.parse_args(argv)
    train_config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=(
            args.learning_rate
            if args.learning_rate is not None
            else defaults.learning_rate
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
    data_dir = (
        args.data_dir.expanduser().resolve()
        if args.data_dir is not None
        else project_dir / "data" / "sft_pilot"
    )
    tokenizer = load_tokenizer(project_dir)
    tokenizer_path = project_dir / "artifacts" / "tokenizer.json"
    checkpoint_path = (
        args.checkpoint.expanduser().resolve() if args.checkpoint else None
    )
    start_epoch = 0
    global_step = 0
    model, checkpoint = load_checkpoint(
        checkpoint_path, device=device, tokenizer_path=tokenizer_path
    )
    model_config = model_config_from_checkpoint(checkpoint)
    start_epoch = 0
    global_step = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.learning_rate)
    print(f"learning_rate={train_config.learning_rate}")
    dataloader = make_sft_dataset(
        project_dir, tokenizer, model_config, train_config, data_dir=data_dir
    )
    print(f"Data directory: {data_dir}")
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_parameters:,}")
    initial_val_loss = evaluate(model, dataloader["validation"], device, -100)
    print(f"initial_val_loss={initial_val_loss:.4f}")

    # Keep separate runs from overwriting earlier experiment checkpoints.
    run_dir = (
        project_dir / "models" / datetime.now().strftime("sft_run_%Y%m%d_%H%M%S_%f")
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    data_record = {
        "data_dir": str(data_dir),
        "splits": {
            split: {
                "rows": len(dataloader[split].dataset),
                "sha256": sha256((data_dir / f"{split}.jsonl").read_bytes()).hexdigest(),
            }
            for split in ("train", "validation")
        },
    }
    (run_dir / "data_manifest.json").write_text(
        json.dumps(data_record, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Run directory: {run_dir}")
    best_val_loss = float("inf")
    for epoch in range(start_epoch + 1, start_epoch + train_config.epochs + 1):
        print(f"Epoch {epoch}:")
        metrics = train_epoch(
            model,
            dataloader["train"],
            device,
            optimizer,
            -100,
            log_every=train_config.log_every,
        )
        global_step += metrics["steps"]
        val_loss = evaluate(model, dataloader["validation"], device, -100)
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
            source_checkpoint=(
                str(checkpoint_path) if checkpoint_path is not None else None
            ),
            stage="sft",
        )
        save_checkpoint(run_dir / f"epoch_{epoch}.pth", **save_args)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(run_dir / "best.pth", **save_args)
        with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
            record = {
                "epoch": epoch,
                "global_step": global_step,
                "val_loss": val_loss,
                **metrics,
            }
            f.write(json.dumps(record) + "\n")
    return run_dir


if __name__ == "__main__":
    main()
