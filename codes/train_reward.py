"""Train a reward model from SFT weights and evaluate its best checkpoint."""

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

if __package__:
    from .checkpoint import save_checkpoint, tokenizer_hash
    from .config import TrainConfig
    from .reward_model import load_reward_model_from_sft, pairwise_reward_loss
else:
    from checkpoint import save_checkpoint, tokenizer_hash
    from config import TrainConfig
    from reward_model import load_reward_model_from_sft, pairwise_reward_loss

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "reward_preferences_v1"


def make_loader(
    split: str, batch_size: int = 8, data_dir: str | Path | None = None
) -> DataLoader:
    """Load paired token sequences and lengths; shuffle only the training split."""
    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    data = torch.load(data_dir / f"{split}.pt", map_location="cpu", weights_only=True)
    count = data["chosen_ids"].shape[0]
    if count == 0:
        raise ValueError(f"{split} contains no preference pairs.")
    for side in ("chosen", "rejected"):
        ids, lengths = data[f"{side}_ids"], data[f"{side}_lengths"]
        if (
            ids.dtype != torch.long
            or lengths.dtype != torch.long
            or ids.ndim != 2
            or ids.shape[0] != count
            or lengths.shape != (count,)
            or torch.any(lengths <= 0)
            or torch.any(lengths > ids.shape[1])
        ):
            raise ValueError(f"Invalid {side} tensors in {split}.")
    dataset = TensorDataset(
        data["chosen_ids"],
        data["chosen_lengths"],
        data["rejected_ids"],
        data["rejected_lengths"],
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"))


def run_reward_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    log_every: int = 250,
) -> dict[str, float]:
    """Train or evaluate one split and return pair-weighted loss and accuracy."""
    tot_loss = 0
    num_correct = 0
    num_pairs = 0

    if optimizer is not None:
        model.train()
        for step, (
            chosen_ids,
            chosen_lengths,
            rejected_ids,
            rejected_lengths,
        ) in enumerate(dataloader, start=1):
            optimizer.zero_grad()
            chosen_rewards = model(chosen_ids.to(device), chosen_lengths.to(device))
            rejected_rewards = model(
                rejected_ids.to(device), rejected_lengths.to(device)
            )
            batch_loss = pairwise_reward_loss(chosen_rewards, rejected_rewards)
            if not torch.isfinite(batch_loss).item():
                raise FloatingPointError(f"Training loss is not finite at step {step}.")
            batch_loss.backward()
            optimizer.step()
            tot_loss += batch_loss.item() * chosen_rewards.shape[0]
            num_correct += (chosen_rewards > rejected_rewards).long().sum().item()
            num_pairs += chosen_rewards.shape[0]
            if step % log_every == 0:
                print(
                    f"step={step}, loss={tot_loss / num_pairs:.4f}, accuracy={num_correct / num_pairs:.4f}",
                    flush=True,
                )

    else:
        model.eval()
        with torch.no_grad():
            for (
                chosen_ids,
                chosen_lengths,
                rejected_ids,
                rejected_lengths,
            ) in dataloader:
                chosen_rewards = model(chosen_ids.to(device), chosen_lengths.to(device))
                rejected_rewards = model(
                    rejected_ids.to(device), rejected_lengths.to(device)
                )
                batch_loss = pairwise_reward_loss(chosen_rewards, rejected_rewards)
                if not torch.isfinite(batch_loss).item():
                    raise FloatingPointError("Evaluation loss is not finite.")
                tot_loss += batch_loss.item() * chosen_rewards.shape[0]
                num_correct += (chosen_rewards > rejected_rewards).long().sum().item()
                num_pairs += chosen_rewards.shape[0]

    if num_pairs == 0:
        raise ValueError("Preference loader contains no pairs.")
    loss = tot_loss / num_pairs
    accuracy = num_correct / num_pairs
    return {"loss": loss, "accuracy": accuracy}


def main(argv: list[str] | None = None) -> Path:
    """Train from SFT weights and test the checkpoint with the lowest validation loss."""
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(
        description="Train the StorySprout reward model from an SFT checkpoint."
    )
    parser.add_argument(
        "--project-dir", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Preference tensor directory; defaults to data/reward_preferences_v1. Relative paths use the current working directory.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Initialize the backbone from SFT weights with a fresh reward head and Adam optimizer.",
    )
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--log-every", type=int, default=defaults.log_every)
    args = parser.parse_args(argv)
    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        log_every=args.log_every,
    )
    device = torch.device(config.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("This project currently supports CPU and CUDA devices.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Check PyTorch or explicitly choose --device cpu."
        )
    project_dir = args.project_dir.expanduser().resolve()
    data_dir = (
        args.data_dir.expanduser().resolve()
        if args.data_dir
        else project_dir / "data/reward_preferences_v1"
    )
    checkpoint_path = args.checkpoint.expanduser().resolve()
    tokenizer_path = project_dir / "artifacts/tokenizer.json"
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["tokenizer_sha256"] != tokenizer_hash(tokenizer_path):
        raise ValueError(
            "Preference data tokenizer does not match the project tokenizer."
        )
    # Check all input files before creating a run or taking an optimization step.
    data_record = {"data_dir": str(data_dir), "manifest": manifest, "files": {}}
    for split in ("train", "validation", "test"):
        path = data_dir / f"{split}.pt"
        actual_hash = tokenizer_hash(path)
        if actual_hash != manifest["splits"][split]["tensors_sha256"]:
            raise ValueError(f"Preference tensor fingerprint mismatch: {split}.")
        data_record["files"][path.name] = actual_hash
    torch.manual_seed(config.seed)
    model = load_reward_model_from_sft(checkpoint_path, tokenizer_path, device=device)
    loaders = {
        split: make_loader(split, config.batch_size, data_dir)
        for split in ("train", "validation", "test")
    }
    for split, loader in loaders.items():
        tensors = loader.dataset.tensors
        for ids, lengths in ((tensors[0], tensors[1]), (tensors[2], tensors[3])):
            if (
                ids.shape[1] > model.config.max_seq_len
                or torch.any(ids < 0)
                or torch.any(ids >= model.config.vocab_size)
            ):
                raise ValueError(
                    f"Preference tokens exceed the model context or vocabulary: {split}."
                )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    run_dir = (
        project_dir / "models" / datetime.now().strftime("reward_run_%Y%m%d_%H%M%S_%f")
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    data_record["source_checkpoint_sha256"] = tokenizer_hash(checkpoint_path)
    (run_dir / "data_manifest.json").write_text(
        json.dumps(data_record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "model_config": asdict(model.config),
                "train_config": asdict(config),
                "source_checkpoint": str(checkpoint_path),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Run directory: {run_dir}", flush=True)
    print(
        f"learning_rate={config.learning_rate}, trainable_parameters={sum(p.numel() for p in model.parameters()):,}",
        flush=True,
    )
    initial = run_reward_epoch(model, loaders["validation"], device)
    print(
        f"initial_val_loss={initial['loss']:.4f}, initial_val_accuracy={initial['accuracy']:.4f}",
        flush=True,
    )
    best_val_loss, global_step = float("inf"), 0
    best_epoch, best_validation = None, None
    for epoch in range(1, config.epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = perf_counter()
        train_metrics = run_reward_epoch(
            model, loaders["train"], device, optimizer, log_every=config.log_every
        )
        validation = run_reward_epoch(model, loaders["validation"], device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        global_step += len(loaders["train"])
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "train": train_metrics,
            "validation": validation,
            "elapsed_seconds": perf_counter() - started,
        }
        save_args = dict(
            model=model,
            optimizer=optimizer,
            model_config=model.config,
            train_config=config,
            epoch=epoch,
            global_step=global_step,
            val_loss=validation["loss"],
            tokenizer_path=tokenizer_path,
            source_checkpoint=str(checkpoint_path),
            stage="reward",
        )
        save_checkpoint(run_dir / f"epoch_{epoch}.pth", **save_args)
        if validation["loss"] < best_val_loss:
            best_val_loss, best_epoch, best_validation = (
                validation["loss"],
                epoch,
                validation,
            )
            save_checkpoint(run_dir / "best.pth", **save_args)
        with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        print(
            f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f}, train_accuracy={train_metrics['accuracy']:.4f}, "
            f"val_loss={validation['loss']:.4f}, val_accuracy={validation['accuracy']:.4f}",
            flush=True,
        )
    # Select on validation loss; consult test only once after restoring best weights.
    best_checkpoint = torch.load(
        run_dir / "best.pth", map_location="cpu", weights_only=True
    )
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_metrics = run_reward_epoch(model, loaders["test"], device)
    summary = {
        "run_dir": str(run_dir),
        "source_checkpoint": str(checkpoint_path),
        "initial_validation": initial,
        "best_epoch": best_epoch,
        "best_validation": best_validation,
        "test": test_metrics,
        "label_policy": manifest.get("label_policy"),
        "data_purpose": manifest.get("purpose"),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"best_epoch={best_epoch}, test_loss={test_metrics['loss']:.4f}, test_accuracy={test_metrics['accuracy']:.4f}",
        flush=True,
    )
    return run_dir


if __name__ == "__main__":
    main()
