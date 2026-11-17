"""Load local stories and build right-padded next-token prediction batches."""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from pathlib import Path

import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

if __package__:
    from .config import ModelConfig, TrainConfig
else:
    from config import ModelConfig, TrainConfig


def special_token_ids(tokenizer: Tokenizer) -> dict[str, int]:
    """Return distinct PAD, BOS, and EOS IDs, rejecting missing tokens."""
    ids = {name: tokenizer.token_to_id(name) for name in ("<pad>", "<bos>", "<eos>")}
    if any(value is None for value in ids.values()):
        raise ValueError("Tokenizer must define <pad>, <bos>, and <eos>.")
    if len(set(ids.values())) != len(ids):
        raise ValueError("Special token IDs must be distinct.")
    return ids


def load_tokenizer(project_dir: str | Path) -> Tokenizer:
    """Load and validate the tokenizer stored under the project artifacts."""
    path = Path(project_dir) / "artifacts" / "tokenizer.json"
    if not path.is_file():
        raise FileNotFoundError(f"Tokenizer not found: {path}")
    tokenizer = Tokenizer.from_file(str(path))
    special_token_ids(tokenizer)
    return tokenizer


def collate_sequences(
    batch: Sequence[torch.Tensor], pad_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad 1D samples and return shifted long tensors [batch, time]."""
    if not batch or any(item.ndim != 1 or item.numel() < 2 for item in batch):
        raise ValueError("Each sample must be a 1D sequence with at least two tokens.")
    # Right-pad sequences to the longest sequence in this batch.
    padded = pad_sequence(batch, batch_first=True, padding_value=pad_id)
    # Shift each sequence to construct next-token targets.
    return padded[:, :-1], padded[:, 1:]


def make_dataset(
    project_dir: str | Path,
    tokenizer: Tokenizer,
    model_config: ModelConfig,
    train_config: TrainConfig,
) -> dict[str, DataLoader]:
    """Build train and validation loaders from the local JSONL story files."""
    project_dir = Path(project_dir)
    data_files = {
        split: str(project_dir / "data" / f"{split}.jsonl")
        for split in ("train", "validation")
    }
    for path in data_files.values():
        if not Path(path).is_file():
            raise FileNotFoundError(f"Dataset not found: {path}")
    if tokenizer.get_vocab_size() != model_config.vocab_size:
        raise ValueError("Tokenizer vocabulary does not match ModelConfig.")
    ids_by_name = special_token_ids(tokenizer)
    pad_id = ids_by_name["<pad>"]
    bos_id = ids_by_name["<bos>"]
    eos_id = ids_by_name["<eos>"]
    text_dataset = load_dataset(
        "json", data_files=data_files, cache_dir=str(project_dir / "data" / ".cache")
    )
    dataloader = {}
    for split in ("train", "validation"):
        sequences = []
        for row in text_dataset[split]:
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{split} contains an empty or invalid story.")
            ids = [bos_id] + tokenizer.encode(text).ids + [eos_id]
            # Keep one extra token for constructing shifted targets.
            ids = ids[: model_config.max_seq_len + 1]
            sequences.append(torch.tensor(ids, dtype=torch.long))
        if not sequences:
            raise ValueError(f"{split} must contain at least one story.")
        dataloader[split] = DataLoader(
            sequences,
            batch_size=train_config.batch_size,
            shuffle=(split == "train"),
            collate_fn=partial(collate_sequences, pad_id=pad_id),
            num_workers=0,
            drop_last=False,
        )
    return dataloader
