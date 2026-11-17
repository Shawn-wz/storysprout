import torch
from tokenizers import Tokenizer
from torch.nn.utils.rnn import pad_sequence
from collections.abc import Sequence
from pathlib import Path
from torch.utils.data import DataLoader
from datasets import load_dataset
from functools import partial

if __package__:
    from .data import special_token_ids
    from .config import ModelConfig, TrainConfig
else:
    from data import special_token_ids
    from config import ModelConfig, TrainConfig


def encode_sft_example(
    tokenizer: Tokenizer,
    prompt: str,
    response: str,
    max_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    ids_by_name = special_token_ids(tokenizer)
    bos_id = ids_by_name["<bos>"]
    user_id = tokenizer.token_to_id("<user>")
    assistant_id = tokenizer.token_to_id("<assistant>")
    if user_id is None:
        raise KeyError("user_id does not exist.")
    if assistant_id is None:
        raise KeyError("assistant_id does not exist.")
    eos_id = ids_by_name["<eos>"]
    prompt_ids = tokenizer.encode(prompt).ids
    response_ids = tokenizer.encode(response).ids
    features = torch.tensor(
        [bos_id, user_id] + prompt_ids + [assistant_id] + response_ids
    )
    if features.numel() + 1 > max_seq_len:
        raise ValueError("SFT example exceeds the context limit.")
    labels = torch.tensor([-100] * (2 + len(prompt_ids)) + response_ids + [eos_id])
    return features, labels


def collate_sft_examples(
    batch: Sequence[tuple[torch.Tensor, torch.Tensor]],
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad shifted 1D samples and return long tensors [batch, time]."""
    if not batch:
        raise ValueError("Batch must not be empty.")
    for features, labels in batch:
        if (
            features.ndim != 1
            or labels.ndim != 1
            or features.shape != labels.shape
            or features.numel() < 2
        ):
            raise ValueError("Each sample must contain matching nonempty 1D tensors.")
    # Right-pad sequences to the longest sequence in this batch.
    features_batch, labels_batch = zip(*batch)
    padded_features = pad_sequence(
        features_batch, batch_first=True, padding_value=pad_id
    )
    padded_labels = pad_sequence(labels_batch, batch_first=True, padding_value=-100)
    return padded_features, padded_labels


def make_sft_dataset(
    project_dir: str | Path,
    tokenizer: Tokenizer,
    model_config: ModelConfig,
    train_config: TrainConfig,
    data_dir: str | Path | None = None,
) -> dict[str, DataLoader]:
    """Build local JSONL loaders, using the pilot dataset by default."""
    project_dir = Path(project_dir)
    dataset_dir = (
        Path(data_dir).expanduser().resolve()
        if data_dir is not None
        else project_dir / "data" / "sft_pilot"
    )
    data_files = {
        split: str(dataset_dir / f"{split}.jsonl")
        for split in ("train", "validation")
    }
    for path in data_files.values():
        if not Path(path).is_file():
            raise FileNotFoundError(f"Dataset not found: {path}")
    if tokenizer.get_vocab_size() != model_config.vocab_size:
        raise ValueError("Tokenizer vocabulary does not match ModelConfig.")
    ids_by_name = special_token_ids(tokenizer)
    pad_id = ids_by_name["<pad>"]
    text_dataset = load_dataset(
        "json", data_files=data_files, cache_dir=str(project_dir / "data" / ".cache")
    )
    dataloader = {}
    for split in ("train", "validation"):
        sequences = []
        for row in text_dataset[split]:
            prompt = row.get("prompt")
            response = row.get("response")
            if (
                not isinstance(prompt, str)
                or not prompt.strip()
                or not isinstance(response, str)
                or not response.strip()
            ):
                raise ValueError(f"{split} contains an empty or invalid story.")
            sequences.append(
                encode_sft_example(
                    tokenizer, prompt, response, model_config.max_seq_len
                )
            )
        if not sequences:
            raise ValueError(f"{split} must contain at least one task.")
        dataloader[split] = DataLoader(
            sequences,
            batch_size=train_config.batch_size,
            shuffle=(split == "train"),
            collate_fn=partial(collate_sft_examples, pad_id=pad_id),
            num_workers=0,
            drop_last=False,
        )
    return dataloader
