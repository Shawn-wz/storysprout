"""Check SFT wiring by memorizing one training example with a fresh optimizer."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer

from codes.checkpoint import load_checkpoint
from codes.config import TrainConfig
from codes.data import load_tokenizer
from codes.model import Transformer
from codes.sft_data import collate_sft_examples, encode_sft_example
from codes.training import evaluate, train_epoch


def chat_prefix(tokenizer: Tokenizer, prompt: str) -> list[int]:
    """Build the exact SFT input prefix without exposing any answer tokens."""
    markers = [
        tokenizer.token_to_id(name) for name in ("<bos>", "<user>", "<assistant>")
    ]
    if any(token is None for token in markers):
        raise ValueError("Tokenizer must define BOS, user, and assistant tokens.")
    return [markers[0], markers[1]] + tokenizer.encode(prompt).ids + [markers[2]]


@torch.no_grad()
def greedy_response(
    model: Transformer,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
) -> dict[str, Any]:
    """Generate from the request alone, retaining EOS and stop details for diagnosis."""
    if type(max_new_tokens) is not int or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be a nonnegative integer.")
    ids = chat_prefix(tokenizer, prompt)
    if len(ids) > model.max_seq_len:
        raise ValueError("Prompt exceeds the model context limit.")
    eos_id = tokenizer.token_to_id("<eos>")
    if eos_id is None:
        raise ValueError("Tokenizer must define EOS.")
    prefix_length = len(ids)
    device = next(model.parameters()).device
    stop_reason = "token_budget"
    model.eval()
    for _ in range(max_new_tokens):
        if len(ids) >= model.max_seq_len:
            stop_reason = "context_limit"
            break
        features = torch.tensor([ids], dtype=torch.long, device=device)
        next_id = model(features)[:, -1, :].argmax(dim=-1).item()
        ids.append(next_id)
        if next_id == eos_id:
            stop_reason = "eos"
            break
    generated = ids[prefix_length:]
    return {
        "text": tokenizer.decode(generated, skip_special_tokens=True),
        "token_ids": generated,
        "stop_reason": stop_reason,
    }


def main(argv: list[str] | None = None) -> Path:
    """Run a bounded single-example diagnostic and save a report, not model weights."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--project-dir", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args(argv)
    config = TrainConfig(
        seed=args.seed,
        epochs=1,
        batch_size=1,
        learning_rate=args.learning_rate,
        device=args.device,
        log_every=args.log_every,
    )
    if args.steps <= 0 or args.sample_index < 0 or args.max_new_tokens < 0:
        raise ValueError(
            "steps must be positive; sample-index and max-new-tokens nonnegative."
        )
    device = torch.device(config.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Choose a CPU or CUDA device.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Choose --device cpu explicitly if needed."
        )
    torch.manual_seed(config.seed)
    project_dir = args.project_dir.resolve()
    tokenizer = load_tokenizer(project_dir)
    model, _ = load_checkpoint(
        args.checkpoint,
        device=device,
        tokenizer_path=project_dir / "artifacts" / "tokenizer.json",
    )
    rows = [
        json.loads(line)
        for line in (project_dir / "data" / "sft_pilot" / "train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    if args.sample_index >= len(rows):
        raise ValueError("sample-index is outside the training dataset.")
    sample = rows[args.sample_index]
    prompt, response = sample["prompt"], sample["response"]
    encoded = encode_sft_example(tokenizer, prompt, response, model.max_seq_len)
    batch = collate_sft_examples([encoded], pad_id=tokenizer.token_to_id("<pad>"))
    prefix = chat_prefix(tokenizer, prompt)
    # A first-answer prediction belongs to the final input-prefix position.
    mask_length = len(prefix) - 1
    expected = tokenizer.encode(response).ids + [tokenizer.token_to_id("<eos>")]
    if (
        batch[0][0, : len(prefix)].tolist() != prefix
        or not (batch[1][0, :mask_length] == -100).all().item()
        or batch[1][0, mask_length:].tolist() != expected
    ):
        raise ValueError("Training labels and generation prefix are not aligned.")

    initial_loss = evaluate(model, [batch], device, -100)
    before = greedy_response(model, tokenizer, prompt, args.max_new_tokens)
    print(f"Request: {prompt}\nTarget: {response}\nBefore: {before['text']}")
    print(f"initial_sample_loss={initial_loss:.6f}")
    # Reuse the existing SFT update on the same batch; do not restore Adam state.
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    metrics = train_epoch(
        model,
        (batch for _ in range(args.steps)),
        device,
        optimizer,
        -100,
        log_every=config.log_every,
    )
    final_loss = evaluate(model, [batch], device, -100)
    after = greedy_response(model, tokenizer, prompt, args.max_new_tokens)
    exact_match = after["token_ids"] == expected
    print(f"final_sample_loss={final_loss:.6f}\nAfter: {after['text']}")
    print(f"stop_reason={after['stop_reason']}, exact_target_match={exact_match}")
    print("This checks training-example memorization, not generalization.")

    report = {
        "purpose": "single_training_example_memorization_check",
        "source_checkpoint": str(args.checkpoint.resolve()),
        "sample_id": sample.get("id"),
        "sample_index": args.sample_index,
        "prompt": prompt,
        "target": response,
        "target_token_ids": expected,
        "steps": args.steps,
        "learning_rate": config.learning_rate,
        "seed": config.seed,
        "initial_sample_loss": initial_loss,
        "final_sample_loss": final_loss,
        "training_metrics": metrics,
        "before": before,
        "after": after,
        "exact_target_match": exact_match,
    }
    output_dir = (
        project_dir
        / "reports"
        / datetime.now().strftime("overfit_sft_%Y%m%d_%H%M%S_%f")
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    output_path = output_dir / "result.json"
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Report: {output_path}")
    return output_path


if __name__ == "__main__":
    main()
