"""Train a GRPO policy from an SFT checkpoint and a frozen reward model."""

import argparse
import json
import math
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

if __package__:
    from .checkpoint import (
        load_checkpoint,
        model_config_from_checkpoint,
        save_checkpoint,
        tokenizer_hash,
    )
    from .config import TrainConfig
    from .grpo import collect_rollout, compute_token_log_probs, grpo_loss
    from .reward_model import RewardModel
else:
    from checkpoint import (
        load_checkpoint,
        model_config_from_checkpoint,
        save_checkpoint,
        tokenizer_hash,
    )
    from config import TrainConfig
    from grpo import collect_rollout, compute_token_log_probs, grpo_loss
    from reward_model import RewardModel


def update_policy(
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    old_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
    beta: float = 0.04,
) -> float:
    """Apply one policy update to a fixed rollout and return its scalar loss."""
    optimizer.zero_grad()
    new_log_probs = compute_token_log_probs(
        policy(batch["input_ids"]), batch["input_ids"]
    )
    loss = grpo_loss(
        new_log_probs,
        old_log_probs.detach(),
        ref_log_probs.detach(),
        advantages,
        batch["response_mask"],
        clip_eps,
        beta,
    )
    loss.backward()
    optimizer.step()
    return loss.item()


def load_prompts(
    path: Path, tokenizer: Tokenizer, tokens: dict[str, int], max_prompt_length: int
) -> list[torch.Tensor]:
    """Encode prompt text only, leaving space for a response and reward EOS."""
    prompts = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            prompt = json.loads(line)["prompt"]
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"Empty or invalid prompt at {path}:{line_number}.")
            ids = [
                tokens["<bos>"],
                tokens["<user>"],
                *tokenizer.encode(prompt).ids,
                tokens["<assistant>"],
            ]
            if len(ids) > max_prompt_length:
                raise ValueError(
                    f"Prompt at {path}:{line_number} exceeds the context budget."
                )
            prompts.append(torch.tensor(ids, dtype=torch.long))
    if not prompts:
        raise ValueError(f"No prompts in {path}.")
    return prompts


def main(argv: list[str] | None = None) -> Path:
    """Load models and prompts, collect rollouts, and checkpoint each epoch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-dir", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Directory containing train.jsonl with a prompt field.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="SFT checkpoint for policy and reference.",
    )
    parser.add_argument(
        "--reward-checkpoint",
        type=Path,
        required=True,
        help="Trained reward checkpoint.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--batch-size", type=int, default=2, help="Prompts per rollout batch."
    )
    parser.add_argument(
        "--group-size", type=int, default=4, help="Responses per prompt."
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--updates-per-rollout", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-every", type=int, default=1)
    args = parser.parse_args(argv)
    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        log_every=args.log_every,
    )
    if args.group_size < 2 or min(args.max_new_tokens, args.updates_per_rollout) < 1:
        parser.error(
            "Need at least two responses per prompt and positive token/update budgets."
        )
    if not 0 < args.clip_eps < 1 or not math.isfinite(args.beta) or args.beta < 0:
        parser.error("Expected 0 < clip-eps < 1 and finite beta >= 0.")
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Use --device cpu.")

    torch.manual_seed(config.seed)
    project = args.project_dir.resolve()
    tokenizer_path = project / "artifacts/tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokens = {
        name: tokenizer.token_to_id(name)
        for name in ("<bos>", "<user>", "<assistant>", "<eos>", "<pad>")
    }
    if None in tokens.values():
        raise ValueError("Tokenizer is missing required special tokens.")
    policy, saved = load_checkpoint(
        args.checkpoint, device="cpu", tokenizer_path=tokenizer_path
    )
    if saved.get("stage") != "sft":
        raise ValueError("--checkpoint must be an SFT checkpoint.")
    model_config = model_config_from_checkpoint(saved)
    del saved
    ref_model = deepcopy(policy).requires_grad_(False).to(device).eval()
    policy.to(device)

    saved = torch.load(args.reward_checkpoint, map_location="cpu", weights_only=True)
    if saved.get("stage") != "reward" or saved.get(
        "tokenizer_sha256"
    ) != tokenizer_hash(tokenizer_path):
        raise ValueError(
            "Expected a trained reward checkpoint with the project tokenizer."
        )
    reward_config = model_config_from_checkpoint(saved)
    if (
        tokenizer.get_vocab_size() != model_config.vocab_size
        or reward_config.vocab_size != model_config.vocab_size
    ):
        raise ValueError(
            "Policy, reward model and tokenizer must share the vocabulary."
        )
    reward_model = RewardModel(reward_config)
    reward_model.load_state_dict(saved["model_state_dict"])
    del saved
    reward_model.requires_grad_(False).to(device).eval()

    data_dir = args.data_dir or project / "data/sft_v1"
    prompts = load_prompts(
        data_dir / "train.jsonl",
        tokenizer,
        tokens,
        min(policy.max_seq_len - 1, reward_config.max_seq_len - 2),
    )
    loader = DataLoader(
        prompts, batch_size=config.batch_size, shuffle=True, collate_fn=list
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    run_dir = project / "models" / datetime.now().strftime("grpo_run_%Y%m%d_%H%M%S_%f")
    global_step = 0
    for epoch in range(1, config.epochs + 1):
        for prompt_batch in loader:
            old_log_probs, ref_log_probs, rewards, advantages, batch = collect_rollout(
                policy,
                ref_model,
                reward_model,
                [prompt.to(device) for prompt in prompt_batch],
                tokens["<eos>"],
                tokens["<pad>"],
                args.group_size,
                args.max_new_tokens,
            )
            for _ in range(args.updates_per_rollout):
                policy.train()
                loss = update_policy(
                    policy,
                    optimizer,
                    batch,
                    old_log_probs,
                    ref_log_probs,
                    advantages,
                    args.clip_eps,
                    args.beta,
                )
                global_step += 1
                if global_step == 1:
                    run_dir.mkdir(parents=True, exist_ok=False)
                    (run_dir / "config.json").write_text(
                        json.dumps(vars(args), default=str, indent=2), encoding="utf-8"
                    )
                    print(f"Run directory: {run_dir}", flush=True)
                if global_step % config.log_every == 0:
                    record = dict(
                        epoch=epoch,
                        step=global_step,
                        loss=loss,
                        mean_reward=rewards.mean().item(),
                    )
                    with (run_dir / "metrics.jsonl").open(
                        "a", encoding="utf-8"
                    ) as stream:
                        stream.write(json.dumps(record) + "\n")
                    print(record, flush=True)
        save_checkpoint(
            run_dir / f"epoch_{epoch}.pth",
            policy,
            optimizer,
            model_config,
            config,
            epoch,
            global_step,
            float("nan"),
            tokenizer_path,
            source_checkpoint=str(args.checkpoint.resolve()),
            stage="grpo",
        )
    return run_dir


if __name__ == "__main__":
    main()
