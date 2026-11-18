import torch


def compute_group_advantages(
    rewards: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Standardize rewards within each prompt group along the last dimension."""
    means = rewards.mean(dim=-1, keepdim=True)
    stds = rewards.std(dim=-1, unbiased=False, keepdim=True)
    advantages = (rewards - means) / (stds + eps)
    return advantages.detach()


def compute_token_log_probs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Return log probabilities of the observed next tokens, shaped [N, T-1]."""
    log_probs = logits[:, :-1, :].log_softmax(dim=-1)
    gathered = torch.gather(log_probs, -1, input_ids[:, 1:].unsqueeze(-1))
    return gathered.squeeze(-1)


def make_response_mask(
    prompt_lengths: torch.Tensor,
    sequence_lengths: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    """Select response targets, including sampled EOS and excluding padding."""
    # Column zero in the log probabilities predicts the token at position one.
    target = torch.arange(1, seq_len, device=prompt_lengths.device)
    return (prompt_lengths.unsqueeze(-1) <= target) & (
        target < sequence_lengths.unsqueeze(-1)
    )


def grpo_policy_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """Average the clipped policy loss over tokens per response, then over responses."""
    valid_counts = response_mask.sum(dim=-1)
    if (valid_counts == 0).any():
        raise ValueError("Each response must contain at least one valid token.")
    probs_ratio = torch.exp(new_log_probs - old_log_probs.detach())
    adv = advantages.detach().unsqueeze(-1)
    scores = torch.minimum(
        probs_ratio * adv, torch.clamp(probs_ratio, 1 - clip_eps, 1 + clip_eps) * adv
    )
    loss = -((response_mask * scores).sum(dim=-1) / valid_counts).mean()
    return loss


def grpo_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_eps: float = 0.2,
    beta: float = 0.04,
) -> torch.Tensor:
    """Add a sampled KL penalty against the reference policy to the clipped loss."""
    valid_counts = response_mask.sum(dim=-1)
    if (valid_counts == 0).any():
        raise ValueError("Each response must contain at least one valid token.")
    d = ref_log_probs.detach() - new_log_probs
    k = torch.expm1(d) - d
    KL_loss = ((response_mask * k).sum(dim=-1) / valid_counts).mean()
    loss = (
        grpo_policy_loss(
            new_log_probs, old_log_probs, advantages, response_mask, clip_eps
        )
        + beta * KL_loss
    )
    return loss


@torch.no_grad()
def sample_response_ids(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    eos_id: int,
    max_new_tokens: int = 64,
) -> dict:
    """Sample one completion and return its full token sequence and lengths."""
    model.eval()
    sequence = prompt_ids
    for _ in range(max_new_tokens):
        if sequence[-1] == eos_id or len(sequence) >= model.max_seq_len:
            break
        logits = model(sequence.unsqueeze(0))
        probs = (logits[:, -1, :]).softmax(dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        sequence = torch.cat((sequence, next_token.reshape(1)), dim=0)
    return {
        "input_ids": sequence,
        "prompt_length": len(prompt_ids),
        "sequence_length": len(sequence),
        "ended_with_eos": (sequence[-1] == eos_id).item(),
    }


def sample_response_groups(
    model: torch.nn.Module,
    prompts: list[torch.Tensor],
    eos_id: int,
    group_size: int = 4,
    max_new_tokens: int = 64,
) -> list[dict]:
    """Sample responses in prompt order, keeping each group contiguous."""
    samples = []
    for prompt in prompts:
        for _ in range(group_size):
            sample = sample_response_ids(model, prompt, eos_id, max_new_tokens)
            samples.append(sample)
    return samples


def collate_rollouts(samples: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    """Right-pad samples and align the response mask with next-token targets."""
    if not samples:
        raise ValueError("Cannot collate an empty rollout batch.")
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [sample["input_ids"] for sample in samples],
        batch_first=True,
        padding_value=pad_id,
    )
    prompt_lengths = torch.tensor(
        [sample["prompt_length"] for sample in samples],
        dtype=torch.long,
        device=input_ids.device,
    )
    sequence_lengths = torch.tensor(
        [sample["sequence_length"] for sample in samples],
        dtype=torch.long,
        device=input_ids.device,
    )
    return {
        "input_ids": input_ids,
        "prompt_lengths": prompt_lengths,
        "sequence_lengths": sequence_lengths,
        "response_mask": make_response_mask(
            prompt_lengths, sequence_lengths, input_ids.size(1)
        ),
    }


@torch.no_grad()
def score_rollouts(
    reward_model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    eos_id: int,
) -> torch.Tensor:
    """Return one scalar reward per rollout without changing the policy batch."""
    input_ids = batch["input_ids"]
    lengths = batch["sequence_lengths"]
    rows = torch.arange(input_ids.size(0), device=input_ids.device)
    needs_eos = input_ids[rows, lengths - 1] != eos_id
    score_lengths = lengths + needs_eos.long()
    width = score_lengths.max().item()
    if width > reward_model.config.max_seq_len:
        raise ValueError("Reward input including EOS exceeds the model context length.")

    # The reward model expects EOS; an added EOS is not a sampled policy action.
    if width > input_ids.size(1):
        score_ids = torch.nn.functional.pad(input_ids, (0, 1))
    else:
        score_ids = input_ids[:, :width].clone()
    score_ids[rows[needs_eos], lengths[needs_eos]] = eos_id
    reward_model.eval()
    return reward_model(score_ids, score_lengths)


def collect_rollout(
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    reward_model: torch.nn.Module,
    prompts: list[torch.Tensor],
    eos_id: int,
    pad_id: int,
    group_size: int = 4,
    max_new_tokens: int = 64,
    eps: float = 1e-6,
):
    """Sample and score completions, then return normalized advantages."""
    samples = []
    for prompt in prompts:
        budget = min(
            max_new_tokens,
            model.max_seq_len - len(prompt),
            reward_model.config.max_seq_len - len(prompt) - 1,
        )
        if budget < 1:
            raise ValueError("No room for a response token and reward EOS.")
        samples.extend(sample_response_groups(model, [prompt], eos_id, group_size, budget))
    batch = collate_rollouts(samples, pad_id)
    rewards = score_rollouts(reward_model, batch, eos_id)
    advantages = compute_group_advantages(
        rewards.reshape(len(prompts), group_size), eps
    ).flatten()
    ids = batch["input_ids"]
    ref_model.eval()
    with torch.no_grad():
        old_log_probs = compute_token_log_probs(model(ids), ids)
        ref_log_probs = compute_token_log_probs(ref_model(ids), ids)
    return old_log_probs, ref_log_probs, rewards, advantages, batch
