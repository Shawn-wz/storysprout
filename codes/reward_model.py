"""Score a complete response using the final non-padding token."""

from pathlib import Path

import torch
from torch import nn

if __package__:
    from .checkpoint import load_checkpoint, model_config_from_checkpoint
    from .config import ModelConfig
    from .model import build_model
else:
    from checkpoint import load_checkpoint, model_config_from_checkpoint
    from config import ModelConfig
    from model import build_model


class RewardModel(nn.Module):
    """Share the language model architecture up to its scalar reward head."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        transformer = build_model(config)
        self.embedding = transformer.embedding
        self.blocks = transformer.blocks
        self.reward_head = nn.Linear(config.d_model, 1, bias=False)

    def forward(self, X: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Return one reward per right-padded sequence."""
        E = self.embedding(X)
        Y = self.blocks(E)
        lasts = Y[torch.arange(Y.shape[0], device=Y.device), lengths - 1, :]
        rewards = self.reward_head(lasts).squeeze(-1)
        return rewards


def load_reward_model_from_sft(
    checkpoint_path: str | Path,
    tokenizer_path: str | Path,
    device: str | torch.device = "cpu",
) -> RewardModel:
    """Copy the SFT backbone, leaving the reward head freshly initialized."""
    sft_model, checkpoint = load_checkpoint(
        checkpoint_path, device="cpu", tokenizer_path=tokenizer_path
    )
    if checkpoint.get("stage") != "sft":
        raise ValueError("Reward initialization requires an SFT checkpoint.")
    model_config = model_config_from_checkpoint(checkpoint)
    reward_model = RewardModel(model_config)
    reward_model.embedding.load_state_dict(sft_model.embedding.state_dict())
    reward_model.blocks.load_state_dict(sft_model.blocks.state_dict())
    reward_model.to(device)
    return reward_model


def pairwise_reward_loss(
    chosen_rewards: torch.Tensor,
    rejected_rewards: torch.Tensor,
) -> torch.Tensor:
    """Minimize the negative log probability of ranking the chosen answer first."""
    loss = -nn.functional.logsigmoid(chosen_rewards - rejected_rewards).mean()
    return loss
