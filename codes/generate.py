"""Sample short continuations from a trained causal language model."""

import math

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

if __package__:
    from .data import special_token_ids
    from .model import Transformer
else:
    from data import special_token_ids
    from model import Transformer


def generate(
    model: Transformer,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    seed: int = 2026
) -> str:
    """Return generated text without the prompt, leaving the model in eval mode.

    Stop at EOS, max_new_tokens, or the total context limit including BOS.
    Reject prompts that already exceed that limit.
    """
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise TypeError("temperature should be a number.")
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("temperature should be positive and finite.")
    if type(max_new_tokens) is not int:
        raise TypeError("max_new_tokens should be an integer.")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens should not be negative.")
    if not isinstance(prompt, str):
        raise TypeError("prompt should be a string.")
    if type(seed) is not int:
        raise TypeError("seed must be an integer.")
    if not 0 <= seed <= 2**64 - 1:
        raise ValueError("seed must be between 0 and 2**64 - 1.")
    torch.manual_seed(seed)
    ids_by_name = special_token_ids(tokenizer)
    bos_id = ids_by_name["<bos>"]
    eos_id = ids_by_name["<eos>"]
    ids = [bos_id] + tokenizer.encode(prompt).ids
    prompt_length = len(ids)
    if prompt_length > model.max_seq_len:
        raise ValueError("Prompt including BOS exceeds the model context limit.")
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        while (
            len(ids) - prompt_length < max_new_tokens
            and ids[-1] != eos_id
            and len(ids) < model.max_seq_len
        ):
            features = torch.tensor([ids], dtype=torch.long, device=device)
            logits = model(features)[:, -1, :]
            probs = F.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            next_id = next_token.item()
            ids.append(next_id)
    # Use token offsets rather than character offsets to isolate the continuation.
    ans = tokenizer.decode(ids[prompt_length:], skip_special_tokens=True)
    return ans
