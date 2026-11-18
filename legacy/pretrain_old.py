"""Original pretraining implementation, retained for compatibility comparisons."""

from pathlib import Path
from datasets import load_dataset
from tokenizers import Tokenizer
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
from functools import partial
from torch.nn.utils.rnn import pad_sequence
from time import perf_counter

D_MODEL = 256
D_HIDDEN = 1024
NUM_HEADS = 4
NUM_BLOCKS = 4
ROPE_BASE = 10000
MAX_SEQ_LEN = 256

SEED = 2026
EPOCH = 3
BATCH = 8
LEARNING_RATE = 0.001


def collate_sequences(batch, pad_id):
    # Right-pad sequences to the longest sequence in this batch.
    padded = pad_sequence(
        batch,
        batch_first=True,
        padding_value=pad_id,
    )

    # Shift each sequence to construct next-token targets.
    features = padded[:, :-1]
    labels = padded[:, 1:]

    return features, labels


class SingleHeadCausalAttention(nn.Module):
    def __init__(self, d_model: int, d_h: int, RoPE_base: int, max_seq_len: int):
        super().__init__()
        self.d_h = d_h
        self.W_Q = nn.Linear(d_model, d_h, bias=False)
        self.W_K = nn.Linear(d_model, d_h, bias=False)
        self.W_V = nn.Linear(d_model, d_h, bias=False)

        # Precompute RoPE rotation factors for all supported positions.
        assert d_h % 2 == 0, "RoPE requires an even head dimension."
        thetas = RoPE_base ** (-torch.arange(0, self.d_h, 2).float() / self.d_h)
        positions = torch.arange(max_seq_len)
        freqs = torch.outer(positions, thetas)
        self.register_buffer(
            "freqs_im",
            torch.polar(torch.ones_like(freqs), freqs),
        )

    def forward(self, X: torch.Tensor):
        Q = self.W_Q(X)
        K = self.W_K(X)
        V = self.W_V(X)
        # Interpret each adjacent pair of features as one complex number.
        Q_complex = torch.view_as_complex(Q.float().reshape(*Q.shape[:-1], -1, 2))
        K_complex = torch.view_as_complex(K.float().reshape(*K.shape[:-1], -1, 2))
        # Select rotation factors for the actual sequence length.
        seq_len = Q.shape[-2]
        if seq_len > self.freqs_im.shape[0]:
            raise ValueError("Sequence length exceeds the RoPE cache.")
        freqs = self.freqs_im[:seq_len]
        # Rotate queries and keys, then restore real-valued features.
        Q_out = torch.view_as_real(Q_complex * freqs).flatten(-2)
        K_out = torch.view_as_real(K_complex * freqs).flatten(-2)
        # scaled dot-product causal attention
        S = Q_out @ K_out.transpose(-1, -2) / math.sqrt(self.d_h)
        M = torch.triu(-torch.inf * torch.ones_like(S), diagonal=1)
        A = torch.softmax(S + M, dim=-1)
        O = A @ V
        return O


class MultiHeadAttention(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, RoPE_base: int, max_seq_len: int
    ):  # requires num_heads | d_model
        super().__init__()
        self.heads = nn.ModuleList(
            [
                SingleHeadCausalAttention(
                    d_model, d_model // num_heads, RoPE_base, max_seq_len
                )
                for _ in range(num_heads)
            ]
        )
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, X: torch.Tensor):
        C = torch.concat([head(X) for head in self.heads], dim=-1)
        O = self.W_O(C)
        return O


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_hidden: int,
        RoPE_base: int,
        max_seq_len: int,
    ):
        super().__init__()
        self.MHA = MultiHeadAttention(d_model, num_heads, RoPE_base, max_seq_len)
        self.FFN = nn.Sequential(
            nn.Linear(d_model, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_model)
        )
        self.LN = nn.LayerNorm(d_model)

    def forward(self, X: torch.Tensor):
        # Pre-LN
        H = X + self.MHA(self.LN(X))
        Y = H + self.FFN(self.LN(H))
        return Y


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_hidden: int,
        num_blocks: int,
        RoPE_base: int,
        max_seq_len: int,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(d_model, num_heads, d_hidden, RoPE_base, max_seq_len)
                for _ in range(num_blocks)
            ]
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, X: torch.Tensor):
        E = self.embedding(X)
        Y = self.blocks(E)
        O = self.lm_head(Y)
        return O


def train(model, dataloader, device, optimizer, vocab_size, pad_id):
    model.train()
    total_loss = 0.0
    total_tokens = 0

    torch.cuda.synchronize(device)
    start_time = perf_counter()

    for step, (features, labels) in enumerate(dataloader["train"], start=1):
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(features)
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size), labels.reshape(-1), ignore_index=pad_id
        )

        assert torch.isfinite(loss).item(), "Loss is not finite."

        loss.backward()
        optimizer.step()

        # Accumulate loss weighted by the number of valid targets.
        batch_tokens = (labels != pad_id).sum().item()
        batch_loss = loss.item()
        total_loss += batch_loss * batch_tokens
        total_tokens += batch_tokens

        if step % 250 == 0:
            average_loss = total_loss / total_tokens
            torch.cuda.synchronize(device)
            elapsed = perf_counter() - start_time
            print(
                f"step={step}, "
                f"batch_loss={batch_loss:.4f}, "
                f"average_loss={average_loss:.4f}, "
                f"elapsed={elapsed:.1f}s"
            )


def evaluate(model, dataloader, device, pad_id):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)

            # Sum losses over valid target tokens.
            loss_sum = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=pad_id,
                reduction="sum",
            )

            total_loss += loss_sum.item()
            total_tokens += (labels != pad_id).sum().item()

    return total_loss / total_tokens


def main():
    torch.manual_seed(SEED)
    project_dir = Path(__file__).resolve().parents[1]

    # Load data
    text_dataset = load_dataset(
        "json",
        data_files={
            "train": str(project_dir / "data" / "train.jsonl"),
            "validation": str(project_dir / "data" / "validation.jsonl"),
        },
    )

    # Prepare tokenizer
    tokenizer = Tokenizer.from_file(str(project_dir / "artifacts" / "tokenizer.json"))
    vocab_size = tokenizer.get_vocab_size()
    pad_id = tokenizer.token_to_id("<pad>")
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")

    # Tokenization
    dataloader = {}
    for split in ["train", "validation"]:
        sequences = []
        for text in text_dataset[split]["text"]:
            ids = [bos_id] + tokenizer.encode(text).ids + [eos_id]
            # Keep one extra token for constructing shifted targets.
            ids = ids[: MAX_SEQ_LEN + 1]
            # Keep each variable-length story as a separate tensor.
            sequences.append(torch.tensor(ids, dtype=torch.long))
        dataloader[split] = DataLoader(
            sequences,
            batch_size=BATCH,
            shuffle=(split == "train"),
            collate_fn=partial(collate_sequences, pad_id=pad_id),
            num_workers=0,
            drop_last=False,
        )

    # Verify that GPU training is available.
    assert torch.cuda.is_available(), "CUDA is not available."
    device = torch.device("cuda")

    # Move the model before constructing the optimizer.
    model = Transformer(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_hidden=D_HIDDEN,
        num_blocks=NUM_BLOCKS,
        RoPE_base=ROPE_BASE,
        max_seq_len=MAX_SEQ_LEN,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Report the number of trainable parameters.
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_parameters:,}\n")

    # Measure validation loss before any training updates.
    initial_val_loss = evaluate(model, dataloader["validation"], device, pad_id)
    print(f"initial_val_loss={initial_val_loss:.4f}")

    # Train
    for epoch in range(1, EPOCH + 1):
        print(f"Epoch {epoch}:")
        train(model, dataloader, device, optimizer, vocab_size, pad_id)
        # Evaluate the same held-out stories after training.
        val_loss = evaluate(model, dataloader["validation"], device, pad_id)
        print(f"val_loss={val_loss:.4f}\n")
        # Save the model
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "vocab_size": vocab_size,
            "d_model": D_MODEL,
            "num_heads": NUM_HEADS,
            "d_hidden": D_HIDDEN,
            "num_blocks": NUM_BLOCKS,
            "RoPE_base": ROPE_BASE,
            "max_seq_len": MAX_SEQ_LEN,
        }
        torch.save(
            checkpoint,
            str(
                project_dir
                / "models"
                / f"pre_trained_model_epoch={epoch}_lr={LEARNING_RATE}.pth"
            ),
        )

    # Check
    new_checkpoint = torch.load(str(project_dir / "models" / "pre_trained_model3.pth"))
    new_model = Transformer(
        vocab_size=new_checkpoint["vocab_size"],
        d_model=new_checkpoint["d_model"],
        num_heads=new_checkpoint["num_heads"],
        d_hidden=new_checkpoint["d_hidden"],
        num_blocks=new_checkpoint["num_blocks"],
        RoPE_base=new_checkpoint["RoPE_base"],
        max_seq_len=new_checkpoint["max_seq_len"],
    ).to(device)
    new_model.load_state_dict(new_checkpoint["model_state_dict"])
    model.eval()
    new_model.eval()
    input = torch.tensor([[1, 2]], dtype=torch.long, device=device)
    with torch.no_grad():
        print(torch.allclose(model(input), new_model(input)))


if __name__ == "__main__":
    main()
