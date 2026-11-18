"""Compare two language models using the same chat prompt and sampling seed."""

import argparse
from pathlib import Path

from tokenizers import Tokenizer

from codes.checkpoint import load_checkpoint
from codes.generate import generate


def main(argv: list[str] | None = None) -> None:
    """Print a continuation from each checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument(
        "--prompt",
        default="Write a short story about Alice and Bob finding a lost dog.",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    path = Path(__file__).resolve().parents[1] / "artifacts/tokenizer.json"
    tokenizer = Tokenizer.from_file(str(path))
    for label, checkpoint in (("Before", args.before), ("After", args.after)):
        model, metadata = load_checkpoint(
            checkpoint, device=args.device, tokenizer_path=path
        )
        del metadata
        answer = generate(model, tokenizer, f"<user>{args.prompt}<assistant>")
        print(f"{label}:\n{answer}")
        del model


if __name__ == "__main__":
    main()
