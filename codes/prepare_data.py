"""Download fixed TinyStories subsets without replacing existing files by default."""

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def main(argv: list[str] | None = None) -> None:
    """Download and validate each split before replacing its destination file."""
    parser = argparse.ArgumentParser(description="Download the TinyStories subset.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    output_dir = Path(__file__).resolve().parents[1] / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    limits = [("train", 100_000), ("validation", 2_000)]
    for split, _ in limits:
        path = output_dir / f"{split}.jsonl"
        if path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Already exists: {path}. Use --overwrite to replace it."
            )
    for split, limit in limits:
        dataset = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
        subset = dataset.filter(lambda row: bool(row["text"].strip())).take(limit)
        output_path = output_dir / f"{split}.jsonl"
        temporary_path = output_path.with_suffix(".jsonl.tmp")
        count = 0
        # A failed download must not truncate a previously complete subset.
        try:
            with temporary_path.open("w", encoding="utf-8") as f:
                for row in subset:
                    record = {"text": row["text"].strip()}
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
            if count != limit:
                raise ValueError(
                    f"Expected {limit} stories in {split}, received {count}."
                )
            temporary_path.replace(output_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        print(f"{split}: {count} stories -> {output_path}")


if __name__ == "__main__":
    main()
