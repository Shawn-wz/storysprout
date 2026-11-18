"""Inspect local story splits and print one training example."""

from pathlib import Path

from datasets import load_dataset


def main() -> None:
    """Print split information and the first training story."""
    project_dir = Path(__file__).resolve().parents[1]
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(project_dir / "data" / "train.jsonl"),
            "validation": str(project_dir / "data" / "validation.jsonl"),
        },
    )
    print(dataset)
    print(dataset["train"][0]["text"])


if __name__ == "__main__":
    main()
