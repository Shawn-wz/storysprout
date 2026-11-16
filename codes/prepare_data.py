import json
from pathlib import Path

from datasets import load_dataset

output_dir = Path(r"data")
output_dir.mkdir(parents=True, exist_ok=True)

for split, limit in [("train", 20_000), ("validation", 1_000)]:
    dataset = load_dataset(
        "roneneldan/TinyStories",
        split=split,
        streaming=True,
    )

    subset = dataset.filter(
        lambda row: bool(row["text"].strip())
    ).take(limit)

    output_path = output_dir / f"{split}.jsonl"
    count = 0

    with output_path.open("w", encoding="utf-8") as f:
        for row in subset:
            record = {"text": row["text"].strip()}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(f"{split}: {count} stories -> {output_path}")