from datasets import load_dataset

dataset = load_dataset(
    "json",
    data_files={
        "train": r"data\train.jsonl",
        "validation": r"data\validation.jsonl",
    },
)

print(dataset)
print(dataset["train"][0]["text"])