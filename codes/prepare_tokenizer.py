import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


# Resolve paths relative to the project rather than the working directory.
PROJECT_DIR = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_DIR / "data" / "train.jsonl"
OUTPUT_PATH = PROJECT_DIR / "artifacts" / "tokenizer.json"
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>", "<user>", "<assistant>"]


def iter_stories():
    # Yield story text without exposing JSON syntax to the trainer.
    with TRAIN_PATH.open("r", encoding="utf-8") as source:
        for line in source:
            yield json.loads(line)["text"]


def main():
    # Use byte-level BPE and preserve the original whitespace.
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    # Reserve control tokens before learning merge rules from training text.
    trainer = trainers.BpeTrainer(
        vocab_size=8192,
        min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )
    tokenizer.train_from_iterator(iter_stories(), trainer=trainer)

    # Save the vocabulary, merge rules, and tokenizer configuration together.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(OUTPUT_PATH))

    # Verify that saving and reloading preserve encoding and decoding.
    loaded = Tokenizer.from_file(str(OUTPUT_PATH))
    text = "Once upon a time, a little cat found a red ball."
    encoded = loaded.encode(text)
    assert loaded.decode(encoded.ids) == text
    assert loaded.encode(text).ids == tokenizer.encode(text).ids

    print("Vocabulary size:", loaded.get_vocab_size())
    print("Special token IDs:", {t: loaded.token_to_id(t) for t in SPECIAL_TOKENS})
    print("Tokens:", encoded.tokens)
    print("IDs:", encoded.ids)
    print("Decoded:", loaded.decode(encoded.ids))
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
