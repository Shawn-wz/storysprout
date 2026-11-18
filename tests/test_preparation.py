"""Check full-data preparation and preflight without optimizer updates."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from codes.config import ModelConfig, TrainConfig
from codes.data import load_tokenizer, make_dataset
from codes.pretrain import main
from tools.data_ablation import digest, write_json
from tools.prepare_full_data import build_cache

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT)
        self.project = Path(self.temporary.name)
        (self.project / "artifacts").mkdir()
        shutil.copyfile(ROOT / "artifacts/tokenizer.json",
                        self.project / "artifacts/tokenizer.json")
        self.directory = self.project / "prepared"
        self.directory.mkdir()
        for split in ("train", "validation"):
            records = [{"text": "A small cat found a red ball."},
                       {"text": "Tom gave a book to Lily."}]
            (self.directory / f"{split}.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        self.tokenizer = load_tokenizer(self.project)
        self.model_config = ModelConfig(vocab_size=self.tokenizer.get_vocab_size())
        self.train_config = TrainConfig(device="cpu", batch_size=2)
        self.metadata = {
            "format_version": 1, "max_seq_len": 256,
            "vocab_size": self.tokenizer.get_vocab_size(),
            "tokenizer_sha256": digest(self.project / "artifacts/tokenizer.json"),
            "splits": {s: build_cache(self.directory, s, self.tokenizer, 256)
                       for s in ("train", "validation")},
        }
        write_json(self.directory / "tokenized_manifest.json", self.metadata)

    def tearDown(self):
        self.temporary.cleanup()

    def test_cache_matches_existing_text_loading(self):
        cached = make_dataset(self.project, self.tokenizer, self.model_config,
                              self.train_config, data_dir=self.directory)
        (self.directory / "tokenized_manifest.json").unlink()
        text = make_dataset(self.project, self.tokenizer, self.model_config,
                            self.train_config, data_dir=self.directory)
        for split in ("train", "validation"):
            self.assertEqual(len(cached[split].dataset), len(text[split].dataset))
            for i in range(2):
                self.assertTrue(torch.equal(cached[split].dataset[i], text[split].dataset[i]))
        for a, b in zip(next(iter(cached["validation"])), next(iter(text["validation"]))):
            self.assertTrue(torch.equal(a, b))

    def test_corrupt_cache_is_rejected(self):
        path = self.directory / "train.tokens.bin"
        with path.open("ab") as output:
            output.write(b"xx")
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            make_dataset(self.project, self.tokenizer, self.model_config,
                         self.train_config, data_dir=self.directory)

    def test_preflight_never_trains_or_creates_checkpoints(self):
        with patch("codes.pretrain.train_epoch", side_effect=AssertionError("training forbidden")), \
             patch.object(torch.optim.Adam, "step", side_effect=AssertionError("update forbidden")):
            main(["--project-dir", str(self.project), "--data-dir", str(self.directory),
                  "--epochs", "1", "--batch-size", "2", "--device", "cpu", "--dry-run"])
        self.assertFalse((self.project / "models").exists())


if __name__ == "__main__":
    unittest.main()
