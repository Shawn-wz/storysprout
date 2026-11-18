"""Integration checks for reward CLI configuration, provenance and checkpoint selection."""
import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import torch

from codes.checkpoint import tokenizer_hash
from codes.config import ModelConfig
from codes.model import build_model
from codes import train_reward


class RewardCLITests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "artifacts").mkdir()
        tokenizer = self.root / "artifacts/tokenizer.json"
        tokenizer.write_text('{"test_fixture": true}', encoding="utf-8")
        config = ModelConfig(32, d_model=16, d_hidden=32, num_heads=4, num_blocks=1, max_seq_len=8)
        self.source = self.root / "sft.pth"
        torch.save({"format_version": 1, "stage": "sft", "epoch": 6,
                    "model_config": asdict(config), "model_state_dict": build_model(config).state_dict(),
                    "tokenizer_sha256": tokenizer_hash(tokenizer)}, self.source)
        self.data_dir = self.root / "custom_preferences"
        self.data_dir.mkdir()
        manifest = {"tokenizer_sha256": tokenizer_hash(tokenizer), "splits": {}, "purpose": "tiny integration fixture"}
        for split, count in [("train", 5), ("validation", 3), ("test", 2)]:
            data = {"chosen_ids": torch.tensor([[2, 4, 10, 3]] * count),
                    "chosen_lengths": torch.full((count,), 4, dtype=torch.long),
                    "rejected_ids": torch.tensor([[2, 4, 11, 12, 3]] * count),
                    "rejected_lengths": torch.full((count,), 5, dtype=torch.long)}
            path = self.data_dir / f"{split}.pt"
            torch.save(data, path)
            manifest["splits"][split] = {"tensors_sha256": tokenizer_hash(path)}
        (self.data_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        self.args = ["--project-dir", str(self.root), "--data-dir", str(self.data_dir),
                     "--checkpoint", str(self.source), "--epochs", "2", "--batch-size", "2",
                     "--device", "cpu", "--log-every", "2"]

    def test_training_saves_and_restores_best_before_single_test_evaluation(self):
        original_hash = tokenizer_hash(self.source)
        original_epoch = train_reward.run_reward_epoch
        evaluations = []

        def observe(model, dataloader, device, optimizer=None, **kwargs):
            if optimizer is None:
                evaluations.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            return original_epoch(model, dataloader, device, optimizer, **kwargs)

        with contextlib.redirect_stdout(io.StringIO()), patch.object(train_reward, "run_reward_epoch", side_effect=observe):
            run_dir = train_reward.main(self.args)
        records = [json.loads(s) for s in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        best = torch.load(run_dir / "best.pth", weights_only=True)
        self.assertEqual(len(evaluations), 4)  # initial validation, two epoch validations, test
        self.assertEqual(best["stage"], "reward")
        self.assertEqual(best["global_step"], best["epoch"] * 3)
        self.assertEqual(best["train_config"]["learning_rate"], 1e-5)
        self.assertEqual(best["val_loss"], min(r["validation"]["loss"] for r in records))
        self.assertEqual(summary["best_epoch"], best["epoch"])
        self.assertTrue(all(torch.equal(v, evaluations[-1][k]) for k, v in best["model_state_dict"].items()))
        self.assertTrue(best["optimizer_state_dict"]["state"])
        self.assertEqual(tokenizer_hash(self.source), original_hash)
        self.assertTrue((run_dir / "epoch_1.pth").is_file())
        self.assertTrue((run_dir / "epoch_2.pth").is_file())

    def test_corrupt_preferences_rejected_before_run_creation(self):
        with (self.data_dir / "train.pt").open("ab") as f:
            f.write(b"tampered")
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            train_reward.main(self.args)
        self.assertFalse((self.root / "models").exists())

    def test_nonpositive_learning_rate_rejected(self):
        with self.assertRaisesRegex(ValueError, "learning_rate"):
            train_reward.main(self.args + ["--learning-rate", "0"])
        self.assertFalse((self.root / "models").exists())

    def test_pretrain_checkpoint_rejected(self):
        checkpoint = torch.load(self.source, weights_only=True)
        checkpoint["stage"] = "pretrain"
        torch.save(checkpoint, self.source)
        with self.assertRaisesRegex(ValueError, "SFT checkpoint"):
            train_reward.main(self.args)


if __name__ == "__main__":
    unittest.main()
