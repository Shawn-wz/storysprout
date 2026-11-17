"""Verify checkpoint continuation through the pretraining entry point."""

import hashlib
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, pre_tokenizers

from codes.checkpoint import load_checkpoint, save_checkpoint
from codes.config import ModelConfig, TrainConfig
from codes.model import build_model
from codes.pretrain import main
from codes.training import train_epoch


ROOT = Path(__file__).resolve().parents[1]


class ResumeTests(unittest.TestCase):
    """Use tiny nondefault models so accidental default reconstruction is detected."""

    def make_project(self, project: Path) -> tuple[Path, ModelConfig]:
        """Create local data and a checkpoint with initialized Adam moments."""
        (project / "data").mkdir()
        (project / "artifacts").mkdir()
        vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3,
                 "hello": 4, "cat": 5, "dog": 6}
        tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.add_special_tokens(["<pad>", "<bos>", "<eos>", "<unk>"])
        tokenizer_path = project / "artifacts" / "tokenizer.json"
        tokenizer.save(str(tokenizer_path))
        for split, stories in (("train", ["hello cat", "hello dog", "cat", "dog"]),
                               ("validation", ["hello cat", "hello dog"])):
            with (project / "data" / f"{split}.jsonl").open("w", encoding="utf-8") as f:
                for story in stories:
                    f.write(json.dumps({"text": story}) + "\n")
        config = ModelConfig(7, d_model=16, d_hidden=32, num_heads=4,
                             num_blocks=2, max_seq_len=12)
        train_config = TrainConfig(learning_rate=0.004, device="cpu")
        model = build_model(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.004)
        batch = (torch.tensor([[1, 4, 5]]), torch.tensor([[4, 5, 2]]))
        train_epoch(model, [batch], torch.device("cpu"), optimizer, 0)
        path = project / "source.pth"
        save_checkpoint(path, model, optimizer, config, train_config,
                        3, 7500, 2.0, tokenizer_path)
        return path, config

    def arguments(self, project: Path, checkpoint: Path) -> list[str]:
        """Request one additional CPU epoch with two local training batches."""
        return ["--project-dir", str(project), "--checkpoint", str(checkpoint),
                "--epochs", "1", "--batch-size", "2", "--device", "cpu"]

    def test_continuation_counters_config_optimizer_and_lr(self) -> None:
        """Preserve architecture, Adam state, counters, source file, and LR policy."""
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            project = Path(directory)
            source, config = self.make_project(project)
            original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            for override in (None, 0.0003):
                with self.subTest(learning_rate=override):
                    args = self.arguments(project, source)
                    if override is not None:
                        args += ["--learning-rate", str(override)]
                    run_dir = main(args)
                    model, checkpoint = load_checkpoint(run_dir / "epoch_4.pth")
                    self.assertEqual(checkpoint["model_config"], asdict(config))
                    self.assertEqual(model.max_seq_len, 12)
                    self.assertEqual(checkpoint["epoch"], 4)
                    self.assertEqual(checkpoint["global_step"], 7502)
                    self.assertEqual(checkpoint["source_checkpoint"], str(source.resolve()))
                    lr = 0.004 if override is None else override
                    self.assertEqual(checkpoint["train_config"]["learning_rate"], lr)
                    optimizer_state = checkpoint["optimizer_state_dict"]
                    self.assertEqual(optimizer_state["param_groups"][0]["lr"], lr)
                    self.assertTrue(all(s["step"].item() == 3
                                        for s in optimizer_state["state"].values()))
                    record = json.loads((run_dir / "metrics.jsonl").read_text())
                    self.assertEqual((record["epoch"], record["global_step"]), (4, 7502))
                    self.assertTrue((run_dir / "best.pth").is_file())
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), original_hash)

    def test_tokenizer_mismatch_is_rejected(self) -> None:
        """Reject a different tokenizer before building data loaders or output files."""
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            project = Path(directory)
            source, _ = self.make_project(project)
            tokenizer_path = project / "artifacts" / "tokenizer.json"
            tokenizer_path.write_text(tokenizer_path.read_text(encoding="utf-8") + "\n",
                                      encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Tokenizer does not match"):
                main(self.arguments(project, source))
            self.assertFalse((project / "models").exists())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_continuation(self) -> None:
        """Restore CPU-saved Adam state and continue updates on CUDA."""
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            project = Path(directory)
            source, _ = self.make_project(project)
            args = self.arguments(project, source)
            args[args.index("--device") + 1] = "cuda"
            run_dir = main(args)
            model, checkpoint = load_checkpoint(run_dir / "epoch_4.pth", device="cuda")
            self.assertEqual(next(model.parameters()).device.type, "cuda")
            self.assertEqual(checkpoint["global_step"], 7502)
            self.assertTrue(all(state["step"].item() == 3
                                for state in checkpoint["optimizer_state_dict"]["state"].values()))

    def test_incomplete_training_state_is_rejected(self) -> None:
        """Require counters and optimizer state instead of silently resetting them."""
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            project = Path(directory)
            source, _ = self.make_project(project)
            original = torch.load(source, weights_only=True)
            for field in ("epoch", "global_step", "optimizer_state_dict"):
                with self.subTest(field=field):
                    incomplete = dict(original)
                    incomplete.pop(field)
                    path = project / f"missing_{field}.pth"
                    torch.save(incomplete, path)
                    with self.assertRaises(ValueError):
                        main(self.arguments(project, path))


if __name__ == "__main__":
    unittest.main()
