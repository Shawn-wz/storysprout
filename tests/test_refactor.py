"""Regression checks for model behavior, data handling, and checkpoint compatibility."""

import importlib.util
import json
import math
import shutil
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional as F

from codes.config import ModelConfig, TrainConfig
from codes.model import build_model
from codes.data import collate_sequences
from codes.training import train_epoch, evaluate
from codes.checkpoint import save_checkpoint, load_checkpoint
from codes.pretrain import main

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


class RefactorTests(unittest.TestCase):
    """Exercise refactored modules with small batches and existing artifacts."""

    def setUp(self) -> None:
        """Create small valid configurations and a right-padded example batch."""
        self.config = ModelConfig(
            32, d_model=16, d_hidden=32, num_heads=4, num_blocks=2, max_seq_len=16
        )
        self.train_config = TrainConfig(
            seed=0, epochs=1, batch_size=2, device="cpu", log_every=100
        )
        self.batch = collate_sequences(
            [torch.tensor([2, 5, 8, 3]), torch.tensor([2, 6, 3])], 0
        )

    def test_config_validation(self) -> None:
        """Accept valid settings and reject invalid dimensions and learning rates."""
        self.assertEqual(ModelConfig(**asdict(self.config)), self.config)
        for kwargs in (
            {"num_heads": 0},
            {"d_model": 15},
            {"d_model": 12},
            {"d_model": 0},
            {"vocab_size": 0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ModelConfig(**{"vocab_size": 32, **kwargs})
        with self.assertRaises(TypeError):
            ModelConfig(vocab_size=32, num_heads=True)
        with self.assertRaises(TypeError):
            ModelConfig(vocab_size=32, tie_word_embeddings=1)
        for lr in (float("nan"), float("inf"), 0, -1):
            with self.assertRaises(ValueError):
                TrainConfig(learning_rate=lr)
        self.assertEqual(TrainConfig(seed=0).seed, 0)

    def test_model_matches_original_forward_and_update(self) -> None:
        """Preserve logits and one optimizer update from the original model."""
        spec = importlib.util.spec_from_file_location(
            "original", ROOT / "legacy" / "pretrain_old.py"
        )
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        kwargs = asdict(self.config)
        kwargs.pop("head_dim")  # Legacy heads always use d_model // num_heads.
        kwargs.pop("tie_word_embeddings")
        kwargs["RoPE_base"] = kwargs.pop("rope_base")
        baseline = original.Transformer(**kwargs)
        model = build_model(self.config)
        model.load_state_dict(baseline.state_dict())
        x, y = self.batch
        self.assertTrue(torch.equal(baseline(x), model(x)))
        old_optimizer = torch.optim.Adam(baseline.parameters(), lr=0.001)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        old_loss = F.cross_entropy(
            baseline(x).reshape(-1, 32), y.reshape(-1), ignore_index=0
        )
        old_loss.backward()
        old_optimizer.step()
        metrics = train_epoch(model, [self.batch], torch.device("cpu"), optimizer, 0)
        self.assertAlmostEqual(metrics["loss"], old_loss.item())
        for key, value in baseline.state_dict().items():
            self.assertTrue(torch.equal(value, model.state_dict()[key]), key)

    def test_causality_padding_and_shapes(self) -> None:
        """Keep valid outputs invariant to future tokens and right padding."""
        model = build_model(self.config).eval()
        x = torch.tensor([[2, 5, 8, 3]])
        altered = x.clone()
        altered[:, 2:] = 7
        with torch.no_grad():
            self.assertTrue(torch.allclose(model(x)[:, :2], model(altered)[:, :2]))
            padded = F.pad(x, (0, 3))
            self.assertTrue(torch.allclose(model(x), model(padded)[:, :4], atol=1e-6))
        self.assertEqual(model(x).shape, (1, 4, 32))
        for invalid in (x[0], x.float(), torch.empty(1, 0, dtype=torch.long)):
            with self.assertRaises((TypeError, ValueError)):
                model(invalid)

    def test_wide_attention_checkpoint(self) -> None:
        """Retain independently widened heads through training and reload."""
        config = ModelConfig(32, d_model=16, d_hidden=16, num_heads=4,
                             head_dim=8, num_blocks=2, max_seq_len=16)
        model = build_model(config)
        self.assertEqual(model.blocks[0].MHA.W_O.in_features, 32)
        train_epoch(model, [self.batch], torch.device("cpu"),
                    torch.optim.Adam(model.parameters()), 0)
        x, _ = self.batch
        altered = x.clone()
        altered[:, 2:] = 7
        model.eval()
        with torch.no_grad():
            expected = model(x)
            self.assertTrue(torch.equal(expected[:, :2], model(altered)[:, :2]))
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "wide.pth"
            tokenizer = ROOT / "artifacts/tokenizer.json"
            save_checkpoint(path, model, None, config, self.train_config,
                            1, 1, 0.0, tokenizer)
            restored, metadata = load_checkpoint(path, tokenizer_path=tokenizer)
            self.assertEqual(metadata["model_config"]["head_dim"], 8)
            with torch.no_grad():
                self.assertTrue(torch.equal(expected, restored(x)))

    def test_optional_clipping_and_scheduler(self) -> None:
        """Clip before updates and advance the learning rate after each update."""
        model = build_model(self.config)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        observed = []

        def observe_update(opt, args, kwargs):
            norm = torch.linalg.vector_norm(torch.stack([
                p.grad.norm() for p in model.parameters() if p.grad is not None
            ])).item()
            observed.append((opt.param_groups[0]["lr"], norm))

        handle = optimizer.register_step_pre_hook(observe_update)
        train_epoch(model, [self.batch, self.batch], torch.device("cpu"),
                    optimizer, 0, scheduler=scheduler, max_grad_norm=0.01)
        handle.remove()
        self.assertEqual([lr for lr, _ in observed], [0.001, 0.0005])
        self.assertTrue(all(norm <= 0.010001 for _, norm in observed))
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.00025)
        other = torch.optim.Adam(model.parameters())
        with self.assertRaisesRegex(ValueError, "Scheduler"):
            train_epoch(model, [self.batch], torch.device("cpu"), other, 0,
                        scheduler=scheduler)
        for value in (0, -1, float("inf"), float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                train_epoch(model, [self.batch], torch.device("cpu"), optimizer, 0,
                            max_grad_norm=value)

    def test_checkpoint_roundtrip_optimizer_and_tokenizer(self) -> None:
        """Restore model and optimizer state and reject a changed tokenizer."""
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            directory = Path(directory)
            tokenizer_path = directory / "tokenizer.json"
            tokenizer_path.write_text("tokenizer-v1", encoding="utf-8")
            model = build_model(self.config)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            train_epoch(model, [self.batch], torch.device("cpu"), optimizer, 0)
            val_loss = evaluate(model, [self.batch], torch.device("cpu"), 0)
            path = directory / "nested" / "model.pth"
            save_checkpoint(
                path,
                model,
                optimizer,
                self.config,
                self.train_config,
                1,
                1,
                val_loss,
                tokenizer_path,
            )
            restored, checkpoint = load_checkpoint(path, tokenizer_path=tokenizer_path)
            with torch.no_grad():
                self.assertTrue(
                    torch.equal(model(self.batch[0]), restored(self.batch[0]))
                )
            restored_optimizer = torch.optim.Adam(restored.parameters(), lr=0.001)
            restored_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            train_epoch(model, [self.batch], torch.device("cpu"), optimizer, 0)
            train_epoch(
                restored, [self.batch], torch.device("cpu"), restored_optimizer, 0
            )
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, restored.state_dict()[key]), key)
            tokenizer_path.write_text("tokenizer-v2", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_checkpoint(path, tokenizer_path=tokenizer_path)

    def test_tied_embeddings_roundtrip_and_parameter_count(self) -> None:
        """A tied checkpoint reloads one shared parameter and its optimizer state."""
        config = ModelConfig(32, d_model=16, d_hidden=32, num_heads=4,
                             num_blocks=2, max_seq_len=16, tie_word_embeddings=True)
        untied = build_model(self.config)
        model = build_model(config)
        self.assertIs(model.embedding.weight, model.lm_head.weight)
        self.assertEqual(sum(p.numel() for p in untied.parameters()) -
                         sum(p.numel() for p in model.parameters()), 32 * 16)
        self.assertLessEqual(float(model.embedding.weight.detach().abs().max()), 16 ** -0.5)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        train_epoch(model, [self.batch], torch.device("cpu"), optimizer, 0)
        tokenizer = ROOT / "artifacts/tokenizer.json"
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "tied.pth"
            save_checkpoint(path, model, optimizer, config, self.train_config,
                            1, 1, 0.0, tokenizer)
            restored, checkpoint = load_checkpoint(path, tokenizer_path=tokenizer)
            self.assertIs(restored.embedding.weight, restored.lm_head.weight)
            self.assertTrue(checkpoint["model_config"]["tie_word_embeddings"])
            opt = torch.optim.Adam(restored.parameters())
            opt.load_state_dict(checkpoint["optimizer_state_dict"])
            self.assertEqual(len(opt.state), len(list(restored.parameters())))
            with torch.no_grad():
                torch.testing.assert_close(restored(self.batch[0]), model(self.batch[0]), rtol=0, atol=0)

    def test_batched_attention_matches_outputs_and_gradients(self) -> None:
        """Preserve checkpoint names and differentiate the same causal attention."""
        config = ModelConfig(32, d_model=16, d_hidden=32, num_heads=4, head_dim=8,
                             num_blocks=2, max_seq_len=16, tie_word_embeddings=True)
        reference = build_model(config)
        batched = build_model(config)
        batched.load_state_dict(reference.state_dict())
        batched.enable_batched_attention()
        self.assertEqual(reference.state_dict().keys(), batched.state_dict().keys())
        inputs, _ = self.batch
        expected = reference(inputs)
        actual = batched(inputs)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        expected.square().mean().backward()
        actual.square().mean().backward()
        for (name, before), (_, after) in zip(reference.named_parameters(), batched.named_parameters()):
            with self.subTest(parameter=name):
                torch.testing.assert_close(after.grad, before.grad, atol=1e-6, rtol=1e-4)

    def test_empty_batches(self) -> None:
        """Reject empty samples and loaders before dividing by token counts."""
        model = build_model(self.config)
        optimizer = torch.optim.Adam(model.parameters())
        with self.assertRaises(ValueError):
            collate_sequences([], 0)
        with self.assertRaises(ValueError):
            evaluate(model, [], torch.device("cpu"), 0)
        with self.assertRaises(ValueError):
            train_epoch(model, [], torch.device("cpu"), optimizer, 0)

    def test_entry_point_with_tiny_local_dataset(self) -> None:
        """Run the entry point on a temporary four-story training dataset."""
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            project = Path(directory)
            (project / "data").mkdir()
            (project / "artifacts").mkdir()
            shutil.copyfile(
                ROOT / "artifacts" / "tokenizer.json",
                project / "artifacts" / "tokenizer.json",
            )
            for split, count in (("train", 4), ("validation", 2)):
                with (project / "data" / f"{split}.jsonl").open(
                    "w", encoding="utf-8"
                ) as f:
                    for i in range(count):
                        story = {"text": "A little cat found a ball." + " Meow." * i}
                        f.write(json.dumps(story) + "\n")
            run_dir = main(
                [
                    "--project-dir",
                    str(project),
                    "--epochs",
                    "1",
                    "--batch-size",
                    "2",
                    "--device",
                    "cpu",
                    "--log-every",
                    "1",
                ]
            )
            model, checkpoint = load_checkpoint(
                run_dir / "best.pth",
                tokenizer_path=project / "artifacts" / "tokenizer.json",
            )
            self.assertEqual(checkpoint["global_step"], 2)
            self.assertEqual(checkpoint["epoch"], 1)
            self.assertTrue(math.isfinite(checkpoint["val_loss"]))
            records = (run_dir / "metrics.jsonl").read_text().splitlines()
            self.assertEqual(len(records), 1)


if __name__ == "__main__":
    unittest.main()
