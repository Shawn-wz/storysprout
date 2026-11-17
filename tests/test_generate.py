"""Check token counting, EOS stopping, context boundaries, and generation devices."""

import unittest

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from torch import nn

from codes.generate import generate


class PredictableModel(nn.Module):
    """Emit predetermined token IDs while recording generation input shapes."""

    def __init__(self, tokens: list[int], max_seq_len: int = 16) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.tokens = tokens
        self.max_seq_len = max_seq_len
        self.inputs: list[tuple[torch.device, torch.dtype, int]] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return an effectively one-hot next-token distribution."""
        if x.device != self.weight.device:
            raise RuntimeError("Input is on the wrong device.")
        index = min(len(self.inputs), len(self.tokens) - 1)
        self.inputs.append((x.device, x.dtype, x.shape[1]))
        logits = torch.full((1, x.shape[1], 7), -1000.0, device=x.device)
        logits[:, :, self.tokens[index]] = 1000.0
        return logits


class GenerationTests(unittest.TestCase):
    """Test control flow independently of the trained model's language quality."""

    def setUp(self) -> None:
        """Create a tokenizer whose words have known token IDs."""
        vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3,
                 "hello": 4, "cat": 5, "dog": 6}
        self.tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        self.tokenizer.add_special_tokens(["<pad>", "<bos>", "<eos>", "<unk>"])

    def test_zero_tokens_skips_forward(self) -> None:
        """A zero generation budget returns an empty continuation."""
        model = PredictableModel([5])
        self.assertEqual(generate(model, self.tokenizer, "hello", 0), "")
        self.assertEqual(model.inputs, [])

    def test_token_budget_and_continuation(self) -> None:
        """Count prompt tokens, not characters, and honor the exact budget."""
        for budget in (1, 2, 4):
            with self.subTest(budget=budget):
                model = PredictableModel([5])
                text = generate(model, self.tokenizer, "hello dog", budget)
                self.assertEqual(text, " ".join(["cat"] * budget))
                self.assertEqual(len(model.inputs), budget)
                self.assertEqual(model.inputs[0][1:], (torch.long, 3))

    def test_eos_stops_and_is_not_decoded(self) -> None:
        """EOS ends generation before the budget and is omitted from the text."""
        model = PredictableModel([5, 2, 6])
        self.assertEqual(generate(model, self.tokenizer, "hello", 8), "cat")
        self.assertEqual(len(model.inputs), 2)

    def test_context_limit(self) -> None:
        """Include BOS and the prompt when limiting the total sequence length."""
        model = PredictableModel([5], max_seq_len=5)
        self.assertEqual(generate(model, self.tokenizer, "hello dog", 8), "cat cat")
        self.assertEqual([item[2] for item in model.inputs], [3, 4])
        full = PredictableModel([5], max_seq_len=3)
        self.assertEqual(generate(full, self.tokenizer, "hello dog"), "")
        self.assertEqual(full.inputs, [])
        with self.assertRaises(ValueError):
            generate(full, self.tokenizer, "hello dog cat")

    def test_empty_prompt(self) -> None:
        """An empty prompt still supplies BOS as the first model input."""
        model = PredictableModel([5])
        self.assertEqual(generate(model, self.tokenizer, "", 1), "cat")
        self.assertEqual(model.inputs[0][2], 1)

    def test_invalid_parameters(self) -> None:
        """Reject invalid temperatures and generation budgets before sampling."""
        model = PredictableModel([5])
        for temperature in (0, -1, float("nan"), float("inf")):
            with self.subTest(temperature=temperature), self.assertRaises(ValueError):
                generate(model, self.tokenizer, "hello", temperature=temperature)
        for temperature in (True, "0.8"):
            with self.assertRaises(TypeError):
                generate(model, self.tokenizer, "hello", temperature=temperature)
        with self.assertRaises(ValueError):
            generate(model, self.tokenizer, "hello", -1)
        for budget in (True, 1.5):
            with self.assertRaises(TypeError):
                generate(model, self.tokenizer, "hello", budget)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_input_device(self) -> None:
        """Generation inputs follow the model's parameter device."""
        model = PredictableModel([5]).to("cuda")
        self.assertEqual(generate(model, self.tokenizer, "hello", 1), "cat")
        self.assertEqual(model.inputs[0][0].type, "cuda")


if __name__ == "__main__":
    unittest.main()
