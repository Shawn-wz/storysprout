"""Verify mixture token budgets, causal labels and generated state assertions."""

import random
import unittest

import torch

from codes.data import load_tokenizer
from codes.sft_data import collate_sft_examples
from tools.data_ablation import ROOT
from tools.state_tasks import make_split, solve
from tools.story_mix_data import BudgetStream, continuation, plain_example, renamed_task
from tools.story_mix_evaluate import checked_state


class StoryMixTests(unittest.TestCase):
    def test_exact_budget_and_reproducible_shared_stream(self):
        examples = [(torch.tensor([2, 4, 5]), torch.tensor([-100, 5, 3])),
                    (torch.tensor([2, 6, 7, 8]), torch.tensor([6, 7, 8, 3]))]
        first, second = BudgetStream(examples, 1), BudgetStream(examples, 1)
        for size in (1, 7, 100):
            a, b = first.take(size), second.take(size)
            self.assertEqual(sum(int((y != -100).sum()) for _, y in a), size)
            self.assertEqual(len(a), len(b))
            collate_sft_examples(a, 0)
            for (ax, ay), (bx, by) in zip(a, b):
                self.assertTrue(torch.equal(ax, bx))
                self.assertTrue(torch.equal(ay, by))
                self.assertEqual(ax.shape, ay.shape)

    def test_plain_continuation_boundary(self):
        tokenizer = load_tokenizer(ROOT)
        prompt, response = "Mia gave the key to Ben.", " Ben held the key."
        x, y = plain_example(tokenizer, prompt, response)
        prefix = [tokenizer.token_to_id("<bos>"), *tokenizer.encode(prompt).ids]
        suffix = tokenizer.encode(response).ids + [tokenizer.token_to_id("<eos>")]
        self.assertEqual(x.tolist(), (prefix + suffix)[:-1])
        self.assertEqual(y.tolist(), [-100] * (len(prefix) - 1) + suffix)

    def test_renamed_scene_oracle_and_assertions(self):
        tokenizer = load_tokenizer(ROOT)
        tasks = make_split(tokenizer, "train", 50, 1, set())
        for i, task in enumerate(tasks):
            task = renamed_task(task, tokenizer, random.Random(i))
            gold = solve(task["initial"], task["events"], task["target"])
            self.assertEqual(gold, task["answer"])
            response, assertions = continuation(task, i)
            self.assertEqual(assertions[0]["value"], gold)
            self.assertIn(gold, response)
            x, y = plain_example(tokenizer, task["story"], response)
            self.assertLessEqual(len(x), 256)
            self.assertEqual(x.shape, y.shape)
            for variant in range(6):
                story, _ = continuation(task, variant)
                result = checked_state(task, story)
                self.assertTrue(result["covered"], story)
                self.assertTrue(result["verified"], str(result))
            wrong = "box" if task["answer"] != "box" and task["domain"] == "location" else "bag"
            if task["domain"] == "owner":
                wrong = "Mia" if task["answer"] != "Mia" else "Ben"
            bad, _ = continuation(task, 0)
            self.assertTrue(checked_state(task, bad.replace(task["answer"], wrong))["violations"])

    def test_checker_tracks_explicit_transfers(self):
        task = {"domain": "owner", "target": "key", "answer": "Ben"}
        text = "Ben gave the key to Mia. Mia held the key carefully."
        self.assertTrue(checked_state(task, text)["verified"])
        self.assertTrue(checked_state(task, "Tom gave the key to Mia.")["violations"])
        self.assertFalse(checked_state(task, "They had a lovely afternoon.")["covered"])


if __name__ == "__main__":
    unittest.main()
