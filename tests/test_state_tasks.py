"""Verify rule rewards, state transitions, and variable-length batched decoding."""

import unittest

import torch
from torch import nn

from codes.data import load_tokenizer
from codes.grpo import collate_rollouts, compute_group_advantages
from tools.data_ablation import ROOT
from tools.paired_state_data import HELD_OUT_EVENTS, TRAIN_EVENTS, chain_key, make_datasets
from tools.paired_state_experiment import paired_metrics
from tools.state_tasks import make_split, reward_ids, solve
from tools.state_rl_experiment import responses, tokens


class NextTokenModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.marker = nn.Parameter(torch.zeros(1))
        self.max_seq_len = 32

    def forward(self, ids):
        logits = torch.full((*ids.shape, 8), -20.0, device=ids.device)
        target = torch.where(ids == 5, 3, 5)
        return logits.scatter(-1, target.unsqueeze(-1), 20.0)


class StateTaskTests(unittest.TestCase):
    def test_paired_splits_hold_out_chains_and_preserve_pair_labels(self):
        tokenizer = load_tokenizer(ROOT)
        data = make_datasets(tokenizer, 12, 6, 2026)
        used_chains = set()
        used_prompts = set()
        self.assertEqual(len(data["control"]), len(data["paired"]))
        for split, rows in data.items():
            prompts = {row["prompt"] for row in rows}
            chains = {chain_key(row) for row in rows}
            self.assertEqual(len(prompts), len(rows))
            self.assertFalse(used_prompts & prompts)
            self.assertFalse(used_chains & chains)
            used_prompts.update(prompts)
            used_chains.update(chains)
            for row in rows:
                self.assertEqual(row["answer"], solve(row["initial"], row["events"], row["target"]))
                self.assertLessEqual(len(tokenizer.encode(row["prompt"]).ids) + 7, 256)
            if split == "control":
                continue
            pool = HELD_OUT_EVENTS if split in ("wording", "combined") else TRAIN_EVENTS
            for j in range(0, len(rows), 4):
                a, a_alt, b, b_alt = rows[j:j + 4]
                self.assertEqual(chain_key(a), chain_key(a_alt))
                self.assertEqual(chain_key(b), chain_key(b_alt))
                self.assertNotEqual(a["story"], a_alt["story"])
                self.assertNotEqual(a["answer"], b["answer"])
                self.assertEqual(a["event_styles"], b["event_styles"])
                self.assertTrue(set(a["event_styles"] + a_alt["event_styles"]).issubset(pool))
                self.assertEqual(a["events"][:-2], b["events"][:-2])
                self.assertEqual([e["destination"] for e in a["events"][-2:]],
                                 [e["destination"] for e in b["events"][-2:]][::-1])

    def test_paired_metrics_require_both_answers(self):
        rows = make_datasets(load_tokenizer(ROOT), 1, 2, 2026)["paired"]
        predictions = [{"id": row["id"], "correct": i != 3} for i, row in enumerate(rows)]
        result = paired_metrics(rows, predictions)["all"]
        self.assertEqual(result, {"pair_total": 2, "paraphrase_both_correct": 1,
                                  "counterfactual_both_correct": 1,
                                  "groups_all_correct": 0, "group_total": 1})

    def test_state_replay_and_impossible_source(self):
        initial = {"key": "Mia", "book": "Ben"}
        events = [{"kind": "move", "object": "key", "source": "Mia",
                   "destination": "Ben"},
                  {"kind": "swap", "a": "Ben", "b": "Tom"}]
        self.assertEqual(solve(initial, events, "key"), "Tom")
        self.assertEqual(solve(initial, events, "book"), "Tom")
        self.assertEqual(initial, {"key": "Mia", "book": "Ben"})
        with self.assertRaises(ValueError):
            solve(initial, [{"kind": "move", "object": "key", "source": "Tom",
                             "destination": "Ben"}], "key")

    def test_generated_tasks_and_split_disjointness(self):
        tokenizer = load_tokenizer(ROOT)
        seen = set()
        train = make_split(tokenizer, "train", 50, 1, seen)
        test = make_split(tokenizer, "combined", 50, 2, seen)
        hard = make_split(tokenizer, "hard_train", 50, 3, seen)
        self.assertFalse({r["prompt"] for r in train} & {r["prompt"] for r in test})
        for row in train + test + hard:
            self.assertEqual(solve(row["initial"], row["events"], row["target"]), row["answer"])
            self.assertLessEqual(len(tokenizer.encode(row["prompt"]).ids), 244)

    def test_reward_rejects_extra_words_and_special_tokens(self):
        tokenizer = load_tokenizer(ROOT)
        special = tokens(tokenizer)
        good = tokenizer.encode(" Ben").ids
        self.assertEqual(reward_ids(tokenizer, good + [special["<eos>"]], "Ben", special), 1)
        self.assertEqual(reward_ids(tokenizer, tokenizer.encode(" Ben.").ids, "Ben", special), 1)
        self.assertEqual(reward_ids(tokenizer, tokenizer.encode(" Ben and Mia").ids, "Ben", special), 0)
        self.assertEqual(reward_ids(tokenizer, [special["<assistant>"]] + good, "Ben", special), 0)
        self.assertEqual(reward_ids(tokenizer, [], "Ben", special), 0)

    def test_batched_decode_and_response_mask(self):
        special = {"<bos>": 0, "<user>": 1, "<assistant>": 2, "<eos>": 3, "<pad>": 4}
        prompts = [torch.tensor([0, 1, 2]), torch.tensor([0, 1, 6, 5])]
        samples = responses(NextTokenModel(), prompts, special)
        self.assertEqual(samples[0]["input_ids"].tolist(), [0, 1, 2, 5, 3])
        self.assertEqual(samples[1]["input_ids"].tolist(), [0, 1, 6, 5, 3])
        batch = collate_rollouts(samples, 4)
        self.assertEqual(batch["response_mask"].sum(1).tolist(), [2, 1])
        advantages = compute_group_advantages(torch.tensor([[0., 0.], [0., 1.]]))
        self.assertTrue(torch.equal(advantages[0], torch.zeros(2)))
        self.assertLess(advantages[1, 0], 0)
        self.assertGreater(advantages[1, 1], 0)


if __name__ == "__main__":
    unittest.main()
