"""Build token-balanced replay pools and state-consistent narrative continuations."""

import json
import random
import re

import numpy as np
import torch

from codes.data import CachedSequences, load_tokenizer
from codes.sft_data import encode_sft_example
from .architecture_ablation import load_json
from .data_ablation import ROOT, digest, write_json
from .state_tasks import make_split, solve

OUT = ROOT / "experiments/story_mix_20261118"
SOURCE = ROOT / "models/run_20261118_183053_375283/best.pth"
TRAIN_NAMES = ("Mia", "Ben", "Tom", "Lily", "Sam", "Max", "Tim", "Anna",
               "Alice", "Peter", "Sara", "Sue", "Jill", "Rose", "Dan", "Nora")
TEST_NAMES = ("Lucy", "Jack", "Bob", "Jane", "Emma", "Kate", "Amy", "Leo")
SETTINGS = {
    "seed": 2026, "updates": 2000, "targets_per_update": 1000,
    "microbatch_size": 16, "learning_rate": 1e-4, "max_grad_norm": 1.0,
    "control": {"common_story": 600, "extra_story": 300, "qa": 100},
    "mixed": {"common_story": 600, "continuation": 300, "qa": 100},
    "selection": "fixed update 2000; development checkpoints at each 500 updates",
}

FREE_PROMPTS = [
    ("Lucy lent her red umbrella to Jack. Later, rain began to fall while Jack was walking home.", "Jack has Lucy's red umbrella."),
    ("Emma put a silver coin inside a blue box. Kate carried the closed box to the kitchen.", "The coin is inside the closed blue box in the kitchen."),
    ("Bob found Amy's lost mitten beside the gate. Amy was still searching for it in the garden.", "Bob has found Amy's mitten; Amy is still searching."),
    ("Jane gave her yellow kite to Leo. Leo took it to the hill to try it in the wind.", "Leo has Jane's yellow kite and wants to fly it."),
    ("Jack left his book on a bench. Lucy picked it up and began looking for Jack.", "Lucy has Jack's book and is looking for Jack."),
    ("Amy planted a bean in a pot beside the window. Every morning, she checked the soil and gave it water.", "Amy is growing a bean in a pot."),
    ("Kate's toy boat was stuck near the far side of a pond. Bob found a long stick and offered to help.", "Kate's boat is stuck; Bob offers help with a stick."),
    ("Leo slipped on the wet path and hurt his knee. Emma brought a clean cloth from the house.", "Leo hurt his knee; Emma brought a cloth to help."),
    ("Lucy put a small bell in her bag. Later, she heard a ringing sound as she walked.", "The bell is in Lucy's bag."),
    ("Jane was too short to reach an apple. Jack brought a low stool and set it under the tree.", "Jane wants an apple; Jack brought a stool to help her reach."),
    ("Amy finished a picture and gave it to Bob. Bob wanted to show the picture to his mother.", "Bob has Amy's finished picture."),
    ("Emma lost her hat in the wind. Leo caught it near a fence and ran toward Emma.", "Leo caught Emma's hat and is going toward Emma."),
    ("Kate filled a cup with water and set it beside a thirsty puppy. The puppy came closer.", "The puppy has water available in Kate's cup."),
    ("Bob put a wooden spoon on the table. Jane moved it to a drawer before she cleaned the table.", "The wooden spoon is now in a drawer."),
    ("Jack found a tiny bird on the ground. Its wing was hurt, so he called Lucy for help.", "Jack found a bird with an injured wing and called Lucy."),
    ("Leo gave his ball to Amy, and Amy passed it to Kate. Then Leo asked who had the ball.", "Kate now has Leo's ball."),
    ("Jane put three seeds in the soil. A week later, one small green shoot appeared.", "Jane planted three seeds and one shoot has appeared."),
    ("Lucy was afraid to cross a narrow bridge. Emma walked beside her and offered a hand.", "Lucy is afraid; Emma is helping her cross."),
    ("Bob hid a gift under his bed. Before Kate came to visit, he wrapped the gift in paper.", "Bob has prepared a hidden gift for Kate."),
    ("Amy left her doll on a chair. Jack moved the doll onto a shelf so the chair would be free.", "The doll is now on a shelf."),
    ("Kate baked a small loaf and gave half to Jane. Jane carried her half to the park.", "Jane has half of Kate's loaf."),
    ("Emma found Leo's red scarf in a basket. Leo was outside looking for something warm to wear.", "Emma found Leo's scarf; Leo is outside looking for warmth."),
    ("Jack's paper plane landed in a bush. Bob held the branches apart so Jack could reach it.", "Jack's plane is in the bush; Bob is holding branches to help."),
    ("Lucy saved a biscuit for Amy and put it in a tin. When Amy arrived, Lucy opened the tin.", "Lucy's tin contains a biscuit saved for Amy."),
]


def write_rows(name: str, rows: list[dict]) -> dict:
    path = OUT / "data" / f"{name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return {"rows": len(rows), "sha256": digest(path)}


def rows(name: str) -> list[dict]:
    return [json.loads(s) for s in (OUT / "data" / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]


def plain_example(tokenizer, prompt: str, response: str):
    prefix = [tokenizer.token_to_id("<bos>"), *tokenizer.encode(prompt).ids]
    suffix = tokenizer.encode(response).ids + [tokenizer.token_to_id("<eos>")]
    if len(prefix) + len(suffix) > 257:
        raise ValueError("Narrative example exceeds the context window.")
    combined = prefix + suffix
    return (torch.tensor(combined[:-1], dtype=torch.long),
            torch.tensor([-100] * (len(prefix) - 1) + suffix, dtype=torch.long))


def continuation(task: dict, variant: int) -> tuple[str, list[dict]]:
    """Narrate the final state, a compatible action, and a complete ending."""
    holder = task["answer"]
    obj = task["target"]
    others = [n for n in task["names"] if n != holder]
    friend = others[variant % len(others)]
    if task["domain"] == "owner":
        choices = [
            f" {holder} held the {obj} carefully. {friend} smiled and watched. "
            f"After playing for a while, {holder} carried the {obj} home. It had been a good day.",
            f" The {obj} was safe with {holder}. {friend} was glad that it was safe. "
            f"Later, {holder} still had the {obj}. The friends waved goodbye and went home.",
            f" {holder} picked up the {obj} and smiled. The friends sat together under a tree. "
            f"{holder} kept the {obj} close. Soon it was time for everyone to go home.",
            f" {holder} had the {obj} now. {friend} asked to see it, and {holder} showed it proudly. "
            f"When the sun began to set, {holder} took the {obj} home. Everyone felt happy.",
            f" The friends looked at the {obj} in {holder}'s hands. They were happy that nothing was lost. "
            f"{holder} held the {obj} and walked toward the gate. The little adventure was over.",
            f" {holder} kept the {obj} safe. The friends had a nice afternoon together. "
            f"At the end of the day, {holder} still had the {obj}. They all said goodbye.",
        ]
    else:
        choices = [
            f" The {obj} was in the {holder}. {friend} looked there and found it. "
            f"It was safe and clean. The friends smiled, glad that they knew where it was. Then they went home.",
            f" The friends looked at the {holder}. The {obj} was in the {holder}, just where it had been left. "
            f"Nobody moved it again. They were happy that it was safe and went outside to play.",
            f" {friend} checked the {holder}. The {obj} was still in the {holder}. "
            f"Everything was where it should be. Soon the friends finished their game and went home.",
            f" The {obj} was safe in the {holder}. {friend} smiled when the friends found it there. "
            f"They left it in the same place. It had been a happy day.",
            f" Nobody needed to search any more. The {obj} was in the {holder}. "
            f"{friend} was glad it had not been lost. The friends left it there and went home together.",
            f" The friends remembered the {holder}. They saw that the {obj} was in the {holder}. "
            f"They were pleased to have found it. Then it was time to go home.",
        ]
    response = choices[variant % len(choices)]
    return response, [{"object": obj, "value": holder, "domain": task["domain"]}]


def renamed_task(task, tokenizer, rng):
    """Expand training names while preserving all symbolic state fields."""
    mapping = dict(zip(task["names"], rng.sample(list(TRAIN_NAMES), len(task["names"]))))
    def rename(text):
        return re.sub(r"\b(" + "|".join(task["names"]) + r")\b", lambda m: mapping[m.group()], text)
    result = dict(task)
    for field in ("prompt", "story", "answer", "response"):
        result[field] = rename(task[field])
    result["names"] = list(TRAIN_NAMES)
    result["initial"] = {k: mapping.get(v, v) for k, v in task["initial"].items()}
    result["events"] = [{k: mapping.get(v, v) if k in ("a", "b", "source", "destination", "actor") else v
                         for k, v in event.items()} for event in task["events"]]
    return result


def prepare():
    """Freeze training pools and separate new test prompts before any update."""
    tokenizer = load_tokenizer(ROOT)
    path = OUT / "protocol.json"
    if path.exists():
        saved = load_json(path)
        assert saved["settings"] == SETTINGS
        assert saved["source_sha256"] == digest(SOURCE)
        for name, record in saved["data"].items():
            assert digest(OUT / "data" / f"{name}.jsonl") == record["sha256"]
        return
    OUT.mkdir(parents=True, exist_ok=True)
    seen = set()
    # Exclude all prior task prompts as an additional guard against reused tests.
    for file in (ROOT / "experiments/state_rl_20261118/data").glob("*.jsonl"):
        seen.update(json.loads(s)["prompt"] for s in file.read_text(encoding="utf-8").splitlines())
    rng = random.Random(8100)
    data = {}
    for name, category, count, seed in (
        ("qa_train", "train", 32768, 8101), ("qa_dev", "dev", 256, 8102),
        ("qa_test_seen", "chain", 240, 8103), ("qa_test_ood", "combined", 240, 8104),
        ("continuation_train", "train", 8192, 8105), ("continuation_dev", "chain", 256, 8106),
        ("story_test", "names", 48, 8107),
    ):
        tasks = make_split(tokenizer, category, count, seed, seen)
        prepared = []
        for i, task in enumerate(tasks):
            if name in ("qa_train", "qa_dev", "continuation_train", "continuation_dev"):
                task = renamed_task(task, tokenizer, rng)
            assert solve(task["initial"], task["events"], task["target"]) == task["answer"]
            if name.startswith("continuation"):
                response, assertions = continuation(task, i)
                prefix = task["story"]
                plain_example(tokenizer, prefix, response)
                prepared.append({**task, "prompt": prefix, "response": response,
                                 "assertions": assertions})
            elif name == "story_test":
                # Plain narrative prefix, with a genuine gap before the continuation.
                marker = "Who has " if task["domain"] == "owner" else "Where is "
                question = task["prompt"].rsplit(marker, 1)[0].strip()
                filler_removed = len(tokenizer.encode(question).ids) > 148
                if filler_removed:
                    question = task["story"]
                prepared.append({**task, "prompt": question, "filler_removed": filler_removed})
            else:
                encode_sft_example(tokenizer, task["prompt"], task["response"], 256)
                prepared.append(task)
        data[name] = write_rows(name, prepared)
        print("Prepared", name, len(prepared), flush=True)
    free = [{"id": f"free-{i:02d}", "prompt": prompt, "facts": facts}
            for i, (prompt, facts) in enumerate(FREE_PROMPTS)]
    data["free_test"] = write_rows("free_test", free)
    manifest = load_json(ROOT / "data/tinystories_full/tokenized_manifest.json")
    indices = np.random.default_rng(8108).choice(manifest["splits"]["train"]["stories"],
                                               size=16384, replace=False)
    np.save(OUT / "ordinary_indices.npy", indices, allow_pickle=False)
    write_json(path, {
        "settings": SETTINGS, "source": str(SOURCE), "source_sha256": digest(SOURCE),
        "tokenizer_sha256": digest(ROOT / "artifacts/tokenizer.json"),
        "ordinary_manifest_sha256": digest(ROOT / "data/tinystories_full/tokenized_manifest.json"),
        "ordinary_indices_sha256": digest(OUT / "ordinary_indices.npy"), "data": data,
        "test_policy": "New tests frozen before training. No test-based model selection. Existing QA prompts excluded during task generation.",
        "budget_note": "Exactly 1000 supervised target tokens per update. Final sample labels may be cropped. Input-token counts and elapsed compute can differ.",
    })


def read_pools(tokenizer):
    manifest = load_json(ROOT / "data/tinystories_full/tokenized_manifest.json")
    source = CachedSequences(ROOT / "data/tinystories_full", "train", manifest, 256)
    indices = np.load(OUT / "ordinary_indices.npy", allow_pickle=False)
    def ordinary(ids):
        return [(source[int(i)][:-1], source[int(i)][1:]) for i in ids]
    return {
        "common_story": ordinary(indices[:8192]),
        "extra_story": ordinary(indices[8192:]),
        "qa": [encode_sft_example(tokenizer, r["prompt"], r["response"], 256)
               for r in rows("qa_train")],
        "continuation": [plain_example(tokenizer, r["prompt"], r["response"])
                         for r in rows("continuation_train")],
    }


class BudgetStream:
    """Shuffle a pool reproducibly and return exactly the requested label count."""
    def __init__(self, examples, seed):
        self.examples = examples
        self.rng = random.Random(seed)
        self.order = list(range(len(examples)))
        self.rng.shuffle(self.order)
        self.position = 0

    def take(self, target_tokens):
        output = []
        while target_tokens:
            if self.position == len(self.order):
                self.rng.shuffle(self.order)
                self.position = 0
            features, labels = self.examples[self.order[self.position]]
            self.position += 1
            valid = (labels != -100).nonzero().flatten()
            take = min(target_tokens, len(valid))
            finish = int(valid[take - 1]) + 1
            end = max(2, finish)
            selected_labels = labels[:end].clone()
            selected_labels[finish:] = -100
            output.append((features[:end], selected_labels))
            target_tokens -= take
        assert sum(int((labels != -100).sum()) for _, labels in output) > 0
        return output
