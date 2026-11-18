"""Repeat the matched RL/SFT comparison with unsaturated state-tracking tasks."""

import json

import torch

from codes.checkpoint import load_checkpoint
from codes.data import load_tokenizer
from codes.training import evaluate
from .architecture_ablation import load_json, make_loader
from .data_ablation import ROOT, digest, write_json
from .evaluate_architecture import make_cases, score, summarize
from .evaluate_full_pretrain import greedy
from .state_rl_experiment import OUT, PROTOCOL, branch, read_rows, task_evaluate, tokens
from .state_tasks import make_split


def prepare(tokenizer):
    path = OUT / "hard_protocol.json"
    if path.exists():
        protocol = load_json(path)
        assert protocol["common_sft_sha256"] == digest(OUT / "sft/best.pth")
        for name, record in protocol["data"].items():
            assert digest(OUT / "data" / f"{name}.jsonl") == record["sha256"]
        return
    seen = set()
    for file in (OUT / "data").glob("*.jsonl"):
        for line in file.read_text(encoding="utf-8").splitlines():
            seen.add(json.loads(line)["prompt"])
    specifications = [
        ("hard_rl", "hard_train", 4096, 4101),
        ("hard_dev", "hard_dev", 512, 4102),
        ("test_hard_seen", "hard_seen", 320, 4103),
        ("test_hard_ood", "hard_ood", 320, 4104),
    ]
    data = {}
    for name, category, count, seed in specifications:
        rows = make_split(tokenizer, category, count, seed, seen)
        file = OUT / "data" / f"{name}.jsonl"
        file.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        data[name] = {"rows": len(rows), "sha256": digest(file)}
    history = load_json(OUT / "grpo/history.json")
    write_json(path, {
        "reason": "First round reward saturation; follow-up difficulty chosen after observing training mixed-reward rate.",
        "first_round_mixed_group_rate": sum(r["mixed_group_rate"] for r in history) / len(history),
        "common_sft_sha256": digest(OUT / "sft/best.pth"),
        "difficulty": "3-5 target changes, distractor changes and 90/110/130-token filler; reject prompts above 244 tokens",
        "branch_settings": PROTOCOL, "data": data,
        "selection": "fixed 400 updates; new tests evaluated after both branches finish",
    })


def main():
    tokenizer = load_tokenizer(ROOT)
    prepare(tokenizer)
    branch(tokenizer, "sft_hard_control", "hard_rl", "hard_dev")
    branch(tokenizer, "grpo_hard", "hard_rl", "hard_dev")
    models = {
        "sft": OUT / "sft/best.pth",
        "sft_control": OUT / "sft_control/step_400.pth",
        "grpo": OUT / "grpo/step_400.pth",
        "sft_hard_control": OUT / "sft_hard_control/step_400.pth",
        "grpo_hard": OUT / "grpo_hard/step_400.pth",
    }
    results = {}
    generations = {}
    val = make_loader(ROOT / "data/validation.jsonl", tokenizer, tokens(tokenizer), 256, 8, False)
    fixed = load_json(OUT / "generations.json")["base"]["plain_greedy"]
    for group, path in models.items():
        model, _ = load_checkpoint(path, "cuda", ROOT / "artifacts/tokenizer.json")
        record = {"checkpoint_sha256": digest(path), "tasks": {}}
        predictions = {}
        for name in ("hard_seen", "hard_ood"):
            record["tasks"][name], predictions[name] = task_evaluate(
                model, tokenizer, read_rows(f"test_{name}"))
        if group in ("sft_hard_control", "grpo_hard"):
            record["validation_loss"] = evaluate(model, val, torch.device("cuda"), tokens(tokenizer)["<pad>"])
            record["original_probe"] = summarize(score(model, tokenizer, make_cases(tokenizer)))
            record["original_test"] = {}
            for name in ("single", "chain", "long", "paraphrase", "names", "combined", "four_hops"):
                record["original_test"][name], predictions[name] = task_evaluate(
                    model, tokenizer, read_rows(f"test_{name}"))
            generations[group] = [greedy(model, tokenizer, row["prompt"]) for row in fixed]
        results[group] = record
        write_json(OUT / f"hard_predictions_{group}.json", predictions)
        write_json(OUT / "hard_evaluation.json", results)
        print("Hard evaluation", group, record, flush=True)
        del model
        torch.cuda.empty_cache()
    write_json(OUT / "hard_generations.json", generations)
    write_json(OUT / "status.json", {"phase": "complete", "follow_up": "hard_completed"})


if __name__ == "__main__":
    main()
