"""Evaluate narrative preservation and checked state assertions after mixed SFT."""

import json
import random
import re

import torch

from codes.checkpoint import load_checkpoint
from codes.data import load_tokenizer
from codes.training import evaluate
from .architecture_ablation import load_json, make_loader
from .data_ablation import ROOT, clean, digest, write_json
from .evaluate_architecture import make_cases, score, summarize
from .state_rl_experiment import responses, task_evaluate, tokens
from .story_mix_data import OUT, SOURCE, TRAIN_NAMES, TEST_NAMES, rows, plain_example
from codes.sft_data import collate_sft_examples

PLACES = ("box", "bag", "desk", "shelf", "basket", "drawer")
EVALUATION = {
    "controlled_budget": 96, "free_budget": 160, "sampling_temperature": 1.0,
    "seed": 8301, "modes": ["greedy", "sampled"], "batch_size": 8,
    "short_output_threshold_tokens": 20, "blind_greedy_prompt_count": 12,
    "blind_review": "One assistant reviewer, model names hidden, before mapping is opened.",
    "automatic_checker": "Explicit named-holder and discrete-place assertions; unsupported claims remain unscored.",
}


def checked_state(task: dict, text: str) -> dict:
    """Conservatively check explicit claims and explicit transfers in text order."""
    obj = re.escape(task["target"])
    name = "(?:" + "|".join(map(re.escape, TRAIN_NAMES + TEST_NAMES)) + ")"
    place = "(?:" + "|".join(PLACES) + ")"
    item = r"(?:(?:the|a|an|his|her|their)\s+)?(?:(?:little|small|red|blue|green|yellow|old|new|silver|wooden)\s+)*" + obj
    patterns = []
    if task["domain"] == "owner":
        patterns = [
            ("give", rf"\b(?P<source>{name})\s+(?:gave|gives|passed|passes|handed)\s+{item}\s+(?:back\s+)?to\s+(?P<value>{name})\b"),
            ("give", rf"\b(?P<value>{name})\s+(?:got|gets|received)\s+{item}\s+from\s+(?P<source>{name})\b"),
            ("claim", rf"\b(?P<value>{name})\s+(?:still\s+)?(?:had|has|held|holds|kept|keeps|carried|carries|used|uses|read|reads|wore|wears|was holding|is holding)\s+{item}\b"),
            ("claim", rf"\b{item}\s+(?:was|is|was still|is still)\s+(?:safe\s+)?with\s+(?P<value>{name})\b"),
            ("claim", rf"\b{item}\s+in\s+(?P<value>{name})'s hands\b"),
            ("acquire", rf"\b(?P<value>{name})\s+(?:picked up|takes|took)\s+{item}\b"),
        ]
    else:
        patterns = [
            ("move", rf"\b{name}\s+(?:put|puts|placed|places|left|leaves|hid|set|dropped)\s+{item}\s+(?:back\s+)?(?:in|on|inside|under)\s+(?:the\s+|a\s+)?(?P<value>{place})\b"),
            ("claim", rf"\b{item}\s+(?:was|is|lay|rested|sat|was still|is still)\s+(?:safe\s+)?(?:in|on|inside|under)\s+(?:the\s+|a\s+)?(?P<value>{place})\b"),
        ]
    matches = []
    for kind, pattern in patterns:
        for match in re.finditer(pattern, text):
            matches.append((match.start(), match.end(), kind, match.groupdict(), match.group()))
    matches.sort(key=lambda r: (r[0], -r[1]))
    state = task["answer"]
    assertions = []
    violations = []
    end = -1
    for start, finish, kind, fields, evidence in matches:
        if start < end:
            continue
        end = finish
        value = fields["value"]
        if kind == "claim":
            assertions.append(evidence)
            if value != state:
                violations.append({"expected": state, "asserted": value, "text": evidence})
        elif kind == "give":
            assertions.append(evidence)
            if fields["source"] != state:
                violations.append({"expected_giver": state, "asserted_giver": fields["source"], "text": evidence})
            state = value
        else:
            # Explicit pickup/placement can introduce a legitimate new state.
            state = value
    return {"assertions": assertions, "violations": violations,
            "covered": bool(assertions), "verified": bool(assertions) and not violations}


def generate_rows(model, tokenizer, tasks, budget, mode):
    """Use the identical full-vocabulary decoder and seed for each model."""
    special = tokens(tokenizer)
    torch.manual_seed(EVALUATION["seed"])
    output = []
    for start in range(0, len(tasks), 8):
        subset = tasks[start:start + 8]
        prefixes = [torch.tensor([special["<bos>"], *tokenizer.encode(r["prompt"]).ids])
                    for r in subset]
        samples = responses(model, prefixes, special, budget=budget, sample=(mode == "sampled"))
        for task, sample in zip(subset, samples):
            ids = sample["input_ids"][sample["prompt_length"]:].tolist()
            text = tokenizer.decode(ids, skip_special_tokens=True)
            record = {"id": task["id"], "prompt": task["prompt"], "mode": mode,
                      "continuation": text, "new_tokens": len(ids),
                      "ended_with_eos": sample["ended_with_eos"],
                      "short": len(ids) < EVALUATION["short_output_threshold_tokens"]}
            if "target" in task:
                record["state_check"] = checked_state(task, text)
                record["expected"] = task["answer"]
            output.append(record)
    return output


def narrative_metrics(output):
    n = len(output)
    result = {"count": n, "short_outputs": sum(r["short"] for r in output),
              "eos_outputs": sum(r["ended_with_eos"] for r in output),
              "mean_new_tokens": sum(r["new_tokens"] for r in output) / n}
    if output and "state_check" in output[0]:
        result.update(covered=sum(r["state_check"]["covered"] for r in output),
                      verified=sum(r["state_check"]["verified"] for r in output),
                      contradicted=sum(bool(r["state_check"]["violations"]) for r in output))
    return result


def main():
    evaluation_path = OUT / "evaluation_protocol.json"
    if evaluation_path.exists():
        assert load_json(evaluation_path) == EVALUATION
    else:
        write_json(evaluation_path, EVALUATION)
    tokenizer = load_tokenizer(ROOT)
    special = tokens(tokenizer)
    val = make_loader(ROOT / "data/validation.jsonl", tokenizer, special, 256, 8, False)
    cleaned = make_loader(ROOT / "data/validation.jsonl", tokenizer, special, 256, 8, False,
                          keep=lambda s: clean(s)[1] is None)
    models = {"base": SOURCE, "control": OUT / "control/step_2000.pth",
              "mixed": OUT / "mixed/step_2000.pth"}
    results, generations = {}, {}
    for group, path in models.items():
        print("Evaluating", group, flush=True)
        model, saved = load_checkpoint(path, "cuda", ROOT / "artifacts/tokenizer.json")
        result = {"checkpoint": str(path), "checkpoint_sha256": digest(path),
                  "validation_loss": evaluate(model, val, torch.device("cuda"), special["<pad>"]),
                  "clean_validation_loss": evaluate(model, cleaned, torch.device("cuda"), special["<pad>"])}
        dev_items = [plain_example(tokenizer, r["prompt"], r["response"]) for r in rows("continuation_dev")]
        dev = torch.utils.data.DataLoader(dev_items, batch_size=16, shuffle=False,
                                         collate_fn=lambda b: collate_sft_examples(b, special["<pad>"]))
        result["continuation_dev_loss"] = evaluate(model, dev, torch.device("cuda"), -100)
        result["qa"] = {}
        for category in ("qa_test_seen", "qa_test_ood"):
            result["qa"][category], predicted = task_evaluate(model, tokenizer, rows(category))
            write_json(OUT / f"{group}_{category}_predictions.json", predicted)
        probe = score(model, tokenizer, make_cases(tokenizer))
        result["old_probe"] = summarize(probe)
        write_json(OUT / f"probe_{group}.json", probe)
        outputs = {}
        for kind, name, budget in (("controlled", "story_test", 96), ("free", "free_test", 160)):
            tasks = rows(name)
            generated = [r for mode in EVALUATION["modes"]
                         for r in generate_rows(model, tokenizer, tasks, budget, mode)]
            outputs[kind] = generated
            result[kind] = narrative_metrics(generated)
        generations[group] = outputs
        results[group] = result
        write_json(OUT / f"generations_{group}.json", outputs)
        write_json(OUT / "evaluation.json", results)
        print(group, result, flush=True)
        del model, saved
        torch.cuda.empty_cache()
    blinded, mapping = [], {}
    facts = {r["id"]: r["facts"] for r in rows("free_test")}
    for group, output in generations.items():
        selected = [r for r in output["free"] if r["mode"] == "greedy"][:12]
        blinded.extend([{**r, "_model": group, "facts": facts[r["id"]]} for r in selected])
    random.Random(8302).shuffle(blinded)
    for i, row in enumerate(blinded):
        key = f"review-{i:02d}"
        mapping[key] = {"model": row.pop("_model"), "prompt_id": row["id"]}
        row["review_id"] = key
    write_json(OUT / "blind_review.json", blinded)
    write_json(OUT / "blind_mapping.json", mapping)
    lines = ["# 新故事开头的生成对照", ""]
    for kind in ("controlled", "free"):
        for index, row in enumerate(generations["base"][kind]):
            lines += [f"## {kind} {row['id']} {row['mode']}", "", row["prompt"], ""]
            for group in models:
                r = generations[group][kind][index]
                lines += [f"### {group}", "", r["continuation"], "",
                          f"生成 {r['new_tokens']} token，EOS 为 {r['ended_with_eos']}。", ""]
    (OUT / "generations.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(OUT / "status.json", {"phase": "evaluation_complete", "review": "pending"})


if __name__ == "__main__":
    main()
