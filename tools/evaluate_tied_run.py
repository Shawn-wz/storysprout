"""Compare full pretraining and replay SFT checkpoints on common story tests."""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

import torch

from codes.checkpoint import load_checkpoint, model_config_from_checkpoint
from codes.data import load_tokenizer, special_token_ids
from codes.training import evaluate
from .architecture_ablation import load_json, make_loader
from .data_ablation import clean, digest, write_json
from .evaluate_architecture import make_cases, score, summarize
from .evaluate_full_pretrain import adjacent_cases
from .paired_state_data import chain_key
from .paired_state_experiment import paired_metrics
from .state_rl_experiment import OUT as HISTORICAL_QA_DIR, task_evaluate
from .story_mix_evaluate import generate_rows, narrative_metrics


def story_cue_cases(tokenizer, rows):
    """Make a forced-choice story completion with the counterpart answer as a foil."""
    answers = {(row["group_id"], row["outcome"]): row["answer"] for row in rows}
    cases = []
    for row in rows:
        cue = "was now with" if row["domain"] == "owner" else "was now at the"
        text = f"{row['story']} The {row['target']} {cue}"
        prefix = tokenizer.encode(text).ids
        foil = answers[row["group_id"], 1 - row["outcome"]]
        continuations = [tokenizer.encode(f"{text} {answer}").ids for answer in (row["answer"], foil)]
        if any(full[:-1] != prefix for full in continuations):
            raise ValueError(f"Expected two single-token story candidates in {row['id']}.")
        cases.append((row["id"], [tokenizer.token_to_id("<bos>"), *prefix],
                      continuations[0][-1], continuations[1][-1]))
    return cases


@torch.inference_mode()
def score_story_cues(model, cases, pad):
    """Score the foil and correct next token at the last actual prompt position."""
    device = next(model.parameters()).device
    ordered = sorted(cases, key=lambda case: len(case[1]))
    results = []
    for start in range(0, len(ordered), 16):
        batch = ordered[start:start + 16]
        row_ids, histories, correct, foil = zip(*batch)
        lengths = [len(ids) for ids in histories]
        width = max(lengths)
        features = torch.tensor([ids + [pad] * (width - len(ids)) for ids in histories], device=device)
        next_logits = model(features)[torch.arange(len(batch), device=device),
                                     torch.tensor(lengths, device=device) - 1]
        margins = (next_logits.gather(1, torch.tensor(correct, device=device)[:, None]) -
                   next_logits.gather(1, torch.tensor(foil, device=device)[:, None])).flatten()
        results.extend({"id": row_id, "margin": margin, "correct": margin > 0}
                       for row_id, margin in zip(row_ids, margins.tolist()))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sft-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()

    asset = args.asset_root.resolve()
    old = asset / "models/run_20261118_183053_375283/best.pth"
    new = args.checkpoint.resolve()
    paths = {"untied_pretrain": old, "tied_pretrain": new,
             "tied_control_sft": args.sft_dir / "control/final.pth",
             "tied_paired_sft": args.sft_dir / "paired/final.pth"}
    corpus_path = asset / "data/tinystories_full/tokenized_manifest.json"
    corpus = load_json(corpus_path)
    tokenizer_path = asset / "artifacts/tokenizer.json"
    validation_path = asset / "data/validation.jsonl"
    if (digest(tokenizer_path) != corpus["tokenizer_sha256"] or
            digest(validation_path) != corpus["splits"]["validation"]["files"]["validation.jsonl"]):
        raise ValueError("The tokenizer or validation corpus does not match the full-data manifest.")
    expected_tokens = corpus["splits"]["train"]["valid_targets"]
    expected_steps = (corpus["splits"]["train"]["stories"] + 7) // 8
    runs = {}
    for label, path in (("untied_pretrain", old), ("tied_pretrain", new)):
        metrics_path = path.parent / "metrics.jsonl"
        metrics, = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
        if metrics["steps"] != expected_steps or metrics["tokens"] != expected_tokens:
            raise ValueError(f"{label} did not complete the recorded full training budget.")
        runs[label] = metrics

    tokenizer = load_tokenizer(asset)
    special = special_token_ids(tokenizer)
    device = torch.device(args.device)
    val = make_loader(validation_path, tokenizer, special, 256, 8, False)
    clean_val = make_loader(validation_path, tokenizer, special, 256, 8, False,
                            keep=lambda story: clean(story)[1] is None)
    history = make_cases(tokenizer)
    cases = history + adjacent_cases(tokenizer, history)
    state_rows = {}
    for split in ("wording", "combined"):
        path = args.sft_dir / f"{split}.jsonl"
        state_rows[split] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    probes = {split: story_cue_cases(tokenizer, rows) for split, rows in state_rows.items()}
    trained_chains = set()
    for split in ("control", "paired"):
        with (args.sft_dir / f"{split}.jsonl").open(encoding="utf-8") as source:
            trained_chains.update(chain_key(json.loads(line)) for line in source)
    historical_qa = {}
    for split in ("chain", "paraphrase", "names", "long", "four_hops", "combined"):
        path = HISTORICAL_QA_DIR / "data" / f"test_{split}.jsonl"
        original = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        retained = [row for row in original if chain_key(row) not in trained_chains]
        historical_qa[split] = (retained, len(original) - len(retained))
    narrative = [{**r, "prompt": r["story"]} for r in state_rows["combined"]
                 if r["wording"] == 0 and len(tokenizer.encode(r["story"]).ids) + 65 <= 256][:24]
    sft_results = load_json(args.sft_dir / "results.json")
    sft_protocol_path = args.sft_dir / "run_protocol.json"
    sft_protocol = load_json(sft_protocol_path)
    if sft_protocol["checkpoint_sha256"] != digest(new):
        raise ValueError("The SFT arms were trained from a different pretraining checkpoint.")
    sft_updates = sft_protocol["updates"]
    sft_tokens = {name: per_step * sft_updates for name, per_step in
                  sft_protocol["targets_per_update"].items()}
    output = {"data_manifest_sha256": digest(corpus_path),
              "sft_protocol_sha256": digest(sft_protocol_path),
              "validation_stories": len(val.dataset), "clean_validation_stories": len(clean_val.dataset),
              "training": runs, "models": {}}

    for name, path in paths.items():
        sha = digest(path)
        model, saved = load_checkpoint(path, device, tokenizer_path)
        cfg = model_config_from_checkpoint(saved)
        loss = evaluate(model, val, device, special["<pad>"])
        if abs(loss - saved["val_loss"]) > 1e-5:
            raise ValueError(f"Recomputed story loss differs for {name}: {loss} vs {saved['val_loss']}.")
        if name in runs and saved["global_step"] != expected_steps:
            raise ValueError(f"{name} is missing pretraining updates.")
        if cfg.tie_word_embeddings != (model.embedding.weight is model.lm_head.weight):
            raise ValueError(f"Incorrect input/output parameter sharing after reloading {name}.")
        legacy = score(model, tokenizer, cases)
        record = {"checkpoint": str(path), "checkpoint_sha256": sha, "model_config": asdict(cfg),
                  "parameters": sum(p.numel() for p in model.parameters()), "validation_loss": loss,
                  "perplexity": math.exp(loss), "clean_validation_loss":
                      evaluate(model, clean_val, device, special["<pad>"]),
                  "legacy_ownership": summarize([row for row in legacy if row["distance"] != 0]),
                  "legacy_by_distance": {str(d): summarize([row for row in legacy if row["distance"] == d])
                                         for d in (0, 20, 80, 160)}, "story_cues": {}}
        for split, probe in probes.items():
            predictions = score_story_cues(model, probe, special["<pad>"])
            record["story_cues"][split] = {"correct": sum(r["correct"] for r in predictions),
                                           "total": len(predictions),
                                           "pairs": paired_metrics(state_rows[split], predictions)}
            write_json(args.output_dir / f"{name}_{split}_cue_predictions.json", predictions)
        continuations = generate_rows(model, tokenizer, narrative, budget=64, mode="greedy")
        record["free_story_continuations"] = narrative_metrics(continuations)
        if name.endswith("_sft"):
            arm = "control" if name == "tied_control_sft" else "paired"
            if sft_results[arm]["checkpoint_sha256"] != sha:
                raise ValueError(f"SFT results do not match the saved weights for {name}.")
            if saved["global_step"] != sft_updates or sft_results[arm]["supervised_tokens"] != sft_tokens:
                raise ValueError(f"SFT did not complete the recorded training budget for {name}.")
            record["qa"] = sft_results[arm]["qa"]
            record["historical_qa"] = {}
            for split, (rows, excluded) in historical_qa.items():
                stats, predictions = task_evaluate(model, tokenizer, rows) if rows else ({}, [])
                record["historical_qa"][split] = {**stats, "excluded_training_chains": excluded}
                write_json(args.output_dir / f"{name}_{split}_historical_predictions.json", predictions)
        write_json(args.output_dir / f"{name}_generations.json", continuations)
        write_json(args.output_dir / f"{name}_legacy_predictions.json", legacy)
        if digest(path) != sha:
            raise ValueError(f"Checkpoint changed while evaluating {name}.")
        output["models"][name] = record
        write_json(args.output_dir / "comparison.json", output)
        print(name, f"loss={loss:.6f}", record["legacy_ownership"], flush=True)
        del model, saved


if __name__ == "__main__":
    main()
