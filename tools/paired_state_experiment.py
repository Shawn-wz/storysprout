"""Prepare paired state data or compare it against ordinary QA in replay SFT."""

import argparse
from functools import partial
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from codes.checkpoint import load_checkpoint, model_config_from_checkpoint, save_checkpoint
from codes.config import TrainConfig
from codes.data import CachedSequences, collate_sequences, load_tokenizer
from codes.sft_data import encode_sft_example
from codes.training import evaluate
from .data_ablation import ROOT, digest, write_json
from .paired_state_data import CONTEXT, make_datasets
from .state_rl_experiment import task_evaluate
from .story_mix_data import BudgetStream
from .story_mix_evaluate import generate_rows, narrative_metrics
from .story_mix_experiment import train_mixture_update


TOKENIZER = ROOT / "artifacts/tokenizer.json"
QUOTAS = {"story": 900, "qa": 100}
SEED = 2026


def prepare(output, tokenizer, group_count, test_groups):
    protocol = {"version": 1, "seed": SEED, "groups": group_count,
                "test_groups_per_split": test_groups, "tokenizer_sha256": digest(TOKENIZER)}
    manifest_path = output / "data_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["protocol"] != protocol:
            raise ValueError("This directory contains prepared data with a different protocol.")
        datasets = {}
        for split, record in manifest["splits"].items():
            path = output / f"{split}.jsonl"
            if digest(path) != record["sha256"]:
                raise ValueError(f"Prepared data fingerprint mismatch: {path}")
            datasets[split] = [json.loads(row) for row in path.read_text(encoding="utf-8").splitlines()]
        return datasets

    datasets = make_datasets(tokenizer, group_count, test_groups, SEED)
    output.mkdir(parents=True, exist_ok=True)
    records = {}
    for split, rows in datasets.items():
        path = output / f"{split}.jsonl"
        temp = path.with_suffix(".jsonl.tmp")
        with temp.open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        temp.replace(path)
        records[split] = {"rows": len(rows), "sha256": digest(path)}
    write_json(manifest_path, {"protocol": protocol, "splits": records})
    return datasets


def paired_metrics(rows, predictions):
    """Count pairs whose two answers are both correct, by pair type and relation."""
    correct = {prediction["id"]: prediction["correct"] for prediction in predictions}
    groups = {}
    for row in rows:
        groups.setdefault((row["domain"], row["group_id"]), {})[
            row["outcome"], row["wording"]
        ] = correct[row["id"]]
    result = {}
    for domain in ("all", "owner", "location"):
        selected = [group for (kind, _), group in groups.items() if domain in ("all", kind)]
        if selected:
            result[domain] = {
                "pair_total": 2 * len(selected),
                "paraphrase_both_correct": sum(g[i, 0] and g[i, 1] for g in selected for i in (0, 1)),
                "counterfactual_both_correct": sum(g[0, i] and g[1, i] for g in selected for i in (0, 1)),
                "groups_all_correct": sum(all(g.values()) for g in selected),
                "group_total": len(selected),
            }
    return result


def run_arm(output, arm, datasets, tokenizer, tokenizer_path, source, replay, validation, cfg, steps):
    result_path = output / arm / "result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if digest(result_path.parent / "final.pth") != result["checkpoint_sha256"]:
            raise ValueError(f"Saved {arm} checkpoint does not match its result.")
        return result
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    model, metadata = load_checkpoint(source, device, tokenizer_path)
    model_config = model_config_from_checkpoint(metadata)
    if model_config.max_seq_len != CONTEXT:
        raise ValueError(f"This experiment expects a {CONTEXT}-token checkpoint.")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    qa = [encode_sft_example(tokenizer, r["prompt"], r["response"], CONTEXT) for r in datasets[arm]]
    streams = {"story": BudgetStream(replay, cfg.seed), "qa": BudgetStream(qa, cfg.seed + 1)}
    pad = tokenizer.token_to_id("<pad>")
    start = time.perf_counter()
    input_tokens = 0
    for step in range(1, steps + 1):
        losses, processed = train_mixture_update(
            model, optimizer, streams, QUOTAS, device, pad, cfg.batch_size, 1.0,
        )
        input_tokens += processed
        if step % cfg.log_every == 0 or step == steps:
            print(f"{arm} {step}/{steps} {losses}", flush=True)
    elapsed = time.perf_counter() - start
    story_loss = evaluate(model, validation, device, pad)
    checkpoint = result_path.parent / "final.pth"
    save_checkpoint(checkpoint, model, optimizer, model_config, cfg, 1, steps, story_loss,
                    tokenizer_path, source_checkpoint=str(source), stage="sft")
    result = {"updates": steps, "supervised_tokens": {k: v * steps for k, v in QUOTAS.items()},
              "input_tokens": input_tokens, "train_seconds": elapsed, "story_val_loss": story_loss,
              "checkpoint_sha256": digest(checkpoint), "qa": {}}
    for split in ("seen", "wording", "combined"):
        rows = datasets[split]
        metrics, predictions = task_evaluate(model, tokenizer, rows)
        result["qa"][split] = {**metrics, "pairs": paired_metrics(rows, predictions)}
        write_json(result_path.parent / f"{split}_predictions.json", predictions)

    story_tasks = [{**r, "prompt": r["story"]} for r in datasets["combined"]
                   if r["wording"] == 0 and len(tokenizer.encode(r["story"]).ids) + 65 <= CONTEXT][:24]
    stories = generate_rows(model, tokenizer, story_tasks, budget=64, mode="greedy")
    result["story_continuations"] = narrative_metrics(stories)
    write_json(result_path.parent / "story_continuations.json", stories)
    write_json(result_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--asset-root", type=Path, default=ROOT,
                        help="Project containing the original checkpoint and TinyStories cache.")
    parser.add_argument("--checkpoint", type=Path, help="Override the full pretraining checkpoint path.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/paired_state")
    parser.add_argument("--groups", type=int, default=1024)
    parser.add_argument("--test-groups", type=int, default=120)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.groups < 1 or args.test_groups < 2 or args.steps < 1:
        parser.error("Use positive group and step counts and at least two test groups per split.")

    output = args.output_dir.resolve()
    tokenizer = load_tokenizer(ROOT)
    datasets = prepare(output, tokenizer, args.groups, args.test_groups)
    print(f"Prepared {', '.join(f'{name}={len(rows)}' for name, rows in datasets.items())}", flush=True)
    if args.action == "prepare":
        return

    source = (args.checkpoint or args.asset_root / "models/run_20261118_183053_375283/best.pth").resolve()
    cache = args.asset_root / "data/tinystories_full"
    manifest_path = cache / "tokenized_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Use the original tokenizer bytes for checkpoint and cache fingerprints.
    # A Windows checkout can have equivalent tokenizer contents with CRLF endings.
    tokenizer_path = args.asset_root / "artifacts/tokenizer.json"
    if load_tokenizer(args.asset_root).to_str() != tokenizer.to_str():
        raise ValueError("The asset tokenizer differs from the tokenizer used for task preparation.")
    if (manifest["tokenizer_sha256"] != digest(tokenizer_path) or manifest["max_seq_len"] != CONTEXT):
        raise ValueError("Story cache uses a different tokenizer or context length.")
    cfg = TrainConfig(seed=SEED, epochs=1, batch_size=16, learning_rate=1e-4,
                      device=args.device, log_every=100)
    protocol = {"checkpoint_sha256": digest(source), "story_cache_sha256": digest(manifest_path),
                "data_sha256": digest(output / "data_manifest.json"), "updates": args.steps,
                "seed": cfg.seed, "learning_rate": cfg.learning_rate, "microbatch_size": cfg.batch_size,
                "device": cfg.device, "targets_per_update": QUOTAS}
    run_protocol = output / "run_protocol.json"
    if run_protocol.is_file() and json.loads(run_protocol.read_text(encoding="utf-8")) != protocol:
        raise ValueError("This directory contains a run with a different training protocol.")
    write_json(run_protocol, protocol)

    train_data = CachedSequences(cache, "train", manifest, CONTEXT)
    indices = np.random.default_rng(SEED).choice(len(train_data), size=min(8192, len(train_data)),
                                                  replace=False)
    replay = [(seq[:-1], seq[1:]) for seq in (train_data[int(i)] for i in indices)]
    validation = DataLoader(CachedSequences(cache, "validation", manifest, CONTEXT), batch_size=8,
                            collate_fn=partial(collate_sequences, pad_id=tokenizer.token_to_id("<pad>")))
    results = {arm: run_arm(output, arm, datasets, tokenizer, tokenizer_path, source,
                            replay, validation, cfg, args.steps)
               for arm in ("control", "paired")}
    write_json(output / "results.json", results)
    print(f"Results: {output / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
