"""Train exact-target-token replay controls and narrative-transfer mixtures."""

import argparse
import gc
import time

import torch
import torch.nn.functional as F

from codes.checkpoint import load_checkpoint, model_config_from_checkpoint, save_checkpoint
from codes.config import TrainConfig
from codes.data import load_tokenizer
from codes.sft_data import collate_sft_examples
from codes.training import evaluate
from .architecture_ablation import load_json, make_loader
from .data_ablation import ROOT, digest, write_json
from .story_mix_data import OUT, SOURCE, SETTINGS, BudgetStream, prepare, read_pools, rows, plain_example


def train_arm(tokenizer, pools, group):
    run = OUT / group
    if (run / "result.json").exists():
        return
    torch.manual_seed(SETTINGS["seed"])
    model, metadata = load_checkpoint(SOURCE, "cuda", ROOT / "artifacts/tokenizer.json")
    opt = torch.optim.Adam(model.parameters(), lr=SETTINGS["learning_rate"])
    streams = {name: BudgetStream(examples, 8200 + i) for i, (name, examples) in enumerate(pools.items())}
    quotas = SETTINGS[group]
    pad = tokenizer.token_to_id("<pad>")
    special = {t: tokenizer.token_to_id(t) for t in ("<bos>", "<eos>", "<pad>")}
    val = make_loader(ROOT / "data/validation.jsonl", tokenizer, special, 256, 8, False)
    development = [plain_example(tokenizer, r["prompt"], r["response"])
                   for r in rows("continuation_dev")]
    dev = torch.utils.data.DataLoader(
        development, batch_size=16, shuffle=False,
        collate_fn=lambda batch: collate_sft_examples(batch, pad))
    history = []
    counts = {name: 0 for name in quotas}
    input_tokens = 0
    elapsed = time.perf_counter()
    for step in range(1, SETTINGS["updates"] + 1):
        model.train()
        opt.zero_grad()
        losses = {}
        for name, quota in quotas.items():
            examples = streams[name].take(quota)
            counted = sum(int((y != -100).sum()) for _, y in examples)
            assert counted == quota
            counts[name] += counted
            total_loss = 0.0
            for start in range(0, len(examples), SETTINGS["microbatch_size"]):
                batch = examples[start:start + SETTINGS["microbatch_size"]]
                features, labels = collate_sft_examples(batch, pad)
                input_tokens += sum(len(x) for x, _ in batch)
                logits = model(features.to("cuda"))
                loss_sum = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                           labels.to("cuda").reshape(-1),
                                           ignore_index=-100, reduction="sum")
                if not torch.isfinite(loss_sum):
                    raise FloatingPointError("Non-finite mixture loss.")
                (loss_sum / SETTINGS["targets_per_update"]).backward()
                total_loss += float(loss_sum.detach())
            losses[name] = total_loss / quota
        torch.nn.utils.clip_grad_norm_(model.parameters(), SETTINGS["max_grad_norm"],
                                      error_if_nonfinite=True)
        opt.step()
        history.append({"step": step, "losses": losses})
        if step % 500 == 0:
            validation = evaluate(model, val, torch.device("cuda"), pad)
            development_loss = evaluate(model, dev, torch.device("cuda"), -100)
            cfg = TrainConfig(epochs=1, batch_size=16, learning_rate=SETTINGS["learning_rate"])
            save_checkpoint(run / f"step_{step}.pth", model, opt,
                            model_config_from_checkpoint(metadata), cfg, 1, step,
                            validation, ROOT / "artifacts/tokenizer.json",
                            source_checkpoint=str(SOURCE), stage="sft")
            record = {"step": step, "story_val_loss": validation,
                      "continuation_dev_loss": development_loss,
                      "target_tokens": dict(counts), "elapsed_seconds": time.perf_counter() - elapsed}
            write_json(run / f"development_{step}.json", record)
            write_json(run / "history.json", history)
            write_json(OUT / "status.json", {"phase": group, "step": step})
            print(group, record, flush=True)
    assert sum(counts.values()) == 2000000
    write_json(run / "result.json", {
        "target_tokens": counts, "updates": SETTINGS["updates"],
        "input_tokens": input_tokens, "elapsed_seconds": time.perf_counter() - elapsed,
        "checkpoint_sha256": digest(run / "step_2000.pth"),
        "source_sha256": digest(SOURCE),
        "protocol_sha256": digest(OUT / "protocol.json"),
    })
    del model, opt
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "train"))
    args = parser.parse_args()
    prepare()
    if args.action == "train":
        tokenizer = load_tokenizer(ROOT)
        pools = read_pools(tokenizer)
        train_arm(tokenizer, pools, "control")
        train_arm(tokenizer, pools, "mixed")
        assert digest(SOURCE) == load_json(OUT / "protocol.json")["source_sha256"]
        write_json(OUT / "status.json", {"phase": "training_complete"})


if __name__ == "__main__":
    main()
