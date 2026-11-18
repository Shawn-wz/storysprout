"""Run controlled state-tracking SFT and verifiable-reward GRPO experiments."""

import argparse
import gc
import json
import random
import time
from copy import deepcopy
from dataclasses import asdict
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from codes.checkpoint import load_checkpoint, model_config_from_checkpoint, save_checkpoint
from codes.config import TrainConfig
from codes.data import load_tokenizer
from codes.generate import generate
from codes.grpo import collate_rollouts, compute_group_advantages, compute_token_log_probs, grpo_loss
from codes.sft_data import collate_sft_examples, encode_sft_example
from codes.training import evaluate, train_epoch
from .data_ablation import ROOT, EXP, digest, write_json
from .architecture_ablation import load_json, make_loader
from .evaluate_architecture import make_cases, score, summarize
from .evaluate_full_pretrain import greedy
from .state_tasks import make_split, reward_ids, solve

OUT = ROOT / "experiments/state_rl_20261118"
SOURCE = ROOT / "models/run_20261118_183053_375283/best.pth"
PROTOCOL = {
    "dataset_version": 2,
    "seed": 2026, "sft_rows": 8192, "development_rows": 512, "rl_rows": 4096,
    "test_rows_per_category": 160, "sft_epochs": 6, "sft_batch_size": 16,
    "sft_learning_rate": 1e-4, "branch_updates": 400,
    "prompts_per_rollout": 4, "group_size": 8, "max_response_tokens": 4,
    "branch_learning_rate": 1e-5, "kl_beta": 0.02, "clip_eps": 0.2,
    "max_grad_norm": 1.0,
    "sft_selection": "highest development greedy exact accuracy; earlier epoch wins ties",
    "branch_selection": "fixed update 400",
    "reward": "exact correct name/place, optionally followed by one punctuation mark",
}


def tokens(tokenizer):
    return {t: tokenizer.token_to_id(t)
            for t in ("<bos>", "<user>", "<assistant>", "<eos>", "<pad>")}


def read_rows(name):
    return [json.loads(line) for line in (OUT / "data" / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]


def prepare(tokenizer):
    """Freeze data and protocol before exposing any test results."""
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        saved = load_json(OUT / "protocol.json")
        assert saved["settings"] == PROTOCOL
        assert saved["source_sha256"] == digest(SOURCE)
        for name, record in saved["data"].items():
            assert digest(OUT / "data" / f"{name}.jsonl") == record["sha256"]
        return
    seen = set()
    specifications = [("train", "train", 8192, 2026), ("dev", "dev", 512, 2027),
                      ("rl", "rl", 4096, 2028)]
    specifications += [(f"test_{c}", c, 160, 2100 + i) for i, c in enumerate(
        ("single", "chain", "long", "paraphrase", "names", "combined", "four_hops"))]
    records = {}
    (OUT / "data").mkdir(exist_ok=True)
    for name, category, count, seed in specifications:
        rows = make_split(tokenizer, category, count, seed, seen)
        path = OUT / "data" / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                assert solve(row["initial"], row["events"], row["target"]) == row["answer"]
                encode_sft_example(tokenizer, row["prompt"], row["response"], 256)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        records[name] = {"rows": len(rows), "sha256": digest(path)}
    write_json(OUT / "protocol.json", {
        "settings": PROTOCOL, "source": str(SOURCE), "source_sha256": digest(SOURCE),
        "tokenizer_sha256": digest(ROOT / "artifacts/tokenizer.json"), "data": records,
        "test_policy": "Prompt-disjoint splits; test names and test paraphrases excluded from training. Test scores are read after all training.",
    })
    print("Prepared task splits and frozen protocol.", flush=True)


def encode_prompt(tokenizer, row):
    special = tokens(tokenizer)
    return torch.tensor([special["<bos>"], special["<user>"],
                         *tokenizer.encode(row["prompt"]).ids, special["<assistant>"]])


@torch.no_grad()
def responses(model, prompts, special, budget=4, sample=False):
    """Decode a padded batch over the full vocabulary with independent stop lengths."""
    device = next(model.parameters()).device
    prompts = [p.to(device) for p in prompts]
    lengths = torch.tensor([len(p) for p in prompts], device=device)
    starts = lengths.clone()
    width = int(lengths.max()) + budget
    if width > model.max_seq_len:
        raise ValueError("Prompt leaves insufficient response room.")
    ids = torch.full((len(prompts), width), special["<pad>"], dtype=torch.long, device=device)
    for i, p in enumerate(prompts):
        ids[i, :len(p)] = p
    finished = torch.zeros(len(prompts), dtype=torch.bool, device=device)
    model.eval()
    rows = torch.arange(len(prompts), device=device)
    for _ in range(budget):
        logits = model(ids[:, :int(lengths.max())])[rows, lengths - 1]
        next_ids = (torch.multinomial(logits.softmax(-1), 1).squeeze(1) if sample
                    else logits.argmax(-1))
        active = ~finished
        ids[rows[active], lengths[active]] = next_ids[active]
        lengths += active.long()
        finished |= active & (next_ids == special["<eos>"])
        if finished.all():
            break
    return [
        {"input_ids": ids[i, :int(lengths[i])].clone(),
         "prompt_length": int(starts[i]), "sequence_length": int(lengths[i]),
         "ended_with_eos": bool(finished[i])}
        for i in range(len(prompts))
    ]


def task_evaluate(model, tokenizer, rows):
    """Measure free greedy answers, rather than restricting candidate names."""
    special = tokens(tokenizer)
    correct = valid = 0
    output = []
    for start in range(0, len(rows), 32):
        subset = rows[start:start + 32]
        samples = responses(model, [encode_prompt(tokenizer, r) for r in subset], special)
        for row, sample in zip(subset, samples):
            answer_ids = sample["input_ids"][sample["prompt_length"]:].tolist()
            reward = reward_ids(tokenizer, answer_ids, row["answer"], special)
            correct += int(reward)
            text = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
            candidates = set(row["names"]) | set(("box", "bag", "desk", "shelf", "basket", "drawer"))
            valid += int(text.rstrip(".!").casefold() in {name.casefold() for name in candidates})
            output.append({"id": row["id"], "domain": row["domain"],
                           "answer": row["answer"], "generated": text, "correct": bool(reward)})
    return {"correct": correct, "total": len(rows), "accuracy": correct / len(rows),
            "valid_answer_rate": valid / len(rows)}, output


def sft_loader(tokenizer, rows, batch=16, shuffle=True):
    items = [encode_sft_example(tokenizer, r["prompt"], r["response"], 256) for r in rows]
    return DataLoader(items, batch_size=batch, shuffle=shuffle,
                      collate_fn=partial(collate_sft_examples, pad_id=tokens(tokenizer)["<pad>"]))


def save(path, model, opt, metadata, step, val, stage, cfg, source, epoch=1):
    save_checkpoint(path, model, opt, model_config_from_checkpoint(metadata), cfg,
                    epoch, step, val, ROOT / "artifacts/tokenizer.json",
                    source_checkpoint=str(source), stage=stage)


def common_sft(tokenizer):
    """Learn the answer format and state task, selecting only on development data."""
    run = OUT / "sft"
    if (run / "result.json").exists():
        return
    torch.manual_seed(PROTOCOL["seed"])
    model, metadata = load_checkpoint(SOURCE, "cuda", ROOT / "artifacts/tokenizer.json")
    cfg = TrainConfig(epochs=6, batch_size=16, learning_rate=1e-4, log_every=256)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    train = sft_loader(tokenizer, read_rows("train"))
    dev_rows = read_rows("dev")
    dev = sft_loader(tokenizer, dev_rows, shuffle=False)
    history, best, step = [], -1.0, 0
    for epoch in range(1, cfg.epochs + 1):
        write_json(OUT / "status.json", {"phase": "sft", "epoch": epoch})
        metric = train_epoch(model, train, torch.device("cuda"), opt, -100, 256,
                             max_grad_norm=1.0)
        step += metric["steps"]
        loss = evaluate(model, dev, torch.device("cuda"), -100)
        scores, _ = task_evaluate(model, tokenizer, dev_rows)
        record = {"epoch": epoch, **metric, "dev_loss": loss, "dev": scores}
        history.append(record)
        if scores["accuracy"] > best:
            best = scores["accuracy"]
            save(run / "best.pth", model, opt, metadata, step, loss, "sft", cfg, SOURCE, epoch)
        save(run / "last.pth", model, opt, metadata, step, loss, "sft", cfg, SOURCE, epoch)
        write_json(run / "history.json", history)
        print("SFT", record, flush=True)
    write_json(run / "result.json", {"selected_dev_accuracy": best, "history": history})
    del model, opt
    torch.cuda.empty_cache()


def branch(tokenizer, mode, task_pool="rl", dev_pool="dev"):
    """Compare additional supervised updates with group-relative policy updates."""
    run = OUT / mode
    if (run / "result.json").exists():
        return
    torch.manual_seed(PROTOCOL["seed"])
    source = OUT / "sft/best.pth"
    model, metadata = load_checkpoint(source, "cuda", ROOT / "artifacts/tokenizer.json")
    cfg = TrainConfig(epochs=1, batch_size=32, learning_rate=1e-5, log_every=100)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    rows = read_rows(task_pool)
    dev_rows = read_rows(dev_pool)
    special = tokens(tokenizer)
    is_rl = mode.startswith("grpo")
    reference = deepcopy(model).requires_grad_(False).eval() if is_rl else None
    rng = random.Random(2029)
    order = []
    while len(order) < PROTOCOL["branch_updates"] * 4:
        block = list(range(len(rows)))
        rng.shuffle(block)
        order.extend(block)
    history, checks = [], []
    elapsed_start = time.perf_counter()
    for step in range(1, PROTOCOL["branch_updates"] + 1):
        selected = [rows[i] for i in order[(step - 1) * 4:step * 4]]
        expanded = [row for row in selected for _ in range(PROTOCOL["group_size"])]
        if not is_rl:
            loader = sft_loader(tokenizer, expanded, batch=32, shuffle=False)
            metric = train_epoch(model, loader, torch.device("cuda"), opt, -100,
                                 max_grad_norm=1.0)
            record = {"step": step, "loss": metric["loss"]}
        else:
            samples = responses(model, [encode_prompt(tokenizer, r) for r in expanded],
                                special, sample=True)
            batch = collate_rollouts(samples, special["<pad>"])
            rewards = torch.tensor([
                reward_ids(tokenizer, sample["input_ids"][sample["prompt_length"]:].tolist(),
                           row["answer"], special)
                for row, sample in zip(expanded, samples)
            ], device="cuda")
            grouped = rewards.reshape(4, PROTOCOL["group_size"])
            advantages = compute_group_advantages(grouped).flatten()
            ids = batch["input_ids"]
            with torch.no_grad():
                old = compute_token_log_probs(model(ids), ids)
                ref = compute_token_log_probs(reference(ids), ids)
            model.train()
            opt.zero_grad()
            new = compute_token_log_probs(model(ids), ids)
            loss = grpo_loss(new, old, ref, advantages, batch["response_mask"],
                             PROTOCOL["clip_eps"], PROTOCOL["kl_beta"])
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite GRPO objective.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
            record = {"step": step, "loss": loss.item(), "reward": rewards.mean().item(),
                      "mixed_group_rate": ((grouped.max(1).values - grouped.min(1).values) > 0).float().mean().item(),
                      "all_wrong_group_rate": (grouped.sum(1) == 0).float().mean().item()}
        history.append(record)
        if step % 100 == 0:
            scores, _ = task_evaluate(model, tokenizer, dev_rows)
            checks.append({"step": step, "dev": scores})
            save(run / f"step_{step}.pth", model, opt, metadata, step, float("nan"),
                 "grpo" if is_rl else "sft", cfg, source)
            write_json(run / "history.json", history)
            write_json(run / "development.json", checks)
            print(mode, step, scores, f"elapsed={time.perf_counter() - elapsed_start:.1f}s", flush=True)
            write_json(OUT / "status.json", {"phase": mode, "step": step})
    write_json(run / "result.json", {
        "updates": 400, "prompt_groups": 1600, "response_slots": 12800,
        "source_sha256": digest(source), "elapsed_seconds": time.perf_counter() - elapsed_start,
        "task_pool": task_pool, "development_pool": dev_pool,
        "development": checks, "last_100": history[-100:],
        "budget_note": "Same prompts, ordering, 400 optimizer steps and 12800 response slots. GRPO also computes sampling/old/reference forwards; compute is unequal.",
    })
    del model, opt, reference
    gc.collect()
    torch.cuda.empty_cache()


def evaluate_all(tokenizer):
    """Open final test splits only after both fixed-budget branches finish."""
    checkpoints = {
        "base": SOURCE, "sft": OUT / "sft/best.pth",
        "sft_control": OUT / "sft_control/step_400.pth",
        "grpo": OUT / "grpo/step_400.pth",
    }
    categories = ("single", "chain", "long", "paraphrase", "names", "combined", "four_hops")
    val = make_loader(ROOT / "data/validation.jsonl", tokenizer, tokens(tokenizer), 256, 8, False)
    cases = make_cases(tokenizer)
    fixed = load_json(EXP / "runs/C/generations.json")[::2]
    results = {}
    generations = {}
    for group, path in checkpoints.items():
        print("Evaluating", group, flush=True)
        model, _ = load_checkpoint(path, "cuda", ROOT / "artifacts/tokenizer.json")
        record = {"checkpoint": str(path), "checkpoint_sha256": digest(path), "tasks": {}}
        predictions = {}
        for category in categories:
            scores, output = task_evaluate(model, tokenizer, read_rows(f"test_{category}"))
            record["tasks"][category] = scores
            predictions[category] = output
        record["validation_loss"] = evaluate(model, val, torch.device("cuda"),
                                              tokens(tokenizer)["<pad>"])
        probe = score(model, tokenizer, cases)
        record["original_probe"] = summarize(probe)
        write_json(OUT / f"probe_{group}.json", probe)
        record["test_accuracy"] = sum(r["correct"] for r in record["tasks"].values()) / 1120
        write_json(OUT / f"predictions_{group}.json", predictions)
        generations[group] = {
            "plain_greedy": [greedy(model, tokenizer, row["prompt"]) for row in fixed],
            "plain_sampled": [
                {"prompt": row["prompt"], "continuation": generate(
                    model, tokenizer, row["prompt"], 160, 0.8, 2026)}
                for row in fixed],
        }
        results[group] = record
        write_json(OUT / "evaluation.json", results)
        write_json(OUT / "generations.json", generations)
        print(group, record, flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    report(results, generations)
    assert digest(SOURCE) == load_json(OUT / "protocol.json")["source_sha256"]
    write_json(OUT / "status.json", {"phase": "complete"})


def report(results, generations):
    lines = [
        "# 可验证状态任务的 SFT 与 GRPO 对照",
        "",
        "全量预训练底座为共同起点。专项 SFT 在 8192 道训练题上训练 6 轮，根据 512 道开发题的贪心精确正确率选取权重。",
        "继续 SFT 与 GRPO 均从同一专项 SFT 权重出发，使用相同的 1600 个提示组、12800 个回答槽位和 400 次参数更新。",
        "GRPO 使用规则奖励、8 个回答一组、完整词表采样、最多 4 个回答 token，KL 系数 0.02，学习率 1e-5。",
        "继续 SFT 使用同一提示及对应正确答案，学习率同为 1e-5。GRPO 额外进行采样及旧策略、参考策略前向，计算量更大。",
        "测试集共 1120 道题，训练与开发阶段均未读取测试成绩。所有 prompt 去重后跨集合无重叠。",
        "",
        "|测试类别|base|专项 SFT|继续 SFT|GRPO|",
        "|---|---:|---:|---:|---:|",
    ]
    for c in results["base"]["tasks"]:
        lines.append("|" + c + "|" + "|".join(f"{results[g]['tasks'][c]['accuracy']:.1%}" for g in results) + "|")
    lines += ["|总体|" + "|".join(f"{results[g]['test_accuracy']:.1%}" for g in results) + "|",
              "", "|其他指标|base|专项 SFT|继续 SFT|GRPO|", "|---|---:|---:|---:|"]
    lines.append("|原验证集 loss|" + "|".join(f"{r['validation_loss']:.4f}" for r in results.values()) + "|")
    lines.append("|原 384 题人物归属|" + "|".join(
        f"{r['original_probe']['correct']/r['original_probe']['items']:.1%}" for r in results.values()) + "|")
    lines += [
        "",
        "single 为单次目标状态变化；chain 为 2–3 次变化；long 加入约 100-token 环境描写。",
        "paraphrase 使用未训练的交接表述和问句；names 使用专项训练未见的人名；combined 同时引入新句式、新人名、三次变化及长间隔；four_hops 为四次变化。",
        "owner 和 location 两类任务等量，另有独立物品的干扰事件，部分持有任务包含交换全部物品。",
        "真实状态由独立事件重放器决定。奖励只接受单个正确姓名或地点，拒绝额外词语与插入角色标记。",
        "自由回答采用全词表贪心生成。plain 自由续写使用原有四个故事开头，以贪心和 temperature 0.8 各生成一次。",
        "单种子、小规模实验的结论针对当前任务和预算。格式学习对 base→SFT 的回答准确率变化也有贡献。",
        "训练后在专项问答上的改善需要结合原人物归属测试、语言模型验证 loss 与自由续写判断迁移。",
        "",
    ]
    (OUT / "results.md").write_text("\n".join(lines), encoding="utf-8")
    lines = ["# 原故事开头的生成迁移检查", ""]
    for mode in ("plain_greedy", "plain_sampled"):
        for i, row in enumerate(generations["base"][mode]):
            lines += [f"## {mode} {i + 1}", "", row["prompt"], ""]
            for group in generations:
                lines += [f"### {group}", "", generations[group][mode][i]["continuation"], ""]
    (OUT / "generations.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    args = parser.parse_args()
    tokenizer = load_tokenizer(ROOT)
    prepare(tokenizer)
    if args.action == "run":
        common_sft(tokenizer)
        branch(tokenizer, "sft_control")
        branch(tokenizer, "grpo")
        evaluate_all(tokenizer)


if __name__ == "__main__":
    main()
