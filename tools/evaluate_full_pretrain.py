"""Evaluate full-data pretraining against saved baselines without updates."""

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import torch

from codes.checkpoint import load_checkpoint, model_config_from_checkpoint
from codes.data import load_tokenizer, special_token_ids
from codes.training import evaluate
from .architecture_ablation import load_json, make_loader
from .data_ablation import EXP, ROOT, clean, digest, write_json
from .evaluate_architecture import make_cases, score, summarize

OUT = ROOT / "reports/full_pretrain_20261118"


@torch.inference_mode()
def greedy(model, tokenizer, prompt: str) -> dict:
    """Continue by argmax with the same context and 160-token limit as sampling."""
    special = special_token_ids(tokenizer)
    ids = [special["<bos>"], *tokenizer.encode(prompt).ids]
    start = len(ids)
    stop = "token_budget"
    device = next(model.parameters()).device
    for _ in range(160):
        if len(ids) >= model.max_seq_len:
            stop = "context_limit"
            break
        token = model(torch.tensor([ids], device=device))[0, -1].argmax().item()
        ids.append(token)
        if token == special["<eos>"]:
            stop = "eos"
            break
    return {"prompt": prompt, "continuation": tokenizer.decode(ids[start:], skip_special_tokens=True),
            "stop_reason": stop, "generated_tokens": len(ids) - start}


def adjacent_cases(tokenizer, original):
    """Remove filler while retaining the existing ownership question and name swaps."""
    cases = []
    for row in original:
        if row["distance"] != 20:
            continue
        event = row["prompt"].split(". ", 1)[0] + "."
        cue = "The " + row["prompt"].rsplit("The ", 1)[-1]
        cases.append({**row, "prompt": event + " " + cue, "distance": 0,
                      "gap_tokens": len(tokenizer.encode(" " + cue).ids)})
    return cases


def main():
    from codes.generate import generate

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    full_dir = ROOT / "data/tinystories_full"
    manifest = load_json(full_dir / "tokenized_manifest.json")
    tokenizer_path = ROOT / "artifacts/tokenizer.json"
    assert digest(tokenizer_path) == manifest["tokenizer_sha256"]
    assert digest(ROOT / "data/validation.jsonl") == manifest["splits"]["validation"]["files"]["validation.jsonl"]
    for split in manifest["splits"].values():
        for name, sha in split["files"].items():
            assert digest(full_dir / name) == sha, name
    metrics_path = args.checkpoint.parent / "metrics.jsonl"
    metrics = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    assert len(metrics) == 1
    assert metrics[0]["steps"] == (manifest["splits"]["train"]["stories"] + 7) // 8
    assert metrics[0]["tokens"] == manifest["splits"]["train"]["valid_targets"]
    tokenizer = load_tokenizer(ROOT)
    special = special_token_ids(tokenizer)
    val = make_loader(ROOT / "data/validation.jsonl", tokenizer, special, 256, 8, False)
    clean_val = make_loader(ROOT / "data/validation.jsonl", tokenizer, special, 256, 8,
                            False, keep=lambda s: clean(s)[1] is None)
    cases = make_cases(tokenizer)
    cases += adjacent_cases(tokenizer, cases)
    prompts = load_json(EXP / "runs/C/generations.json")
    models = {
        "C": EXP / "runs/C/epoch_1.pth",
        "100k_5epochs": ROOT / "experiments/repeated_data_20261118/100k_5epochs/epoch_5.pth",
        "previous_pretrain": ROOT / "models/run_20261117_164236_439489/best.pth",
        "full": args.checkpoint.resolve(),
    }
    results = {}
    reference_config = None
    for group, path in models.items():
        before = digest(path)
        model, checkpoint = load_checkpoint(path, "cuda", tokenizer_path)
        config = asdict(model_config_from_checkpoint(checkpoint))
        reference_config = config if reference_config is None else reference_config
        assert config == reference_config
        assert checkpoint.get("stage", "pretrain") == "pretrain"
        assert all(torch.isfinite(p).all().item() for p in model.parameters())
        loss = evaluate(model, val, torch.device("cuda"), special["<pad>"])
        assert abs(loss - checkpoint["val_loss"]) < 1e-5
        rows = score(model, tokenizer, cases)
        write_json(OUT / f"probe_{group}.json", rows)
        summary = {str(d): summarize([r for r in rows if r["distance"] == d])
                   for d in (0, 20, 80, 160)}
        summary["all_original_384"] = summarize([r for r in rows if r["distance"] != 0])
        record = {
            "checkpoint": str(path), "checkpoint_sha256": before,
            "model_config": config, "train_config": checkpoint["train_config"],
            "epoch": checkpoint["epoch"], "global_step": checkpoint["global_step"],
            "parameters": sum(p.numel() for p in model.parameters()),
            "saved_val_loss": checkpoint["val_loss"], "validation_loss": loss,
            "validation_targets": 366546, "validation_stories": len(val.dataset),
            "clean_validation_loss": evaluate(model, clean_val, torch.device("cuda"), special["<pad>"]),
            "clean_validation_stories": len(clean_val.dataset),
            "perplexity": math.exp(loss), "probe": summary,
        }
        if group == "full":
            assert checkpoint["epoch"] == metrics[0]["epoch"] == 1
            assert checkpoint["global_step"] == metrics[0]["global_step"]
            assert checkpoint["source_checkpoint"] is None
            record["train_metrics"] = metrics[0]
            record["prepared_data_manifest_sha256"] = digest(full_dir / "tokenized_manifest.json")
        results[group] = record
        sampled = [
            {**p, "continuation": generate(model, tokenizer, p["prompt"], 160,
                                            p["temperature"], p["seed"])}
            for p in prompts
        ]
        deterministic = [greedy(model, tokenizer, p["prompt"]) for p in prompts[::2]]
        write_json(OUT / f"generations_{group}.json",
                   {"sampled": sampled, "greedy": deterministic})
        assert digest(path) == before
        print(group, f"loss={loss:.6f}", summary, flush=True)
        write_json(OUT / "comparison.json", results)
        del model, checkpoint
        torch.cuda.empty_cache()
    write_json(OUT / "audit.json", {
        "mode": "evaluation_only", "optimizer_updates": 0,
        "tokenizer_sha256": digest(tokenizer_path),
        "validation_sha256": digest(ROOT / "data/validation.jsonl"),
        "prepared_data_manifest_sha256": digest(full_dir / "tokenized_manifest.json"),
        "models": {g: r["checkpoint_sha256"] for g, r in results.items()},
        "torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(),
    })
    report(results)


def report(results):
    lines = [
        "# 全量 TinyStories 预训练评估",
        "",
        "四个模型结构及 tokenizer 一致。统一使用原有 2,000 篇验证故事、256 上下文和相同目标 token 覆盖。",
        "清洗验证子集只过滤疑似乱码及非法内容，共 1,886 篇。全部权重均重新评估并核对保存的验证 loss。",
        "本轮只进行前向评估和生成，优化器更新次数为 0，权重文件在评估前后核对 SHA-256。",
        "",
        "|模型|累计更新|验证 loss|PPL|清洗验证 loss|人物归属正确率|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for group, r in results.items():
        p = r["probe"]["all_original_384"]
        lines.append(f"|{group}|{r['global_step']:,}|{r['validation_loss']:.6f}|{r['perplexity']:.4f}|{r['clean_validation_loss']:.6f}|{p['correct']}/{p['items']} ({p['correct']/p['items']:.1%})|")
    lines += [
        "",
        "full 对应全量清洗后 1,689,700 篇故事训练 1 轮；C 为 50 万篇训练 1 轮；100k_5epochs 为 10 万篇训练 5 轮。",
        "previous_pretrain 为此前 SFT 使用的预训练底座，其语料和学习率具有多阶段训练历史。",
        "",
        "## 人物归属与距离",
        "",
        "原有 384 条测试包含姓名互换、8 对姓名、4 件物品、gave/got 两种关系及三个距离。",
        "额外加入 128 条移除环境描写的相邻测试。直接比较两个候选单 token 人名在同一续写位置的分数。",
        "间隔包含交接句结束后到候选姓名之前的文本。互换姓名的两条题均正确，可以减少固定人名偏好的影响。",
        "",
        "|模型|实际间隔 token|单条正确率|姓名互换后均正确|",
        "|---|---:|---:|---:|",
    ]
    for group, r in results.items():
        for d in ("0", "20", "80", "160"):
            p = r["probe"][d]
            lines.append(f"|{group}|{p['gap_range']}|{p['correct']}/{p['items']} ({p['correct']/p['items']:.1%})|{p['both_correct']}/{p['pairs']} ({p['both_correct']/p['pairs']:.1%})|")
    lines += [
        "",
        "## 比较范围",
        "",
        "全量模型训练 325,448,604 个有效 token，211,213 次更新，训练耗时约 71.3 分钟。",
        "C 使用 96,262,605 个有效 token。全量模型同时增加了不同故事数量和训练计算，比较收益包含这两项影响。",
        "本轮使用此前反复比较过的验证集及人物归属模板。模板题具有相关性，结果应结合自由续写阅读。",
        "单轮训练只有最终轮验证记录，无法据此判断学习曲线已收敛或最优训练轮数。",
        "源日志的步数与 token 数符合准备清单；checkpoint 没有单独记录训练数据路径或内容指纹。",
        "固定提示的采样续写沿用原来的 4 个故事开头、temperature 0.5/0.8、seed 2026、最多 160 token。",
        "另附相同开头的 argmax 续写及停止原因。所有生成均使用故事续写前缀。",
        "",
    ]
    (OUT / "results.md").write_text("\n".join(lines), encoding="utf-8")
    samples = {g: load_json(OUT / f"generations_{g}.json") for g in results}
    lines = ["# 统一故事开头的生成对照", ""]
    for mode in ("greedy", "sampled"):
        count = len(samples["full"][mode])
        for i in range(count):
            p = samples["full"][mode][i]
            suffix = "" if mode == "greedy" else f"，temperature {p['temperature']}"
            lines += [f"## {mode} {i + 1}{suffix}", "", p["prompt"], ""]
            for group in results:
                row = samples[group][mode][i]
                lines += [f"### {group}", "", row["continuation"], ""]
                if mode == "greedy":
                    lines += [f"停止原因为 {row['stop_reason']}，生成 {row['generated_tokens']} token。", ""]
    (OUT / "generations.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
