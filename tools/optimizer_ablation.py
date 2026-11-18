"""Compare one warmup/cosine Adam configuration with the archived A control."""

import math
from dataclasses import asdict, replace

import torch

from codes.checkpoint import load_checkpoint, save_checkpoint
from codes.config import ModelConfig, TrainConfig
from codes.data import load_tokenizer, special_token_ids
from codes.generate import generate
from codes.model import build_model
from codes.training import evaluate, train_epoch
from .architecture_ablation import A, fingerprint, load_json, make_loader
from .data_ablation import EXP, ROOT, clean, digest, write_json
from .evaluate_architecture import make_cases, score, summarize

OUT = ROOT / "experiments/optimizer_20261118"
OPTIMIZATION = {
    "optimizer": "torch.optim.Adam",
    "betas": [0.9, 0.95],
    "eps": 1e-8,
    "weight_decay": 0.0,
    "peak_learning_rate": 1e-3,
    "final_learning_rate": 1e-4,
    "warmup_steps": 625,
    "total_steps": 12500,
    "max_grad_norm": 1.0,
}


def lr_factor(step: int) -> float:
    """Set the rate for update step+1, including both schedule endpoints."""
    update = step + 1
    warmup = OPTIMIZATION["warmup_steps"]
    if update <= warmup:
        return update / warmup
    progress = min((update - warmup) / (OPTIMIZATION["total_steps"] - warmup), 1.0)
    minimum = OPTIMIZATION["final_learning_rate"] / OPTIMIZATION["peak_learning_rate"]
    return minimum + (1 - minimum) * (1 + math.cos(math.pi * progress)) / 2


def main():
    baseline = load_json(A / "result.json")
    cfg = replace(TrainConfig(**baseline["train_config"]),
                  learning_rate=OPTIMIZATION["peak_learning_rate"])
    config = ModelConfig(**baseline["model_config"])
    tokenizer_path = ROOT / "artifacts/tokenizer.json"
    inputs = ((EXP / "clean_100k.jsonl", "train_sha256"),
              (ROOT / "data/validation.jsonl", "validation_sha256"),
              (tokenizer_path, "tokenizer_sha256"))
    for path, key in inputs:
        assert digest(path) == baseline[key], path
    device = torch.device(cfg.device)
    assert torch.__version__ == baseline["torch_version"]
    tokenizer = load_tokenizer(ROOT)
    special = special_token_ids(tokenizer)
    pad = special["<pad>"]
    train = make_loader(inputs[0][0], tokenizer, special, 256, 8, True)
    val = make_loader(inputs[1][0], tokenizer, special, 256, 8, False)
    run = OUT / "optimized"
    if not (run / "result.json").exists():
        torch.manual_seed(cfg.seed)
        model = build_model(config).to(device)
        assert fingerprint(model) == baseline["initial_parameters_sha256"]
        opt = torch.optim.Adam(
            model.parameters(), lr=cfg.learning_rate, betas=tuple(OPTIMIZATION["betas"]),
            eps=OPTIMIZATION["eps"], weight_decay=OPTIMIZATION["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
        record = {
            "group": "optimized", "model_config": asdict(config),
            "train_config": asdict(cfg), "optimization": OPTIMIZATION,
            "parameters": sum(p.numel() for p in model.parameters()),
            "initial_parameters_sha256": fingerprint(model),
            "source_checkpoint": None, "torch_version": torch.__version__,
            "gpu": torch.cuda.get_device_name(device),
            **{key: baseline[key] for _, key in inputs},
        }
        initial_val = evaluate(model, val, device, pad)
        assert abs(initial_val - baseline["initial_native_val_loss"]) < 1e-5
        record["initial_native_val_loss"] = initial_val
        write_json(run / "config.json", record)
        print("Training optimization experiment", flush=True)
        metrics = train_epoch(model, train, device, opt, pad, cfg.log_every,
                              scheduler=scheduler,
                              max_grad_norm=OPTIMIZATION["max_grad_norm"])
        assert metrics["tokens"] == baseline["tokens"]
        assert metrics["steps"] == baseline["steps"] == OPTIMIZATION["total_steps"]
        loss = evaluate(model, val, device, pad)
        # The saved TrainConfig matches the optimizer's actual ending rate.
        saved_cfg = replace(cfg, learning_rate=opt.param_groups[0]["lr"])
        save_checkpoint(run / "epoch_1.pth", model, opt, config, saved_cfg, 1,
                        metrics["steps"], loss, tokenizer_path)
        write_json(run / "scheduler.json", scheduler.state_dict())
        record.update(metrics, native_val_loss=loss,
                      final_learning_rate=opt.param_groups[0]["lr"])
        samples = []
        for p in load_json(A / "generations.json"):
            text = generate(model, tokenizer, p["prompt"], 160, p["temperature"], p["seed"])
            samples.append({**p, "continuation": text})
        write_json(run / "generations.json", samples)
        write_json(run / "result.json", record)
        print(f"Completed optimized training: validation={loss:.6f}", flush=True)
    compare(tokenizer, special, val)
    for path, key in inputs:
        assert digest(path) == baseline[key], path


def compare(tokenizer, special, val):
    cases = make_cases(tokenizer)
    clean_val = make_loader(ROOT / "data/validation.jsonl", tokenizer, special, 256, 8,
                            False, keep=lambda s: clean(s)[1] is None)
    results = {}
    for group, run in (("A", A), ("optimized", OUT / "optimized")):
        model, checkpoint = load_checkpoint(run / "epoch_1.pth", "cuda",
                                             ROOT / "artifacts/tokenizer.json")
        record = load_json(run / "result.json")
        rows = score(model, tokenizer, cases)
        write_json(OUT / f"probe_{group}.json", rows)
        results[group] = {
            "training": record,
            "validation": evaluate(model, val, torch.device("cuda"), special["<pad>"]),
            "clean_validation": evaluate(model, clean_val, torch.device("cuda"), special["<pad>"]),
            "probe": {str(d): summarize([r for r in rows if r["distance"] == d])
                      for d in (20, 80, 160)},
        }
        results[group]["probe"]["all"] = summarize(rows)
        assert abs(results[group]["validation"] - checkpoint["val_loss"]) < 1e-5
        print(group, results[group]["validation"], results[group]["probe"]["all"], flush=True)
    write_json(OUT / "comparison.json", results)
    report(results)


def report(results):
    lines = [
        "# StorySprout 优化配置实验",
        "",
        "A 使用原有权重。优化组从头训练，结构与 A 一致，初始权重 SHA-256 和初始验证 loss 均与 A 核对一致。",
        "两组使用相同的 10 万篇清洗故事、tokenizer、验证集、256 上下文、batch size 8、seed 2026 及故事顺序。",
        "每组 1 epoch，19,162,135 个有效 token，12,500 次更新。",
        "",
        "|设置|A|优化组|",
        "|---|---|---|",
        "|优化器|Adam|Adam|",
        "|β|0.9, 0.999|0.9, 0.95|",
        "|ε|1e-8|1e-8|",
        "|权重衰减|0|0|",
        "|学习率|固定 3e-4|峰值 1e-3，末尾 1e-4|",
        "|学习率曲线|固定|625 步线性预热，随后余弦衰减|",
        "|梯度范数裁剪|关闭|1.0|",
        "",
        "这组设置尝试加快前期学习，通过预热和裁剪控制大更新，并在训练末期降低更新幅度。属于多项优化配置的联合实验。",
        "",
        "|组别|训练 loss|验证 loss|清洗验证子集 loss|人物归属正确率|",
        "|---|---:|---:|---:|---:|",
    ]
    for g, r in results.items():
        s = r["probe"]["all"]
        lines.append(f"|{g}|{r['training']['loss']:.6f}|{r['validation']:.6f}|{r['clean_validation']:.6f}|{s['correct']}/{s['items']} ({s['correct']/s['items']:.1%})|")
    delta = (results["optimized"]["validation"] / results["A"]["validation"] - 1) * 100
    lines += ["", f"优化组验证 loss 相对 A 变化 {delta:+.2f}%。", "",
              "|间隔 token|A 人物归属正确率|优化组正确率|",
              "|---|---:|---:|"]
    for distance in ("20", "80", "160"):
        a = results["A"]["probe"][distance]
        b = results["optimized"]["probe"][distance]
        lines.append(f"|{a['gap_range']}|{a['correct']}/{a['items']} ({a['correct']/a['items']:.1%})|{b['correct']}/{b['items']} ({b['correct']/b['items']:.1%})|")
    lines += [
        "",
        "人物归属测试沿用上一轮 384 条模板，包含 8 对姓名、4 件物品、gave/got 两种表述、交换姓名及三个间隔。",
        "比较固定续写位置的两个候选姓名，候选均为单 token。结果只描述该模板任务。",
        "实验仅使用一个随机种子和 1 epoch，多个优化选项同时变化，单个选项的贡献与最终收敛质量仍需进一步测量。",
        "共享 train_epoch 通过可选关键字参数接受 scheduler 和 max_grad_norm。",
        "复现可从 storysprout 根目录运行 python -m tools.optimizer_ablation，已完成训练将跳过并重新评估权重。",
        "优化元数据及调度状态保存在 optimized/config.json 和 optimized/scheduler.json。",
        "checkpoint 保存最终学习率和 Adam 状态，可由现有入口继续训练；追加轮次的调度需由调用方设置。",
        "",
    ]
    (OUT / "results.md").write_text("\n".join(lines), encoding="utf-8")
    samples = {"A": load_json(A / "generations.json"),
               "optimized": load_json(OUT / "optimized/generations.json")}
    lines = ["# 相同提示续写对照", ""]
    for i, p in enumerate(samples["A"]):
        lines += [f"## 提示 {i // 2 + 1}，temperature {p['temperature']}", "", p["prompt"], ""]
        for g in samples:
            lines += [f"### {g}", "", samples[g][i]["continuation"], ""]
    (OUT / "generations.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
