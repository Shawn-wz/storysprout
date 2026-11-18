"""Compare 100k stories over five epochs with the archived 500k-story C run."""

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

C = EXP / "runs/C"
OUT = ROOT / "experiments/repeated_data_20261118"
RUN = OUT / "100k_5epochs"


def main():
    baseline = load_json(C / "result.json")
    a = load_json(A / "result.json")
    manifest = load_json(EXP / "data_manifest.json")
    train_path = EXP / "clean_100k.jsonl"
    val_path = ROOT / "data/validation.jsonl"
    tokenizer_path = ROOT / "artifacts/tokenizer.json"
    expected = {
        train_path: manifest["files"]["clean_100k.jsonl"]["sha256"],
        EXP / "clean_500k.jsonl": baseline["train_sha256"],
        val_path: baseline["validation_sha256"],
        tokenizer_path: baseline["tokenizer_sha256"],
    }
    for path, sha in expected.items():
        assert digest(path) == sha, path
    cfg = replace(TrainConfig(**baseline["train_config"]), epochs=5)
    config = ModelConfig(**baseline["model_config"])
    assert torch.__version__ == baseline["torch_version"]
    assert torch.cuda.get_device_name() == baseline["gpu"]
    device = torch.device(cfg.device)
    tokenizer = load_tokenizer(ROOT)
    special = special_token_ids(tokenizer)
    pad = special["<pad>"]
    train = make_loader(train_path, tokenizer, special, 256, cfg.batch_size, True)
    val = make_loader(val_path, tokenizer, special, 256, cfg.batch_size, False)
    record = {
        "group": "100k_5epochs", "control": str(C / "epoch_1.pth"),
        "model_config": asdict(config), "train_config": asdict(cfg),
        "optimizer": baseline["optimizer"], "source_checkpoint": None,
        "train_data": str(train_path), "train_sha256": expected[train_path],
        "validation_sha256": expected[val_path], "tokenizer_sha256": expected[tokenizer_path],
        "torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(),
        "stories": len(train.dataset), "parameters": 7347200,
    }
    if not (RUN / "result.json").exists():
        torch.manual_seed(cfg.seed)
        model = build_model(config).to(device)
        assert fingerprint(model) == baseline["initial_parameters_sha256"]
        record["initial_parameters_sha256"] = fingerprint(model)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
        initial = evaluate(model, val, device, pad)
        assert abs(initial - baseline["initial_native_val_loss"]) < 1e-5
        record["initial_native_val_loss"] = initial
        if (RUN / "config.json").exists():
            assert load_json(RUN / "config.json") == record
        write_json(RUN / "config.json", record)
        epochs = []
        if (RUN / "epochs.json").exists():
            epochs = load_json(RUN / "epochs.json")
        if epochs:
            last_epoch = epochs[-1]["epoch"]
            model, checkpoint = load_checkpoint(RUN / f"epoch_{last_epoch}.pth", device, tokenizer_path)
            opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
            opt.load_state_dict(checkpoint["optimizer_state_dict"])
            rng = torch.load(RUN / f"epoch_{last_epoch}_rng.pth", weights_only=True)
            torch.set_rng_state(rng["cpu"])
            torch.cuda.set_rng_state_all(rng["cuda"])
            print(f"Resuming after epoch {last_epoch}", flush=True)
        for epoch in range(len(epochs) + 1, 6):
            write_json(OUT / "status.json", {"phase": "training", "epoch": epoch})
            print(f"Epoch {epoch}/5", flush=True)
            metrics = train_epoch(model, train, device, opt, pad, cfg.log_every)
            assert metrics["steps"] == a["steps"] == 12500
            assert metrics["tokens"] == a["tokens"] == 19162135
            loss = evaluate(model, val, device, pad)
            if epoch == 1:
                assert abs(loss - a["native_val_loss"]) < 1e-5
                assert abs(metrics["loss"] - a["loss"]) < 1e-5
            save_checkpoint(RUN / f"epoch_{epoch}.pth", model, opt, config, cfg, epoch,
                            epoch * metrics["steps"], loss, tokenizer_path)
            torch.save({"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()},
                       RUN / f"epoch_{epoch}_rng.pth")
            epochs.append({"epoch": epoch, "validation_loss": loss, **metrics})
            write_json(RUN / "epochs.json", epochs)
            print(f"Finished epoch {epoch}: validation={loss:.6f}", flush=True)
        record.update(
            epoch=5, steps=sum(e["steps"] for e in epochs),
            tokens=sum(e["tokens"] for e in epochs),
            elapsed_seconds=sum(e["elapsed_seconds"] for e in epochs),
            loss=sum(e["loss"] * e["tokens"] for e in epochs) / sum(e["tokens"] for e in epochs),
            native_val_loss=epochs[-1]["validation_loss"],
            last_epoch_train_loss=epochs[-1]["loss"],
        )
        assert record["steps"] == baseline["steps"] == 62500
        write_json(RUN / "result.json", record)
    compare(tokenizer, special, val)
    for path, sha in expected.items():
        assert digest(path) == sha, path
    write_json(OUT / "status.json", {"phase": "complete", "epochs": 5})


def compare(tokenizer, special, val):
    write_json(OUT / "status.json", {"phase": "evaluation", "epochs": 5})
    clean_val = make_loader(ROOT / "data/validation.jsonl", tokenizer, special, 256, 8,
                            False, keep=lambda s: clean(s)[1] is None)
    cases = make_cases(tokenizer)
    prompts = load_json(C / "generations.json")
    results = {}
    samples = {}
    for group, run, epoch in (("C", C, 1), ("100k_5epochs", RUN, 5)):
        model, checkpoint = load_checkpoint(run / f"epoch_{epoch}.pth", "cuda",
                                             ROOT / "artifacts/tokenizer.json")
        record = load_json(run / "result.json")
        assert checkpoint["global_step"] == record["steps"] == 62500
        assert checkpoint["epoch"] == epoch
        loss = evaluate(model, val, torch.device("cuda"), special["<pad>"])
        assert abs(loss - record["native_val_loss"]) < 1e-5
        rows = score(model, tokenizer, cases)
        write_json(OUT / f"probe_{group}.json", rows)
        results[group] = {
            "training": record, "validation_loss": loss,
            "clean_validation_loss": evaluate(model, clean_val, torch.device("cuda"), special["<pad>"]),
            "probe": {str(d): summarize([r for r in rows if r["distance"] == d])
                      for d in (20, 80, 160)},
        }
        results[group]["probe"]["all"] = summarize(rows)
        if group == "C":
            samples[group] = prompts
        else:
            samples[group] = [
                {**p, "continuation": generate(model, tokenizer, p["prompt"], 160,
                                               p["temperature"], p["seed"])}
                for p in prompts
            ]
            write_json(RUN / "generations.json", samples[group])
        print(group, results[group], flush=True)
    write_json(OUT / "comparison.json", results)
    report(results, samples)


def report(results, samples):
    lines = [
        "# 10 万篇重复训练与 50 万篇单轮训练",
        "",
        "实验组为 clean_100k.jsonl 训练 5 轮，对照组使用原 C 组的 clean_500k.jsonl 训练 1 轮。",
        "两组都是 7,347,200 参数，4 层、4 头、每头 64 维、FFN 1024、256 上下文。",
        "均采用 Adam 默认参数、固定学习率 3e-4、batch size 8、seed 2026。初始化可训练参数指纹和初始验证 loss 相同。",
        "实验组第 1 轮的训练和验证 loss 与原 A 组核对一致。后续轮次继续保留优化器与随机状态。",
        "",
        "|组别|不同故事数|轮数|累计更新|有效训练 token|验证 loss|清洗验证子集 loss|人物归属正确率|",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for g, r in results.items():
        t = r["training"]
        p = r["probe"]["all"]
        lines.append(f"|{g}|{t['stories']:,}|{t['train_config']['epochs']}|{t['steps']:,}|{t['tokens']:,}|{r['validation_loss']:.6f}|{r['clean_validation_loss']:.6f}|{p['correct']}/{p['items']} ({p['correct']/p['items']:.1%})|")
    c = results["C"]
    e = results["100k_5epochs"]
    difference = (e["validation_loss"] / c["validation_loss"] - 1) * 100
    token_difference = (e["training"]["tokens"] / c["training"]["tokens"] - 1) * 100
    lines += [
        "", f"实验组验证 loss 相对 C 变化 {difference:+.2f}%。",
        f"更新次数相等，实验组有效训练 token 比 C 少 {abs(token_difference):.3f}%。",
        "",
        "## 实验组逐轮结果", "",
        "|轮次|本轮训练 loss|验证 loss|",
        "|---|---:|---:|",
    ]
    for epoch in load_json(RUN / "epochs.json"):
        lines.append(f"|{epoch['epoch']}|{epoch['loss']:.6f}|{epoch['validation_loss']:.6f}|")
    lines += [
        "", "## 人物归属测试", "",
        "|交接后的间隔 token|C 正确率|10 万篇 × 5 正确率|",
        "|---|---:|---:|",
    ]
    for distance in ("20", "80", "160"):
        a = c["probe"][distance]
        b = e["probe"][distance]
        lines.append(f"|{a['gap_range']}|{a['correct']}/{a['items']} ({a['correct']/a['items']:.1%})|{b['correct']}/{b['items']} ({b['correct']/b['items']:.1%})|")
    lines += [
        "",
        "测试沿用 384 条固定模板，包含 8 对姓名、4 件物品、gave/got 两种表述、姓名互换和三个间隔。比较两个单 token 人名的分数。",
        "两组使用同一验证集与 token 覆盖。10 万篇是 50 万篇数据的前缀子集，原有故事与补充故事的来源构成也有差异。",
        "单随机种子、单数据子集的结果主要描述本次训练预算下的效果，数据质量和数量的贡献仍有耦合。",
        "主要对照使用预先指定的第 5 轮权重。每轮 checkpoint、Adam 状态、随机状态和指标独立保存。",
        "复现从 storysprout 根目录运行 python -m tools.repeated_data_ablation。程序可从最近完成的轮次恢复训练。",
        "",
    ]
    (OUT / "results.md").write_text("\n".join(lines), encoding="utf-8")
    lines = ["# 相同提示的续写对照", ""]
    for i, p in enumerate(samples["C"]):
        lines += [f"## 提示 {i // 2 + 1}，temperature {p['temperature']}", "", p["prompt"], ""]
        for g in samples:
            lines += [f"### {g}", "", samples[g][i]["continuation"], ""]
    (OUT / "generations.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
