"""Audit state-tracking trials and summarize paired test differences."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .architecture_ablation import load_json
from .data_ablation import ROOT, digest, write_json
from .state_rl_experiment import OUT, SOURCE
from .state_tasks import solve


def paired_difference(control, rl):
    """Bootstrap paired cases, leaving training-seed variability unmeasured."""
    assert [r["id"] for r in control] == [r["id"] for r in rl]
    delta = np.array([int(b["correct"]) - int(a["correct"])
                      for a, b in zip(control, rl)])
    rng = np.random.default_rng(2026)
    draws = delta[rng.integers(0, len(delta), size=(10000, len(delta)))].mean(1)
    return {
        "net_correct": int(delta.sum()), "cases": len(delta),
        "rl_only_correct": int((delta == 1).sum()),
        "sft_only_correct": int((delta == -1).sum()),
        "accuracy_difference": float(delta.mean()),
        "paired_case_bootstrap_95": np.quantile(draws, [0.025, 0.975]).tolist(),
    }


def main():
    initial = load_json(OUT / "evaluation.json")
    hard = load_json(OUT / "hard_evaluation.json")
    differences = {}
    a = load_json(OUT / "hard_predictions_sft_hard_control.json")
    b = load_json(OUT / "hard_predictions_grpo_hard.json")
    for category in ("hard_seen", "hard_ood"):
        differences[category] = paired_difference(a[category], b[category])
    differences["hard_all"] = paired_difference(
        a["hard_seen"] + a["hard_ood"], b["hard_seen"] + b["hard_ood"])
    domains = {}
    for group, predictions in (("sft_hard_control", a), ("grpo_hard", b)):
        domains[group] = {
            category: {
                domain: {"correct": sum(r["correct"] for r in rows if r["domain"] == domain),
                         "total": sum(r["domain"] == domain for r in rows)}
                for domain in ("owner", "location")
            }
            for category, rows in predictions.items() if category.startswith("hard")
        }
    learning = {}
    for group in ("grpo", "grpo_hard"):
        rows = load_json(OUT / group / "history.json")
        learning[group] = {
            "mean_reward": sum(r["reward"] for r in rows) / len(rows),
            "mixed_group_fraction": sum(r["mixed_group_rate"] for r in rows) / len(rows),
            "steps_with_mixed_groups": sum(r["mixed_group_rate"] > 0 for r in rows),
            "first_100_reward": sum(r["reward"] for r in rows[:100]) / 100,
            "last_100_reward": sum(r["reward"] for r in rows[-100:]) / 100,
        }
    paths = {g: r["checkpoint"] for g, r in initial.items()}
    paths.update({g: str(OUT / g / "step_400.pth")
                  for g in ("sft_hard_control", "grpo_hard")})
    fingerprints = {}
    for group, path in paths.items():
        path = Path(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if group == "sft":
            assert checkpoint["epoch"] == 5 and checkpoint["global_step"] == 2560
        elif group != "base":
            assert checkpoint["global_step"] == 400
            assert all(float(s["step"]) == 400 for s in checkpoint["optimizer_state_dict"]["state"].values())
            assert load_json(OUT / group / "result.json")["source_sha256"] == digest(OUT / "sft/best.pth")
        fingerprints[group] = digest(path)
        del checkpoint
    protocol = load_json(OUT / "protocol.json")
    assert digest(SOURCE) == protocol["source_sha256"]
    assert fingerprints["sft"] == load_json(OUT / "hard_protocol.json")["common_sft_sha256"]
    prompts = set()
    data = {}
    for file in sorted((OUT / "data").glob("*.jsonl")):
        rows = [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines()]
        for row in rows:
            assert row["prompt"] not in prompts
            prompts.add(row["prompt"])
            assert solve(row["initial"], row["events"], row["target"]) == row["answer"]
        data[file.stem] = {
            "rows": len(rows), "sha256": digest(file),
            "domain": dict(Counter(r["domain"] for r in rows)),
            "hops": dict(Counter(r["hops"] for r in rows)),
            "distance": dict(Counter(r["distance"] for r in rows)),
        }
    for record in (protocol, load_json(OUT / "hard_protocol.json")):
        for name, expected in record["data"].items():
            assert data[name]["sha256"] == expected["sha256"]
    write_json(OUT / "audit.json", {
        "source_unchanged": True, "checkpoint_fingerprints": fingerprints,
        "all_prompts_unique": len(prompts), "state_replay_verified": True,
        "data": data, "paired_differences": differences, "learning_signal": learning,
        "hard_domain_breakdown": domains,
        "torch_version": torch.__version__,
    })
    lines = [
        "# 状态追踪实验的补充对照",
        "",
        "第一轮 GRPO 中仅 2.1875% 的回答组具有奖励差异。在检查训练饱和后追加难题对照，使用新的独立测试题。",
        "两个难题分支仍从同一个专项 SFT 权重出发，均为 400 次更新、1600 个相同提示组、12800 个回答槽位。",
        "难题包含 3–5 次目标状态变化、独立物品干扰，以及 90/110/130-token 的环境描写长度目标。仅接受 prompt 不超过 244 token 的题目，实际分布见 audit.json。",
        "所有生成均未限制候选词表。任务奖励和参考策略、KL 系数、学习率与第一轮一致。",
        "",
        "|模型|新难题，同分布|新难题，新人名及句式|",
        "|---|---:|---:|",
    ]
    for group, r in hard.items():
        lines.append(f"|{group}|{r['tasks']['hard_seen']['accuracy']:.1%}|{r['tasks']['hard_ood']['accuracy']:.1%}|")
    lines += [
        "",
        "|GRPO 减去继续 SFT|净多答对|正确率差|配对题目自助法 95% 区间|",
        "|---|---:|---:|---|",
    ]
    for name, r in differences.items():
        lo, hi = r["paired_case_bootstrap_95"]
        lines.append(f"|{name}|{r['net_correct']}/{r['cases']}|{r['accuracy_difference']:+.1%}|[{lo:+.1%}, {hi:+.1%}]|")
    lines += [
        "",
        "自助法区间仅反映这批题目的抽样变化。训练只使用一个随机种子，多个指标同时比较，区间未覆盖训练随机性及多重比较。",
        "",
        "|训练信号|第一轮 GRPO|难题 GRPO|",
        "|---|---:|---:|",
        f"|奖励不同的组|{learning['grpo']['mixed_group_fraction']:.1%}|{learning['grpo_hard']['mixed_group_fraction']:.1%}|",
        f"|前 100 次更新平均奖励|{learning['grpo']['first_100_reward']:.3f}|{learning['grpo_hard']['first_100_reward']:.3f}|",
        f"|后 100 次更新平均奖励|{learning['grpo']['last_100_reward']:.3f}|{learning['grpo_hard']['last_100_reward']:.3f}|",
        "",
        "|最终难题分支|故事验证 loss|原 384 题人物归属|第一轮 1120 题总体|",
        "|---|---:|---:|---:|",
    ]
    for group in ("sft_hard_control", "grpo_hard"):
        r = hard[group]
        p = r["original_probe"]
        accuracy = sum(t["correct"] for t in r["original_test"].values()) / 1120
        lines.append(f"|{group}|{r['validation_loss']:.4f}|{p['correct']/p['items']:.1%}|{accuracy:.1%}|")
    lines += [
        "",
        "第一轮测试题已在追加难题训练前评估，因此追加分支在第一轮 1120 题上的分数属于探索性复测。新增的 640 道难题在两个追加分支训练结束后统一评估。",
        "新人名及句式难题中，继续 SFT 的人物题为 3/160，位置题为 74/160；GRPO 分别为 16/160 和 80/160。人物类组合泛化仍然薄弱。",
        "原 384 题为此前长期使用的开发测试，包含专项训练接触的人名和交接词，应用于衡量格式之外的迁移。",
        "新生成的短答任务与自然故事生成不同。难题分支的故事续写保存在 hard_generations.json。",
        "本次新增实验采用独立工具，复用既有 SFT 编码、训练循环、GRPO 损失和检查点格式。",
        "",
        "复现第一轮可运行 python -m tools.state_rl_experiment run，追加对照运行 python -m tools.state_rl_hard，最后运行 python -m tools.summarize_state_rl。",
        "已完成的训练阶段会读取 result.json 并跳过更新，评估部分会重新运行。",
        "",
    ]
    (OUT / "hard_results.md").write_text("\n".join(lines), encoding="utf-8")
    print("Paired differences", differences)
    print("Training signals", learning)
    print("Verified distinct prompts", len(prompts))


if __name__ == "__main__":
    main()
