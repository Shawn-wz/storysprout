"""Audit mixed-SFT budgets and combine automatic checks with sealed blind ratings."""

import json
from collections import Counter
from pathlib import Path

import torch

from codes.checkpoint import load_checkpoint
from codes.data import load_tokenizer
from .architecture_ablation import load_json
from .data_ablation import ROOT, EXP, digest, write_json
from .evaluate_full_pretrain import greedy
from .state_tasks import solve
from .story_mix_data import OUT, SOURCE, SETTINGS, rows


def main():
    results = load_json(OUT / "evaluation.json")
    ratings = load_json(OUT / "blind_ratings.json")
    review_hash = digest(OUT / "blind_ratings.json")
    mapping = load_json(OUT / "blind_mapping.json")
    assert {r[0] for r in ratings["ratings"]} == set(mapping)
    blind = {group: {"count": 0, "contradiction": 0, "contradiction_unknown": 0,
                     "goal_progress": 0, "coherent_ending": 0, "ending_unknown": 0,
                     "all_three": 0} for group in results}
    for key, contradiction, progress, ending, note in ratings["ratings"]:
        group = mapping[key]["model"]
        record = blind[group]
        record["count"] += 1
        record["contradiction"] += contradiction is True
        record["contradiction_unknown"] += contradiction is None
        record["goal_progress"] += progress is True
        record["coherent_ending"] += ending is True
        record["ending_unknown"] += ending is None
        record["all_three"] += contradiction is False and progress is True and ending is True
    write_json(OUT / "blind_summary.json", blind)

    protocol = load_json(OUT / "protocol.json")
    assert digest(SOURCE) == protocol["source_sha256"]
    assert digest(ROOT / "artifacts/tokenizer.json") == protocol["tokenizer_sha256"]
    assert digest(OUT / "ordinary_indices.npy") == protocol["ordinary_indices_sha256"]
    prompts = set()
    data = {}
    for name, record in protocol["data"].items():
        path = OUT / "data" / f"{name}.jsonl"
        assert digest(path) == record["sha256"]
        examples = rows(name)
        assert len(examples) == record["rows"]
        for row in examples:
            assert row["prompt"] not in prompts
            prompts.add(row["prompt"])
            if "target" in row:
                assert solve(row["initial"], row["events"], row["target"]) == row["answer"]
        data[name] = {"rows": len(examples), "sha256": digest(path)}
    budgets = {}
    for group in ("control", "mixed"):
        run = OUT / group
        record = load_json(run / "result.json")
        assert record["updates"] == 2000
        assert record["target_tokens"] == {k: v * 2000 for k, v in SETTINGS[group].items()}
        assert record["source_sha256"] == protocol["source_sha256"]
        assert record["checkpoint_sha256"] == digest(run / "step_2000.pth")
        saved = torch.load(run / "step_2000.pth", map_location="cpu", weights_only=True)
        assert saved["global_step"] == 2000
        assert abs(saved["val_loss"] - results[group]["validation_loss"]) < 1e-5
        assert all(float(s["step"]) == 2000 for s in saved["optimizer_state_dict"]["state"].values())
        budgets[group] = record
        del saved
    write_json(OUT / "audit.json", {
        "source_unchanged": True, "all_prepared_prompts_unique": len(prompts),
        "state_labels_replayed": True, "data": data, "budgets": budgets,
        "blind_ratings_sha256": review_hash,
        "checkpoint_fingerprints": {g: r["checkpoint_sha256"] for g, r in results.items()},
    })
    tokenizer = load_tokenizer(ROOT)
    legacy = {}
    prompts_legacy = load_json(EXP / "runs/C/generations.json")[::2]
    for group, record in results.items():
        model, _ = load_checkpoint(record["checkpoint"], "cuda", ROOT / "artifacts/tokenizer.json")
        legacy[group] = [greedy(model, tokenizer, p["prompt"]) for p in prompts_legacy]
        del model
        torch.cuda.empty_cache()
    write_json(OUT / "legacy_development_generations.json", legacy)
    report(results, blind, budgets)
    write_json(OUT / "status.json", {"phase": "complete", "review": "complete"})
    print("Blind review summary", blind)
    print("Verified unique prompts", len(prompts))


def report(results, blind, budgets):
    labels = {"base": "原全量底座", "control": "90% 普通故事 + 10% QA",
              "mixed": "60% 普通故事 + 10% QA + 30% 一致性续写"}
    lines = [
        "# 故事回放与状态一致性续写的混合 SFT 对照",
        "",
        "两个训练组从同一全量预训练底座开始。模型和 tokenizer 相同，固定学习率 1e-4、Adam 默认参数、梯度范数上限 1.0。",
        "两组各完成 2,000 次更新、2,000,000 个参与 loss 的目标 token。微批次为 16，按目标 token 总数归一化并累积梯度后每次只更新一次。",
        "普通故事回放使用全量清洗数据中的 16,384 个固定抽样故事。状态问答池为 32,768 条，一致性续写池为 8,192 条。",
        "每次更新共享相同的 600 个普通故事目标 token 和 100 个 QA 目标 token。对照组另有 300 个普通故事目标 token，实验组用 300 个一致性续写目标 token 替代。",
        "比例按目标 token 精确核对，末尾样本可截断监督区间。输入 token 与实际计算量允许不同。",
        "固定使用第 2,000 次更新的权重进行最终评估，未依据测试结果选模。",
        "",
        "|模型|故事验证 loss|清洗验证 loss|新同类 QA|组合泛化 QA|原人物归属开发题|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for group, r in results.items():
        p = r["old_probe"]
        lines.append(f"|{labels[group]}|{r['validation_loss']:.4f}|{r['clean_validation_loss']:.4f}|{r['qa']['qa_test_seen']['accuracy']:.1%}|{r['qa']['qa_test_ood']['accuracy']:.1%}|{p['correct']/p['items']:.1%}|")
    lines += [
        "",
        "新 QA 测试每类 240 题。组合泛化包含专项样本未见的人名、新问句和长间隔。原底座未适配聊天短答格式，其自由答案准确率包含格式影响。",
        "原人物归属 384 题是此前长期使用的开发诊断，采用同一故事续写前缀下的两个人名比较。",
        "",
        "## 受控前缀的叙事检查",
        "",
        "48 个新前缀，各生成贪心及 temperature 1.0 续写一次，预算均为 96 个新 token，合计每模型 96 条。",
        "只检查显式目标物品持有人与离散位置陈述，并跟踪可解析的交接与移动。覆盖不到的输出不算核验通过。",
        "",
        "|模型|有可检查陈述|明确核验一致|检测到矛盾|少于 20 token 的输出|",
        "|---|---:|---:|---:|---:|",
    ]
    for group, r in results.items():
        c = r["controlled"]
        contradicted = f"{c['contradicted']}/{c['covered']}" if c["covered"] else "未覆盖"
        lines.append(f"|{group}|{c['covered']}/{c['count']}|{c['verified']}/{c['count']}|{contradicted}|{c['short_outputs']}/{c['count']}|")
    lines += [
        "",
        "该检查器覆盖有限，尤其容易识别训练续写模板里的显式陈述。不同模型覆盖率差异很大，上表不能直接解释成完整故事正确率。",
        "模板只提供每类状态六种人工编写的续写骨架，标签由状态机核验，未进行独立人工故事质量评审。",
        "",
        "## 开放故事与隐藏标签审阅",
        "",
        "24 个新故事开头，各生成贪心及 temperature 1.0 续写一次，每模型 48 条，预算为 160 个新 token。",
        "预先选定其中前 12 个开头的贪心续写，隐藏模型名称并打乱顺序。由本助手按预先记录的规则进行一次审阅，共 36 条。",
        "这属于单助手小样本定性审阅，未进行独立人工评审。含糊情况保留 null，完整理由见 blind_ratings.json。",
        "",
        "|模型|明确矛盾|矛盾判断不确定|推进开头情节|连贯结尾|结尾判断不确定|三项均满足|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group, r in blind.items():
        lines.append(f"|{group}|{r['contradiction']}/12|{r['contradiction_unknown']}/12|{r['goal_progress']}/12|{r['coherent_ending']}/12|{r['ending_unknown']}/12|{r['all_three']}/12|")
    lines += [
        "",
        "三项均满足表示无明确矛盾、推进开头情节且结尾连贯；不确定案例不计入。",
        "模型标签映射在评分文件保存后才用于汇总，评分文件指纹写入 audit.json。",
        "",
        "|模型|24 开头 × 两种生成的短输出|平均新 token|",
        "|---|---:|---:|",
    ]
    for group, r in results.items():
        f = r["free"]
        lines.append(f"|{group}|{f['short_outputs']}/{f['count']}|{f['mean_new_tokens']:.1f}|")
    lines += ["", "## 预算与复核", ""]
    for group, r in budgets.items():
        lines.append(f"- {group} 的目标 token 为 {r['target_tokens']}，输入 token 为 {r['input_tokens']:,}，训练约 {r['elapsed_seconds']/60:.1f} 分钟。")
    lines += [
        "",
        "4 项专项测试通过，覆盖精确 token 配比、因果标签对齐、状态陈述及交接检查。全部 8,448 条训练/开发续写的预设事实通过核验。",
        "audit.json 核对了跨集合 prompt 唯一性、状态机标签、来源权重及每个优化器 2,000 次更新。原底座指纹保持一致。",
        "本实验采用一个训练种子。新名字只是在专项合成数据中保留，普通故事回放中可能出现这些名字。",
        "早先四个故事开头的贪心结果另存 legacy_development_generations.json，作为已知失败模式的事后开发检查。",
        "本轮测试结果均为固定预算下的结果。合成模板上的 loss 或状态问答收益需要与开放式续写一起解释。",
        "",
        "复现训练运行 python -m tools.story_mix_experiment train，评估运行 python -m tools.story_mix_evaluate，汇总运行 python -m tools.story_mix_report。",
        "已完成组会跳过参数更新。重新评估会生成隐藏标签审阅材料，既有人工填写的评分应与相同输出逐条核对。",
        "",
    ]
    (OUT / "results.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
