"""Score saved checkpoints on paired, counterbalanced ownership continuations."""

from collections import defaultdict
from statistics import mean

import torch

from codes.checkpoint import load_checkpoint
from codes.data import load_tokenizer, special_token_ids
from codes.training import evaluate
from .architecture_ablation import A, OUT, ROOT, load_json, make_loader
from .data_ablation import EXP, clean, digest, write_json

NAMES = [("Mia", "Ben"), ("Lily", "Tom"), ("Anna", "Max"), ("Lucy", "Jack"),
         ("Bob", "Sam"), ("Jane", "Emma"), ("Tim", "Peter"), ("Amy", "Sara")]
OBJECTS = ("key", "book", "ball", "hat")
FILLER = (
    "A little bird sang in the tree. The sun was warm.",
    "A white cloud floated over the garden and a soft wind moved the grass.",
    "The flowers by the gate were bright and pretty in the morning light.",
    "Somewhere down the road, a little bell rang twice and then all was quiet again.",
    "The path through the garden was dry and the air smelled sweet.",
    "A yellow butterfly rested on a leaf, then rose slowly into the air.",
    "At the far end of the garden a tiny brown rabbit came out of a hole.",
    "It sat very still for a while and listened to the birds in the old tree.",
    "Beyond the trees there was a hill with a small white house right at the top.",
    "The red gate at the bottom of the hill stood open in the sunshine.",
    "A little stream ran under the wooden bridge and down to the pond.",
    "Small waves made circles on the water until the surface was smooth again.",
    "The afternoon was quiet and bright.",
)


def nearest_filler(tokenizer, cue, target):
    choices = [" ".join(FILLER[:n]) for n in range(len(FILLER) + 1)]
    def gap(text):
        return len(tokenizer.encode(" " + (text + " " if text else "") + cue).ids)
    text = min(choices, key=lambda x: abs(gap(x) - target))
    return text, gap(text)


def make_cases(tokenizer):
    cases = []
    for target in (20, 80, 160):
        for obj in OBJECTS:
            cue = f"The {obj} was now with"
            filler, gap = nearest_filler(tokenizer, cue, target)
            for first, second in NAMES:
                for verb in ("gave", "got"):
                    for subject, other in ((first, second), (second, first)):
                        if verb == "gave":
                            event = f"{subject} gave the {obj} to {other}."
                            holder, foil = other, subject
                        else:
                            event = f"{subject} got the {obj} from {other}."
                            holder, foil = subject, other
                        prompt = " ".join(s for s in (event, filler, cue) if s)
                        cases.append(dict(pair=f"{first}_{second}_{obj}_{verb}", verb=verb,
                                          distance=target, gap_tokens=gap, prompt=prompt,
                                          holder=holder, foil=foil))
    return cases


@torch.inference_mode()
def score(model, tokenizer, cases):
    rows = []
    device = next(model.parameters()).device
    for case in cases:
        prefix = tokenizer.encode(case["prompt"]).ids
        candidates = []
        for name in (case["holder"], case["foil"]):
            full = tokenizer.encode(case["prompt"] + " " + name).ids
            assert full[:-1] == prefix
            candidates.append(full[-1])
        ids = [tokenizer.token_to_id("<bos>"), *prefix]
        assert len(ids) <= model.max_seq_len
        logits = model(torch.tensor([ids], device=device))[0, -1]
        margin = (logits[candidates[0]] - logits[candidates[1]]).item()
        rows.append({**case, "margin": margin})
    return rows


def summarize(rows):
    pairs = defaultdict(list)
    for row in rows:
        pairs[row["distance"], row["pair"]].append(row["margin"])
    assert all(len(p) == 2 for p in pairs.values())
    return dict(items=len(rows), correct=sum(r["margin"] > 0 for r in rows),
                pairs=len(pairs), both_correct=sum(min(p) > 0 for p in pairs.values()),
                mean_margin=mean(r["margin"] for r in rows),
                gap_range=[min(r["gap_tokens"] for r in rows), max(r["gap_tokens"] for r in rows)])


def main():
    tokenizer = load_tokenizer(ROOT)
    cases = make_cases(tokenizer)
    summary = {}
    special = special_token_ids(tokenizer)
    val_args = (ROOT / "data/validation.jsonl", tokenizer, special, 256, 8, False)
    val = make_loader(*val_args)
    clean_val = make_loader(*val_args, keep=lambda text: clean(text)[1] is None)
    validation = {}
    for group, run in (("A", A), ("3", OUT / "3"), ("4", OUT / "4")):
        model, ckpt = load_checkpoint(run / "epoch_1.pth", "cuda", ROOT / "artifacts/tokenizer.json")
        record = load_json(run / "result.json")
        assert ckpt["epoch"] == 1 and ckpt["global_step"] == record["steps"] == 12500
        assert ckpt["val_loss"] == record["native_val_loss"]
        validation[group] = {
            "full": evaluate(model, val, torch.device("cuda"), special["<pad>"]),
            "clean": evaluate(model, clean_val, torch.device("cuda"), special["<pad>"]),
            "stories_full": len(val.dataset), "stories_clean": len(clean_val.dataset),
        }
        assert abs(validation[group]["full"] - ckpt["val_loss"]) < 1e-5
        rows = score(model, tokenizer, cases)
        write_json(OUT / f"probe_{group}.json", rows)
        summary[group] = {str(d): summarize([r for r in rows if r["distance"] == d])
                          for d in (20, 80, 160)}
        summary[group]["all"] = summarize(rows)
        print(group, summary[group], flush=True)
    write_json(OUT / "probe_summary.json", summary)
    write_json(OUT / "validation.json", validation)
    records = {g: load_json(r / "result.json")
               for g, r in (("A", A), ("3", OUT / "3"), ("4", OUT / "4"))}
    assert records["3"]["pre_shuffle_rng_sha256"] == records["4"]["pre_shuffle_rng_sha256"]
    for path, key in ((EXP / "clean_100k.jsonl", "train_sha256"),
                      (ROOT / "data/validation.jsonl", "validation_sha256"),
                      (ROOT / "artifacts/tokenizer.json", "tokenizer_sha256")):
        assert all(digest(path) == r[key] for r in records.values())
    write_json(OUT / "comparison.json", {
        g: dict(training=records[g], validation=validation[g], probe=summary[g])
        for g in records
    })
    lines = [
        "# StorySprout 注意力与 FFN 参数分配实验",
        "",
        "A 组使用既有 checkpoint。第 3、4 组从头训练，随机种子为 2026。",
        "三组使用同一份 clean_100k.jsonl、原验证集和 tokenizer，均为 256 上下文、batch size 8、Adam 默认参数、学习率 3e-4、1 epoch。",
        "每组训练 19,162,135 个有效目标 token，12,500 次更新。",
        "新结构初始化后，将 CPU 随机状态恢复到 A 组初始化结束的状态，再进行相同的初始验证与训练，保持故事打乱顺序一致。",
        "结构变化使初始化参数的形状和取值发生变化。三组比较采用等数据和更新次数的预算，结构改变也会改变实际计算量。",
        "",
        "|组别|层数|头数 × 头维度|FFN 中间维度|总参数|训练 loss|验证 loss|清洗验证子集 loss|训练秒数|",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for g, r in records.items():
        c = r["model_config"]
        heads = c["num_heads"]
        dim = c.get("head_dim") or c["d_model"] // heads
        parameters = r.get("parameters", 7347200)
        lines.append(f"|{g}|{c['num_blocks']}|{heads} × {dim}|{c['d_hidden']}|{parameters:,}|{r['loss']:.6f}|{validation[g]['full']:.6f}|{validation[g]['clean']:.6f}|{r['elapsed_seconds']:.1f}|")
    lines += [
        "",
        "## 人物归属测试",
        "",
        "共 384 条，使用 8 对人名、4 件物品、gave/got 两种交接表述、双方姓名互换及三个距离。",
        "比较固定前缀 The [物品] was now with 后，正确人名与另一人名的下一个 token 分数。所有候选均为单 token。",
        "两个姓名均正确的配对比例要求互换人物后仍判断正确，可以抑制固定人名偏好。两种交接表述同时覆盖主语持有和另一人持有。",
        "填充句只增加无关环境描写。测试为人工构造的单次交接模板，结果只描述该任务。",
        "",
        "|组别|交接后的间隔 token|单条正确率|姓名互换后均正确|",
        "|---|---:|---:|---:|",
    ]
    for g in records:
        for distance in ("20", "80", "160"):
            s = summary[g][distance]
            lines.append(f"|{g}|{s['gap_range']}|{s['correct']}/{s['items']} ({s['correct']/s['items']:.1%})|{s['both_correct']}/{s['pairs']} ({s['both_correct']/s['pairs']:.1%})|")
    lines += [
        "",
        "单一随机种子和 1 epoch 主要反映当前训练预算下的效果，最终收敛质量需要进一步测量。",
        "第 3 组验证 loss 较 A 上升约 0.31%，第 4 组下降约 0.78%。清洗验证子集呈相同方向。",
        "三组在人物归属二选一测试上的总体正确率均接近 50%。第 4 组在最远距离有更多姓名互换配对全部正确，但整体结果尚不足以支持稳定的逻辑提升。",
        "旧 A 组计时来自历史运行，本次第 3 组训练初期并行执行了 CPU 测试且系统负载有变化，训练秒数仅供参考。",
        "原始续写按 A 组相同 4 个提示、temperature 0.5/0.8、seed 2026、最多生成 160 token，保存在各组 generations.json；汇总见 generations.md。",
        "",
        "复现训练可运行 python -m tools.architecture_ablation，评估可运行 python -m tools.evaluate_architecture。",
        "已完成组会读取 result.json 并跳过训练，评估会重新读取保存的权重。命令均从 storysprout 根目录执行。",
        "",
    ]
    (OUT / "results.md").write_text("\n".join(lines), encoding="utf-8")
    examples = ["# 三组相同提示续写", ""]
    samples = {g: load_json(r / "generations.json")
               for g, r in (("A", A), ("3", OUT / "3"), ("4", OUT / "4"))}
    for i, sample in enumerate(samples["A"]):
        examples += [f"## 提示 {i // 2 + 1}，temperature {sample['temperature']}",
                     "", sample["prompt"], ""]
        for g in samples:
            s = samples[g][i]
            assert (s["prompt"], s["temperature"], s["seed"]) == (
                sample["prompt"], sample["temperature"], sample["seed"])
            examples += [f"### 组 {g}", "", s["continuation"], ""]
    (OUT / "generations.md").write_text("\n".join(examples), encoding="utf-8")


if __name__ == "__main__":
    main()
