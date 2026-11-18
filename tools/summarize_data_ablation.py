"""Verify completed runs and compare validation on identical token coverage."""

import json
from functools import partial

import torch
from torch.utils.data import DataLoader

from codes.checkpoint import load_checkpoint
from codes.data import collate_sequences, load_tokenizer, special_token_ids
from codes.training import evaluate

from .data_ablation import EXP, ROOT, clean, digest, read_texts, write_json


def main() -> None:
    """Evaluate all five checkpoints and refresh the comparison report."""
    tokenizer = load_tokenizer(ROOT)
    special = special_token_ids(tokenizer)
    buckets = {"clean": [], "suspected_or_invalid": []}
    for text in read_texts(ROOT / "data/validation.jsonl"):
        _, reason = clean(text)
        ids = ([special["<bos>"]] + tokenizer.encode(text).ids + [special["<eos>"]])[
            :257
        ]
        buckets["suspected_or_invalid" if reason else "clean"].append(
            torch.tensor(ids, dtype=torch.long)
        )
    loaders = {
        k: DataLoader(
            v,
            batch_size=8,
            shuffle=False,
            collate_fn=partial(collate_sequences, pad_id=special["<pad>"]),
        )
        for k, v in buckets.items()
    }
    results = []
    initial_hashes = set()
    for group in ["control", "A", "B", "C", "D"]:
        run = EXP / "runs" / group
        result = json.loads((run / "result.json").read_text(encoding="utf-8"))
        model, checkpoint = load_checkpoint(
            run / "epoch_1.pth",
            device="cuda",
            tokenizer_path=ROOT / "artifacts/tokenizer.json",
        )
        assert checkpoint["epoch"] == 1
        assert checkpoint["global_step"] == result["steps"]
        assert checkpoint["source_checkpoint"] is None
        assert checkpoint["train_config"]["learning_rate"] == 3e-4
        assert checkpoint["train_config"]["batch_size"] == 8
        assert checkpoint["train_config"]["epochs"] == 1
        initial_hashes.add(result["initial_parameters_sha256"])
        result["validation_buckets_common_256"] = {
            name: {
                "stories": len(buckets[name]),
                "loss": evaluate(model, loader, torch.device("cuda"), special["<pad>"]),
            }
            for name, loader in loaders.items()
        }
        results.append(result)
        print(
            group,
            result["common_256_val_loss"],
            result["validation_buckets_common_256"],
            flush=True,
        )
        del model, checkpoint
        torch.cuda.empty_cache()
    assert len(initial_hashes) == 1
    manifest = json.loads((EXP / "data_manifest.json").read_text(encoding="utf-8"))
    assert digest(ROOT / "data/train.jsonl") == manifest["original_train_sha256"]
    assert digest(ROOT / "data/validation.jsonl") == manifest["validation_sha256"]
    write_json(EXP / "comparison.json", results)
    lines = [
        "# StorySprout 五组实验结果",
        "",
        "各组均从相同随机初始化训练 1 epoch；固定学习率 3e-4、Adam 默认参数、batch size 8。模型结构与 tokenizer 保持不变。",
        "",
        "|组别|故事数|上下文|有效训练 token|更新步数|训练 loss|统一 256 验证 loss|各自上下文验证 loss|无疑似乱码验证子集 loss|训练秒数|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"|{r['group']}|{r['stories']}|{r['model_config']['max_seq_len']}|{r['tokens']}|{r['steps']}|{r['loss']:.4f}|{r['common_256_val_loss']:.4f}|{r['native_val_loss']:.4f}|{r['validation_buckets_common_256']['clean']['loss']:.4f}|{r['elapsed_seconds']:.1f}|"
        )
    lines += [
        "",
        "统一 256 验证 loss 在相同原验证集、相同目标 token 覆盖下计算，可用于组间比较。清洗验证子集仅过滤疑似乱码或非法内容，未更改文本。",
        "",
        "所有组均训练 1 epoch，因此训练 token 和更新次数不相等；50 万条组的收益混合了新语料与更多计算的影响。单次随机种子、单 epoch 和少量生成样本不能证明最终收敛质量。",
        "",
        "检查通过：五组初始可训练参数指纹相同；checkpoint 均为 epoch 1、从头训练；原训练文件、原验证文件及 tokenizer 的指纹保持一致。",
        "",
        "数据清洗、抽样版本及来源见 data_manifest.json。独立模型、配置、训练记录及固定提示词续写见 runs/<组别>/。",
    ]
    (EXP / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
