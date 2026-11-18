"""Verify the complete official shards and prepare cleaned, cached training data."""

import json
import shutil
import sys
from collections import Counter
from dataclasses import asdict
from itertools import islice
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from codes.config import ModelConfig, TrainConfig
from codes.data import load_tokenizer, special_token_ids
from .data_ablation import EXP, ROOT, clean, digest, download, key, read_texts, write_json

DESTINATION = ROOT / "data/tinystories_full"


def build_cache(directory: Path, split: str, tokenizer, context: int) -> dict:
    """Write compact token IDs and sequence offsets using bounded text batches."""
    special = special_token_ids(tokenizer)
    offsets = [0]
    stats = Counter()
    temporary = directory / f"{split}.tokens.bin.tmp"
    texts = iter(read_texts(directory / f"{split}.jsonl"))
    with temporary.open("wb") as output:
        while batch := list(islice(texts, 1000)):
            flat = []
            for encoded in tokenizer.encode_batch(batch):
                ids = [special["<bos>"], *encoded.ids, special["<eos>"]]
                stats["truncated_stories"] += len(ids) > context + 1
                ids = ids[:context + 1]
                if not 2 <= len(ids) <= context + 1:
                    raise ValueError("Invalid tokenized sequence.")
                stats["valid_targets"] += len(ids) - 1
                stats["eos_retained"] += ids[-1] == special["<eos>"]
                flat.extend(ids)
                offsets.append(offsets[-1] + len(ids))
            np.asarray(flat, dtype="<u2").tofile(output)
            if (len(offsets) - 1) % 100000 == 0:
                print(f"{split}: tokenized {len(offsets) - 1:,} stories", flush=True)
    temporary.replace(directory / f"{split}.tokens.bin")
    temporary = directory / f"{split}.offsets.npy.tmp"
    with temporary.open("wb") as output:
        np.save(output, np.asarray(offsets, dtype="<u8"), allow_pickle=False)
    temporary.replace(directory / f"{split}.offsets.npy")
    files = [f"{split}.jsonl", f"{split}.tokens.bin", f"{split}.offsets.npy"]
    return {
        "stories": len(offsets) - 1, "stored_tokens": offsets[-1], **stats,
        "files": {name: digest(directory / name) for name in files},
    }


def main() -> None:
    """Prepare all files and startup arguments without creating a training run."""
    previous = json.loads((EXP / "data_manifest.json").read_text(encoding="utf-8"))
    metadata = json.loads((EXP / "source/hub_metadata.json").read_text(encoding="utf-8"))
    source = EXP / "source"
    source_records = {}
    for name, expected in previous["source_files"].items():
        path = source / name
        download(f"https://huggingface.co/datasets/roneneldan/TinyStories/resolve/"
                 f"{metadata['sha']}/data/{name}?download=true", path)
        if digest(path) != expected:
            raise ValueError(f"Official source fingerprint mismatch: {path}")
        source_records[name] = {
            "path": str(path), "sha256": expected,
            "bytes": path.stat().st_size, "rows": pq.ParquetFile(path).metadata.num_rows,
        }
    assert len(source_records) == 4
    assert sum(r["rows"] for r in source_records.values()) == 2119719
    print("Verified all four official training shards (2,119,719 rows).", flush=True)
    tokenizer_path = ROOT / "artifacts/tokenizer.json"
    validation_path = ROOT / "data/validation.jsonl"
    protected = {path: digest(path) for path in (
        tokenizer_path, validation_path, ROOT / "data/train.jsonl",
        EXP / "clean_100k.jsonl", EXP / "clean_500k.jsonl",
    )}
    tokenizer = load_tokenizer(ROOT)
    assert tokenizer.get_vocab_size() <= 65536
    DESTINATION.mkdir(parents=True, exist_ok=True)
    manifest_path = DESTINATION / "tokenized_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["sources"] == source_records
        assert manifest["tokenizer_sha256"] == protected[tokenizer_path]
        assert manifest["splits"]["validation"]["files"]["validation.jsonl"] == protected[validation_path]
        for split in manifest["splits"].values():
            for name, expected in split["files"].items():
                assert digest(DESTINATION / name) == expected, name
        print("Verified existing full-data preparation.", flush=True)
    else:
        banned = {key(t) for t in read_texts(validation_path)}
        seen = set()
        counts = Counter()
        temporary = DESTINATION / "train.jsonl.tmp"
        with temporary.open("w", encoding="utf-8", newline="\n") as output:
            for name in source_records:
                parquet = pq.ParquetFile(source / name)
                for batch in parquet.iter_batches(batch_size=4096, columns=["text"]):
                    for text in batch.column(0).to_pylist():
                        counts["source_rows"] += 1
                        text, reason = clean(text)
                        if reason:
                            counts[reason] += 1
                            continue
                        hashed = key(text)
                        if hashed in banned:
                            counts["validation_overlap"] += 1
                            continue
                        if hashed in seen:
                            counts["duplicate"] += 1
                            continue
                        seen.add(hashed)
                        output.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                        counts["retained"] += 1
                print(f"Scanned {name}: {dict(counts)}", flush=True)
        temporary.replace(DESTINATION / "train.jsonl")
        del seen
        shutil.copyfile(validation_path, DESTINATION / "validation.jsonl")
        splits = {split: build_cache(DESTINATION, split, tokenizer, 256)
                  for split in ("train", "validation")}
        assert splits["train"]["stories"] == counts["retained"]
        manifest = {
            "format_version": 1, "dataset": "roneneldan/TinyStories",
            "revision": metadata["sha"], "sources": source_records,
            "source_rows": counts["source_rows"], "cleaning": dict(counts),
            "policy": "All official train shards; NFC+strip; filter empty, suspected mojibake "
                      "and control characters; deduplicate normalized whitespace; exclude "
                      "normalized matches to the existing 2000-story validation set.",
            "max_seq_len": 256, "vocab_size": tokenizer.get_vocab_size(),
            "tokenizer_sha256": protected[tokenizer_path],
            "token_dtype": "<u2", "splits": splits,
        }
        write_json(manifest_path, manifest)
    cfg = TrainConfig(epochs=1, batch_size=8, learning_rate=3e-4, log_every=1000)
    model = ModelConfig(vocab_size=tokenizer.get_vocab_size())
    training = {
        "python": str(Path(sys.executable).resolve()),
        "model_config": asdict(model), "train_config": asdict(cfg),
        "optimizer": "torch.optim.Adam defaults", "start": "from scratch",
        "data_manifest_sha256": digest(manifest_path),
        "steps_per_epoch": (manifest["splits"]["train"]["stories"] + 7) // 8,
        "valid_targets_per_epoch": manifest["splits"]["train"]["valid_targets"],
        "arguments": ["--project-dir", str(ROOT), "--data-dir", str(DESTINATION),
                      "--epochs", "1", "--batch-size", "8", "--learning-rate", "0.0003",
                      "--log-every", "1000", "--device", "cuda"],
    }
    write_json(DESTINATION / "training_config.json", training)
    retained = manifest["splits"]["train"]
    lines = [
        "# 全量 TinyStories 训练准备",
        "",
        f"官方默认训练集共 {manifest['source_rows']:,} 篇，4 个完整原始 Parquet 分片已下载并通过 SHA-256 校验。",
        f"固定版本为 {manifest['revision']}，原始文件位置及指纹见 tokenized_manifest.json 的 sources 字段。",
        f"沿用此前清洗规则，得到 {retained['stories']:,} 篇训练故事。完整排除统计见 manifest 的 cleaning 字段。",
        "规则包括 NFC 标准化、过滤空白与疑似乱码、按规范化空白去重，以及排除与现有验证集重复的文本。",
        "验证集使用已有 2,000 篇故事。tokenizer 使用项目现有的 8,192 词表版本。",
        "",
        "训练从随机初始化开始，4 层、4 个 64 维注意力头、FFN 1024、735 万参数。",
        "启动设置为 1 epoch、batch size 8、Adam 默认参数、固定学习率 3e-4、seed 2026、CUDA。",
        f"每轮 {training['steps_per_epoch']:,} 次更新，{retained['valid_targets']:,} 个有效目标 token。",
        f"上下文为 256，缓存保存每篇前 257 个 token 以构造下一词目标，{retained['truncated_stories']:,} 篇在此长度被截断。",
        "JSONL 中保留清洗后完整故事文本。分词缓存以 uint16 token 和 uint64 偏移存储，并在加载时逐样本转成 long tensor。",
        "",
        "## 手动启动",
        "",
        "在 PowerShell 中执行以下命令。",
        "",
        "```powershell",
        f"& '{ROOT / 'start_full_pretrain.ps1'}'",
        "```",
        "",
        "仅检查数据与前向计算时使用下面的命令。",
        "",
        "```powershell",
        f"& '{ROOT / 'start_full_pretrain.ps1'}' -CheckOnly",
        "```",
        "",
        "启动配置保存在 training_config.json，完整数据和缓存指纹保存在 tokenized_manifest.json。",
        "正式训练会在项目 models 下创建新的运行目录，记录逐轮指标并保存每轮权重及 best.pth。",
        "修改轮数时同时修改 training_config.json 的 train_config.epochs 和 arguments 中 --epochs 后面的值。",
        "缓存对应 256 上下文及当前 tokenizer。更改这两项需要重新生成缓存。",
        "",
        "官方数据源为 https://huggingface.co/datasets/roneneldan/TinyStories。",
        "",
    ]
    (DESTINATION / "README.md").write_text("\n".join(lines), encoding="utf-8")
    for path, expected in protected.items():
        assert digest(path) == expected, path
    print("Full dataset and startup configuration are ready.", flush=True)


if __name__ == "__main__":
    main()
