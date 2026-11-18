"""Run the wide-attention and deep models against the archived A control."""

import hashlib
import json
from dataclasses import asdict, replace
from functools import partial

import torch
from torch.utils.data import DataLoader

from codes.checkpoint import save_checkpoint
from codes.config import ModelConfig, TrainConfig
from codes.data import collate_sequences, load_tokenizer, special_token_ids
from codes.generate import generate
from codes.model import build_model
from codes.training import evaluate, train_epoch
from .data_ablation import EXP, ROOT, digest, read_texts, write_json

A = EXP / "runs/A"
OUT = ROOT / "experiments/architecture_20261118"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def make_loader(path, tokenizer, special, context, batch_size, shuffle, keep=None):
    texts = [s for s in read_texts(path) if keep is None or keep(s)]
    sequences = []
    for i in range(0, len(texts), 1000):
        for row in tokenizer.encode_batch(texts[i : i + 1000]):
            ids = [special["<bos>"], *row.ids, special["<eos>"]]
            sequences.append(torch.tensor(ids[:context + 1], dtype=torch.long))
    return DataLoader(sequences, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=partial(collate_sequences, pad_id=special["<pad>"]))


def fingerprint(model):
    h = hashlib.sha256()
    for p in model.parameters():
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    baseline = load_json(A / "result.json")
    cfg = TrainConfig(**baseline["train_config"])
    base = ModelConfig(**baseline["model_config"])
    train_path = EXP / "clean_100k.jsonl"
    val_path = ROOT / "data/validation.jsonl"
    tokenizer_path = ROOT / "artifacts/tokenizer.json"
    for path, key in ((train_path, "train_sha256"), (val_path, "validation_sha256"),
                      (tokenizer_path, "tokenizer_sha256")):
        assert digest(path) == baseline[key], path
    device = torch.device(cfg.device)
    assert torch.__version__ == baseline["torch_version"]
    assert torch.cuda.get_device_name(device) == baseline["gpu"]
    tokenizer = load_tokenizer(ROOT)
    special = special_token_ids(tokenizer)
    pad = special["<pad>"]
    train = make_loader(train_path, tokenizer, special, base.max_seq_len, cfg.batch_size, True)
    val = make_loader(val_path, tokenizer, special, base.max_seq_len, cfg.batch_size, False)
    prompts = load_json(A / "generations.json")

    # A used the global CPU generator for shuffling after model init and validation.
    torch.manual_seed(cfg.seed)
    reference = build_model(base)
    assert fingerprint(reference) == baseline["initial_parameters_sha256"]
    before_validation = torch.get_rng_state()
    del reference

    variants = {
        "3": replace(base, num_heads=8, head_dim=64, d_hidden=512),
        "4": replace(base, num_blocks=6, d_hidden=512),
    }
    for group, config in variants.items():
        run = OUT / group
        if (run / "result.json").exists():
            continue
        torch.manual_seed(cfg.seed)
        model = build_model(config).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
        record = {
            "group": group, "model_config": asdict(config), "train_config": asdict(cfg),
            "parameters": sum(p.numel() for p in model.parameters()),
            "initial_parameters_sha256": fingerprint(model),
            "train_sha256": baseline["train_sha256"], "tokenizer_sha256": baseline["tokenizer_sha256"],
            "validation_sha256": baseline["validation_sha256"], "optimizer": baseline["optimizer"],
            "torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(device),
        }
        torch.set_rng_state(before_validation)
        record["initial_native_val_loss"] = evaluate(model, val, device, pad)
        print(f"Starting group {group}: {record['parameters']} parameters", flush=True)
        record["pre_shuffle_rng_sha256"] = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
        record.update(train_epoch(model, train, device, opt, pad, log_every=cfg.log_every))
        assert (record["tokens"], record["steps"]) == (baseline["tokens"], baseline["steps"])
        record["native_val_loss"] = evaluate(model, val, device, pad)
        save_checkpoint(run / "epoch_1.pth", model, opt, config, cfg, 1, record["steps"],
                        record["native_val_loss"], tokenizer_path)
        samples = []
        for p in prompts:
            continuation = generate(model, tokenizer, p["prompt"], max_new_tokens=160,
                                    temperature=p["temperature"], seed=p["seed"])
            samples.append({**p, "continuation": continuation})
        write_json(run / "generations.json", samples)
        write_json(run / "result.json", record)
        print(f"Finished group {group}: validation {record['native_val_loss']:.6f}", flush=True)


if __name__ == "__main__":
    main()
