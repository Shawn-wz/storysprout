"""Prepare and run the five requested data/context experiments without changing training."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import re
import time
import unicodedata
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'experiments' / 'data_context_20261117'
SEED = 2026


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def key(text):
    return hashlib.sha256(' '.join(unicodedata.normalize('NFC', text).split()).encode()).digest()


def clean(text):
    text = unicodedata.normalize('NFC', text).strip()
    if not text:
        return None, 'empty'
    if any(s in text for s in ['\ufffd', '\u00e2\u20ac', '\u00c3', '\u00c2']):
        return None, 'suspected_mojibake'
    if any(ord(c) < 32 and c not in '\n\r\t' for c in text):
        return None, 'control_character'
    return text, None


def read_texts(path):
    with path.open(encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)['text']


def download(url, destination):
    if destination.exists():
        return
    for attempt in range(3):
        try:
            temp = destination.with_suffix('.partial')
            with urllib.request.urlopen(url, timeout=90) as src, temp.open('wb') as out:
                while chunk := src.read(4 * 1024 * 1024):
                    out.write(chunk)
            temp.replace(destination)
            print('Downloaded', destination.name, destination.stat().st_size, flush=True)
            return
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)


def prepare():
    import pyarrow.parquet as pq
    EXP.mkdir(parents=True, exist_ok=True)
    manifest_path = EXP / 'data_manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        for name, record in manifest['files'].items():
            assert digest(EXP / name) == record['sha256'], name
        print('Verified existing prepared data.', flush=True)
        return
    source = EXP / 'source'
    source.mkdir(exist_ok=True)
    metadata_path = source / 'hub_metadata.json'
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    else:
        with urllib.request.urlopen('https://huggingface.co/api/datasets/roneneldan/TinyStories', timeout=60) as response:
            metadata = json.load(response)
        write_json(metadata_path, metadata)
    revision = metadata['sha']
    names = sorted(r['rfilename'] for r in metadata['siblings'] if re.match(r'data/train-.*\.parquet$', r['rfilename']))
    assert len(names) == 4
    def fetch(name):
        download(f'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/{revision}/{name}?download=true', source / Path(name).name)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, names))
    validation = list(read_texts(ROOT / 'data/validation.jsonl'))
    banned = {key(t) for t in validation}
    seen = set(banned)
    counts = Counter()
    original_clean = []
    for text in read_texts(ROOT / 'data/train.jsonl'):
        text, reason = clean(text)
        if reason:
            counts['original_' + reason] += 1
        elif key(text) in seen:
            counts['original_duplicate_or_validation_overlap'] += 1
        else:
            seen.add(key(text))
            original_clean.append(text)
    counts['original_retained'] = len(original_clean)
    capacity = 500_000 - len(original_clean)
    rng = random.Random(SEED)
    reservoir = []
    accepted = 0
    for name in names:
        parquet = pq.ParquetFile(source / Path(name).name)
        for batch in parquet.iter_batches(batch_size=4096, columns=['text']):
            for text in batch.column(0).to_pylist():
                counts['source_scanned'] += 1
                text, reason = clean(text)
                if reason:
                    counts['source_' + reason] += 1
                    continue
                hashed = key(text)
                if hashed in seen:
                    counts['source_duplicate_or_validation_overlap'] += 1
                    continue
                seen.add(hashed)
                accepted += 1
                if len(reservoir) < capacity:
                    reservoir.append(text)
                else:
                    index = rng.randrange(accepted)
                    if index < capacity:
                        reservoir[index] = text
        print('Scanned', name, dict(counts), flush=True)
    assert len(reservoir) == capacity
    rng.shuffle(reservoir)
    cleaned = original_clean + reservoir
    files = {}
    for name, texts in [('clean_100k.jsonl', cleaned[:100_000]), ('clean_500k.jsonl', cleaned)]:
        path = EXP / name
        with path.open('w', encoding='utf-8') as f:
            for text in texts:
                f.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
        files[name] = {'rows': len(texts), 'sha256': digest(path)}
    write_json(manifest_path, {
        'seed': SEED, 'dataset': 'roneneldan/TinyStories', 'revision': revision,
        'policy': 'NFC + strip; filter empty, suspicious mojibake and control characters; deduplicate normalized whitespace; exclude original validation; retain clean local stories, top up using reservoir sampling across all four official training shards; 100k is prefix subset of 500k. No semantic or near-duplicate filtering.',
        'counts': dict(counts), 'files': files,
        'original_train_sha256': digest(ROOT / 'data/train.jsonl'),
        'validation_sha256': digest(ROOT / 'data/validation.jsonl'),
        'tokenizer_sha256': digest(ROOT / 'artifacts/tokenizer.json'),
        'source_files': {Path(n).name: digest(source / Path(n).name) for n in names},
    })


def train(selected=None):
    import torch
    from torch.utils.data import DataLoader
    from .checkpoint import load_checkpoint, save_checkpoint
    from .config import ModelConfig, TrainConfig
    from .data import collate_sequences, load_tokenizer, special_token_ids
    from .model import build_model
    from .training import evaluate, train_epoch
    from .generate import generate

    manifest_path = EXP / 'data_manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    elif selected == ['control']:
        manifest = {'original_train_sha256': digest(ROOT/'data/train.jsonl'),
            'validation_sha256': digest(ROOT/'data/validation.jsonl'),
            'tokenizer_sha256': digest(ROOT/'artifacts/tokenizer.json')}
    else:
        raise RuntimeError('Prepare experimental data first.')
    assert digest(ROOT / 'data/train.jsonl') == manifest['original_train_sha256']
    assert digest(ROOT / 'data/validation.jsonl') == manifest['validation_sha256']
    assert digest(ROOT / 'artifacts/tokenizer.json') == manifest['tokenizer_sha256']
    tokenizer = load_tokenizer(ROOT)
    special = special_token_ids(tokenizer)
    cfg = TrainConfig(seed=SEED, epochs=1, batch_size=8, learning_rate=3e-4, log_every=1000, device='cuda')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for the requested experiment.')
    device = torch.device('cuda')
    # Reuse the original model, Adam, loss, collator, and training loop exactly.
    def sequences(path, context):
        result = []
        texts = list(read_texts(path))
        for start in range(0, len(texts), 1000):
            for encoded in tokenizer.encode_batch(texts[start:start+1000]):
                ids = ([special['<bos>']] + encoded.ids + [special['<eos>']])[:context+1]
                result.append(torch.tensor(ids, dtype=torch.long))
        return result
    def loader(seqs, shuffle=False):
        return DataLoader(seqs, batch_size=8, shuffle=shuffle, num_workers=0, drop_last=False,
            collate_fn=partial(collate_sequences, pad_id=special['<pad>']))
    val256 = loader(sequences(ROOT / 'data/validation.jsonl', 256))
    val512 = loader(sequences(ROOT / 'data/validation.jsonl', 512))
    # Identical initial trainable parameters, despite context-dependent RoPE buffers.
    expected_initial_hash = None
    summary = []
    groups = [('control', ROOT/'data/train.jsonl', 256), ('A', EXP/'clean_100k.jsonl',256),
        ('B', EXP/'clean_100k.jsonl',512), ('C', EXP/'clean_500k.jsonl',256), ('D', EXP/'clean_500k.jsonl',512)]
    prompts = [
        'Once upon a time, a girl named Alice found a small dog by the road. The dog looked lost.',
        'Tom and his cat Milo played with a red ball in the yard. The ball rolled under a bush.',
        'A little rabbit wanted to cross a stream, but the water was too deep.',
        'Lily planted a seed in a pot. Every morning, she gave it a little water.',
    ]
    for name, path, context in groups:
        if selected and name not in selected:
            continue
        run = EXP / 'runs' / name
        if (run / 'result.json').exists():
            summary.append(json.loads((run / 'result.json').read_text(encoding='utf-8')))
            continue
        run.mkdir(parents=True, exist_ok=True)
        write_json(EXP/'status.json', {'phase': 'training', 'group': name, 'completed': [r['group'] for r in summary]})
        torch.manual_seed(SEED)
        model_config = ModelConfig(vocab_size=tokenizer.get_vocab_size(), max_seq_len=context)
        model = build_model(model_config).to(device)
        h = hashlib.sha256()
        for p in model.parameters():
            h.update(p.detach().cpu().numpy().tobytes())
        initial_hash = h.hexdigest()
        if expected_initial_hash is None:
            expected_initial_hash = initial_hash
        assert expected_initial_hash == initial_hash
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
        if name != 'control':
            assert digest(path) == manifest['files'][path.name]['sha256']
        config_record = {'group': name, 'train_config': asdict(cfg), 'model_config': asdict(model_config),
            'optimizer': 'torch.optim.Adam defaults', 'initial_parameters_sha256': initial_hash,
            'train_data': str(path), 'train_sha256': digest(path),
            'validation_sha256': manifest['validation_sha256'], 'tokenizer_sha256': manifest['tokenizer_sha256'],
            'source_checkpoint': None, 'torch_version': torch.__version__, 'gpu': torch.cuda.get_device_name(0)}
        write_json(run/'config.json', config_record)
        train_sequences = sequences(path, context)
        train_loader = loader(train_sequences, True)
        native_val = val256 if context == 256 else val512
        initial_val = evaluate(model, native_val, device, special['<pad>'])
        print(f'GROUP {name} stories={len(train_sequences)} context={context} initial_val={initial_val:.6f}', flush=True)
        # Reproduce the original seed ordering: model initialization, data preparation,
        # initial validation, then the shuffled training iterator.
        metrics = train_epoch(model, train_loader, device, optimizer, special['<pad>'], log_every=1000)
        val_loss = evaluate(model, native_val, device, special['<pad>'])
        save_checkpoint(run/'epoch_1.pth', model, optimizer, model_config, cfg, 1, metrics['steps'], val_loss,
            ROOT/'artifacts/tokenizer.json', source_checkpoint=None)
        common_val = val_loss if context == 256 else evaluate(model, val256, device, special['<pad>'])
        record = {**config_record, 'epoch': 1, 'initial_native_val_loss': initial_val,
            'native_val_loss': val_loss, 'common_256_val_loss': common_val,
            'stories': len(train_sequences), **metrics}
        with (run/'metrics.jsonl').open('w',encoding='utf-8') as f:
            f.write(json.dumps(record,ensure_ascii=False)+'\n')
        # Give every group exactly the same short-prefix generation budget.
        generations = [{'prompt': p, 'temperature': t, 'seed': SEED,
            'continuation': generate(model,tokenizer,p,max_new_tokens=160,temperature=t,seed=SEED)}
            for p in prompts for t in (0.5,0.8)]
        write_json(run/'generations.json', generations)
        write_json(run/'result.json', record)
        summary.append(record)
        write_json(EXP/'summary.json', summary)
        print(f'FINISHED {name}: {json.dumps(record,ensure_ascii=False)}',flush=True)
        del train_loader, train_sequences, model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
    summary = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((EXP/'runs').glob('*/result.json'))]
    assert len({r['initial_parameters_sha256'] for r in summary}) == 1
    write_json(EXP/'summary.json', summary)
    complete = len(summary) == 5
    write_json(EXP/'status.json', {'phase':'complete' if complete else 'partial','completed':[r['group'] for r in summary]})
    print('ALL FIVE GROUPS COMPLETE' if complete else 'SELECTED GROUPS COMPLETE',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare','train'])
    parser.add_argument('--groups', nargs='+', choices=['control','A','B','C','D'])
    args = parser.parse_args()
    prepare() if args.action == 'prepare' else train(args.groups)
