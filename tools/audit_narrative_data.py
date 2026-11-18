"""Read-only corpus scan; lexical proxies are NOT narrative-quality labels."""
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/narrative_data_audit_20261118'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / 'data/tinystories_full/tokenized_manifest.json').read_text('utf-8'))
    n = manifest['splits']['train']['stories']
    seed = 20261118
    chosen = set(random.Random(seed).sample(range(n), 48))
    patterns = {
        'transfer_verb': r'\b(?:gave|give|gives|given|handed|passed|lent|borrowed|received)\b',
        'gave_to': r'\bgave\b[^.!?\n]{0,90}\bto\b',
        'got_from': r'\b(?:got|received)\b[^.!?\n]{0,90}\bfrom\b',
        'explicit_now_with': r'\b(?:was|is) now with\b',
        'gratitude': r'\b(?:thank|thanks|thanked|thankful)\b',
        'named_self_thank': r'\b([A-Z][a-z]+) thanked \1\b',
        'named_self_help': r'\b([A-Z][a-z]+) helped \1\b',
    }
    compiled = {k: re.compile(v, 0 if k.startswith('named_') else re.I) for k, v in patterns.items()}
    counts = Counter()
    samples = []
    targeted = {k: [] for k in patterns if k.startswith('named_')}
    rngs = {k: random.Random(seed + i) for i, k in enumerate(targeted)}
    digest = hashlib.sha256()
    with (ROOT / 'data/tinystories_full/train.jsonl').open('rb') as f:
        for i, line in enumerate(f):
            digest.update(line)
            text = json.loads(line)['text']
            for key, pattern in compiled.items():
                if pattern.search(text):
                    counts[key] += 1
                    if key in targeted:
                        row = {'row_index_0based': i, 'text': text}
                        if len(targeted[key]) < 8:
                            targeted[key].append(row)
                        else:
                            j = rngs[key].randrange(counts[key])
                            if j < 8:
                                targeted[key][j] = row
            if i in chosen:
                samples.append({'row_index_0based': i, 'text': text})
            if (i+1) % 400000 == 0:
                print(f'Scanned {i+1:,}', flush=True)
    assert i + 1 == n
    assert digest.hexdigest() == manifest['splits']['train']['files']['train.jsonl']
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(ROOT / 'artifacts/tokenizer.json'))
    for row in samples:
        ids = tokenizer.encode(row['text']).ids
        row['text_tokens'] = len(ids)
        row['training_text_prefix'] = tokenizer.decode(ids[:256])
        row['truncated'] = len(ids) > 255
    summary = {
        'source': 'data/tinystories_full/train.jsonl', 'sha256': digest.hexdigest(),
        'stories': n, 'seed': seed, 'uniform_sample_size': len(samples),
        'patterns': patterns, 'story_counts': dict(counts),
        'caveat': 'Lexical matches measure coverage only, not correctness; named-self matches require full-text review.',
        'truncated_stories_from_manifest': manifest['splits']['train']['truncated_stories'],
    }
    for name, value in [('scan.json', summary), ('uniform_sample.json', samples), ('targeted_sample.json', targeted)]:
        (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), 'utf-8')
    (OUT / 'uniform_sample.md').write_text('\n\n'.join(
        f"## Row {s['row_index_0based']} ({s['text_tokens']} text tokens)\n\n{s['text']}" for s in samples), 'utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
