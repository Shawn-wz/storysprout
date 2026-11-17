# StorySprout

Run from the project directory with Python 3.11 and the existing `storysprout_env` environment:

```powershell
python -m codes.pretrain
```

The direct entry point `python codes/pretrain.py` also works. Training defaults remain 3 epochs, batch size 8, learning rate 0.001, and CUDA. Use `--help` for optional overrides.

Module responsibilities:

- `codes/config.py`: model and training settings and validation.
- `codes/model.py`: the existing Transformer and `build_model(config)`.
- `codes/data.py`: local tokenizer, story encoding, padding and loaders.
- `codes/training.py`: `train_epoch(...)` and `evaluate(...)`.
- `codes/checkpoint.py`: `save_checkpoint(...)` and `load_checkpoint(...)`.
- `codes/pretrain.py`: training orchestration.

Each training run creates a new directory under `models/`, containing per-epoch checkpoints, `best.pth`, and `metrics.jsonl`. Existing checkpoints are preserved. The loader accepts both the original flat configuration and the new nested configuration. New checkpoints record the tokenizer SHA-256; pass `tokenizer_path` to the loader to verify it. Legacy checkpoints do not contain this fingerprint.

`load_checkpoint(path, device="cpu", tokenizer_path=None)` returns `(model, checkpoint)`, with the model in evaluation mode. To continue training, create an optimizer from the returned model and load `checkpoint["optimizer_state_dict"]`. The saved epoch and global step are available in new checkpoints; exact replay of random sampling is not implemented.

Data and tokenizer preparation are separate steps. They refuse to overwrite existing artifacts unless explicitly called with `--overwrite`. Do not retrain the tokenizer used by an existing model.

The previous implementation remains in `codes/pretrain_old.py` as a reference.

Run the refactor checks with:

```powershell
python -m unittest discover -s tests -p test_refactor.py -v
```

These checks include existing checkpoint compatibility, unchanged model computation, padding and causality, optimizer restoration, invalid configurations, and a tiny local training run. They do not rerun full pretraining.
