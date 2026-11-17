# SFT pilot data

This assistant-authored synthetic dataset contains 32 training examples and 8
development examples for a single-turn English short-story assistant. It is a
pipeline pilot, not a sufficient final training corpus or an independent test.
Expand it after checking the encoding, response-only labels, and small-batch
training behavior. Do not treat passing format checks as proof of training gains.

Each JSONL record contains `id`, `prompt`, `response`, and `required_terms`.
Some records also specify `sentence_count`. Only `prompt` and `response` are
model text; the remaining fields support simple constraint checks. These checks
cannot establish coherence, safety, or complete instruction compliance. The
examples have not received independent human review.

Use the existing tokenizer. Assemble IDs separately in this exact order:

```text
<bos><user>{prompt}<assistant>{response}<eos>
```

Do not insert implicit spaces or newlines between components. Train on response
tokens and EOS only. Mask the prompt and control-prefix labels with `-100`;
use `-100` for padded labels too. Shift the inputs and labels exactly once.
The assistant marker remains part of the input prefix. Its output predicts the
first response token, so that prediction must remain supervised.

Validate complete encoded lengths against the model context limit. Reject an
oversized example instead of truncating its answer or EOS. At inference time,
assemble the same prefix ending at `<assistant>` and generate the response.

SFT starts from a selected completed pretraining checkpoint, with a fresh
optimizer. Save SFT models in their own run directory. The generic story
continuation helper does not yet assemble this dialogue prefix.
