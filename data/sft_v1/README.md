# StorySprout SFT v1

This is a small development dataset for the first SFT scaling experiment.
All responses are assistant-authored synthetic stories. They have not received
independent human review.

| File | Contents |
| --- | --- |
| `train.jsonl` | 200 examples: the original 32 pilot examples plus 168 new stories |
| `validation.jsonl` | An unchanged copy of the 8 pilot validation examples |
| `evaluation_prompts.jsonl` | 20 development prompts for generation comparisons; excluded from training |
| `new_stories.tsv` | The 168 new stories with their topics and categories |
| `manifest.json` | Dataset provenance, checks, lengths, counts, and file hashes |

The new stories cover lost objects, cooperation, small problems, kindness,
feelings, discovery, safe choices, and nature care. Each new response has three
sentences. Prompts vary between a sentence-count request, consistent character
names, clear events, and a clear ending. The dataset remains intentionally small;
it is not sufficient evidence of broad instruction-following ability.

## Encoding and checks

Keep the existing tokenizer and pretrained model. The full sequence is:

```text
<bos><user>{prompt}<assistant>{response}<eos>
```

The existing encoder shifts inputs and labels for next-token prediction. Only
response tokens and EOS contribute to loss; prompt and padding labels use `-100`.
No example needs truncation: full sequences contain 56–96 tokens, below the
current context limit of 256. The shifted model input is one token shorter.

Automatic checks covered unique IDs, exact prompt/response duplicates across
training and validation, response word-trigram similarity, required terms,
explicit sentence counts, special-token text, and token lengths. These checks
do not establish semantic quality or improved model behavior.

Validation is kept fixed for comparison with the pilot. Its eight examples are
too few for a stable estimate. The generation prompts are also development data,
not a final independent benchmark. Compare the same prompts before and after SFT
with the same decoding settings; check names, requested events, coherence, and EOS.

## Use

From the project directory, add this option to the existing SFT command:

```powershell
--data-dir data/sft_v1
```

Start from a completed pretraining checkpoint with a fresh optimizer. A first
trial can use `--epochs 3 --batch-size 8 --learning-rate 0.0001 --log-every 5`.
This gives 25 updates per epoch. Select the checkpoint using validation loss and
generation inspection together. The default dataset remains `data/sft_pilot` if
`--data-dir` is omitted. Each new SFT run records its data paths, counts, and hashes
in `data_manifest.json`.
