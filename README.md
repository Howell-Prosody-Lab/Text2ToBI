# Text2ToBI

A CLI tool for ToBI prosodic annotation from text alone. No audio required at inference time.

This is an early research prototype accompanying my thesis *Text2ToBI: Recovering Prosodic Structure from Text*. A pip-installable package and Python API are planned for a future release.

---

## Requirements

- Python 3.10+
- PyTorch
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [HuggingFace Hub](https://github.com/huggingface/huggingface_hub)

```bash
pip install torch transformers huggingface_hub
```

---

## Setup

Clone the repo and run via `python -m`:

```bash
git clone github.com/Howell-Prosody-Lab/Text2ToBI
cd text2tobi
python -m text2tobi download
python -m text2tobi "yesterday I bought some beans some arugula a rutabaga and an onion."
```

Make sure this is executed one level above the folder containing code (current directory should be the one with the README in it).

To use a local checkpoint instead:

```bashF
python -m text2tobi "some text" --checkpoint /path/to/checkpoint
```

---

## Usage

```
python -m text2tobi "input text"              annotate a string
python -m text2tobi path/to/file.txt          annotate a .txt file
python -m text2tobi help                      show usage
python -m text2tobi info                      model and limitation details
python -m text2tobi download [model_name]     download checkpoint from HuggingFace Hub
```
Note: Input text does not require punctuation at all, it will be stripped at inference time and the model will receive a continuous stream of words.

**Flags** (annotation commands only):

| Flag | Description |
|---|---|
| `--ssml` | Output SSML XML |
| `--raw` | Output inline bracket annotations |
| `--clean` | Strip timestamps and headers (file input only) |
| `--model NAME` | Select a registered model name |
| `--checkpoint PATH` | Load from an explicit local checkpoint path |
| `output_path` | Optional output file (default: stdout) |

---

## Output formats

**Default (table)**

```
word        boundary    intonation    break_index
the         -           -             -
cat         -           -             -
sat         B           L%            4
on          -           -             -
the         -           -             -
mat         B           H%            3
```

**`--raw`**

```
the cat sat[B/L%/4] on the mat[B/H%/3]
```

**`--ssml`**

```xml
<speak>
  the cat <prosody pitch="-15%">sat</prosody><break strength="strong"/>
  on the <prosody pitch="+15%">mat</prosody><break strength="medium"/>
</speak>
```

---

## Model checkpoint

The `libri+peoples+sbc` checkpoint is hosted publicly on HuggingFace Hub. To download it:

```bash
python -m text2tobi download
```

This downloads the checkpoint to `~/.cache/text2tobi/libri+peoples+sbc/`. Subsequent runs load from local cache.

---

## Known limitations

- **Break index** is experimental. Current evaluation figures are against LibriTTS silver labels, not a human-annotated test set. Boundary detection and intonation type are the reliably evaluated outputs.
- **Intonation** labels apply to boundary words only. Non-boundary intonation is not modeled.
- **Register coverage** is read speech (LibriTTS + People's Speech) and spontaneous conversational speech (SBCSAE). Generalization to telephony, noisy speech, or non-native speakers is not yet tested.
- **Chunking fallback**: unpunctuated input is split at a 100-token word boundary. This is not linguistically motivated and may affect predictions near split points.

---

## Architecture

`ProsodyBoundaryModel` is a multi-task token classifier fine-tuned on `distilbert-base-uncased`:

```
DistilBERT encoder
    └─► dropout
         ├─► boundary_head    Linear(768 → 2)   boundary / non-boundary
         ├─► intonation_head  Linear(768 → 3)   H% / L% / !H%
         └─► break_idx_head   Linear(768 → 2)   index-3 / index-4
```
## License

Licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
