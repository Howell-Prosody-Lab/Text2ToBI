# text2tobi

A CLI tool for ToBI prosodic annotation from text alone. No audio required at inference time.

This is an early research prototype accompanying the paper *Text2ToBI: Recovering Prosodic Structure from Text*. A pip-installable package and Python API are planned for a future release.

---

## Requirements

- Python 3.10+
- PyTorch
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [spaCy](https://spacy.io/) + `en_core_web_sm`

```bash
pip install torch transformers spacy
python -m spacy download en_core_web_sm
```

---

## Setup

Clone the repo and point `--checkpoint` at a local model directory (see below for obtaining the checkpoint):

```bash
git clone https://github.com/your-handle/text2tobi
cd text2tobi
python text2tobi.py "the cat sat on the mat"
```

Or make it executable:

```bash
chmod +x text2tobi.py
./text2tobi.py "the cat sat on the mat"
```

---

## Usage

```
text2tobi "input text"                    annotate a string
text2tobi path/to/file.txt                annotate a .txt file
text2tobi help                            show usage
text2tobi info                            model and limitation details
text2tobi download [model_name]           download checkpoint from HuggingFace Hub
```

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

The `libri+sbc_pos_stl` checkpoint is hosted on HuggingFace Hub (private repository due to licensing — see below). To access it, you must be added as a collaborator by the author.

Once you have access:

```bash
python text2tobi.py download
```

This downloads the checkpoint to `~/.cache/text2tobi/libri+sbc_pos_stl/`. Subsequent runs load from local cache.

Alternatively, point directly at a local checkpoint directory:

```bash
python text2tobi.py "some text" --checkpoint /path/to/checkpoint
```

---

## Licensing note

The `libri+sbc_pos_stl` model was trained on the Santa Barbara Corpus of Spoken American English (SBCSAE), which is licensed CC BY-ND 3.0 US. Distributing model weights derived from it is a legal gray area under the no-derivatives clause. The Hub repository is therefore private. A future model trained exclusively on CC BY data (LibriTTS + People's Speech) may be released publicly.

---

## Known limitations

- **Break index** is experimental. Current evaluation figures are against LibriTTS silver labels, not a human-annotated test set. Boundary detection and intonation type are the reliably evaluated outputs.
- **Intonation** labels apply to boundary words only. Non-boundary intonation is not modeled.
- **Register coverage** is read speech (LibriTTS) and spontaneous conversational speech (SBCSAE). Generalization to telephony, noisy speech, or non-native speakers is not yet tested.
- **Chunking fallback**: unpunctuated input is split at a 100-token word boundary. This is not linguistically motivated and may affect predictions near split points.

---

## Architecture

`ProsodyBoundaryModel` is a multi-task token classifier fine-tuned on `distilbert-base-uncased`:

```
DistilBERT encoder
    [+ POS embedding injection, post-transformer]
    └─► dropout
         ├─► boundary_head    Linear(768 → 2)   boundary / non-boundary
         ├─► intonation_head  Linear(768 → 3)   H% / L% / !H%
         └─► break_idx_head   Linear(768 → 2)   index-3 / index-4
```

POS tags (spaCy Universal Dependencies) are embedded (64-dim) and projected to 768-dim, then added to DistilBERT's last hidden state before the classification heads.
