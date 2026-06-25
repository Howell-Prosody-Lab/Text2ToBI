#!/usr/bin/env python3
# text2tobi/__main__.py
# Entry point and argument parser. Called by text2tobi.py (shebang script).

from __future__ import annotations

import argparse
import os
import sys

from .registry import MODEL_REGISTRY, DEFAULT_MODEL
from .model import load_model_and_tokenizer
from .inference import (
    predict,
    words_from_string,
    words_from_file,
    _get_spacy,
)
from .output import format_table, format_raw, format_ssml, write_output

# ── Help / info strings ────────────────────────────────────────────────────────

USAGE = """\
text2tobi — ToBI prosody annotation from text

USAGE
  text2tobi "input text"            annotate a string (default table output)
  text2tobi path/to/file.txt        annotate a .txt file
  text2tobi help                    show this message
  text2tobi info                    show model and limitation details
  text2tobi download [model_name]   download a model checkpoint from HuggingFace Hub

ANNOTATION FLAGS (apply to string or file input)
  --ssml                            output SSML XML instead of the default table
  --raw                             output inline bracket annotations
  --clean                           strip timestamps, headers, and empty lines (file input only)
  --model MODEL_NAME                select a registered model (default: libri+peoples+sbc)
  --checkpoint PATH                 override registry; load from an explicit local path
  output_path                       optional output file path (default: stdout)

EXAMPLES
  text2tobi "the cat sat on the mat"
  text2tobi transcript.txt output.txt --ssml
  text2tobi "hello world" --raw --checkpoint /path/to/my/checkpoint
"""

INFO = """\
text2tobi — model and limitation details

MODEL
  Default model : libri+peoples+sbc
  Architecture  : DistilBERT multi-task token classifier
  Heads         : boundary detection (2-class)
                  intonation type    (3-class: H% / L% / !H%)
                  break index        (2-class: 3 / 4)

TRAINING DATA
  LibriTTS   (~145k samples, silver-standard via PSST + Wav2ToBI consensus)
  SBCSAE     (gold-standard, conversational speech)
  Test set   : SBCSAE SBC001–005 (held out from all training)

KNOWN LIMITATIONS
  - Only boundary detection and intonation type are reliably evaluated against
    a gold-standard test set. Break index figures are evaluated against
    LibriTTS silver labels, not human-annotated data. Break index output
    should be treated as experimental.
  - Intonation labels apply only to boundary words. Non-boundary intonation
    is not modeled.
  - Training data covers read speech (LibriTTS) and conversational speech
    (SBCSAE). Generalization to telephony, noisy speech, or non-native
    speakers is not yet tested.
  - For unpunctuated input, chunking falls back to a 100-token word-boundary
    split. This is not linguistically motivated and may affect predictions
    near split points.
  - This is an early research prototype. A pip-installable package and
    Python API are not yet available.
"""


# ── Dispatch ───────────────────────────────────────────────────────────────────

def cmd_help(_args):
    print(USAGE)


def cmd_info(_args):
    print(INFO)


def cmd_download(args):
    model_name = args.model_name or DEFAULT_MODEL
    if model_name not in MODEL_REGISTRY:
        _die(f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY)}")

    entry = MODEL_REGISTRY[model_name]
    hub_id = entry.get("hub_path")
    if not hub_id:
        _die(
            f"Model '{model_name}' has no HuggingFace Hub ID configured.\n"
            "  Set hub_path in text2tobi/registry.py once the checkpoint is uploaded."
        )

    cache_dir = os.path.expanduser(f"~/.cache/text2tobi/{model_name}")
    os.makedirs(cache_dir, exist_ok=True)

    print(f"Downloading '{model_name}' from {hub_id} → {cache_dir} ...")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=hub_id, local_dir=cache_dir)
        print(f"✓ Downloaded to {cache_dir}")
    except Exception as e:
        _die(f"Download failed: {e}")


def cmd_annotate(args):
    # ── 1. Resolve checkpoint path ─────────────────────────────────────────────
    if args.checkpoint:
        ckpt_path = args.checkpoint
        requires_pos = True   # conservative default for explicit paths
    else:
        model_name = args.model or DEFAULT_MODEL
        if model_name not in MODEL_REGISTRY:
            _die(f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY)}")
        entry = MODEL_REGISTRY[model_name]
        requires_pos = entry.get("requires_pos", False)

        # Prefer local cache, fall back to hub_path.
        cache_dir = os.path.expanduser(f"~/.cache/text2tobi/{model_name}")
        hub_path  = entry.get("hub_path")

        if os.path.isdir(cache_dir) and os.listdir(cache_dir):
            ckpt_path = cache_dir
        elif hub_path:
            ckpt_path = hub_path
        else:
            _die(
                f"Model '{model_name}' is not downloaded and has no Hub ID configured.\n"
                f"  Run:  text2tobi download {model_name}\n"
                f"  Or:   set hub_path in text2tobi/registry.py"
            )

    # ── 2. Load model and tokenizer ────────────────────────────────────────────
    print(f"Loading model from {ckpt_path} ...", file=sys.stderr)
    try:
        model, tokenizer = load_model_and_tokenizer(ckpt_path)
    except RuntimeError as e:
        _die(str(e))

    # ── 3. Load spaCy if needed ────────────────────────────────────────────────
    nlp = None
    if requires_pos:
        print("Loading spaCy en_core_web_sm ...", file=sys.stderr)
        try:
            nlp = _get_spacy()
        except RuntimeError as e:
            _die(str(e))

    # ── 4. Build word list ─────────────────────────────────────────────────────
    input_arg = args.input

    if os.path.isfile(input_arg):
        if not input_arg.endswith(".txt"):
            _die("Only .txt file input is supported.")
        words = words_from_file(input_arg, clean=args.clean)
    else:
        if args.clean:
            print(
                "Warning: --clean has no effect on string input.", file=sys.stderr
            )
        words = words_from_string(input_arg)

    if not words:
        _die("No words to annotate after processing input.")

    # ── 5. Run inference ───────────────────────────────────────────────────────
    annotations = predict(words, model, tokenizer,
                          requires_pos=requires_pos, nlp=nlp)

    # ── 6. Format output ───────────────────────────────────────────────────────
    if args.ssml:
        text = format_ssml(annotations)
        fmt  = "ssml"
    elif args.raw:
        text = format_raw(annotations)
        fmt  = "txt"
    else:
        text = format_table(annotations)
        fmt  = "txt"

    # Determine output path.
    output_path = args.output_path
    if output_path is None and os.path.isfile(input_arg):
        # File input with no explicit output → default to stdout (no auto-naming).
        pass

    write_output(text, output_path, fmt)


# ── Argument parsing ───────────────────────────────────────────────────────────

def main(argv: list[str] | None = None):
    argv = argv if argv is not None else sys.argv[1:]

    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE)
        return

    subcommand = argv[0]

    if subcommand == "help":
        print(USAGE)
        return

    if subcommand == "info":
        print(INFO)
        return

    if subcommand == "download":
        parser = argparse.ArgumentParser(prog="text2tobi download", add_help=False)
        parser.add_argument("model_name", nargs="?", default=None)
        args = parser.parse_args(argv[1:])
        cmd_download(args)
        return

    # Otherwise: annotation command.
    parser = argparse.ArgumentParser(prog="text2tobi", add_help=False)
    parser.add_argument("input",        help="Input string or path to .txt file.")
    parser.add_argument("output_path",  nargs="?", default=None,
                        help="Output file path (default: stdout).")
    parser.add_argument("--ssml",       action="store_true")
    parser.add_argument("--raw",        action="store_true")
    parser.add_argument("--clean",      action="store_true")
    parser.add_argument("--model",      default=None)
    parser.add_argument("--checkpoint", default=None,
                        help="Explicit local checkpoint path (overrides registry).")

    args = parser.parse_args(argv)

    if args.ssml and args.raw:
        _die("--ssml and --raw are mutually exclusive.")

    cmd_annotate(args)


def _die(msg: str):
    print(f"text2tobi error: {msg}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
