# text2tobi/inference.py
# Tokenization, POS tagging, chunking, and prediction logic.

import torch
from .model import POS_TAG_TO_ID, UNIVERSAL_TO_TOKEN, UNK_POS_TOKEN

# Speaker change marker used internally (matches training pipeline).
SPK_CHANGE_TOKEN = "/"

# Chunk token limit (with headroom below the 128-token model max).
CHUNK_TOKEN_LIMIT = 100


def _get_spacy():
    """Lazy-load spaCy en_core_web_sm. Raises a clean error if not installed."""
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
            # Add sentencizer for sentence boundary detection.
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
            return nlp
        except OSError:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' not found.\n"
                "  Install it with:  python -m spacy download en_core_web_sm"
            )
    except ImportError:
        raise RuntimeError(
            "spaCy is not installed. Install it with:  pip install spacy"
        )


def _pos_ids_for_words(words: list[str], nlp) -> list[int]:
    """
    Return a list of POS tag IDs (integers) for each word in `words`,
    aligned 1-to-1. Speaker-change tokens get PAD (0).

    Uses forced tokenization (Doc with words= and spaces=) over the full word
    list at once, matching how the training pipeline tagged words. This gives
    contextual POS tags (tok2vec sees the whole sequence) and avoids the
    IndexError risk of tagging single isolated words via nlp(word).
    """
    from spacy.tokens import Doc

    pos_ids = [0] * len(words)  # default PAD; overwritten for non-SPK positions

    # Collect indices and words that need tagging (skip SPK_CHANGE_TOKEN).
    non_spk = [(i, w) for i, w in enumerate(words) if w != SPK_CHANGE_TOKEN]
    if not non_spk:
        return pos_ids

    indices, word_seq = zip(*non_spk)
    # spaces list: True between words, False after the last one.
    spaces = [True] * (len(word_seq) - 1) + [False]
    doc = Doc(nlp.vocab, words=list(word_seq), spaces=spaces)

    # Run only the pipes needed for POS tagging.
    for pipe_name in ("tok2vec", "tagger"):
        if nlp.has_pipe(pipe_name):
            nlp.get_pipe(pipe_name)(doc)

    for i, token in zip(indices, doc):
        tag = token.pos_ or "X"
        pos_ids[i] = POS_TAG_TO_ID.get(tag, POS_TAG_TO_ID.get("X", 0))

    return pos_ids


def _split_into_chunks(words: list[str], tokenizer) -> list[list[str]]:
    """
    Split a flat word list into chunks that fit within CHUNK_TOKEN_LIMIT tokens.

    Strategy:
      1. Try to split at sentence boundaries (SPK_CHANGE_TOKEN is always a hard break).
      2. If any resulting chunk exceeds the limit, split at the nearest word boundary.

    Returns a list of word-lists (chunks). SPK_CHANGE_TOKEN is NOT included in
    output chunks — it is used as a split signal only.
    """
    # First pass: split on speaker change tokens.
    segments: list[list[str]] = []
    current: list[str] = []
    for word in words:
        if word == SPK_CHANGE_TOKEN:
            if current:
                segments.append(current)
            current = []
        else:
            current.append(word)
    if current:
        segments.append(current)

    # Second pass: split any segment that is too long.
    chunks: list[list[str]] = []
    for seg in segments:
        # Estimate token count for the whole segment.
        encoding = tokenizer(
            seg,
            is_split_into_words=True,
            add_special_tokens=True,
            return_attention_mask=False,
        )
        if len(encoding["input_ids"]) <= CHUNK_TOKEN_LIMIT:
            chunks.append(seg)
        else:
            # Split greedily at CHUNK_TOKEN_LIMIT token boundary.
            buf: list[str] = []
            for word in seg:
                trial = buf + [word]
                enc = tokenizer(
                    trial,
                    is_split_into_words=True,
                    add_special_tokens=True,
                    return_attention_mask=False,
                )
                if len(enc["input_ids"]) > CHUNK_TOKEN_LIMIT and buf:
                    chunks.append(buf)
                    buf = [word]
                else:
                    buf = trial
            if buf:
                chunks.append(buf)

    return chunks


def _tokenize_chunk(words: list[str], tokenizer, pos_ids: list[int] | None,
                    max_length: int = 128):
    """
    Tokenize a single chunk and align POS IDs to sub-words.

    Returns
    -------
    input_ids      : (1, T) LongTensor
    attention_mask : (1, T) LongTensor
    word_ids       : list[int | None]  — sub-word → word index mapping
    aligned_pos    : (1, T) LongTensor | None
    """
    encoding = tokenizer(
        words,
        is_split_into_words=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    word_ids = encoding.word_ids(batch_index=0)

    aligned_pos = None
    if pos_ids is not None:
        raw = []
        prev = None
        for wid in word_ids:
            if wid is None:
                raw.append(0)   # PAD for [CLS]/[SEP]
            elif wid != prev:
                raw.append(pos_ids[wid] if wid < len(pos_ids) else 0)
            else:
                raw.append(0)   # PAD for continuation sub-words
            prev = wid
        aligned_pos = torch.tensor([raw], dtype=torch.long)

    return (
        encoding["input_ids"],
        encoding["attention_mask"],
        word_ids,
        aligned_pos,
    )


# Label decode maps.
BOUNDARY_LABELS   = {0: "-", 1: "B"}
INTONATION_LABELS = {0: "H%", 1: "L%", 2: "!H%"}
BREAK_IDX_LABELS  = {0: "3",  1: "4"}


def predict(
    words: list[str],
    model,
    tokenizer,
    requires_pos: bool = True,
    nlp=None,
) -> list[dict]:
    """
    Run inference on a flat word list and return per-word annotation dicts.

    Parameters
    ----------
    words        : flat list of word strings (speaker change tokens stripped)
    model        : ProsodyBoundaryModel in eval mode
    tokenizer    : tokenizer matching the checkpoint
    requires_pos : whether to inject POS embeddings
    nlp          : loaded spaCy pipeline (required if requires_pos=True)

    Returns
    -------
    list of dicts, one per input word:
        {
            "word":        str,
            "boundary":    "-" | "B",
            "intonation":  "-" | "H%" | "L%" | "!H%",
            "break_index": "-" | "3" | "4",
        }
    """
    if requires_pos and nlp is None:
        nlp = _get_spacy()

    # POS tag the full word list once before chunking.
    pos_ids = _pos_ids_for_words(words, nlp) if requires_pos else None

    chunks = _split_into_chunks(words, tokenizer)

    results: list[dict] = []

    # `pos_ids` is indexed against the original `words` list, which includes
    # SPK_CHANGE_TOKEN entries. `_split_into_chunks` strips those tokens from
    # output chunks, so we can't use `word_offset += len(chunk)` — that would
    # desync whenever a speaker-change token appears between chunks.
    # Instead we walk `words` in parallel and advance past SPK tokens explicitly.
    original_pos = 0  # cursor into the original `words` list

    with torch.no_grad():
        for chunk in chunks:
            # Advance past any SPK_CHANGE_TOKEN entries that were consumed as
            # chunk delimiters by _split_into_chunks.
            while original_pos < len(words) and words[original_pos] == SPK_CHANGE_TOKEN:
                original_pos += 1

            # Slice pos_ids for the words in this chunk (no SPK tokens).
            chunk_pos = (
                pos_ids[original_pos: original_pos + len(chunk)]
                if pos_ids is not None else None
            )

            input_ids, attention_mask, word_ids, aligned_pos = _tokenize_chunk(
                chunk, tokenizer, chunk_pos
            )

            forward_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if aligned_pos is not None:
                forward_kwargs["pos_ids"] = aligned_pos

            output = model(**forward_kwargs)

            b_logits = output["boundary_logits"][0]    # (T, 2)
            i_logits = output["intonation_logits"][0]  # (T, 3)
            x_logits = output["break_idx_logits"][0]   # (T, 2)

            b_preds = b_logits.argmax(dim=-1).tolist()
            i_preds = i_logits.argmax(dim=-1).tolist()
            x_preds = x_logits.argmax(dim=-1).tolist()

            # Collect first-subword predictions for each word in the chunk.
            seen = set()
            for pos, wid in enumerate(word_ids):
                if wid is None or wid in seen:
                    continue
                seen.add(wid)
                word = chunk[wid]
                b = b_preds[pos]
                boundary   = BOUNDARY_LABELS[b]
                intonation = INTONATION_LABELS[i_preds[pos]] if b == 1 else "-"
                break_idx  = BREAK_IDX_LABELS[x_preds[pos]] if b == 1 else "-"
                results.append({
                    "word":        word,
                    "boundary":    boundary,
                    "intonation":  intonation,
                    "break_index": break_idx,
                })

            original_pos += len(chunk)

    return results


def words_from_string(text: str) -> list[str]:
    """
    Split a quoted CLI string into words.
    Internal speaker change marker ' ## ' becomes SPK_CHANGE_TOKEN.
    """
    # Normalise speaker change marker.
    text = text.replace(" ## ", f" {SPK_CHANGE_TOKEN} ")
    return text.split()


def words_from_file(path: str, clean: bool = False) -> list[str]:
    """
    Read a .txt file and return a flat word list.
    Blank lines become SPK_CHANGE_TOKEN (speaker turn signal).

    Parameters
    ----------
    path  : path to .txt file
    clean : if True, strip timestamps, ALL-CAPS headers, and colon-terminated lines
    """
    import re

    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    if clean:
        lines = raw.splitlines()
        cleaned = []
        for line in lines:
            # Strip timestamps like 00:12:34 or 1:23:45
            line = re.sub(r"\b\d{1,2}:\d{2}(:\d{2})?\b", "", line).strip()
            # Drop ALL-CAPS lines (section headers)
            if line and line == line.upper() and any(c.isalpha() for c in line):
                continue
            # Drop lines ending in ':'
            if line.endswith(":"):
                continue
            cleaned.append(line)
        raw = "\n".join(cleaned)

    # Split on blank lines → speaker turns → insert SPK_CHANGE_TOKEN.
    paragraphs = re.split(r"\n\s*\n", raw.strip())
    words: list[str] = []
    for i, para in enumerate(paragraphs):
        if i > 0:
            words.append(SPK_CHANGE_TOKEN)
        words.extend(para.split())

    return words
