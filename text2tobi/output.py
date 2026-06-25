# text2tobi/output.py
# Output formatters: default table, --raw inline, --ssml XML.

from __future__ import annotations


# ── SSML mappings ──────────────────────────────────────────────────────────────

_PITCH_MAP = {
    "H%":  '+15%',
    "L%":  '-15%',
    "!H%": None,    # level — no pitch markup
}

_BREAK_STRENGTH_MAP = {
    "3": "medium",
    "4": "strong",
}


# ── Formatters ─────────────────────────────────────────────────────────────────

def format_table(annotations: list[dict]) -> str:
    col_w = max(len(a["word"]) for a in annotations) + 2
    header = f"{'word':<{col_w}}{'boundary':<12}{'intonation':<14}{'break_index'}"
    lines = [header]
    for ann in annotations:
        lines.append(
            f"{ann['word']:<{col_w}}{ann['boundary']:<12}{ann['intonation']:<14}{ann['break_index']}"
        )
    return "\n".join(lines)


def format_raw(annotations: list[dict]) -> str:
    """
    Inline annotation: bare words are output as-is; boundary words get
    [B/intonation/break_index] tags appended.

    Example:  the cat sat[B/L%/4] on the mat[B/H%/3]
    """
    parts = []
    for ann in annotations:
        if ann["boundary"] == "B":
            parts.append(
                f"{ann['word']}[B/{ann['intonation']}/{ann['break_index']}]"
            )
        else:
            parts.append(ann["word"])
    return " ".join(parts)


def format_ssml(annotations: list[dict]) -> str:
    """
    SSML output. Pitch markup wraps boundary words; break tags follow them.

    Mapping:
        H%   → <prosody pitch="+15%">word</prosody>
        L%   → <prosody pitch="-15%">word</prosody>
        !H%  → no pitch markup (level)
        brk3 → <break strength="medium"/>
        brk4 → <break strength="strong"/>
    """
    inner_parts = []
    for ann in annotations:
        if ann["boundary"] == "B":
            pitch_val = _PITCH_MAP.get(ann["intonation"])
            break_val = _BREAK_STRENGTH_MAP.get(ann["break_index"], "medium")

            if pitch_val:
                word_markup = (
                    f'<prosody pitch="{pitch_val}">{ann["word"]}</prosody>'
                )
            else:
                word_markup = ann["word"]

            inner_parts.append(f"{word_markup}<break strength=\"{break_val}\"/>")
        else:
            inner_parts.append(ann["word"])

    inner = "\n  ".join(inner_parts)
    return f"<speak>\n  {inner}\n</speak>"


def write_output(
    text: str,
    output_path: str | None,
    mode: str,   # "ssml" → .ssml extension; else .txt
) -> None:
    """
    Write `text` to `output_path`, or print to stdout if output_path is None.
    """
    if output_path is None:
        print(text)
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Written to {output_path}")
