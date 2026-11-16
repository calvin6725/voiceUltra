"""Text frontend: normalisation, sentence chunking and prosody pauses.

ZipVoice speaks whatever text it is given, including its punctuation.  Long
single-shot synthesis flattens intonation and rushes pauses, which is the
main "robotic" tell.  This module splits text into speakable chunks, keeps
terminal punctuation (so the model inflects correctly) and records a
natural pause after each chunk for :func:`audio_dsp.join_chunks`.
"""

from __future__ import annotations

import hashlib
import re

# Terminal punctuation -> pause after the chunk (ms, before pause_scale).
_PAUSE_MS = {
    "。": 220, "！": 260, "!": 260, "？": 260, "?": 260,
    "；": 160, ";": 160, "：": 120, ":": 120,
    "，": 90, ",": 90, "、": 90, "…": 200, "...": 200,
    "”": 120, "”": 120, "’": 60, "'": 60,
}

_SENTENCE_END = re.compile(r"(?<=[。！？；.!?;])\s*")
# Protect common English abbreviations from sentence splitting.
_ABBREV = re.compile(r"\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|e\.g|i\.e)\.", re.IGNORECASE)
_PLACEHOLDER = "\uE000"  # private-use sentinel, never in real text


def normalize_text(text: str) -> str:
    """Clean user text into a form ZipVoice pronounces reliably.

    * full-width ASCII -> half-width (except CJK punctuation)
    * collapse whitespace, strip control characters
    * remove emphasis markers (``*``/``**``/``#``) ZipVoice cannot hear
    * ``[pause]`` / ``[停顿]`` -> sentence boundary with a long pause
    * ensure the text ends with terminal punctuation
    """
    if text is None:
        return ""
    text = str(text)
    # Full-width ASCII letters/digits/punct -> half-width.
    out = []
    for ch in text:
        code = ord(ch)
        if 0xFF01 <= code <= 0xFF5E:
            out.append(chr(code - 0xFEE0))
        elif ch == "\u3000":  # ideographic space
            out.append(" ")
        else:
            out.append(ch)
    text = "".join(out)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    text = text.replace("**", "").replace("*", "").replace("#", "")
    text = re.sub(r"\[(pause|停顿|长停顿)\]", "。", text, flags=re.IGNORECASE)
    text = re.sub(r"([。！？；.!?;])\1+", r"\1", text)  # collapse duplicates
    text = re.sub(r"\s+", " ", text).strip()
    if text and text[-1] not in "。！？；.!?;…":
        text += "。"
    return text


def split_sentences(text: str) -> list[str]:
    """Split into sentences, keeping terminal punctuation attached."""
    text = normalize_text(text)
    if not text:
        return []
    protected = _ABBREV.sub(lambda m: m.group(0).replace(".", _PLACEHOLDER), text)
    parts = [p.strip() for p in _SENTENCE_END.split(protected) if p.strip()]
    return [p.replace(_PLACEHOLDER, ".") for p in parts]


def _pause_for(sentence: str) -> int:
    for punct, ms in _PAUSE_MS.items():
        if sentence.endswith(punct):
            return ms
    return 140


def chunk_text(
    text: str,
    max_chars: int = 140,
    pause_scale: float = 1.0,
) -> list[dict]:
    """Group sentences into speakable chunks (<= ``max_chars`` chars each).

    Returns ``[{"text": ..., "pause_ms": ...}, ...]``.  Short sentences are
    merged so the model gets enough context; long sentences are hard-split
    at clause punctuation (，、；) so no chunk is rushed.
    """
    sentences = split_sentences(text)
    if not sentences:
        return []
    # Hard-split over-long sentences at clause boundaries.
    expanded: list[str] = []
    for sent in sentences:
        if len(sent) <= max_chars:
            expanded.append(sent)
            continue
        pieces = re.split(r"(?<=[，、；,;])", sent)
        buf = ""
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            if len(buf) + len(piece) <= max_chars:
                buf += piece
            else:
                if buf:
                    expanded.append(buf)
                # Still too long: slice by characters.
                while len(piece) > max_chars:
                    expanded.append(piece[:max_chars])
                    piece = piece[max_chars:]
                buf = piece
        if buf:
            expanded.append(buf)
    chunks: list[dict] = []
    buf = ""
    for sent in expanded:
        if buf and len(buf) + len(sent) > max_chars:
            chunks.append({"text": buf, "pause_ms": int(_pause_for(buf) * pause_scale)})
            buf = sent
        else:
            buf += sent
    if buf:
        # Last chunk needs no trailing pause.
        chunks.append({"text": buf, "pause_ms": 0})
    return chunks


def prosody_speed_variation(chunk_index: int, base_speed: float, amount: float = 0.02) -> float:
    """Deterministic per-chunk micro speed variation (±``amount``).

    Human readers never hit exactly the same tempo twice.  A tiny
    deterministic wobble breaks metronomic monotony without changing
    overall pace.  Deterministic so re-renders are reproducible.
    """
    digest = hashlib.md5(f"chunk:{chunk_index}".encode()).digest()
    unit = (digest[0] / 255.0) * 2.0 - 1.0  # [-1, 1]
    varied = base_speed * (1.0 + unit * amount)
    return round(max(0.75, min(1.35, varied)), 4)


def estimate_speech_seconds(text: str, speed: float = 1.0) -> float:
    """Rough duration estimate for progress display (not for synthesis)."""
    text = normalize_text(text)
    if not text:
        return 0.0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    latin = len(re.findall(r"[A-Za-z]", text))
    # ~4.2 CJK chars/sec, ~13 latin chars/sec at 1.0x.
    seconds = cjk / 4.2 + latin / 13.0 + max(0, len(text) - cjk - latin) / 20.0
    return round(max(0.5, seconds / max(0.5, speed)), 2)


__all__ = [
    "chunk_text",
    "estimate_speech_seconds",
    "normalize_text",
    "prosody_speed_variation",
    "split_sentences",
]
