from __future__ import annotations

import html
import re
import unicodedata
import uuid
from typing import Any, Optional, overload

NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "datasets/raw")


def _uuid_from_parts(*parts: Any) -> uuid.UUID:
    return uuid.uuid5(NAMESPACE, ":".join(str(p) for p in parts))


# ---- Text cleaning helpers ---- #
_SOUND_RE = re.compile(r"\[sound:[^\]]*\]")
_CLOZE_RE = re.compile(r"\{\{c\d+::.*?\}\}", re.DOTALL)
_SVG_RE = re.compile(r"<svg\b.*?</svg>", re.DOTALL | re.IGNORECASE)
_SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b.*?</\1>", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_FURIGANA_RE = re.compile(r"\[[^\]]*\]")  # Anki furigana:  漢字[かんじ] -> 漢字
_WS_RE = re.compile(r"\s+")
_SENTENCE_PUNCT_RE = re.compile(r"[。、！？!?．…「」『』,]")
_INTERROGATIVE_RE = re.compile(r"^(what|which|how|when|where|why|who)\b", re.IGNORECASE)


def _strip_html(text: str) -> str:
    text = _SVG_RE.sub(" ", text)
    text = _SCRIPT_STYLE_RE.sub(" ", text)
    text = _TAG_RE.sub(" ", text)
    return html.unescape(text)


def _clean_text(text: Optional[str]) -> Optional[str]:
    """Baseline normalisation for any extracted field."""
    if text is None:
        return None
    text = _SOUND_RE.sub(" ", text)
    text = _strip_html(text)
    text = _WS_RE.sub(" ", text).strip()
    return text or None


def _strip_furigana(text: Optional[str]) -> Optional[str]:
    """Drop Anki furigana brackets, keeping only the base form."""
    if text is None:
        return None
    return _FURIGANA_RE.sub("", text).strip() or None


def _norm_key(text: Optional[str]) -> str:
    """NFKC-normalised key for dedup (does not alter stored value)."""
    if not text:
        return ""
    return unicodedata.normalize("NFKC", text).strip()


def _to_hiragana(text: str | None) -> str | None:
    if text is None:
        return None
    text = unicodedata.normalize("NFKC", text)
    return "".join(
        chr(ord(char) - 0x60)
        if "ァ" <= char <= "ヶ"
        else {"ヽ": "ゝ", "ヾ": "ゞ"}.get(char, char)
        for char in text
    )


def _has_cloze(text: Optional[str]) -> bool:
    return bool(text and _CLOZE_RE.search(text))


@overload
def _valid_string(value: None, error_msg: str | None) -> None: ...
@overload
def _valid_string(value: str, error_msg: str | None) -> str: ...


def _valid_string(value: str | None, error_msg: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        raise ValueError(error_msg or "String is empty or whitespace only")
    return value
