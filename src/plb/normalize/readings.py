from __future__ import annotations

import unicodedata

from fugashi import Tagger

from .helpers import _to_hiragana
from .schema import ReadingSource

_TAGGER = Tagger()
_KANA_MARKS = frozenset("ーゝゞヽヾ")


def _is_kana_letter(char: str) -> bool:
    name = unicodedata.name(char, "")
    return "HIRAGANA LETTER" in name or "KATAKANA LETTER" in name


def _is_clean_kana(text: str) -> bool:
    """Return whether text can be converted to a reading without lexical lookup."""
    saw_letter = False
    for char in text:
        if _is_kana_letter(char):
            saw_letter = True
        elif char not in _KANA_MARKS:
            return False
    return saw_letter


def resolve_reading(
    form: str, reading: str | None, language: str
) -> tuple[str | None, ReadingSource | None]:
    """Canonicalize an original reading or derive a missing Japanese reading."""
    if reading:
        canonical = _to_hiragana(reading) if language == "ja" else reading
        return canonical, "original"

    if language != "ja":
        return None, None

    normalized_form = unicodedata.normalize("NFKC", form)
    if _is_clean_kana(normalized_form):
        return _to_hiragana(normalized_form), "derived"

    kana_parts: list[str] = []
    for token in _TAGGER(normalized_form):
        kana = token.feature.kana
        if not kana or kana == "*":
            return None, None
        kana_parts.append(kana)

    derived = _to_hiragana("".join(kana_parts))
    return (derived, "derived") if derived else (None, None)
