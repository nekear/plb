from __future__ import annotations

import html
import re

from .helpers import _clean_text
from .process import Policy, Submission

# --------------------------------------------------------------------------- #
# Utilities for the extractors
# --------------------------------------------------------------------------- #

_SOUND = re.compile(r"\[sound:[^\]]*\]")
_TAG = re.compile(r"<[^>]+>")
_FURI = re.compile(r"([^\s\[\]]+)\[([^\]]+)\]")
_RT = re.compile(r"<rt>.*?</rt>", re.S)


def furigana_reading(s: str | None) -> str | None:
    """
    Anki furigana ('参考[さんこう]', '倒[たお]す') OR plain kana ('たおす')
    -> pure kana reading ('さんこう', 'たおす'). No-op on kana-only input.
    """
    if s is None:
        return None
    s = _SOUND.sub(" ", s)
    s = _TAG.sub("", s)
    s = html.unescape(s)
    s = _FURI.sub(lambda m: m.group(2), s)  # base[read] -> read
    s = re.sub(r"[\s\u3000]", "", s)
    return s or None


def deruby_base(s: str | None) -> str | None:
    """'<ruby>千<rt>せん</rt></ruby>' -> '千' (drop <rt> readings, keep base)."""
    return _clean_text(_RT.sub("", s)) if s else None


def ruby_reading(s: str | None) -> str | None:
    """'<ruby>千<rt>せん</rt></ruby>' -> 'せん' (join <rt> contents)."""
    if not s:
        return None
    return _clean_text("".join(re.findall(r"<rt>(.*?)</rt>", s, re.S)))


# --------------------------------------------------------------------------- #
# Extractors  (return {form, reading, meaning, pitch}; None fields allowed)    #
# --------------------------------------------------------------------------- #


def jp_expression(f):  # L04 M1, M2 -- Expression / Reading(furigana) / Meaning
    return {
        "form": _clean_text(f.get("Expression")),
        "reading": furigana_reading(f.get("Reading")),
        "meaning": _clean_text(f.get("Meaning")),
        "pitch": None,
    }


def jp_media(f):  # L03 M1 -- expression / (no reading field) / 英語 ; pitch=SVG
    return {
        "form": _clean_text(f.get("expression")),
        "reading": None,  # no clean reading field; derive later from pitch/g2p
        "meaning": _clean_text(f.get("英語")),
        "pitch": f.get("pitch") or None,
    }


def front_back(f):  # L08 M1, M3 -- alphabetic Front / Back (es, pt)
    return {
        "form": _clean_text(f.get("Front")),
        "meaning": _clean_text(f.get("Back")),
        "reading": None,
        "pitch": None,
    }


def it_front_back(f):  # L07 M2 -- strip trailing article: "biologia, la" -> "biologia"
    fr = _clean_text(f.get("Front"))
    return {
        "form": fr.split(",")[0].strip() if fr else None,
        "meaning": _clean_text(f.get("Back")),
        "reading": None,
        "pitch": None,
    }


def kaishi_vocab(f):  # L05 M4, L06 M4 -- Word / Word Reading / Word Meaning
    return {
        "form": _clean_text(f.get("Word")),
        "reading": furigana_reading(f.get("Word Reading")),
        "meaning": _clean_text(f.get("Word Meaning")),
        "pitch": f.get("Pitch Accent") or None,
    }


def one_k_vocab(f):  # L05 M5 -- Word / Vocabulary-Kana / Vocabulary-English
    return {
        "form": _clean_text(f.get("Word")),
        "reading": furigana_reading(f.get("Vocabulary-Kana")),
        "meaning": _clean_text(f.get("Vocabulary-English")),
        "pitch": None,
    }


def yomitan_cap(f):  # L05 M6 -- Expression / ExpressionReading / MainDefinition
    return {
        "form": _clean_text(f.get("Expression")),
        "reading": furigana_reading(f.get("ExpressionReading")),
        "meaning": _clean_text(f.get("MainDefinition"))
        or _clean_text(f.get("Glossary")),
        "pitch": None,
    }


def yomitan_lc(f):  # L06 M5 -- word / reading / (definition | glossary)
    return {
        "form": _clean_text(f.get("word")),
        "reading": furigana_reading(f.get("reading")),
        "meaning": _clean_text(f.get("definition")) or _clean_text(f.get("glossary")),
        "pitch": None,
    }


def qa_vocab(f):  # L06 M1 -- Question(ruby) / Answer
    return {
        "form": deruby_base(f.get("Question")),
        "reading": ruby_reading(f.get("Question")),
        "meaning": _clean_text(f.get("Answer")),
        "pitch": None,
    }


def jp_tango(f):  # L02 M1 -- VocabKanji / VocabFurigana / VocabDef
    return {
        "form": _clean_text(f.get("VocabKanji")),
        "reading": furigana_reading(f.get("VocabFurigana")),
        "meaning": _clean_text(f.get("VocabDef")),
        "pitch": f.get("VocabPitchPattern") or None,
    }


def jp_scheduler(f):  # L01 M2 -- Front / Reading / Back (Jitendex glossary)
    return {
        "form": _clean_text(f.get("Front")),
        "reading": furigana_reading(f.get("Reading")),
        "meaning": _clean_text(f.get("Back")),
        "pitch": None,
    }


def nl_vocab(f):  # L01 M3 -- Dutch / (alphabetic) / Glossary (French glosses)
    return {
        "form": _clean_text(f.get("Dutch")),
        "reading": None,
        "meaning": _clean_text(f.get("Glossary")),
        "pitch": None,
    }


# --------------------------------------------------------------------------- #
# Submissions  (model_extractors keyed by str(release model_id))               #
# "Mk" in comments is the k-th note type of that learner in note_types.jsonl.  #
# --------------------------------------------------------------------------- #

SUBMISSIONS = [
    # L01 -- fr/en; ja Unsure / nl Unsure
    Submission(
        learner_id="L01",
        model_extractors={
            "3": jp_scheduler,  # M2 ja mining vocab (MWEs filtered)
            "4": nl_vocab,  # M3 nl vocab
            # dropped: M1 kanji-reading aggregation, M4 radicals, M5 grammar Q&A,
            # M6 kana, M7 single-kanji study, M8 nl sentences, M9 ja sentences,
            # M10 kana mnemonics
        },
    ),
    # L02 -- hi/en; ja Unsure
    Submission(
        learner_id="L02",
        model_extractors={
            "2": jp_tango,  # M1 JLPT Tango vocab
            # dropped: M2 katakana
        },
    ),
    # L03 -- en; ja B2
    Submission(
        learner_id="L03",
        model_extractors={
            "1": jp_media,  # M1 mined vocab (no reading field)
        },
    ),
    # L04 -- id native; ja B2 / en C1
    Submission(
        learner_id="L04",
        model_extractors={
            "2": jp_expression,  # M1 vocab
            "1": jp_expression,  # M2 vocab (Core-style)
        },
    ),
    # L05 -- en/fr; ja B1
    Submission(
        learner_id="L05",
        model_extractors={
            "5": kaishi_vocab,  # M4 Kaishi vocab
            "1": one_k_vocab,  # M5 1k-merged vocab
            "3": yomitan_cap,  # M6 Yomitan vocab
            # dropped: M1/M2 empty, M3 single-kanji study (RTK/Kodansha/WaniKani)
        },
    ),
    # L06 -- en; ja A1
    Submission(
        learner_id="L06",
        model_extractors={
            "1": qa_vocab,  # M1 Q/A vocab (ruby)
            "6": kaishi_vocab,  # M4 Kaishi vocab (same deck template as L05)
            "5": yomitan_lc,  # M5 Yomitan vocab
            # dropped: M2/M3 kana, M6 kanji components, M7/M8 empty,
            # M9 radicals, M10 single-kanji study
        },
    ),
    # L07 -- zh-Hant/en; it A2
    Submission(
        learner_id="L07",
        model_extractors={
            "3": it_front_back,  # M2 AnkiMorphs vocab
            # dropped: M1 tense grammar, M3 sentence pairs, M4 function-word table,
            # M5 occlusion, M6 grammar Q&A, M7 particle sentences, M8 conjugation,
            # M9 cloze
        },
    ),
    # L08 -- en; es B1 / pt A1
    Submission(
        learner_id="L08",
        model_extractors={
            "1": front_back,  # M1 es vocab
            "2": front_back,  # M3 es/pt vocab (MWEs filtered)
            # dropped: M2 cloze
        },
    ),
]

# --------------------------------------------------------------------------- #
# Policy
# --------------------------------------------------------------------------- #

# fugashi's short-unit segmentation splits compound nouns (展覧会 -> 展覧+会) and
# kana-spelled words (がくせい -> がく+せい), so a token-count rule over-filters
# real vocabulary. Instead: reject on a case/binding particle (助詞), whitespace,
# or sentence punctuation - the marks of a phrase/sentence - and otherwise
# accept. This keeps compounds while dropping "お腹が空いた" (has 助詞 が) and
# multi-token sentences. Minor known leak: set phrases whose only glue is a
# 助動詞 (e.g. "お大事に") are kept.

import fugashi

_TAGGER = fugashi.Tagger()
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uf900-\ufaff]")
_JA_STOP = re.compile(r"[\s\u3000。、！？!?…‥「」『』（）()｛｝【】〜~・,.]")
_ALPHA_STOP = re.compile(r"[。、！？…「」]|[.!?]$")


def should_keep_lexeme(form: str, lang: str | None) -> bool:
    if not form:
        return False
    form = form.strip()
    if _CJK.search(form):  # Japanese (covers ja and ja-"Unsure")
        if _JA_STOP.search(form) or len(form) > 15:
            return False
        return not any(m.feature.pos1 == "助詞" for m in _TAGGER(form))
    if _ALPHA_STOP.search(form):  # alphabetic: nl / it / es / pt / en
        return False
    return len(form.split()) <= 1


POLICY = Policy(should_keep_lexeme=should_keep_lexeme)
