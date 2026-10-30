from uuid import UUID

import pytest
from pydantic import ValidationError

from plb.normalize.extractors import should_keep_lexeme
from plb.normalize.helpers import _to_hiragana
from plb.normalize.process import (
    Policy,
    WordCandidate,
    dedup_candidates,
    process_submission,
)
from plb.normalize.readings import resolve_reading
from plb.normalize.schema import LexemeRow, Submission


def test_to_hiragana_normalizes_width_and_iteration_marks() -> None:
    assert _to_hiragana("ﾊﾟｰﾃｨｰ") == "ぱーてぃー"
    assert _to_hiragana("クヽヾ") == "くゝゞ"
    assert _to_hiragana(None) is None


def test_original_japanese_reading_is_canonicalized() -> None:
    assert resolve_reading("復帰", "フッキ", "ja") == ("ふっき", "original")


@pytest.mark.parametrize(
    ("form", "expected"),
    [
        ("アミュプラザ", "あみゅぷらざ"),
        ("パーティーピーポー", "ぱーてぃーぴーぽー"),
        ("復帰", "ふっき"),
    ],
)
def test_missing_japanese_reading_is_derived(form: str, expected: str) -> None:
    assert resolve_reading(form, None, "ja") == (expected, "derived")


@pytest.mark.parametrize("form", ["なん／なに", "コロコロ1", "サクサク★", "１０日"])
def test_unresolvable_japanese_form_has_no_reading(form: str) -> None:
    assert resolve_reading(form, None, "ja") == (None, None)


def test_process_submission_keeps_unresolved_japanese_lexeme() -> None:
    submission = Submission(
        learner_id="L99",
        model_extractors={
            "1": lambda _: {
                "form": "１０日",
                "meaning": "ten days",
                "reading": None,
                "pitch": None,
            }
        },
    )
    raw = {
        "decks": [
            {
                "scheduler": "FSRS",
                "deck_id": 1,
                "target_language": "ja",
                "fsrs_weights": [],
                "desired_retention": 0.9,
            }
        ],
        "models": [{"model_id": 1, "field_names": []}],
        "notes": [{"note_id": 1, "model_id": 1, "fields": [], "created_at_ms": 0}],
        "cards": [
            {
                "note_id": 1,
                "card_id": 1,
                "deck_id": 1,
                "fsrs_stability": 1.0,
                "reps_total": 1,
                "lapses_total": 0,
            }
        ],
        "reviews": [],
    }

    result = process_submission(
        submission,
        raw,
        Policy(should_keep_lexeme=lambda form, language: True),
    )

    assert len(result.candidates) == 1
    assert result.candidates[0].reading is None
    assert result.candidates[0].reading_source is None


def test_non_japanese_reading_is_not_transliterated_or_derived() -> None:
    assert resolve_reading("casa", "KASA", "es") == ("KASA", "original")
    assert resolve_reading("casa", None, "es") == (None, None)


def test_policy_uses_renamed_callable() -> None:
    assert should_keep_lexeme("復帰", "ja")


def test_lexeme_reading_and_source_must_be_set_together() -> None:
    with pytest.raises(ValidationError):
        LexemeRow(
            lexeme_id=UUID(int=1),
            language="ja",
            form="復帰",
            reading="ふっき",
        )


def test_original_source_wins_during_candidate_deduplication() -> None:
    common = {
        "user_id": UUID(int=1),
        "deck_id": UUID(int=2),
        "target_language": "ja",
        "form": "復帰",
        "meaning": "return",
        "reading": "ふっき",
        "pitch": None,
        "created_at_ms": 0,
        "note_id": 1,
        "reviews": [],
        "first_reviewed_at_ms": None,
    }
    original = WordCandidate(
        **common,
        reading_source="original",
        card_id=1,
        select_score=0.0,
    )
    derived = WordCandidate(
        **common,
        reading_source="derived",
        card_id=2,
        select_score=1.0,
    )

    result = dedup_candidates([original, derived])

    assert len(result) == 1
    assert result[0].card_id == 2
    assert result[0].reading_source == "original"
