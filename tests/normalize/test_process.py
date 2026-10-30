"""The study analyses Japanese only, so normalization drops every other target
language. These tests pin that the cut happens at the deck level and cascades to
notes, lexemes, word instances and reviews."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from plb.learners import LEARNER_CODE_BY_USER_ID
from plb.normalize.process import Policy, Submission, build_dataset, process_submission

A_LEARNER_ID = next(iter(LEARNER_CODE_BY_USER_ID))
UNMAPPED_USER_ID = "L99"

# Keeps every form, so these tests exercise the language filter alone rather than
# the phrase/sentence heuristics of the real policy.
KEEP_EVERYTHING = Policy(should_keep_lexeme=lambda form, lang: True)


def _extractor(fields: dict) -> dict:
    return {
        "form": fields.get("Word"),
        "reading": fields.get("Reading"),
        "meaning": fields.get("Meaning"),
        "pitch": None,
    }


def _deck(deck_id: int, target_language: str) -> dict:
    return {
        "deck_id": deck_id,
        "scheduler": "FSRS",
        "target_language": target_language,
        "fsrs_weights": [0.1, 0.2],
        "desired_retention": 0.9,
    }


def _raw(decks: list[dict]) -> dict:
    """A submission carrying one note per deck, each with one reviewed card."""
    notes = []
    cards = []
    reviews = []
    for index, deck in enumerate(decks):
        note_id = 10 + index
        card_id = 100 + index
        notes.append(
            {
                "note_id": note_id,
                "model_id": "m1",
                "fields": [f"form-{deck['target_language']}", "かな", "meaning"],
                "created_at_ms": 1_700_000_000_000,
            }
        )
        cards.append(
            {
                "card_id": card_id,
                "note_id": note_id,
                "deck_id": deck["deck_id"],
                "fsrs_stability": 5.0,
                "reps_total": 2,
                "lapses_total": 0,
            }
        )
        reviews.append(
            {
                "card_id": card_id,
                "timestamp_ms": 1_700_000_100_000,
                "rating": 3,
                "interval_before": 0,
                "interval_after": 1,
                "factor_after": 2500,
                "time_taken_ms": 1000,
                "review_type": 0,
            }
        )
    return {
        "consent": {"publish_userinfo": True},
        "user_profile": {
            "native_languages": ["en"],
            "target_languages": [{"lang": "ja", "proficiency": "B2"}],
            "domain_interests": ["technology"],
        },
        "decks": decks,
        "models": [{"model_id": "m1", "field_names": ["Word", "Reading", "Meaning"]}],
        "notes": notes,
        "cards": cards,
        "reviews": reviews,
    }


def _submission(user_id: str = A_LEARNER_ID) -> Submission:
    return Submission(learner_id=user_id, model_extractors={"m1": _extractor})


def _run(decks: list[dict], policy: Policy = KEEP_EVERYTHING):
    return process_submission(_submission(), _raw(decks), policy)


def test_non_japanese_decks_and_their_notes_are_dropped() -> None:
    result = _run([_deck(1, "ja"), _deck(2, "it")])

    assert [deck.target_language for deck in result.decks] == ["ja"]
    assert [candidate.target_language for candidate in result.candidates] == ["ja"]
    # The Italian note's card lives in a filtered deck, so it carries no reviews
    # into the dataset either.
    assert [candidate.form for candidate in result.candidates] == ["form-ja"]


def test_jp_alias_survives_because_the_filter_runs_after_the_compatibility_remap() -> None:
    result = _run([_deck(1, "jp")])

    assert [deck.target_language for deck in result.decks] == ["ja"]
    assert len(result.candidates) == 1


def test_submission_without_japanese_decks_yields_nothing() -> None:
    result = _run([_deck(1, "it"), _deck(2, "es")])

    assert result.decks == []
    assert result.candidates == []


def test_kept_languages_are_policy_driven() -> None:
    decks = [_deck(1, "ja"), _deck(2, "it")]

    italian_only = _run(decks, Policy(should_keep_lexeme=lambda f, l: True, keep_languages=["it"]))
    assert [deck.target_language for deck in italian_only.decks] == ["it"]

    both = _run(
        decks, Policy(should_keep_lexeme=lambda f, l: True, keep_languages=["ja", "it"])
    )
    assert [deck.target_language for deck in both.decks] == ["ja", "it"]


def _write_release(release_dir: Path, decks_by_learner: dict[str, list[dict]]) -> list[Submission]:
    """Writes the given learners in the public release's table layout."""
    release_dir.mkdir()
    tables: dict[str, list[dict]] = {
        "learners": [], "decks": [], "note_types": [], "notes": [], "cards": [], "reviews": []
    }
    for learner_id, decks in decks_by_learner.items():
        raw = _raw(decks)
        tables["learners"].append({"learner_id": learner_id, **raw["user_profile"]})
        for table, key in (
            ("decks", "decks"), ("note_types", "models"), ("notes", "notes"),
            ("cards", "cards"), ("reviews", "reviews"),
        ):
            tables[table].extend({"learner_id": learner_id, **row} for row in raw[key])
    for table, rows in tables.items():
        (release_dir / f"{table}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    return [_submission(learner_id) for learner_id in decks_by_learner]


def test_build_dataset_writes_only_learners_who_kept_a_deck(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    release_dir = tmp_path / "release"
    submissions = _write_release(
        release_dir,
        {
            A_LEARNER_ID: [_deck(1, "ja")],
            # An entirely non-Japanese learner: filtered away, and so never written
            # as a user row that owns nothing.
            UNMAPPED_USER_ID: [_deck(2, "it")],
        },
    )

    counts = build_dataset(submissions, release_dir, output_dir, policy=KEEP_EVERYTHING)

    assert counts["users"] == 1
    assert counts["decks"] == 1
    assert counts["failed_submissions"] == 0
    written = [json.loads(line) for line in (output_dir / "users.jsonl").read_text().splitlines()]
    assert [row["user_id"] for row in written] == [A_LEARNER_ID]


def test_build_dataset_rejects_a_surviving_learner_without_a_code(tmp_path: Path) -> None:
    release_dir = tmp_path / "release"
    submissions = _write_release(release_dir, {UNMAPPED_USER_ID: [_deck(1, "ja")]})

    with pytest.raises(ValueError, match="Unmapped learner user_id"):
        build_dataset(submissions, release_dir, tmp_path / "out", policy=KEEP_EVERYTHING)
