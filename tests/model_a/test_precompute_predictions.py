import pandas as pd
import pytest

from plb.model_a.multils.prompts import MultiLSCondition, make_multils_prompt
from plb.model_a.precompute.precompute_predictions import (
    build_prompt_frame,
    language_name,
    make_deploy_prompt,
    primary_native_language,
)


def make_frames(
    *,
    target_language: str = "ja",
    native_languages: list[str] | None = None,
    lexeme_id: str = "lex-1",
    deck_id: str = "deck-1",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    word_instances = pd.DataFrame(
        [
            {
                "word_instance_id": "wi-1",
                "lexeme_id": lexeme_id,
                "deck_id": deck_id,
                "meaning": "movie",
            },
            {
                "word_instance_id": "wi-2",
                "lexeme_id": "lex-2",
                "deck_id": "deck-1",
                "meaning": "music",
            },
        ]
    )
    lexemes = pd.DataFrame(
        [
            {"lexeme_id": "lex-1", "form": "映画"},
            {"lexeme_id": "lex-2", "form": "音楽"},
        ]
    )
    decks = pd.DataFrame(
        [{"deck_id": "deck-1", "user_id": "user-1", "target_language": target_language}]
    )
    users = pd.DataFrame(
        [
            {
                "user_id": "user-1",
                "native_languages": (
                    ["fr", "en"] if native_languages is None else native_languages
                ),
            }
        ]
    )
    return word_instances, lexemes, decks, users


def test_japanese_prompt_matches_the_evaluated_full_l2_deploy_prompt() -> None:
    """The precomputed scores are only comparable with the reported MultiLS results
    while this module's prompt and the evaluated condition agree character for
    character."""
    kwargs = {
        "target": "映画",
        "l1_name": "French",
        "l1_gloss": "movie",
    }
    assert make_deploy_prompt(l2_name="Japanese", **kwargs) == make_multils_prompt(
        MultiLSCondition.FULL_L2_DEPLOY, **kwargs
    )


def test_prompt_substitutes_a_non_japanese_target_language() -> None:
    prompt = make_deploy_prompt(
        l2_name="Italian",
        target="amare",
        l1_name="English",
        l1_gloss="to love",
    )
    assert "Italian word: amare" in prompt
    assert "English word: to love" in prompt
    assert "Japanese" not in prompt


def test_language_name_rejects_unmapped_codes() -> None:
    assert language_name("ja") == "Japanese"
    with pytest.raises(ValueError, match="No language name mapping"):
        language_name("xx")


def test_primary_native_language_takes_the_first_l1() -> None:
    assert primary_native_language(["fr", "en"]) == "fr"
    with pytest.raises(ValueError, match="no native language"):
        primary_native_language([])


def test_build_prompt_frame_conditions_on_the_first_native_language() -> None:
    frame = build_prompt_frame(*make_frames())

    assert frame["word_instance_id"].tolist() == ["wi-1", "wi-2"]
    assert frame.columns.tolist() == ["word_instance_id", "text"]
    first = frame.iloc[0]["text"]
    assert "Japanese word: 映画" in first
    assert "French word: movie" in first
    assert "English" not in first


def test_build_prompt_frame_uses_the_deck_target_language() -> None:
    frame = build_prompt_frame(*make_frames(target_language="it"))
    assert "Italian word: 映画" in frame.iloc[0]["text"]


def test_build_prompt_frame_rejects_unmapped_language_codes() -> None:
    with pytest.raises(ValueError, match=r"\['xx'\]"):
        build_prompt_frame(*make_frames(target_language="xx"))

    with pytest.raises(ValueError, match=r"\['qq'\]"):
        build_prompt_frame(*make_frames(native_languages=["qq"]))


def test_build_prompt_frame_rejects_learners_without_a_native_language() -> None:
    with pytest.raises(ValueError, match="no native language"):
        build_prompt_frame(*make_frames(native_languages=[]))


def test_build_prompt_frame_rejects_missing_lexemes_and_decks() -> None:
    with pytest.raises(ValueError, match="missing from lexemes"):
        build_prompt_frame(*make_frames(lexeme_id="lex-absent"))

    with pytest.raises(ValueError, match="missing from decks"):
        build_prompt_frame(*make_frames(deck_id="deck-absent"))


def test_build_prompt_frame_rejects_missing_columns() -> None:
    word_instances, lexemes, decks, users = make_frames()
    with pytest.raises(ValueError, match="lexemes is missing required columns"):
        build_prompt_frame(word_instances, lexemes.drop(columns="form"), decks, users)
