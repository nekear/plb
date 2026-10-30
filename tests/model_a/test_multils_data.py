
import pandas as pd
import pytest

from plb.utils import get_dataset_as_path
from plb.model_a.multils.data import (
    derive_expression_type,
    derive_source_word_class,
    derive_word_origin,
    load_multils_items,
    load_original_profiles,
    make_l2_clue,
    mark_target_span,
    original_l1_names,
    primary_l1,
    required_l1_names,
)

DATA_DIR = get_dataset_as_path("raw/multils_japanese")


@pytest.mark.parametrize(
    ("target", "expected"),
    [("掲載した", "掲___"), ("連れ戻す", "連___"), ("2児", "2_")],
)
def test_make_l2_clue(target: str, expected: str) -> None:
    assert make_l2_clue(target) == expected


def test_primary_l1_uses_first_listed_language() -> None:
    assert primary_l1("Basque, Spanish") == "Basque"
    assert primary_l1(" French, English ") == "French"


def test_original_profile_language_distribution() -> None:
    profiles = load_original_profiles(DATA_DIR)
    assert profiles["l1_name"].value_counts().to_dict() == {
        "English": 5,
        "French": 2,
        "Portuguese": 1,
        "Basque": 1,
        "Swedish": 1,
    }
    assert original_l1_names(profiles) == (
        "English",
        "Portuguese",
        "French",
        "Basque",
        "Swedish",
    )


def test_required_languages_add_chinese_only_for_trial() -> None:
    original = ("English", "French")
    assert required_l1_names("test", original) == original
    assert required_l1_names("trial", original) == (*original, "Chinese")


@pytest.mark.parametrize(
    ("base", "origin", "pos", "expected"),
    [
        ("掲載|する|た", "漢|和|和", "動詞-サ変|動詞-非自立可能|助動詞", "kango"),
        ("恩|を|売る", "漢|和|和", "名詞-普通名詞-一般|助詞-格助詞|動詞-一般", "mixed"),
        ("連れ戻す", "和", "動詞-一般-x", "wago"),
        ("ピックアップ|する|て", "外|和|和", "動詞-サ変|動詞-非自立可能|助詞-接続助詞", "gairaigo"),
    ],
)
def test_derive_word_origin_matches_paper_examples(
    base: str, origin: str, pos: str, expected: str
) -> None:
    assert derive_word_origin(base, origin, pos) == expected


def test_expression_type_uses_author_main_pos() -> None:
    assert derive_expression_type("MWE") == "mwe"
    assert derive_expression_type("N") == "single_word"
    assert derive_expression_type(pd.NA) == "single_word"


@pytest.mark.parametrize(
    ("main_pos", "expected"),
    [("PART", "function_word"), ("AUX", "function_word"), ("N", "content_word")],
)
def test_source_word_class_uses_author_main_pos(main_pos: str, expected: str) -> None:
    assert derive_source_word_class(main_pos) == expected


def test_mark_target_span_validates_author_offsets() -> None:
    assert mark_target_span("回答も掲載した。", "掲載した", "3:7") == (
        "回答も<TARGET>掲載した</TARGET>。"
    )
    with pytest.raises(ValueError, match="mismatch"):
        mark_target_span("回答も掲載した。", "掲載した", "0:4")


def test_local_item_counts_and_mwe_counts_match_release() -> None:
    items = load_multils_items(DATA_DIR)
    assert items["id"].nunique() == 600
    assert items.groupby("source_split").size().to_dict() == {"test": 570, "trial": 30}
    assert (
        items[items["expression_type"] == "mwe"].groupby("source_split").size().to_dict()
        == {"test": 40, "trial": 2}
    )
    assert items.groupby("source_split")["source_word_class"].apply(
        lambda values: int((values == "function_word").sum())
    ).to_dict() == {"test": 20, "trial": 1}
    assert items["marked_context"].str.count("<TARGET>").eq(1).all()
