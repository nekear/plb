import numpy as np
import pandas as pd
import pytest

from plb.model_b.core.features import (
    add_intralexical_features,
    load_tubelex_frequencies,
)

LEXEMES = pd.DataFrame({"lexeme_id": ["lex-1", "lex-2"], "form": ["猫", "犬"]})
TUBELEX = pd.DataFrame({"form": ["猫"], "count": [999]})
KVL = pd.DataFrame({"word_instance_id": ["wi-1", "wi-2"], "kvl_score": [3.5, 2.0]})


def _frame(rows: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "word_instance_id": [r[0] for r in rows],
            "lexeme_id": [r[1] for r in rows],
        }
    )


def test_kvl_scores_are_merged_by_word_instance_id() -> None:
    result = add_intralexical_features(
        _frame([("wi-1", "lex-1"), ("wi-2", "lex-2")]),
        lexemes=LEXEMES,
        tubelex=TUBELEX,
        kvl=KVL,
    )

    assert result["kvl_score"].tolist() == [3.5, 2.0]


def test_a_row_without_a_kvl_prediction_raises_with_its_id() -> None:
    with pytest.raises(ValueError, match=r"1 rows have no entry in KVL.*wi-9"):
        add_intralexical_features(
            _frame([("wi-9", "lex-1")]), lexemes=LEXEMES, tubelex=TUBELEX, kvl=KVL
        )


def test_a_row_whose_lexeme_has_no_form_raises() -> None:
    with pytest.raises(ValueError, match=r"1 rows have no entry in lexemes.*lex-9"):
        add_intralexical_features(
            _frame([("wi-1", "lex-9")]), lexemes=LEXEMES, tubelex=TUBELEX, kvl=KVL
        )


def test_an_oov_form_gets_the_flag_and_zero_log_frequency() -> None:
    result = add_intralexical_features(
        _frame([("wi-2", "lex-2")]),  # 犬 is not in TUBELEX
        lexemes=LEXEMES,
        tubelex=TUBELEX,
        kvl=KVL,
    )

    assert result["is_oov"].tolist() == [1]
    assert result["log_freq"].tolist() == [0.0]


def test_a_known_form_gets_log10_of_count_plus_one() -> None:
    result = add_intralexical_features(
        _frame([("wi-1", "lex-1")]), lexemes=LEXEMES, tubelex=TUBELEX, kvl=KVL
    )

    assert result["is_oov"].tolist() == [0]
    assert result["log_freq"].tolist() == [np.log10(1000)]  # count 999 + 1


def test_duplicate_tubelex_forms_are_summed_before_merging(tmp_path) -> None:
    path = tmp_path / "tubelex.tsv"
    path.write_text("word\tcount\tpos\n猫\t10\tnoun\n猫\t5\tverb\n 犬 \t7\tnoun\n")

    frequencies = load_tubelex_frequencies(path)

    by_form = dict(zip(frequencies["form"], frequencies["count"]))
    assert by_form == {"猫": 15, "犬": 7}  # whitespace stripped, counts summed


def test_negative_tubelex_counts_raise(tmp_path) -> None:
    path = tmp_path / "tubelex.tsv"
    path.write_text("word\tcount\n猫\t-1\n")

    with pytest.raises(ValueError, match="non-negative"):
        load_tubelex_frequencies(path)


def test_a_frame_without_lexeme_id_resolves_it_via_the_word_instance_index() -> None:
    index = pd.DataFrame({"word_instance_id": ["wi-1"], "lexeme_id": ["lex-1"]})

    result = add_intralexical_features(
        pd.DataFrame({"word_instance_id": ["wi-1"]}),
        lexemes=LEXEMES,
        tubelex=TUBELEX,
        kvl=KVL,
        word_instances=index,
    )

    assert result["kvl_score"].tolist() == [3.5]
    assert result["is_oov"].tolist() == [0]


def test_a_frame_without_lexeme_id_and_without_an_index_raises() -> None:
    with pytest.raises(ValueError, match="pass word_instances"):
        add_intralexical_features(
            pd.DataFrame({"word_instance_id": ["wi-1"]}),
            lexemes=LEXEMES,
            tubelex=TUBELEX,
            kvl=KVL,
        )
