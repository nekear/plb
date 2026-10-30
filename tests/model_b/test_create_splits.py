import pandas as pd
import pytest

from plb.cohort import Cohort
from plb.learners import get_user_id
from plb.model_b.config import SplitsConfig
from plb.model_b.preparation.create_splits import (
    LEXEME_ID_COLUMN,
    SHARED_TEST_COLUMN,
    SPLIT_COLUMN,
    TEST,
    TRAIN,
    assign_splits,
    shared_lexeme_pool,
)

COHORT = Cohort(learners=["L01", "L02", "L03", "L06"], train_only=["L06"])


def _dataset(vocab: dict[str, list[str]]) -> pd.DataFrame:
    """One labelled row per (learner code, lexeme)."""
    rows = []
    for code, lexemes in vocab.items():
        for lexeme in lexemes:
            rows.append(
                {
                    "user_id": get_user_id(code),
                    "deck_id": f"deck-{code}",
                    LEXEME_ID_COLUMN: lexeme,
                    "word_instance_id": f"{code}-{lexeme}",
                    "language": "ja",
                    "plb_14": 3,
                    "log_plb_14": 1.0986,
                    "first_reviewed_at": pd.Timestamp("2025-01-01", tz="UTC"),
                    "last_reviewed_at": pd.Timestamp("2025-02-01", tz="UTC"),
                    "plb_14_achieved_at": pd.Timestamp("2025-01-15", tz="UTC"),
                    SHARED_TEST_COLUMN: False,
                }
            )
    return pd.DataFrame(rows)


def _vocab(shared: int = 20, unique: int = 60) -> dict[str, list[str]]:
    common = [f"common-{i}" for i in range(shared)]
    return {
        code: common + [f"{code}-only-{i}" for i in range(unique)]
        for code in ("L01", "L02", "L03", "L06")
    }


def _config(**overrides) -> SplitsConfig:
    return SplitsConfig(
        **{
            "shared_test_lexemes": 10,
            "test_fraction": 0.2,
            "eval_fraction": 0.2,
            "seed": 7,
            **overrides,
        }
    )


def test_pool_ignores_train_only_learners() -> None:
    vocab = _vocab(shared=20, unique=5)
    # L06 knows only a fraction of the common words; as a train-only learner it must
    # not be able to shrink the pool the others share.
    vocab["L06"] = vocab["L06"][:3]

    pool = shared_lexeme_pool(_dataset(vocab), COHORT.test_eligible_user_ids())

    assert len(pool) == 20


def test_shared_test_lexemes_are_identical_across_learners() -> None:
    frame = _dataset(_vocab())
    pool = shared_lexeme_pool(frame, COHORT.test_eligible_user_ids())
    dataset = assign_splits(frame, pool, _config(), COHORT)

    test_rows = dataset[dataset[SPLIT_COLUMN] == TEST]
    shared = test_rows[test_rows[SHARED_TEST_COLUMN]]
    by_learner = shared.groupby("user_id")[LEXEME_ID_COLUMN].apply(frozenset)

    assert by_learner.nunique() == 1
    assert len(by_learner.iloc[0]) == 10


def test_train_only_learner_stays_in_train_even_on_shared_lexemes() -> None:
    frame = _dataset(_vocab())
    pool = shared_lexeme_pool(frame, COHORT.test_eligible_user_ids())
    dataset = assign_splits(frame, pool, _config(), COHORT)

    l06 = dataset[dataset["user_id"] == get_user_id("L06")]

    assert set(l06[SPLIT_COLUMN]) == {TRAIN}
    # It does own shared-core lexemes; they simply stay in its train rows.
    assert l06[LEXEME_ID_COLUMN].str.startswith("common-").any()


def test_fill_never_comes_from_the_shared_pool() -> None:
    frame = _dataset(_vocab())
    pool = shared_lexeme_pool(frame, COHORT.test_eligible_user_ids())
    dataset = assign_splits(frame, pool, _config(), COHORT)

    fill = dataset[(dataset[SPLIT_COLUMN] == TEST) & (~dataset[SHARED_TEST_COLUMN])]

    # Not merely outside the drawn core — outside the pool entirely, so a leftover
    # pool lexeme cannot be one learner's test row and another's train row.
    assert not fill[LEXEME_ID_COLUMN].isin(pool).any()


def test_fractions_are_respected_per_learner() -> None:
    frame = _dataset(_vocab())
    pool = shared_lexeme_pool(frame, COHORT.test_eligible_user_ids())
    dataset = assign_splits(frame, pool, _config(), COHORT)

    for code in ("L01", "L02", "L03"):
        rows = dataset[dataset["user_id"] == get_user_id(code)]
        assert (rows[SPLIT_COLUMN] == TEST).sum() == round(len(rows) * 0.2)
        assert (rows[SPLIT_COLUMN] == "eval").sum() == round(len(rows) * 0.2)


def test_the_seed_alone_determines_the_assignment() -> None:
    frame = _dataset(_vocab())
    pool = shared_lexeme_pool(frame, COHORT.test_eligible_user_ids())

    first = assign_splits(frame, pool, _config(seed=11), COHORT)
    same = assign_splits(frame, pool, _config(seed=11), COHORT)
    other = assign_splits(frame, pool, _config(seed=12), COHORT)

    assert first[SPLIT_COLUMN].equals(same[SPLIT_COLUMN])
    assert not first[SPLIT_COLUMN].equals(other[SPLIT_COLUMN])


def test_a_lexeme_never_lands_in_two_splits_for_one_learner() -> None:
    frame = _dataset(_vocab())
    pool = shared_lexeme_pool(frame, COHORT.test_eligible_user_ids())
    dataset = assign_splits(frame, pool, _config(), COHORT)

    per_learner_lexeme = dataset.groupby(["user_id", LEXEME_ID_COLUMN])[
        SPLIT_COLUMN
    ].nunique()

    assert per_learner_lexeme.max() == 1


def test_too_small_a_pool_names_both_numbers() -> None:
    frame = _dataset(_vocab(shared=5))
    pool = shared_lexeme_pool(frame, COHORT.test_eligible_user_ids())

    with pytest.raises(ValueError, match=r"Only 5 lexemes.*shared_test_lexemes is 10"):
        assign_splits(frame, pool, _config(shared_test_lexemes=10), COHORT)


def test_shared_core_larger_than_the_test_budget_is_rejected() -> None:
    # 20 shared lexemes already exceed a 10% test budget, so no fill is possible and
    # the requested test size cannot be honoured.
    frame = _dataset(_vocab(shared=20, unique=60))
    pool = shared_lexeme_pool(frame, COHORT.test_eligible_user_ids())

    with pytest.raises(ValueError, match="more than the .* its test_fraction allows"):
        assign_splits(
            frame, pool, _config(shared_test_lexemes=20, test_fraction=0.1), COHORT
        )
