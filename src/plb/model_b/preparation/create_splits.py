"""Build the Model B train/eval/test splits over the study cohort.

Each test set is two scopes at once. A core of lexemes that *every* test-eligible
learner has studied is held out for all of them, so cross-learner performance is read
on identical items rather than on whatever vocabulary each learner happens to own. The
rest of each test set is filled from that learner's own words, so the metric also says
something about their idiosyncratic vocabulary.

Assignment is a seeded random draw rather than chronological. A chronological split
cannot produce a shared core — measured on this data, zero lexemes land in all three
learners' test sets — and the neighbour features that could otherwise leak across a
random boundary are anonymous aggregates with no item identity to exploit.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from plb.cohort import Cohort, load_cohort
from plb.model_b.config import SplitsConfig, load_model_b_config
from plb.utils import get_dataset_as_path

TARGET_COLUMN = "plb_14"
LOG_TARGET_COLUMN = "log_plb_14"
LEXEME_ID_COLUMN = "lexeme_id"
SPLIT_COLUMN = "split"
SHARED_TEST_COLUMN = "is_shared_test"

TRAIN, EVAL, TEST = "train", "eval", "test"

OUTPUT_COLUMNS = [
    "user_id",
    "deck_id",
    LEXEME_ID_COLUMN,
    "word_instance_id",
    "language",
    TARGET_COLUMN,
    LOG_TARGET_COLUMN,
    "first_reviewed_at",
    "last_reviewed_at",
    "plb_14_achieved_at",
    SHARED_TEST_COLUMN,
]

OUTPUT_PATHS = {
    split: get_dataset_as_path(f"processed/model_b/{split}_word_instances.jsonl")
    for split in (TRAIN, EVAL, TEST)
}


def _require_columns(
    dataframe: pd.DataFrame,
    required: Iterable[str],
    dataset_name: str,
) -> None:
    # Computing the difference between two sets which will keep items from the first set which are not in the second set
    missing = sorted(set(required) - set(dataframe.columns))
    if missing:
        raise ValueError(f"{dataset_name} is missing required columns: {missing}")


def _require_unique(
    dataframe: pd.DataFrame,
    column: str,
    dataset_name: str,
) -> None:
    duplicates = dataframe.loc[
        # Returns `True` for *all* duplicate values. By the default `keep="first"` and shows only the second and later occurrences
        dataframe[column].duplicated(keep=False),
        column,
    ]
    if not duplicates.empty:
        examples = duplicates.astype(str).drop_duplicates().head(5).tolist()
        raise ValueError(
            f"{dataset_name}.{column} must be unique; examples: {examples}"
        )


def _parse_required_timestamps(
    dataframe: pd.DataFrame,
    column: str,
) -> pd.Series:
    parsed = pd.to_datetime(dataframe[column], utc=True, errors="coerce")
    missing = parsed.isna()
    if missing.any():
        examples = (
            dataframe.loc[missing, "word_instance_id"].astype(str).head(5).tolist()
        )
        raise ValueError(
            f"Eligible word instances have invalid {column} values; "
            f"examples: {examples}"
        )
    return parsed


def build_split_dataset(
    decks_df: pd.DataFrame,
    lexemes_df: pd.DataFrame,
    word_instances_df: pd.DataFrame,
    cohort: Cohort,
) -> pd.DataFrame:
    """Join IDs, retain the cohort's labelled rows, and derive the log target."""
    _require_columns(decks_df, {"deck_id", "user_id"}, "decks")
    _require_columns(lexemes_df, {"lexeme_id", "language"}, "lexemes")
    _require_columns(
        word_instances_df,
        {
            "word_instance_id",
            LEXEME_ID_COLUMN,
            "deck_id",
            TARGET_COLUMN,
            "first_reviewed_at",
            "last_reviewed_at",
            "plb_14_achieved_at",
        },
        "word instances",
    )
    _require_unique(decks_df, "deck_id", "decks")
    _require_unique(lexemes_df, LEXEME_ID_COLUMN, "lexemes")
    _require_unique(word_instances_df, "word_instance_id", "word instances")

    words = word_instances_df[
        [
            "word_instance_id",
            LEXEME_ID_COLUMN,
            "deck_id",
            TARGET_COLUMN,
            "first_reviewed_at",
            "last_reviewed_at",
            "plb_14_achieved_at",
        ]
    ].merge(
        lexemes_df[[LEXEME_ID_COLUMN, "language"]],
        on=LEXEME_ID_COLUMN,
        how="left",
        validate="many_to_one",
        indicator="_lexeme_merge",
    )
    missing_lexemes = words["_lexeme_merge"] != "both"
    if missing_lexemes.any():
        examples = (
            words.loc[missing_lexemes, LEXEME_ID_COLUMN]
            .astype(str)
            .drop_duplicates()
            .head(5)
            .tolist()
        )
        raise ValueError(
            f"Word instances reference missing lexemes; examples: {examples}"
        )

    words = words.drop(columns="_lexeme_merge").merge(
        decks_df[["deck_id", "user_id"]],
        on="deck_id",
        how="left",
        validate="many_to_one",
        indicator="_deck_merge",
    )
    missing_decks = (words["_deck_merge"] != "both") | words["user_id"].isna()
    if missing_decks.any():
        examples = (
            words.loc[missing_decks, "deck_id"]
            .astype(str)
            .drop_duplicates()
            .head(5)
            .tolist()
        )
        raise ValueError(
            f"Word instances reference missing deck users; examples: {examples}"
        )
    words = words.drop(columns="_deck_merge")

    # Normalization emits Japanese only, so eligibility is just "has a burden label".
    targets = pd.to_numeric(words[TARGET_COLUMN], errors="raise")
    eligible = targets.notna() & words["user_id"].astype(str).isin(cohort.user_ids())
    words = words.loc[eligible].copy()
    targets = targets.loc[eligible]
    if words.empty:
        raise ValueError(
            "No labelled word instances remain for the configured cohort; check "
            "cohort.learners in params.yaml."
        )

    target_values = targets.to_numpy(dtype=np.float64)
    invalid_targets = (
        ~np.isfinite(target_values)
        | (target_values < 1.0)
        | (target_values != np.floor(target_values))
    )
    if invalid_targets.any():
        examples = (
            words.loc[
                words.index[invalid_targets],
                ["word_instance_id", TARGET_COLUMN],
            ]
            .head(5)
            .to_dict(orient="records")
        )
        raise ValueError(
            f"{TARGET_COLUMN} must contain positive integers; examples: {examples}"
        )

    words[TARGET_COLUMN] = targets.astype("int64")
    words[LOG_TARGET_COLUMN] = np.log(words[TARGET_COLUMN].astype(np.float64))
    words["first_reviewed_at"] = _parse_required_timestamps(
        words,
        "first_reviewed_at",
    )
    words["last_reviewed_at"] = _parse_required_timestamps(
        words,
        "last_reviewed_at",
    )
    # Eligible rows all have a burden label, so the label-availability timestamp
    # must exist too; _parse_required_timestamps raises on any null.
    words["plb_14_achieved_at"] = _parse_required_timestamps(
        words,
        "plb_14_achieved_at",
    )

    chronology_violations = words["last_reviewed_at"] < words["first_reviewed_at"]
    if chronology_violations.any():
        examples = (
            words.loc[chronology_violations, "word_instance_id"]
            .astype(str)
            .head(5)
            .tolist()
        )
        raise ValueError(
            "last_reviewed_at precedes first_reviewed_at for eligible word "
            f"instances; examples: {examples}"
        )

    words[SHARED_TEST_COLUMN] = False
    return (
        words[OUTPUT_COLUMNS]
        .sort_values(
            ["first_reviewed_at", LEXEME_ID_COLUMN, "word_instance_id"],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def shared_lexeme_pool(
    dataset: pd.DataFrame,
    test_eligible_user_ids: set[str],
) -> set[str]:
    """Lexemes with a labelled row for *every* test-eligible learner.

    Train-only learners are excluded from the intersection on purpose: they never
    contribute a test row, so requiring them here would only shrink the pool.
    """
    users = dataset["user_id"].astype(str)
    eligible = dataset.loc[users.isin(test_eligible_user_ids)]
    per_learner = [
        set(group[LEXEME_ID_COLUMN].astype(str))
        for _, group in eligible.groupby(users[eligible.index], sort=True)
    ]
    if len(per_learner) != len(test_eligible_user_ids):
        missing = len(test_eligible_user_ids) - len(per_learner)
        raise ValueError(
            f"{missing} test-eligible learner(s) have no labelled rows, so no shared "
            "lexeme pool can be formed."
        )
    return set.intersection(*per_learner)


def assign_splits(
    dataset: pd.DataFrame,
    pool: set[str],
    config: SplitsConfig,
    cohort: Cohort,
) -> pd.DataFrame:
    """Label every row `train`, `eval` or `test`.

    The shared core is drawn once and held out for every test-eligible learner, so
    their test sets contain identical lexemes. Fill and eval are then drawn per
    learner from what remains, with fill restricted to lexemes outside the pool so the
    two test scopes stay cleanly separable.
    """
    _require_columns(dataset, OUTPUT_COLUMNS, "split dataset")
    if dataset.empty:
        raise ValueError("Cannot split an empty dataset.")
    if len(pool) < config.shared_test_lexemes:
        raise ValueError(
            f"Only {len(pool):,} lexemes are studied by every test-eligible learner, "
            f"but shared_test_lexemes is {config.shared_test_lexemes:,}. Lower it, or "
            "widen the cohort."
        )

    generator = np.random.default_rng(config.seed)
    shared_core = set(
        generator.choice(
            sorted(pool),  # sorted so the seed alone determines the draw
            size=config.shared_test_lexemes,
            replace=False,
        )
    )

    dataset = dataset.copy()
    dataset[SPLIT_COLUMN] = TRAIN
    dataset[SHARED_TEST_COLUMN] = False
    lexemes = dataset[LEXEME_ID_COLUMN].astype(str)
    users = dataset["user_id"].astype(str)
    train_only = cohort.train_only_user_ids()

    for user_id, positions in dataset.groupby(users, sort=True).groups.items():
        if user_id in train_only:
            continue

        rows = dataset.loc[positions]
        in_core = lexemes.loc[positions].isin(shared_core)
        dataset.loc[positions[in_core], SPLIT_COLUMN] = TEST
        dataset.loc[positions[in_core], SHARED_TEST_COLUMN] = True

        # Fill from outside the pool entirely, not merely outside the drawn core, so
        # a leftover pool lexeme never lands in one learner's test and another's train.
        test_budget = round(len(rows) * config.test_fraction)
        fill_count = test_budget - int(in_core.sum())
        if fill_count < 0:
            raise ValueError(
                f"The shared core gives learner {user_id} {int(in_core.sum()):,} test "
                f"rows, more than the {test_budget:,} its test_fraction allows. Lower "
                "shared_test_lexemes or raise test_fraction."
            )
        outside_pool = positions[(~in_core) & (~lexemes.loc[positions].isin(pool))]
        if fill_count > len(outside_pool):
            raise ValueError(
                f"Learner {user_id} needs {fill_count:,} fill rows but only "
                f"{len(outside_pool):,} of its rows lie outside the shared pool."
            )
        if fill_count > 0:
            chosen = generator.choice(outside_pool, size=fill_count, replace=False)
            dataset.loc[chosen, SPLIT_COLUMN] = TEST

        remaining = positions[dataset.loc[positions, SPLIT_COLUMN] == TRAIN]
        eval_count = round(len(rows) * config.eval_fraction)
        if eval_count > len(remaining):
            raise ValueError(
                f"Learner {user_id} needs {eval_count:,} eval rows but only "
                f"{len(remaining):,} remain after the test split."
            )
        if eval_count > 0:
            dataset.loc[
                generator.choice(remaining, size=eval_count, replace=False),
                SPLIT_COLUMN,
            ] = EVAL

    _validate_splits(dataset, config, cohort)
    return dataset


def _validate_splits(
    dataset: pd.DataFrame,
    config: SplitsConfig,
    cohort: Cohort,
) -> None:
    users = dataset["user_id"].astype(str)
    lexemes = dataset[LEXEME_ID_COLUMN].astype(str)
    train_only = cohort.train_only_user_ids()

    shared_sets = {
        user_id: frozenset(lexemes[(users == user_id) & dataset[SHARED_TEST_COLUMN]])
        for user_id in sorted(set(users) - train_only)
    }
    distinct = set(shared_sets.values())
    if len(distinct) != 1:
        raise AssertionError(
            "Shared test lexemes differ between learners, so cross-learner "
            f"comparison would not be on identical items; sizes: "
            f"{ {user: len(items) for user, items in shared_sets.items()} }"
        )
    shared_size = len(next(iter(distinct)))
    if shared_size != config.shared_test_lexemes:
        raise AssertionError(
            f"Shared test core holds {shared_size:,} lexemes; expected "
            f"{config.shared_test_lexemes:,}."
        )

    for user_id in sorted(set(users)):
        rows = dataset.loc[users == user_id]
        splits = set(rows[SPLIT_COLUMN])
        if user_id in train_only:
            if splits != {TRAIN}:
                raise AssertionError(
                    f"Train-only learner {user_id} has rows in {sorted(splits)}."
                )
            continue
        missing = {TRAIN, EVAL, TEST} - splits
        if missing:
            raise AssertionError(
                f"Learner {user_id} is missing from split(s) {sorted(missing)}."
            )
        per_lexeme = rows.groupby(LEXEME_ID_COLUMN)[SPLIT_COLUMN].nunique()
        straddling = per_lexeme[per_lexeme > 1]
        if not straddling.empty:
            raise AssertionError(
                f"Learner {user_id} has {len(straddling):,} lexeme(s) in more than "
                f"one split; examples: {straddling.index[:5].tolist()}"
            )

    fill_in_pool = dataset[
        (dataset[SPLIT_COLUMN] == TEST) & (~dataset[SHARED_TEST_COLUMN])
    ]
    pool_lexemes = frozenset(lexemes[dataset[SHARED_TEST_COLUMN]])
    overlap = set(fill_in_pool[LEXEME_ID_COLUMN].astype(str)) & pool_lexemes
    if overlap:
        raise AssertionError(
            f"{len(overlap):,} fill lexeme(s) also appear in the shared core, so the "
            "two test scopes are not separable."
        )


def _write_jsonl(dataframe: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_json(
        output_path,
        orient="records",
        lines=True,
        date_format="iso",
    )


def _print_summary(dataset: pd.DataFrame, pool: set[str], cohort: Cohort) -> None:
    from plb.learners import get_learner_code

    print(
        f"Cohort data: {len(dataset):,} labelled word instances, "
        f"{dataset[LEXEME_ID_COLUMN].nunique():,} lexemes, "
        f"{dataset['user_id'].nunique():,} learners."
    )
    print(f"Shared pool (every test-eligible learner): {len(pool):,} lexemes.")
    header = f"{'learner':9}{'rows':>7}{'train':>8}{'eval':>7}{'test':>7}{'shared':>8}{'fill':>7}"
    print(header)
    for user_id, rows in dataset.groupby(dataset["user_id"].astype(str), sort=True):
        counts = rows[SPLIT_COLUMN].value_counts()
        test_rows = rows[rows[SPLIT_COLUMN] == TEST]
        shared = int(test_rows[SHARED_TEST_COLUMN].sum())
        print(
            f"{get_learner_code(user_id):9}{len(rows):>7,}"
            f"{counts.get(TRAIN, 0):>8,}{counts.get(EVAL, 0):>7,}"
            f"{counts.get(TEST, 0):>7,}{shared:>8,}{len(test_rows) - shared:>7,}"
        )


def main() -> None:
    cohort = load_cohort()
    config = load_model_b_config().splits

    decks_df = pd.read_json(
        get_dataset_as_path("interim/decks.jsonl"),
        lines=True,
    )
    lexemes_df = pd.read_json(
        get_dataset_as_path("interim/lexemes.jsonl"),
        lines=True,
    )
    word_instances_df = pd.read_json(
        get_dataset_as_path("processed/word_instances_w_stability_targets.jsonl"),
        lines=True,
    )

    dataset = build_split_dataset(
        decks_df,
        lexemes_df,
        word_instances_df,
        cohort,
    )
    pool = shared_lexeme_pool(dataset, cohort.test_eligible_user_ids())
    dataset = assign_splits(dataset, pool, config, cohort)

    _print_summary(dataset, pool, cohort)
    sort_columns = ["first_reviewed_at", LEXEME_ID_COLUMN, "word_instance_id"]
    for split, output_path in OUTPUT_PATHS.items():
        rows = (
            dataset.loc[dataset[SPLIT_COLUMN] == split, OUTPUT_COLUMNS]
            .sort_values(sort_columns, kind="stable")
            .reset_index(drop=True)
        )
        _write_jsonl(rows, output_path)
        print(f"Wrote {len(rows):,} {split} rows to {output_path}")


if __name__ == "__main__":
    raise SystemExit(main())
