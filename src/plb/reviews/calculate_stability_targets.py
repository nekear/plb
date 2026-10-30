import pandas as pd

from plb.utils import get_dataset_as_path

STABILITY_TARGET = 14.0


def calculate_stability_targets(
    df_word_instances: pd.DataFrame,
    df_fsrs_reviews: pd.DataFrame,
    stability_target: float = STABILITY_TARGET,
) -> pd.DataFrame:
    """
    Adds the following columns to `word_instances.jsonl`:
    - `plb_14` - personalized learning burden at a stability of 14 days: the
      number of the first clean review whose replayed stability is at least
      `stability_target`. Null when the target has not yet been reached. The
      threshold is named in the column, so a different `STABILITY_TARGET` must
      be written to a differently named column rather than silently changing
      what `plb_14` means downstream.
    - `plb_14_achieved_at` - timestamp of that crossing review: the moment the
      `plb_14` label became knowable. Null when the target has not been
      reached. Anything windowing on "recent burdens" must use this, not
      `first_reviewed_at`, or it reads labels from the future.
    - `current_review_count`: the number of clean replayed reviews for the
      word instance. Zero when it has no clean replayed reviews.
    - `current_stability`: stability after the most recent clean replayed
      review, or null when no such review exists.
    - `current_difficulty`: difficulty after the most recent clean replayed
      review, or null when no such review exists.
    - `last_reviewed_at`: timestamp of the most recent clean replayed
      review, or null when no such review exists.
    """

    reviews = df_fsrs_reviews.sort_values(
        ["word_instance_id", "timestamp_ms", "review_id"],
        kind="stable",
    ).copy()

    reviews["review_number"] = reviews.groupby("word_instance_id").cumcount() + 1

    review_counts = (
        reviews.groupby("word_instance_id", as_index=False)
        .size()
        .rename(columns={"size": "current_review_count"})
    )

    # The frame is sorted (word_instance_id, timestamp_ms, review_id), so the min
    # review_number and the min timestamp among crossing rows are the same review.
    first_target_reviews = (
        reviews.loc[
            reviews["fsrs_stability"] >= stability_target,
            ["word_instance_id", "review_number", "timestamp_ms"],
        ]
        .groupby("word_instance_id", as_index=False)
        .agg(
            plb_14=("review_number", "min"),
            plb_14_achieved_at=("timestamp_ms", "min"),
        )
    )
    current_state = (
        reviews.groupby("word_instance_id", as_index=False)
        .tail(1)[
            [
                "word_instance_id",
                "fsrs_stability",
                "fsrs_difficulty",
                "timestamp_ms",
            ]
        ]
        .rename(
            columns={
                "fsrs_stability": "current_stability",
                "fsrs_difficulty": "current_difficulty",
                "timestamp_ms": "last_reviewed_at",
            }
        )
    )

    stability_targets = review_counts.merge(
        first_target_reviews,
        on="word_instance_id",
        how="left",
        validate="one_to_one",
    ).merge(
        current_state,
        on="word_instance_id",
        how="left",
        validate="one_to_one",
    )
    result = df_word_instances.merge(
        stability_targets,
        on="word_instance_id",
        how="left",
        validate="one_to_one",
    )

    result["plb_14"] = result["plb_14"].astype("Int64")
    result["current_review_count"] = (
        result["current_review_count"].fillna(0).astype("Int64")
    )
    return result


def main():
    word_instances_path = get_dataset_as_path("interim/word_instances.jsonl")
    fsrs_reviews_path = get_dataset_as_path("processed/clean_reviews_w_fsrs.jsonl")
    output_path = get_dataset_as_path(
        "processed/word_instances_w_stability_targets.jsonl"
    )

    df_word_instances = pd.read_json(word_instances_path, lines=True)
    df_fsrs_reviews = pd.read_json(fsrs_reviews_path, lines=True)
    df_stability_targets = calculate_stability_targets(
        df_word_instances,
        df_fsrs_reviews,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_stability_targets.to_json(
        output_path,
        lines=True,
        orient="records",
        date_format="iso",
    )

    reached_target = df_stability_targets["plb_14"].notna().sum()
    print(
        f"Word instances reaching stability {STABILITY_TARGET:g}: "
        f"{reached_target:,}/{len(df_stability_targets):,}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
