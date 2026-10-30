from typing import cast

import pandas as pd

from plb.utils import get_dataset_as_path


def clean_reviews(df_reviews: pd.DataFrame) -> pd.DataFrame:
    # Columns: review_id, word_instance_id, timestamp_ms, rating, interval_before, interval_after, factor_after, time_taken_ms, review_type
    # print(df_reviews.head())

    # Clean reviews
    df = df_reviews.copy()

    # Sorting values by card_id, review
    df = df.sort_values(["word_instance_id", "timestamp_ms"])

    # print(df_reviews[["word_instance_id", "timestamp_ms", "interval_before", "interval_after"]].iloc[0:10])

    # Confirming that `interval_before` equals to the number of days passed since the last review
    df[["prev_timestamp_ms", "prev_interval_after"]] = df.groupby("word_instance_id")[
        ["timestamp_ms", "interval_after"]
    ].shift(1)
    df["days_since_prev"] = (
                                    df["timestamp_ms"] - df["prev_timestamp_ms"]
                            ).dt.total_seconds() / 86400

    # print(df_reviews[["timestamp_ms", "interval_before", "interval_after", "prev_timestamp_ms", "prev_interval_after", "days_since_prev"]].iloc[0:10])

    # Determining whether the reviews were the first in the day
    df["review_day"] = df["timestamp_ms"].dt.floor("D")
    df["review_rank_in_day"] = df.groupby(["word_instance_id", "review_day"]).cumcount()
    df["is_first_of_day"] = df["review_rank_in_day"] == 0

    # Filtering out reviews which do not fall under the criteria
    clean_mask = (
            (df["review_type"] == 1)
            & (df["days_since_prev"] >= 1)
    )

    # WARN: returning *only* columns used in the original reviews dataset
    return cast(pd.DataFrame, df.loc[clean_mask, df_reviews.columns].copy())


def main() -> int:
    input_path = get_dataset_as_path("interim/reviews.jsonl")
    output_path = get_dataset_as_path("processed/clean_reviews.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_reviews = pd.read_json(input_path, lines=True)
    df_clean_reviews = clean_reviews(df_reviews)

    print(f"Raw reviews: {len(df_reviews.index)}")
    print(f"Clean reviews: {len(df_clean_reviews.index)}")
    print(
        f"Which is {round(len(df_clean_reviews.index) / len(df_reviews.index) * 100, 2)}% reduction."
    )

    df_clean_reviews.to_json(
        output_path,
        lines=True,
        orient="records",
        date_format="iso",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
