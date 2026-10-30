from numbers import Number

import pandas as pd
from fsrs import Card, Rating, Scheduler
from tqdm import tqdm

from plb.utils import get_dataset_as_path

RATING_MAP = {
    1: Rating.Again,
    2: Rating.Hard,
    3: Rating.Good,
    4: Rating.Easy,
}


def _build_schedulers(df_decks: pd.DataFrame) -> dict[str, Scheduler]:
    """Builds one FSRS scheduler per deck configuration."""
    schedulers: dict[str, Scheduler] = {}

    for deck in df_decks.to_dict(orient="records"):
        deck_id = deck.get("deck_id")
        if deck_id is None or bool(pd.isna(deck_id)):
            continue

        scheduler_kwargs: dict[str, object] = {}

        weights = deck.get("fsrs_weights")
        if isinstance(weights, (list, tuple)) and weights:
            scheduler_kwargs["parameters"] = tuple(float(weight) for weight in weights)

        desired_retention = deck.get("desired_retention")
        if desired_retention is not None and not bool(pd.isna(desired_retention)):
            scheduler_kwargs["desired_retention"] = float(desired_retention)

        schedulers[str(deck_id)] = Scheduler(**scheduler_kwargs)

    return schedulers


def replay_card_history(
    group: pd.DataFrame,
    scheduler_by_deck: dict[str, Scheduler],
) -> pd.DataFrame:
    """Given one card's reviews (sorted, clean), replays them through FSRS
    and returns the resulting stability/difficulty after each review."""
    group = group.sort_values("timestamp_ms")
    card = Card()
    word_instance_id = group["word_instance_id"].iloc[0]

    deck_ids = group["deck_id"].dropna().unique()
    deck_id = str(deck_ids[0])

    scheduler = scheduler_by_deck.get(deck_id)
    if scheduler is None:
        raise ValueError(f"No FSRS configuration found for deck {deck_id}")

    stabilities = []
    difficulties = []

    for row in group.itertuples():
        rating = RATING_MAP[row.rating]
        if isinstance(row.timestamp_ms, Number):
            review_dt = pd.to_datetime(row.timestamp_ms, unit="ms", utc=True)
        else:
            review_dt = pd.to_datetime(row.timestamp_ms, utc=True)

        card, _ = scheduler.review_card(card, rating, review_datetime=review_dt)

        stabilities.append(card.stability)
        difficulties.append(card.difficulty)

    group["fsrs_stability"] = stabilities
    group["fsrs_difficulty"] = difficulties

    return group.drop(columns="_replay_group_id", errors="ignore")


def calculate_fsrs(
    df_reviews: pd.DataFrame,
    df_decks: pd.DataFrame,
    df_word_instances: pd.DataFrame,
) -> pd.DataFrame:
    scheduler_by_deck = _build_schedulers(df_decks)
    df = df_reviews.merge(
        df_word_instances[["word_instance_id", "deck_id"]],
        on="word_instance_id",
        how="left",
        validate="many_to_one",
    )

    missing_decks = df.loc[df["deck_id"].isna(), "word_instance_id"].unique()
    if len(missing_decks):
        raise ValueError(
            f"Could not find a deck for word instances: {missing_decks[:5].tolist()}"
        )

    # I'm keeping the original word_instance_id in each group, becase Pandas
    # excludes the column used directly by groupby(...).apply from the group.
    df["_replay_group_id"] = df["word_instance_id"]

    tqdm.pandas(desc="Replaying FSRS")

    result = (
        df.groupby("_replay_group_id", group_keys=False)
        .progress_apply(
            replay_card_history,
            scheduler_by_deck=scheduler_by_deck,
        )
        .reset_index(drop=True)
    )

    return result


def main():
    decks_path = get_dataset_as_path("interim/decks.jsonl")
    df_decks = pd.read_json(decks_path, lines=True)

    word_instances_path = get_dataset_as_path("interim/word_instances.jsonl")
    df_word_instances = pd.read_json(word_instances_path, lines=True)

    clean_reviews_path = get_dataset_as_path("processed/clean_reviews.jsonl")
    df_clean_reviews = pd.read_json(clean_reviews_path, lines=True)

    df_clean_reviews = df_clean_reviews.sort_values(
        ["word_instance_id", "timestamp_ms"]
    )

    df_fsrs_replayed = calculate_fsrs(
        df_clean_reviews,
        df_decks,
        df_word_instances,
    )

    output_path = get_dataset_as_path("processed/clean_reviews_w_fsrs.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_fsrs_replayed.to_json(
        output_path,
        lines=True,
        orient="records",
        date_format="iso",
    )

    total_words_count = len(df_word_instances["word_instance_id"].index)
    words_with_fsrs_s_over_14 = df_fsrs_replayed.loc[
        df_fsrs_replayed["fsrs_stability"] > 14, "word_instance_id"
    ].nunique()

    print(f"Total words: {total_words_count}")
    print(
        f"Words with FSRS-S over 14: {words_with_fsrs_s_over_14}, which is {words_with_fsrs_s_over_14 / total_words_count * 100:.2f}% of the total"
    )


if __name__ == "__main__":
    raise SystemExit(main())
