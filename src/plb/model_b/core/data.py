"""Load Model B split files, optionally merged with the similarity neighbors.

Every ablation reads the same split files; neighbor-aware models additionally opt into
the ``neighbors`` column, which carries each row's raw list of neighbor dicts from
either ``similarities.jsonl`` or ``randomities.jsonl`` (based on the ablation).

Feature extraction from those dicts is each model's own job, so this logic is only about loading the data.
"""

from __future__ import annotations

import pandas as pd

from plb.model_b.preparation.create_splits import (
    EVAL,
    LOG_TARGET_COLUMN,
    OUTPUT_PATHS,
    TARGET_COLUMN,
    TEST,
    TRAIN,
)
from plb.utils import get_dataset_as_path

NEIGHBOR_PATHS = {
    "similar": get_dataset_as_path("processed/similarities.jsonl"),
    "random": get_dataset_as_path("processed/randomities.jsonl"),
}

REQUIRED_COLUMNS = (
    "user_id",
    "word_instance_id",
    TARGET_COLUMN,
    LOG_TARGET_COLUMN,
    "first_reviewed_at",
    "plb_14_achieved_at",
    "is_shared_test",
)


def load_split(
    split: str,
    *,
    with_neighbors: bool = False,
    neighbors_source: str = "similar",
) -> pd.DataFrame:
    """
    Loads a specific train, test or eval split.

    Args:
        - split (str): Either "train", "test" or "eval, and should be aligned with the name in the global constants;
        - with_neighbors (bool, optional): Whether to load the neighbor data. Defaults to False. Required because we have intralexical-only models, and intralexical + interlexical models;
        - neighbors_source (str, optional): We support neighbors based on similarity to a target (similarities.jsonl), or random (randomities.jsonl). This field can either be "similar" or "random" to load the respective neighbor list.
    """

    # Checking the validity of the split
    if split not in (TRAIN, EVAL, TEST):
        raise ValueError(f"split must be one of {(TRAIN, EVAL, TEST)}; got {split!r}")

    # Loading the dataframe
    frame = pd.read_json(OUTPUT_PATHS[split], lines=True)

    # Checking that all required columns are present (based on a single, global list of required columns)
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"{OUTPUT_PATHS[split]} is missing columns: {missing}")

    # Converting specific columns to datetime
    for column in ("first_reviewed_at", "last_reviewed_at", "plb_14_achieved_at"):
        frame[column] = pd.to_datetime(frame[column], utc=True, format="ISO8601")

    # If neighbors are required (we have intralexical-only models, and intralexical + interlexical models)
    if with_neighbors:
        # Neighbors can be either similar (similarities.jsonl) or random (randomities.jsonl) - therefore we have `neighbors_source`
        if neighbors_source not in NEIGHBOR_PATHS:
            raise ValueError(
                f"neighbors_source must be one of {sorted(NEIGHBOR_PATHS)}; "
                f"got {neighbors_source!r}"
            )


        neighbors = pd.read_json(NEIGHBOR_PATHS[neighbors_source], lines=True)
        frame = frame.merge(
            neighbors[["word_instance_id", "neighbors"]],
            on="word_instance_id",
            how="left",
            validate="one_to_one",
        )

        # Protecting from cases where the preceding neighbor computation pipeline misses some words that still appear in any of the splits
        uncovered = frame[frame["neighbors"].isna()]["word_instance_id"]
        if not uncovered.empty:
            raise ValueError(
                f"{len(uncovered)} split rows have no {neighbors_source} neighbors "
                f"entry; "
                f"examples: {uncovered.head(5).tolist()}"
            )

    return frame
