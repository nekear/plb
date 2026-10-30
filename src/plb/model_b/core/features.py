"""Intralexical features shared by every content-aware Model B ablation.

One extractor defines the word-level feature list (currently ``kvl_score``,
``log_freq``, ``is_oov``), so extending it — script type, character count,
L1-cognateness — reaches every model without touching their code. The transform is a
pure function over injected frames; ``load_intralexical_inputs`` is the one place that
touches the filesystem, which keeps unit tests free of real data.

``learner_median_plb`` is deliberately not produced here: it is fit-time state derived
from the training targets, so it belongs to each model's ``fit``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from plb.tubelex.config import load_tubelex_config
from plb.utils import get_dataset_as_path

INTRALEXICAL_COLUMNS = ("kvl_score", "log_freq", "is_oov")

LEXEMES_PATH = get_dataset_as_path("interim/lexemes.jsonl")
KVL_PATH = get_dataset_as_path("processed/word_instances_kvl_predictions.jsonl")
WORD_INSTANCE_INDEX_PATH = get_dataset_as_path(
    "processed/word_instances_w_stability_targets.jsonl"
)


def load_word_instance_index() -> pd.DataFrame:
    """word_instance_id -> lexeme_id, for rows that carry only an instance id.

    Similarity neighbors reference words by ``word_instance_id`` alone; this index
    lets ``add_intralexical_features`` resolve their lexeme forms.
    """
    frame = pd.read_json(WORD_INSTANCE_INDEX_PATH, lines=True)
    return frame[["word_instance_id", "lexeme_id"]]


def load_tubelex_frequencies(path: Path) -> pd.DataFrame:
    """Load one total Tubelex count per surface form (columns: form, count)."""
    raw = pd.read_csv(path, sep="\t")
    missing = sorted({"word", "count"} - set(raw.columns))
    if missing:
        raise ValueError(f"Tubelex frequencies file is missing columns: {missing}")

    frequencies = raw[["word", "count"]].copy()
    frequencies["word"] = frequencies["word"].astype("string").str.strip()
    frequencies["count"] = pd.to_numeric(frequencies["count"], errors="raise")
    frequencies = frequencies.loc[
        frequencies["word"].notna() & frequencies["word"].ne("")
    ]
    if (frequencies["count"] < 0).any():
        raise ValueError("Tubelex counts must be non-negative")

    return (
        frequencies.groupby("word", as_index=False)["count"]
        .sum()
        .rename(columns={"word": "form"})
    )


def load_intralexical_inputs() -> dict[str, pd.DataFrame]:
    """Load the reference frames once; keys match add_intralexical_features kwargs."""
    return {
        "lexemes": pd.read_json(LEXEMES_PATH, lines=True),
        "tubelex": load_tubelex_frequencies(
            load_tubelex_config().get_frequencies_file_path()
        ),
        "kvl": pd.read_json(KVL_PATH, lines=True),
    }


def add_intralexical_features(
    frame: pd.DataFrame,
    *,
    lexemes: pd.DataFrame,
    tubelex: pd.DataFrame,
    kvl: pd.DataFrame,
    word_instances: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return ``frame`` with the INTRALEXICAL_COLUMNS appended.

    KVL scores and lexeme forms must cover every row (both are produced for the full
    cohort, so a miss is a pipeline bug); Tubelex is a left join where a missing form
    means out-of-vocabulary: ``is_oov = 1`` and ``log_freq = log10(0 + 1) = 0``.

    A frame without a ``lexeme_id`` column (e.g. neighbor word instances) can pass
    ``word_instances`` (see ``load_word_instance_index``) to resolve it first.
    """
    if "lexeme_id" not in frame.columns:
        if word_instances is None:
            raise ValueError(
                "frame has no lexeme_id column; pass word_instances to resolve it"
            )
        frame = frame.merge(
            word_instances,
            on="word_instance_id",
            how="left",
            validate="many_to_one",
        )
        # Verifying that all rows have `lexeme_id`
        _require_covered(frame, "lexeme_id", "word_instance_id", "word instances")

    # Adding KVL scores
    enriched = frame.merge(
        kvl[["word_instance_id", "kvl_score"]],
        on="word_instance_id",
        how="left",
        validate="many_to_one",
    )
    # Verifying that all rows contain the added KVL scores
    _require_covered(enriched, "kvl_score", "word_instance_id", "KVL predictions")

    # Adding lexeme forms to further merge the Tubelex frequencies
    enriched = enriched.merge(
        lexemes[["lexeme_id", "form"]],
        on="lexeme_id",
        how="left",
        validate="many_to_one",
    )
    # Verifying that all rows contain the lexeme forms
    _require_covered(enriched, "form", "lexeme_id", "lexemes")

    # Adding Tubelex frequencies with the `is OOV` flag and a log-scaled frequency
    enriched = enriched.merge(tubelex, on="form", how="left", validate="many_to_one")
    enriched["is_oov"] = enriched["count"].isna().astype(np.int64)
    enriched["log_freq"] = np.log10(enriched["count"].fillna(0.0) + 1.0)

    # Dropping the form as it shouldn't be used as a feature and returning the result
    return enriched.drop(columns=["form", "count"])


def _require_covered(
    frame: pd.DataFrame, column: str, key: str, source_name: str
) -> None:
    """
        Ensure that all values in a specific column contain the values.
        Args:
            - ``frame``: ``pd.DataFrame`` to check in;
            - ``column``: ``str`` to check in;
            - ``key``: ``str`` to report back the exact identifiable rows instead of a pure count.
    """

    uncovered = frame.loc[frame[column].isna(), key]
    if not uncovered.empty:
        raise ValueError(
            f"{len(uncovered)} rows have no entry in {source_name}; "
            f"examples: {uncovered.head(5).tolist()}"
        )
