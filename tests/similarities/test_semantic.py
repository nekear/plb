import json
from pathlib import Path
from typing import Callable

import pandas as pd
import pytest

from plb.similarities.semantic import compute_semantic_similarity
from plb.similarities.types import SimilarityEntry


def similar_entries() -> list[SimilarityEntry]:
    return [
        SimilarityEntry(id="1", form="お父さん", meaning="father"),
        SimilarityEntry(id="2", form="お母さん", meaning="mother"),
    ]


def different_entries() -> list[SimilarityEntry]:
    return [
        SimilarityEntry(id="3", form="大人", meaning="adults"),
        SimilarityEntry(id="4", form="野菜", meaning="vegetables"),
    ]


@pytest.fixture
def get_embedding() -> Callable[[str], pd.Series]:
    path = Path(__file__).with_name("embeddings.json")

    with path.open(encoding="utf-8") as file:
        data = json.load(file)

    by_meaning = {
        meaning: pd.Series(vector, dtype=float)
        for meaning, vector in zip(
            data["inputs"],
            data["results"]["embeddings"],
            strict=True,
        )
    }

    entries = similar_entries() + different_entries()
    meaning_by_id = {entry.id: entry.meaning for entry in entries}

    return lambda entry_id: by_meaning[meaning_by_id[entry_id]]


def test_semantic_similarity(get_embedding) -> None:
    related = similar_entries()
    unrelated = different_entries()

    related_score = compute_semantic_similarity(
        related[0],
        related[1],
        get_embedding,
    )
    unrelated_score = compute_semantic_similarity(
        unrelated[0],
        unrelated[1],
        get_embedding,
    )

    assert related_score == pytest.approx(0.757633, abs=1e-6)
    assert unrelated_score == pytest.approx(0.545804, abs=1e-6)
    assert related_score > unrelated_score


def test_semantic_similarity_clamps_negative_cosine_to_zero() -> None:
    entries = similar_entries()
    embeddings = {
        entries[0].id: [1.0, 0.0],
        entries[1].id: [-1.0, 0.0],
    }

    assert (
        compute_semantic_similarity(entries[0], entries[1], embeddings.__getitem__)
        == 0.0
    )


def test_semantic_similarity_rejects_zero_norm_embedding() -> None:
    entries = similar_entries()
    embeddings = {
        entries[0].id: [0.0, 0.0],
        entries[1].id: [1.0, 0.0],
    }

    with pytest.raises(ValueError, match="zero-norm"):
        compute_semantic_similarity(entries[0], entries[1], embeddings.__getitem__)
