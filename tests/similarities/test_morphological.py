import pytest
from plb.similarities.morphological import compute_morphological_similarity
from plb.similarities.types import SimilarityEntry


def test_morphological_similarity_full():
    w1 = SimilarityEntry(
        id="1",
        form="父親",
        meaning="father",
    )

    w2 = SimilarityEntry(
        id="2",
        form="父親",
        meaning="father",
    )

    score = compute_morphological_similarity(w1, w2)

    assert score == pytest.approx(1.0), (
        f"Expected identical forms to have full similarity, got {score}"
    )


@pytest.mark.parametrize(
    ("form_1", "form_2", "minimum", "maximum"),
    [
        pytest.param(
            "日本語",
            "日本文化",
            0.20,
            0.80,
            id="shared-japan-token",
        ),
        pytest.param(
            "東京駅",
            "東京大学",
            0.20,
            0.80,
            id="shared-tokyo-token",
        ),
    ],
)
def test_morphological_similarity_partial(
        form_1,
        form_2,
        minimum,
        maximum,
):
    w1 = SimilarityEntry(
        id="1",
        form=form_1,
        meaning="...",
    )

    w2 = SimilarityEntry(
        id="2",
        form=form_2,
        meaning="...",
    )

    score = compute_morphological_similarity(w1, w2)

    assert minimum <= score <= maximum, (
        f"Similarity for {form_1!r} and {form_2!r} "
        f"should be between {minimum} and {maximum}, "
        f"but got {score}"
    )


@pytest.mark.parametrize(
    ("form_1", "form_2"),
    [
        pytest.param(
            "猫",
            "狼",
            id="cat-and-wolf",
        ),
        pytest.param(
            "山",
            "魚",
            id="mountain-and-fish",
        ),
        pytest.param(
            "走る",
            "静か",
            id="run-and-quiet",
        ),
    ],
)
def test_morphological_similarity_none(form_1, form_2):
    w1 = SimilarityEntry(
        id="1",
        form=form_1,
        meaning="...",
    )

    w2 = SimilarityEntry(
        id="2",
        form=form_2,
        meaning="...",
    )

    score = compute_morphological_similarity(w1, w2)

    assert score == pytest.approx(0.0), (
        f"Expected unrelated forms {form_1!r} and {form_2!r} "
        f"to have no similarity, got {score}"
    )
