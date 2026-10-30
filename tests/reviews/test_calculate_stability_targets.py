import pandas as pd

from plb.reviews.calculate_stability_targets import calculate_stability_targets

WORD_INSTANCES = pd.DataFrame({"word_instance_id": ["crosses", "never-crosses"]})

REVIEWS = pd.DataFrame(
    [
        # "crosses" reaches stability 14 on its SECOND review.
        ("crosses", "2025-01-01T10:00:00.000", "r1", 3.0, 5.0),
        ("crosses", "2025-01-05T10:00:00.000", "r2", 20.0, 4.8),
        ("crosses", "2025-02-01T10:00:00.000", "r3", 50.0, 4.5),
        ("never-crosses", "2025-01-02T10:00:00.000", "r4", 2.0, 6.0),
    ],
    columns=[
        "word_instance_id",
        "timestamp_ms",
        "review_id",
        "fsrs_stability",
        "fsrs_difficulty",
    ],
)


def test_the_achievement_timestamp_is_the_crossing_reviews_timestamp() -> None:
    result = calculate_stability_targets(WORD_INSTANCES, REVIEWS).set_index(
        "word_instance_id"
    )

    assert result.loc["crosses", "plb_14"] == 2
    # The label became knowable at the crossing (r2), not at the start or the end.
    assert result.loc["crosses", "plb_14_achieved_at"] == "2025-01-05T10:00:00.000"


def test_words_that_never_cross_have_null_burden_and_null_achievement() -> None:
    result = calculate_stability_targets(WORD_INSTANCES, REVIEWS).set_index(
        "word_instance_id"
    )

    assert pd.isna(result.loc["never-crosses", "plb_14"])
    assert pd.isna(result.loc["never-crosses", "plb_14_achieved_at"])
