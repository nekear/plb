import numpy as np
import pandas as pd
import pytest

from plb.similarities import pipeline
from plb.similarities.calibration import (
    CHANNELS,
    MORPHOLOGICAL,
    ORTHO_KANA,
    ORTHO_STROKE,
    PHONOLOGICAL,
    SEMANTIC,
    ChannelNull,
    NullCalibration,
    build_null_calibration,
    channel_values,
)
from plb.similarities.types import (
    OrthographicSimilarityResult,
    PhonologicalSimilarityEntry,
    SimilarityEntry,
)


def _null(values: list[float]) -> NullCalibration:
    grid = np.quantile(np.asarray(values, dtype=np.float64), np.linspace(0, 1, 1001))
    return NullCalibration(
        sample_pairs=len(values),
        seed=0,
        channels={
            channel: ChannelNull(
                sample_size=len(values),
                quantiles=[float(value) for value in grid],
            )
            for channel in CHANNELS
        },
    )


def _scores(
    *,
    sem: float = 0.0,
    phon: float | None = 0.0,
    mor: float = 0.0,
    stroke: float = 0.0,
    stroke_coverage: float = 1.0,
    kana: float = 0.0,
    kana_coverage: float = 1.0,
) -> pipeline._PairScores:
    return pipeline._PairScores(
        sem_sim=sem,
        phon_sim=phon,
        mor_sim=mor,
        orthographic=OrthographicSimilarityResult(
            ortho_stroke_sim=stroke,
            ortho_stroke_coverage=stroke_coverage,
            ortho_kana_confusion=kana,
            ortho_kana_coverage=kana_coverage,
        ),
    )


def test_percentile_rises_with_the_score() -> None:
    calibration = _null([index / 100 for index in range(101)])

    percentiles = [
        calibration.percentile(SEMANTIC, value) for value in (0.1, 0.5, 0.9)
    ]

    assert percentiles == sorted(percentiles)
    assert percentiles[0] < percentiles[-1]


def test_an_atom_at_zero_maps_to_zero_not_to_the_top() -> None:
    """The morphological null is a point mass at 0: almost every random pair shares no
    lemma. A `<=` comparison would rate a zero-overlap pair at the 99.9th percentile,
    letting noise through; strictly-less maps it to 0."""
    calibration = _null([0.0] * 999 + [0.4, 0.6])

    assert calibration.percentile(MORPHOLOGICAL, 0.0) == 0.0
    assert calibration.percentile(MORPHOLOGICAL, 0.5) > 0.9


def test_missing_channels_have_no_percentile() -> None:
    calibration = _null([index / 100 for index in range(101)])

    assert calibration.percentile(PHONOLOGICAL, None) is None


def test_unknown_channel_is_rejected() -> None:
    calibration = _null([0.0, 1.0])

    with pytest.raises(KeyError, match="No null distribution for channel"):
        calibration.percentile("pixel", 0.5)


def test_channel_values_drop_uncovered_orthographic_axes() -> None:
    """Zero coverage means "no comparable characters", which is not the same as a
    similarity of zero — counting it as zero would drag the null down and inflate
    every real score's percentile."""
    values = channel_values(
        _scores(sem=0.4, phon=None, stroke=0.0, stroke_coverage=0.0, kana=0.7)
    )

    assert values[SEMANTIC] == 0.4
    assert values[PHONOLOGICAL] is None
    assert values[ORTHO_STROKE] is None
    assert values[ORTHO_KANA] == 0.7


def _word(word_instance_id: str, *, reading: str | None = "かな") -> pipeline._Word:
    entry = SimilarityEntry(
        id=word_instance_id,
        form=word_instance_id,
        meaning=f"meaning-{word_instance_id}",
        reading=reading,
    )
    return pipeline._Word(
        user_id="user",
        first_reviewed_at=pd.Timestamp("2025-01-01", tz="UTC"),
        entry=entry,
        phonological_entry=(
            PhonologicalSimilarityEntry.model_validate(entry.model_dump())
            if reading is not None
            else None
        ),
        plb_14=None,
        log_plb_14=None,
    )


def test_building_a_null_is_reproducible_from_the_seed() -> None:
    words_by_user = {
        "a": [_word(f"a{index}") for index in range(20)],
        "b": [_word(f"b{index}") for index in range(5)],
    }
    counter = {"calls": 0}

    def score_pair(left, right):
        counter["calls"] += 1
        return _scores(sem=hash((left.entry.id, right.entry.id)) % 100 / 100)

    first = build_null_calibration(words_by_user, score_pair, sample_pairs=200, seed=11)
    second = build_null_calibration(words_by_user, score_pair, sample_pairs=200, seed=11)
    other = build_null_calibration(words_by_user, score_pair, sample_pairs=200, seed=12)

    assert counter["calls"] == 600
    assert first.channels[SEMANTIC].quantiles == second.channels[SEMANTIC].quantiles
    assert first.channels[SEMANTIC].quantiles != other.channels[SEMANTIC].quantiles


def test_a_channel_with_no_comparable_values_is_rejected() -> None:
    words_by_user = {"a": [_word(f"a{index}") for index in range(5)]}

    with pytest.raises(ValueError, match="cannot be calibrated"):
        build_null_calibration(
            words_by_user,
            lambda left, right: _scores(phon=None, kana_coverage=0.0),
            sample_pairs=50,
            seed=3,
        )


def test_sampled_pairs_are_within_a_learner_and_never_self_paired() -> None:
    words_by_user = {
        "a": [_word(f"a{index}") for index in range(6)],
        "b": [_word(f"b{index}") for index in range(4)],
    }
    seen: list[tuple[str, str]] = []

    def score_pair(left, right):
        seen.append((left.entry.id, right.entry.id))
        return _scores(sem=0.5)

    build_null_calibration(words_by_user, score_pair, sample_pairs=300, seed=5)

    assert all(left != right for left, right in seen)
    assert all(left[0] == right[0] for left, right in seen)  # same learner prefix
    # Both learners are represented, weighted by their pair counts.
    assert {left[0] for left, _ in seen} == {"a", "b"}
