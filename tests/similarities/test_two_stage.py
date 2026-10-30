"""The scoring stage and the selection stage must agree with the single-pass path.

The cache is positional and float32, so the guarantee worth testing is that splitting
the work changes neither which neighbours are chosen nor what is recorded about them.
"""

import numpy as np
import pandas as pd
import pytest

from plb.similarities import cache, pipeline
from plb.similarities.calibration import (
    CHANNELS,
    ORTHO_STROKE,
    PHONOLOGICAL,
    QUANTILE_POINTS,
    ChannelNull,
    NullCalibration,
)
from plb.similarities.config import SimilaritiesConfig
from plb.similarities.score_pairs import score_all_pairs
from plb.similarities.select_neighbors import (
    channel_matrix,
    iter_selected_rows,
    percentiles_of,
    sample_null_from_cache,
)
from plb.similarities.types import PhonologicalSimilarityEntry, SimilarityEntry
from plb.utils import get_dataset_as_path

EMBEDDINGS_PATH = get_dataset_as_path(
    "processed/embeddings/word_instance_embeddings_direct.jsonl"
)


def _config(tmp_path, **overrides) -> SimilaritiesConfig:
    return SimilaritiesConfig(
        percentiles=dict.fromkeys(CHANNELS, 0.8),
        cache={"path": tmp_path / "scores"},
        embeddings_path=EMBEDDINGS_PATH,
        workers=1,
        target_chunk_size=2,
        **overrides,
    )


def _calibration() -> NullCalibration:
    grid = [index / (QUANTILE_POINTS - 1) for index in range(QUANTILE_POINTS)]
    return NullCalibration(
        sample_pairs=100,
        seed=0,
        channels={
            channel: ChannelNull(sample_size=100, quantiles=grid)
            for channel in CHANNELS
        },
    )


def _word(word_instance_id: str, day: int, *, form: str, reading: str | None):
    entry = SimilarityEntry(
        id=word_instance_id,
        form=form,
        meaning=f"meaning of {form}",
        reading=reading,
    )
    return pipeline._Word(
        user_id="user",
        first_reviewed_at=pd.Timestamp(f"2025-01-{day:02d}", tz="UTC"),
        entry=entry,
        phonological_entry=(
            PhonologicalSimilarityEntry.model_validate(entry.model_dump())
            if reading is not None
            else None
        ),
        plb_14=3,
        log_plb_14=float(np.log(3)),
    )


def _words() -> list[pipeline._Word]:
    # Mixed scripts on purpose: kanji forms give the stroke channel coverage, kana
    # forms give the kana-confusion channel coverage, and one word without a reading
    # exercises the phonological NaN path. All five channels must see some coverage
    # or calibration legitimately refuses to build a null for the empty one.
    specs = [
        ("w1", "猫", "ねこ"),
        ("w2", "いぬ", "いぬ"),
        ("w3", "ねこじた", "ねこじた"),
        ("w4", "本", None),
        ("w5", "日本", "にほん"),
    ]
    return [
        _word(word_id, day + 1, form=form, reading=reading)
        for day, (word_id, form, reading) in enumerate(specs)
    ]


def _embeddings(words) -> dict[str, np.ndarray]:
    generator = np.random.default_rng(3)
    return {
        word.word_instance_id: generator.normal(size=8) for word in words
    }


def test_two_stage_matches_the_single_pass_path(tmp_path) -> None:
    words = _words()
    embeddings = _embeddings(words)
    config = _config(tmp_path)
    calibration = _calibration()

    single_pass = [
        row.model_dump()
        for row in pipeline._iter_similarity_rows(
            words, embeddings, {}, config, set(), calibration, show_progress=False
        )
    ]

    manifest = score_all_pairs(words, embeddings, config, show_progress=False)
    two_stage = [
        row.model_dump()
        for row, _ in iter_selected_rows(
            pipeline._group_words_by_user(words),
            manifest,
            {},
            calibration,
            config,
            show_progress=False,
        )
    ]

    assert [row["word_instance_id"] for row in two_stage] == [
        row["word_instance_id"] for row in single_pass
    ]
    for after, before in zip(two_stage, single_pass, strict=True):
        assert [n["word_instance_id"] for n in after["neighbors"]] == [
            n["word_instance_id"] for n in before["neighbors"]
        ]
        for neighbor_after, neighbor_before in zip(
            after["neighbors"], before["neighbors"], strict=True
        ):
            assert neighbor_after["selected_by"] == neighbor_before["selected_by"]
            assert neighbor_after["is_censored"] == neighbor_before["is_censored"]
            assert neighbor_after["reviews"] == neighbor_before["reviews"]
            # float32 in the cache; the scores themselves stay within its epsilon.
            for key in ("sem_sim", "mor_sim", "ortho_kana_confusion"):
                assert neighbor_after[key] == pytest.approx(
                    neighbor_before[key], abs=1e-6
                )


def test_scores_land_at_the_offsets_selection_reads(tmp_path) -> None:
    words = _words()
    config = _config(tmp_path)

    manifest = score_all_pairs(words, _embeddings(words), config, show_progress=False)

    ends = pipeline._candidate_ends(words)
    offsets = cache.pair_offsets(ends)
    scores = cache.open_scores(
        config.cache.path, "user", manifest.learners["user"].pair_count, mode="r"
    )
    assert scores.shape == (offsets[-1], len(cache.SCORE_COLUMNS))
    # w4 has no reading, so every pair involving it is NaN in the phonological column.
    w4_block = np.asarray(scores[offsets[3] : offsets[4]])
    assert np.isnan(w4_block[:, list(cache.SCORE_COLUMNS).index("phon_sim")]).all()


def test_resuming_skips_completed_chunks_and_preserves_bytes(tmp_path) -> None:
    words = _words()
    embeddings = _embeddings(words)
    config = _config(tmp_path)

    score_all_pairs(words, embeddings, config, show_progress=False)
    first = cache.scores_path(config.cache.path, "user").read_bytes()

    # A complete progress file means there is nothing left to do.
    score_all_pairs(words, embeddings, config, show_progress=False)
    assert cache.scores_path(config.cache.path, "user").read_bytes() == first

    # Dropping a chunk from the progress file re-scores exactly that chunk.
    cache.clear_progress(config.cache.path, "user")
    cache.record_completed_chunk(config.cache.path, "user", 0)
    score_all_pairs(words, embeddings, config, show_progress=False)
    assert cache.scores_path(config.cache.path, "user").read_bytes() == first


def test_a_reordered_word_list_invalidates_the_cache(tmp_path) -> None:
    words = _words()
    config = _config(tmp_path)
    manifest = score_all_pairs(words, _embeddings(words), config, show_progress=False)

    # Same words, different order: every offset would still resolve, to the wrong pair.
    with pytest.raises(ValueError, match="different word list"):
        manifest.verify_against({"user": list(reversed(words))})


def test_percentile_mapping_matches_the_scalar_path() -> None:
    calibration = _calibration()
    block = np.zeros((3, len(cache.SCORE_COLUMNS)), dtype=np.float32)
    block[:, list(cache.SCORE_COLUMNS).index("sem_sim")] = [0.1, 0.5, 0.9]

    values = channel_matrix(block)["semantic"]
    vectorised = percentiles_of(values, calibration, "semantic")

    # Compare against the value as stored, not as written: float32(0.1) is a hair
    # above 0.1, which lands on the far side of the grid point and shifts the
    # percentile by one step. That is the cache's rounding, not a mapping difference.
    for index in range(len(values)):
        assert vectorised[index] == pytest.approx(
            calibration.percentile("semantic", float(values[index]))
        )


def test_uncovered_channels_map_to_nan_not_zero() -> None:
    calibration = _calibration()
    block = np.zeros((1, len(cache.SCORE_COLUMNS)), dtype=np.float32)
    block[0, list(cache.SCORE_COLUMNS).index("phon_sim")] = np.nan
    block[0, list(cache.SCORE_COLUMNS).index("ortho_stroke_coverage")] = 0.0

    columns = channel_matrix(block)

    # Zero would claim "maximally dissimilar" and drag the null down; absent is absent.
    assert np.isnan(columns[PHONOLOGICAL][0])
    assert np.isnan(columns[ORTHO_STROKE][0])
    assert np.isnan(percentiles_of(columns[PHONOLOGICAL], calibration, PHONOLOGICAL)[0])


def test_calibration_from_cache_is_seed_reproducible(tmp_path) -> None:
    words = _words()
    config = _config(tmp_path, calibration={"sample_pairs": 100, "seed": 5})
    manifest = score_all_pairs(words, _embeddings(words), config, show_progress=False)
    words_by_user = pipeline._group_words_by_user(words)

    first = sample_null_from_cache(words_by_user, manifest, config)
    again = sample_null_from_cache(words_by_user, manifest, config)
    other = sample_null_from_cache(
        words_by_user,
        manifest,
        config.model_copy(
            update={"calibration": config.calibration.model_copy(update={"seed": 6})}
        ),
    )

    assert first.channels["semantic"].quantiles == again.channels["semantic"].quantiles
    assert first.channels["semantic"].quantiles != other.channels["semantic"].quantiles
