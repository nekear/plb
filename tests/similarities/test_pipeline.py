from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from plb.similarities import pipeline
from plb.similarities.calibration import (
    CHANNELS,
    QUANTILE_POINTS,
    ChannelNull,
    NullCalibration,
)
from plb.similarities.config import SimilaritiesConfig
from plb.similarities.types import (
    NeighborSimilarityResult,
    OrthographicSimilarityResult,
    PhonologicalSimilarityEntry,
    SimilarityEntry,
)
from plb.utils import get_dataset_as_path

# Only the pipeline entry point reads this file; these tests drive the ranking
# functions with in-memory words, so the path just has to satisfy the config.
EMBEDDINGS_PATH = get_dataset_as_path(
    "processed/embeddings/word_instance_embeddings_direct.jsonl"
)


def _config(
    *,
    top_k: int = 30,
    workers: int = 1,
    target_chunk_size: int = 32,
    percentile: float = 0.8,
    percentiles: dict[str, float] | None = None,
) -> SimilaritiesConfig:
    return SimilaritiesConfig(
        percentiles=percentiles or dict.fromkeys(CHANNELS, percentile),
        top_k=top_k,
        workers=workers,
        target_chunk_size=target_chunk_size,
        embeddings_path=EMBEDDINGS_PATH,
    )


def _calibration() -> NullCalibration:
    """A uniform null on [0, 1], so `percentile(x) == x` to within grid resolution.

    That keeps these tests readable in raw-score terms while still exercising the
    calibrated selection path.
    """
    grid = [index / (QUANTILE_POINTS - 1) for index in range(QUANTILE_POINTS)]
    return NullCalibration(
        sample_pairs=100,
        seed=0,
        channels={
            channel: ChannelNull(sample_size=100, quantiles=grid)
            for channel in CHANNELS
        },
    )


def _word(
    word_instance_id: str,
    first_reviewed_at: str,
    *,
    reading: str | None = "かな",
    plb_14: int | None = 3,
) -> pipeline._Word:
    entry = SimilarityEntry(
        id=word_instance_id,
        form=word_instance_id,
        meaning=f"meaning-{word_instance_id}",
        reading=reading,
    )
    phonological_entry = (
        PhonologicalSimilarityEntry.model_validate(entry.model_dump())
        if reading is not None
        else None
    )
    return pipeline._Word(
        user_id="user",
        first_reviewed_at=pd.Timestamp(first_reviewed_at, tz="UTC"),
        entry=entry,
        phonological_entry=phonological_entry,
        plb_14=plb_14,
        log_plb_14=(
            None
            if plb_14 is None
            else float(np.log(plb_14))
        ),
    )


def _history(
    timestamps: list[str],
    stabilities: list[float],
) -> pipeline._ReviewHistory:
    return pipeline._ReviewHistory(
        # UTC-naive datetime64, matching what the loader builds. Constructing int64
        # nanoseconds here previously hid a unit mismatch against the parse path.
        timestamps=pd.to_datetime(timestamps, utc=True).tz_localize(None).to_numpy(),
        stabilities=np.asarray(stabilities, dtype=np.float64),
        difficulties=np.asarray(stabilities, dtype=np.float64) + 0.5,
    )


def _orthographic_result() -> OrthographicSimilarityResult:
    return OrthographicSimilarityResult(
        ortho_stroke_sim=0.0,
        ortho_stroke_coverage=0.0,
        ortho_kana_confusion=0.0,
        ortho_kana_coverage=0.0,
    )


@pytest.mark.parametrize(
    "orthographic",
    [
        OrthographicSimilarityResult(
            ortho_stroke_sim=0.9,
            ortho_stroke_coverage=1.0,
            ortho_kana_confusion=0.0,
            ortho_kana_coverage=0.0,
        ),
        OrthographicSimilarityResult(
            ortho_stroke_sim=0.0,
            ortho_stroke_coverage=0.0,
            ortho_kana_confusion=0.9,
            ortho_kana_coverage=1.0,
        ),
    ],
)
def test_each_supported_orthographic_axis_can_qualify_a_neighbor(
    orthographic: OrthographicSimilarityResult,
) -> None:
    scores = pipeline._PairScores(
        sem_sim=0.1,
        phon_sim=None,
        mor_sim=0.1,
        orthographic=orthographic,
    )

    percentiles = pipeline._channel_percentiles(scores, _calibration())
    thresholds = _config().percentiles.by_channel()
    qualifying = [
        channel
        for channel, value in percentiles.items()
        if value is not None and value >= thresholds[channel]
    ]

    assert qualifying  # the axis with coverage carries the pair on its own
    # A channel with zero coverage yields no percentile rather than a zero one, so it
    # can neither qualify a pair nor drag a combined score down.
    assert percentiles[pipeline.PHONOLOGICAL] is None
    assert (
        percentiles[pipeline.ORTHO_KANA] is None
        or percentiles[pipeline.ORTHO_STROKE] is None
    )


def test_prepare_words_keeps_only_reviewed_japanese_word_instances() -> None:
    lexemes = pd.DataFrame(
        [
            {
                "lexeme_id": "ja-reviewed",
                "language": "ja",
                "form": "猫",
                "reading": "ねこ",
            },
            {
                "lexeme_id": "ja-unreviewed",
                "language": "ja",
                "form": "犬",
                "reading": "いぬ",
            },
            {
                "lexeme_id": "en-reviewed",
                "language": "en",
                "form": "cat",
                "reading": None,
            },
        ]
    )
    word_instances = pd.DataFrame(
        [
            {
                "word_instance_id": "word-ja",
                "lexeme_id": "ja-reviewed",
                "deck_id": "deck",
                "meaning": "cat",
                "first_reviewed_at": "2025-01-01T00:00:00Z",
                "plb_14": 4,
            },
            {
                "word_instance_id": "word-unreviewed",
                "lexeme_id": "ja-unreviewed",
                "deck_id": "deck",
                "meaning": "dog",
                "first_reviewed_at": None,
                "plb_14": None,
            },
            {
                "word_instance_id": "word-en",
                "lexeme_id": "en-reviewed",
                "deck_id": "deck",
                "meaning": "cat",
                "first_reviewed_at": "2025-01-01T00:00:00Z",
                "plb_14": 2,
            },
        ]
    )
    decks = pd.DataFrame([{"deck_id": "deck", "user_id": "user"}])

    words = pipeline._prepare_words(lexemes, word_instances, decks)

    assert [word.word_instance_id for word in words] == ["word-ja"]
    assert words[0].plb_14 == 4
    assert words[0].log_plb_14 == pytest.approx(np.log(4))


def test_prepare_words_preserves_missing_burden_target() -> None:
    lexemes = pd.DataFrame(
        [
            {
                "lexeme_id": "lexeme",
                "language": "ja",
                "form": "猫",
                "reading": "ねこ",
            }
        ]
    )
    word_instances = pd.DataFrame(
        [
            {
                "word_instance_id": "word",
                "lexeme_id": "lexeme",
                "deck_id": "deck",
                "meaning": "cat",
                "first_reviewed_at": "2025-01-01T00:00:00Z",
                "plb_14": None,
            }
        ]
    )
    decks = pd.DataFrame([{"deck_id": "deck", "user_id": "user"}])

    words = pipeline._prepare_words(lexemes, word_instances, decks)

    assert words[0].plb_14 is None
    assert words[0].log_plb_14 is None


def test_review_state_is_read_as_of_the_cutoff_through_the_real_parse_path() -> None:
    """Drives `_build_review_histories` from a DataFrame rather than hand-building a
    `_ReviewHistory`. pandas parses timestamps at microsecond resolution while
    `pd.Timestamp.value` is nanoseconds, and comparing those as raw integers made every
    cutoff land past the end of the history, returning the word's final state instead
    of its state at the time."""
    reviews = pd.DataFrame(
        [
            {
                "review_id": f"r{index}",
                "word_instance_id": "word",
                "timestamp_ms": timestamp,
                "fsrs_stability": stability,
                "fsrs_difficulty": stability + 0.5,
            }
            for index, (timestamp, stability) in enumerate(
                [
                    ("2025-01-01T00:00:00Z", 1.0),
                    ("2025-01-05T00:00:00Z", 5.0),
                    ("2025-06-01T00:00:00Z", 900.0),  # long after the cutoff
                ]
            )
        ]
    )

    histories = pipeline._build_review_histories(reviews, {"word"})
    stability, difficulty, review_count = pipeline._state_before(
        histories["word"],
        pd.Timestamp("2025-01-10", tz="UTC"),
    )

    assert review_count == 2
    assert stability == 5.0
    assert difficulty == 5.5

    # Before the word's first review at all: nothing is known about it yet.
    assert pipeline._state_before(
        histories["word"], pd.Timestamp("2024-12-31", tz="UTC")
    ) == (None, None, 0)


def test_burden_label_is_censored_at_the_cutoff() -> None:
    candidate = _word("candidate", "2025-01-01", plb_14=5)

    # Mastery came at review 5, but only 3 reviews had happened by the target's first
    # review, so the label is not yet known and the row carries the bound instead.
    censored = pipeline._CandidateState(
        candidate=candidate, stability=2.0, difficulty=5.0, review_count=3
    )
    assert pipeline._censored_burden(censored) is None

    # Once the fifth review is in the past, the label is legitimately observed.
    observed = pipeline._CandidateState(
        candidate=candidate, stability=20.0, difficulty=5.0, review_count=5
    )
    assert pipeline._censored_burden(observed) == 5


def test_censored_neighbour_reports_no_burden_but_keeps_its_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    words = [
        _word("candidate", "2025-01-01", plb_14=4),
        _word("target", "2025-01-10"),
    ]
    embeddings = {word.word_instance_id: np.array([1.0, 0.0]) for word in words}
    histories = {
        # Two reviews before the target, mastery would only come at the fourth.
        "candidate": _history(["2025-01-02", "2025-01-05"], [1.0, 3.0])
    }
    monkeypatch.setattr(
        pipeline,
        "_compute_pair_scores",
        lambda target, candidate, get_embedding, invalid_pairs: _scores(sem=0.99),
    )

    rows = list(
        pipeline._iter_similarity_rows(
            words,
            embeddings,
            histories,
            _config(),
            set(),
            _calibration(),
            show_progress=False,
        )
    )

    neighbor = rows[-1].neighbors[0]
    assert neighbor.is_censored
    assert neighbor.plb_14 is None
    assert neighbor.log_plb_14 is None
    assert neighbor.reviews == 2  # the censoring bound
    assert neighbor.s == 3.0


def test_censoring_flag_cannot_disagree_with_the_burden_label() -> None:
    fields = dict(
        word_instance_id="neighbour",
        reviews=4,
        sem_sim=0.5,
        mor_sim=0.0,
        ortho_stroke_sim=0.0,
        ortho_stroke_coverage=0.0,
        ortho_kana_confusion=0.0,
        ortho_kana_coverage=0.0,
        sem_pct=0.5,
        mor_pct=0.0,
        selected_by=("semantic",),
    )

    # Claiming censored while carrying an observed burden would let a downstream
    # filter keep a row it believes it dropped.
    with pytest.raises(ValidationError, match="is_censored must agree"):
        NeighborSimilarityResult(
            **fields, is_censored=True, plb_14=3
        )
    with pytest.raises(ValidationError, match="is_censored must agree"):
        NeighborSimilarityResult(**fields, is_censored=False)
    with pytest.raises(ValidationError, match="cannot carry log_"):
        NeighborSimilarityResult(
            **fields, is_censored=True, log_plb_14=1.4
        )


def test_missing_embeddings_are_reported_upfront() -> None:
    embeddings = pd.DataFrame(
        [{"word_instance_id": "present", "embedding": [1.0, 0.0]}]
    )

    with pytest.raises(ValueError, match=r"Missing semantic embeddings for 1.*missing"):
        pipeline._build_embedding_lookup(embeddings, {"present", "missing"})


def _embedding_frame(vectors: dict[str, list[float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"word_instance_id": word_instance_id, "embedding": vector}
            for word_instance_id, vector in vectors.items()
        ]
    )


def test_embeddings_are_mean_centered_so_a_shared_component_stops_inflating_scores() -> None:
    # A dominant shared first dimension is exactly E5's anisotropy in miniature:
    # every raw pair looks near-identical, and the discriminating dimensions are
    # swamped.
    raw = {
        "related_a": [10.0, 1.0, 0.0],
        "related_b": [10.0, 0.9, 0.1],
        "unrelated": [10.0, 0.0, 1.0],
    }
    frame = _embedding_frame(raw)
    lookup = pipeline._build_embedding_lookup(frame, set(raw))

    def cosine(vectors: dict[str, np.ndarray], left: str, right: str) -> float:
        a, b = np.asarray(vectors[left]), np.asarray(vectors[right])
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    raw_vectors = {key: np.asarray(value) for key, value in raw.items()}
    assert cosine(raw_vectors, "related_a", "unrelated") > 0.98  # indistinguishable
    assert cosine(lookup, "related_a", "unrelated") < cosine(
        raw_vectors, "related_a", "unrelated"
    )
    # Centering must sharpen the contrast, not invert the ordering.
    assert cosine(lookup, "related_a", "related_b") > cosine(
        lookup, "related_a", "unrelated"
    )


def test_centered_embeddings_are_unit_norm_and_read_only() -> None:
    raw = {"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [1.0, 1.0]}
    lookup = pipeline._build_embedding_lookup(_embedding_frame(raw), set(raw))

    stacked = np.vstack(list(lookup.values()))
    assert np.allclose(np.linalg.norm(stacked, axis=1), 1.0)
    with pytest.raises(ValueError):
        lookup["a"][0] = 0.5


def test_identical_embeddings_are_rejected_rather_than_producing_nan() -> None:
    raw = {"a": [1.0, 2.0], "b": [1.0, 2.0]}

    with pytest.raises(ValueError, match="equal the corpus centroid"):
        pipeline._build_embedding_lookup(_embedding_frame(raw), set(raw))


def test_similarity_rows_use_strict_history_and_deterministic_top_k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    words = [
        _word("b", "2025-01-01"),
        _word("a", "2025-01-02"),
        _word(
            "c",
            "2025-01-03",
            reading=None,
            plb_14=7,
        ),
        _word("target", "2025-01-05"),
    ]
    embeddings = {
        word.word_instance_id: np.array([1.0, 0.0]) for word in words
    }
    histories = {
        "b": _history(
            ["2025-01-02", "2025-01-04"],
            [1.0, 2.0],
        ),
        "a": _history(
            ["2025-01-03", "2025-01-04"],
            [1.0, 2.0],
        ),
        "c": _history(
            [
                "2025-01-03T12:00:00",
                "2025-01-04T00:00:00",
                "2025-01-04T12:00:00",
                "2025-01-05T00:00:00",
            ],
            [1.0, 2.0, 3.0, 99.0],
        ),
    }
    orthographic_calls: list[tuple[str, str]] = []

    # Distinct scores so the top_k=2 semantic selection is unambiguous: c and b in,
    # a out. Selection no longer depends on review counts, so it has to be driven by
    # the channel score here.
    semantic_by_candidate = {"c": 0.99, "b": 0.95, "a": 0.10}
    monkeypatch.setattr(
        pipeline,
        "compute_semantic_similarity",
        lambda target, candidate, get_embedding: semantic_by_candidate[candidate.id],
    )
    monkeypatch.setattr(
        pipeline,
        "compute_phonological_similarity",
        lambda target, candidate: 0.2,
    )
    monkeypatch.setattr(
        pipeline,
        "compute_morphological_similarity",
        lambda target, candidate: 0.2,
    )

    def compute_orthographic(target, candidate):
        orthographic_calls.append((target.id, candidate.id))
        return _orthographic_result()

    monkeypatch.setattr(
        pipeline,
        "compute_orthographic_similarity",
        compute_orthographic,
    )

    stats = pipeline._SimilarityRunStats()
    rows = list(
        pipeline._iter_similarity_rows(
            words,
            embeddings,
            histories,
            _config(top_k=2),
            set(),
            _calibration(),
            show_progress=False,
            stats=stats,
        )
    )

    target_row = rows[-1]
    by_id = {neighbor.word_instance_id: neighbor for neighbor in target_row.neighbors}
    assert set(by_id) == {"c", "b"}

    # The state attached to a neighbour is its state *strictly before* the target's
    # first review: c's fourth review lands exactly on the cutoff, so the 99.0
    # stability must not leak in.
    assert by_id["c"].reviews == 3
    assert by_id["c"].s == 3.0
    assert by_id["c"].d == 3.5
    assert by_id["c"].phon_sim is None
    # c reaches stability 14 on its 7th review but had only 3 by the cutoff, so its
    # burden is still unknown at this point and is reported as censored.
    assert by_id["c"].is_censored
    assert by_id["c"].plb_14 is None
    assert by_id["c"].log_plb_14 is None
    assert by_id["b"].reviews == 2

    # Every eligible pair is scored now that rank-first stopping is gone.
    assert len(orthographic_calls) == 6
    assert stats.candidate_pairs == 6
    assert stats.scored_pairs == 6


def test_words_first_reviewed_at_same_time_are_not_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    words = [
        _word("same-a", "2025-01-01"),
        _word("same-b", "2025-01-01"),
        _word("later", "2025-01-02"),
    ]
    embeddings = {
        word.word_instance_id: np.array([1.0, 0.0]) for word in words
    }
    calls: list[tuple[str, str]] = []

    monkeypatch.setattr(
        pipeline,
        "_compute_pair_scores",
        lambda target, candidate, get_embedding, invalid_pairs: (
            calls.append((target.word_instance_id, candidate.word_instance_id))
            or pipeline._PairScores(
                sem_sim=0.9,
                phon_sim=0.9,
                mor_sim=0.9,
                orthographic=_orthographic_result(),
            )
        ),
    )

    rows = list(
        pipeline._iter_similarity_rows(
            words,
            embeddings,
            {},
            _config(),
            set(),
            _calibration(),
            show_progress=False,
        )
    )

    assert rows[0].neighbors == []
    assert rows[1].neighbors == []
    assert calls == [("later", "same-a"), ("later", "same-b")]


def test_candidate_without_clean_review_state_is_retained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    words = [
        _word(
            "candidate",
            "2025-01-01",
            plb_14=None,
        ),
        _word("target", "2025-01-02"),
    ]
    embeddings = {
        word.word_instance_id: np.array([1.0, 0.0]) for word in words
    }
    monkeypatch.setattr(
        pipeline,
        "_compute_pair_scores",
        lambda target, candidate, get_embedding, invalid_pairs: pipeline._PairScores(
            sem_sim=0.9,
            phon_sim=None,
            mor_sim=0.0,
            orthographic=_orthographic_result(),
        ),
    )

    rows = list(
        pipeline._iter_similarity_rows(
            words,
            embeddings,
            {},
            _config(),
            set(),
            _calibration(),
            show_progress=False,
        )
    )

    neighbor = rows[-1].neighbors[0]
    assert neighbor.s is None
    assert neighbor.d is None
    assert neighbor.reviews == 0
    assert neighbor.plb_14 is None
    assert neighbor.log_plb_14 is None
    assert '"s":null' in rows[-1].model_dump_json()
    assert '"plb_14":null' in rows[-1].model_dump_json()


def _run_target_row(
    candidate_scores: dict[str, pipeline._PairScores],
    top_k: int = 2,
    percentiles: dict[str, float] | None = None,
):
    """Drives `_compute_target_row` with a fixed score per candidate id."""
    candidates = [
        _word(word_instance_id, f"2025-01-{index + 1:02d}")
        for index, word_instance_id in enumerate(candidate_scores)
    ]
    target = _word("target", "2025-02-01")

    return pipeline._compute_target_row(
        target=target,
        user_words=[*candidates, target],
        candidate_end=len(candidates),
        get_embedding=lambda word_instance_id: np.array([1.0, 0.0]),
        review_histories={},
        calibration=_calibration(),
        percentiles=percentiles or dict.fromkeys(CHANNELS, 0.8),
        top_k=top_k,
        invalid_phonological_pairs=set(),
    )


def _scores(
    *,
    sem: float = 0.0,
    phon: float | None = 0.0,
    mor: float = 0.0,
) -> pipeline._PairScores:
    return pipeline._PairScores(
        sem_sim=sem,
        phon_sim=phon,
        mor_sim=mor,
        orthographic=_orthographic_result(),
    )


def test_each_channel_selects_its_own_top_k(monkeypatch: pytest.MonkeyPatch) -> None:
    by_id = {
        "sem-best": _scores(sem=0.99),
        "sem-second": _scores(sem=0.95),
        "sem-third": _scores(sem=0.90),  # above threshold but outside top_k=2
        "phon-best": _scores(phon=0.99),
        "below": _scores(sem=0.5, phon=0.5, mor=0.5),
    }
    monkeypatch.setattr(
        pipeline,
        "_compute_pair_scores",
        lambda target, candidate, get_embedding, invalid_pairs: by_id[
            candidate.word_instance_id
        ],
    )

    row, scored_pairs, channel_counts = _run_target_row(by_id)
    selected = {neighbor.word_instance_id: neighbor for neighbor in row.neighbors}

    # Semantic keeps its two best; the phonological pick rides in on its own channel
    # rather than competing against semantic scores.
    assert set(selected) == {"sem-best", "sem-second", "phon-best"}
    assert channel_counts[pipeline.SEMANTIC] == 2
    assert channel_counts[pipeline.PHONOLOGICAL] == 1
    assert selected["phon-best"].selected_by == (pipeline.PHONOLOGICAL,)
    # Every candidate is scored now: no rank-first early stop.
    assert scored_pairs == len(by_id)


def test_a_pair_qualifying_on_two_channels_is_stored_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    by_id = {"both": _scores(sem=0.99, phon=0.99), "neither": _scores()}
    monkeypatch.setattr(
        pipeline,
        "_compute_pair_scores",
        lambda target, candidate, get_embedding, invalid_pairs: by_id[
            candidate.word_instance_id
        ],
    )

    row, _, _ = _run_target_row(by_id)

    assert [neighbor.word_instance_id for neighbor in row.neighbors] == ["both"]
    assert row.neighbors[0].selected_by == (
        pipeline.SEMANTIC,
        pipeline.PHONOLOGICAL,
    )


def test_equal_scores_break_ties_by_candidate_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    by_id = {name: _scores(sem=0.99) for name in ("earliest", "middle", "latest")}
    monkeypatch.setattr(
        pipeline,
        "_compute_pair_scores",
        lambda target, candidate, get_embedding, invalid_pairs: by_id[
            candidate.word_instance_id
        ],
    )

    row, _, _ = _run_target_row(by_id, top_k=2)

    # Candidates are ordered by first_reviewed_at then word_instance_id, so ties
    # resolve to the earlier-reviewed pair deterministically.
    assert [neighbor.word_instance_id for neighbor in row.neighbors] == [
        "earliest",
        "middle",
    ]


def test_a_channel_threshold_can_be_loosened_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Middling on every channel: excluded at the shared default, admitted once its
    # own channel is loosened, and only through that channel.
    by_id = {"middling": _scores(sem=0.5, phon=0.5, mor=0.5)}
    monkeypatch.setattr(
        pipeline,
        "_compute_pair_scores",
        lambda target, candidate, get_embedding, invalid_pairs: by_id[
            candidate.word_instance_id
        ],
    )

    strict, _, _ = _run_target_row(by_id)
    assert strict.neighbors == []

    loosened, _, channel_counts = _run_target_row(
        by_id,
        percentiles={**dict.fromkeys(CHANNELS, 0.8), pipeline.MORPHOLOGICAL: 0.4},
    )
    assert [n.word_instance_id for n in loosened.neighbors] == ["middling"]
    assert loosened.neighbors[0].selected_by == (pipeline.MORPHOLOGICAL,)
    assert channel_counts[pipeline.SEMANTIC] == 0


def test_percentiles_are_recorded_alongside_raw_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    by_id = {"neighbour": _scores(sem=0.99, phon=None, mor=0.95)}
    monkeypatch.setattr(
        pipeline,
        "_compute_pair_scores",
        lambda target, candidate, get_embedding, invalid_pairs: by_id[
            candidate.word_instance_id
        ],
    )

    row, _, _ = _run_target_row(by_id)
    neighbor = row.neighbors[0]

    assert neighbor.sem_sim == 0.99
    assert neighbor.sem_pct == pytest.approx(0.99, abs=1e-3)
    assert neighbor.mor_pct == pytest.approx(0.95, abs=1e-3)
    # No reading on one side means no phonological score and no percentile for it.
    assert neighbor.phon_sim is None
    assert neighbor.phon_pct is None


def test_parallel_target_chunks_match_sequential_results() -> None:
    words = [
        _word("first", "2025-01-01"),
        _word("second", "2025-01-02"),
        _word("third", "2025-01-03"),
        _word("fourth", "2025-01-04"),
    ]
    embeddings = {
        word.word_instance_id: np.array([1.0, 0.0]) for word in words
    }

    sequential_stats = pipeline._SimilarityRunStats()
    sequential = list(
        pipeline._iter_similarity_rows(
            words,
            embeddings,
            {},
            _config(top_k=2, workers=1, target_chunk_size=1),
            set(),
            _calibration(),
            show_progress=False,
            stats=sequential_stats,
        )
    )
    parallel_stats = pipeline._SimilarityRunStats()
    parallel = list(
        pipeline._iter_similarity_rows(
            words,
            embeddings,
            {},
            _config(top_k=2, workers=2, target_chunk_size=1),
            set(),
            _calibration(),
            show_progress=False,
            stats=parallel_stats,
        )
    )

    assert [row.model_dump() for row in parallel] == [
        row.model_dump() for row in sequential
    ]
    assert parallel_stats == sequential_stats
