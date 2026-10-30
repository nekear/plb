from __future__ import annotations

import sys
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from multiprocessing import get_all_start_methods, get_context

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from tqdm import tqdm

from .calibration import (
    CHANNELS,
    MORPHOLOGICAL,
    ORTHO_KANA,
    ORTHO_STROKE,
    PHONOLOGICAL,
    SEMANTIC,
    NullCalibration,
    build_null_calibration,
    channel_values,
)
from .config import SimilaritiesConfig, load_similarities_config
from .morphological import compute_morphological_similarity
from .orthographic import compute_orthographic_similarity
from .phonological import (
    PhonologicalSimilarityComputationError,
    compute_phonological_similarity,
)
from .semantic import compute_semantic_similarity
from plb.cohort import load_cohort
from plb.utils import write_json
from .types import (
    NeighborSimilarityResult,
    OrthographicSimilarityResult,
    PhonologicalSimilarityEntry,
    SimilarityEntry,
    WordSimilaritiesResult,
)

JAPANESE_LANGUAGE = "ja"


@dataclass(frozen=True)
class _Word:
    user_id: str
    first_reviewed_at: pd.Timestamp
    entry: SimilarityEntry
    phonological_entry: PhonologicalSimilarityEntry | None
    plb_14: int | None
    log_plb_14: float | None

    @property
    def word_instance_id(self) -> str:
        return self.entry.id


@dataclass(frozen=True)
class _ReviewHistory:
    # datetime64, not int64: numpy then reconciles resolutions on comparison instead
    # of us having to assume both sides use the same unit.
    timestamps: NDArray[np.datetime64]
    stabilities: NDArray[np.float64]
    difficulties: NDArray[np.float64]


@dataclass(frozen=True)
class _PairScores:
    sem_sim: float
    phon_sim: float | None
    mor_sim: float
    orthographic: OrthographicSimilarityResult


@dataclass(frozen=True)
class _CandidateState:
    candidate: _Word
    stability: float | None
    difficulty: float | None
    review_count: int


@dataclass(frozen=True)
class _TargetChunk:
    user_id: str
    start: int
    stop: int


@dataclass(frozen=True)
class _WorkerContext:
    words_by_user: dict[str, list[_Word]]
    candidate_ends_by_user: dict[str, list[int]]
    embeddings: dict[str, NDArray[np.float64]]
    review_histories: dict[str, _ReviewHistory]
    calibration: NullCalibration
    percentiles: dict[str, float]
    top_k: int


@dataclass(frozen=True)
class _ChunkResult:
    rows: tuple[WordSimilaritiesResult, ...]
    candidate_pairs: int
    scored_pairs: int
    channel_counts: dict[str, int]
    saturated_targets: dict[str, int]
    invalid_phonological_pairs: frozenset[tuple[str, str]]


@dataclass
class _SimilarityRunStats:
    candidate_pairs: int = 0
    scored_pairs: int = 0
    targets: int = 0
    # Selected neighbours per channel, and how often a channel filled its top_k —
    # the two numbers needed to tell whether `percentile` is set sensibly.
    channel_counts: dict[str, int] = field(default_factory=lambda: dict.fromkeys(CHANNELS, 0))
    saturated_targets: dict[str, int] = field(default_factory=lambda: dict.fromkeys(CHANNELS, 0))


_WORKER_CONTEXT: _WorkerContext | None = None


def _require_columns(
    dataframe: pd.DataFrame,
    required: set[str],
    dataset_name: str,
) -> None:
    missing = sorted(required - set(dataframe.columns))
    if missing:
        raise ValueError(f"{dataset_name} is missing required columns: {missing}")


def _optional_string(value: object) -> str | None:
    if value is None or bool(pd.isna(value)):
        return None

    text = str(value).strip()
    return text or None


def _parse_utc_timestamps(values: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(values.dtype):
        return pd.to_datetime(values, unit="ms", utc=True, errors="raise")
    return pd.to_datetime(values, utc=True, errors="raise")


def _prepare_words(
    lexemes_df: pd.DataFrame,
    word_instances_df: pd.DataFrame,
    decks_df: pd.DataFrame,
) -> list[_Word]:
    _require_columns(
        lexemes_df,
        {"lexeme_id", "language", "form", "reading"},
        "lexemes",
    )
    _require_columns(
        word_instances_df,
        {
            "word_instance_id",
            "lexeme_id",
            "deck_id",
            "meaning",
            "first_reviewed_at",
            "plb_14",
        },
        "word instances",
    )
    _require_columns(decks_df, {"deck_id", "user_id"}, "decks")

    duplicate_word_ids = word_instances_df.loc[
        word_instances_df["word_instance_id"].duplicated(keep=False),
        "word_instance_id",
    ]
    if not duplicate_word_ids.empty:
        examples = duplicate_word_ids.astype(str).drop_duplicates().head(5).tolist()
        raise ValueError(f"Duplicate word_instance_id values: {examples}")

    words = word_instances_df[
        [
            "word_instance_id",
            "lexeme_id",
            "deck_id",
            "meaning",
            "first_reviewed_at",
            "plb_14",
        ]
    ].merge(
        lexemes_df[["lexeme_id", "language", "form", "reading"]],
        on="lexeme_id",
        how="left",
        validate="many_to_one",
        indicator="_lexeme_merge",
    )
    missing_lexemes = words.loc[
        words["_lexeme_merge"] != "both", "lexeme_id"
    ].astype(str)
    if not missing_lexemes.empty:
        missing_lexeme_ids = missing_lexemes.drop_duplicates()
        raise ValueError(
            f"Word instances reference {len(missing_lexeme_ids):,} missing lexemes; "
            f"examples: {missing_lexeme_ids.head(5).tolist()}"
        )

    words = words.loc[words["language"] == JAPANESE_LANGUAGE].drop(
        columns="_lexeme_merge"
    )
    words = words.merge(
        decks_df[["deck_id", "user_id"]],
        on="deck_id",
        how="left",
        validate="many_to_one",
        indicator="_deck_merge",
    )
    missing_decks = words.loc[
        (words["_deck_merge"] != "both") | words["user_id"].isna(),
        "deck_id",
    ].astype(str)
    if not missing_decks.empty:
        raise ValueError(
            "Japanese word instances reference missing deck users: "
            f"{missing_decks.drop_duplicates().head(5).tolist()}"
        )

    words["first_reviewed_at"] = _parse_utc_timestamps(
        words["first_reviewed_at"]
    )
    words = words.loc[words["first_reviewed_at"].notna()].drop(
        columns="_deck_merge"
    )

    burden_targets = pd.to_numeric(
        words["plb_14"],
        errors="raise",
    )
    available_targets = burden_targets.dropna()
    target_values = available_targets.to_numpy(dtype=np.float64)
    invalid_target_positions = (
        ~np.isfinite(target_values)
        | (target_values < 1.0)
        | (target_values != np.floor(target_values))
    )
    if invalid_target_positions.any():
        invalid_indices = available_targets.index[invalid_target_positions]
        examples = words.loc[
            invalid_indices,
            ["word_instance_id", "plb_14"],
        ].head(5)
        raise ValueError(
            "Reviewed Japanese word instances have invalid "
            "plb_14 values; expected positive integers or null. "
            f"Examples: {examples.to_dict(orient='records')}"
        )
    words["plb_14"] = burden_targets

    prepared: list[_Word] = []
    for row in words.itertuples(index=False):
        reading = _optional_string(row.reading)
        plb_14 = (
            None
            if pd.isna(row.plb_14)
            else int(row.plb_14)
        )
        entry = SimilarityEntry(
            id=str(row.word_instance_id),
            form=str(row.form),
            meaning=str(row.meaning),
            reading=reading,
        )
        phonological_entry = (
            PhonologicalSimilarityEntry.model_validate(entry.model_dump())
            if reading is not None
            else None
        )
        prepared.append(
            _Word(
                user_id=str(row.user_id),
                first_reviewed_at=pd.Timestamp(row.first_reviewed_at),
                entry=entry,
                phonological_entry=phonological_entry,
                plb_14=plb_14,
                log_plb_14=(
                    None
                    if plb_14 is None
                    else float(np.log(plb_14))
                ),
            )
        )

    return sorted(
        prepared,
        key=lambda word: (
            word.user_id,
            word.first_reviewed_at,
            word.word_instance_id,
        ),
    )


def _build_embedding_lookup(
    embeddings_df: pd.DataFrame,
    required_word_ids: set[str],
) -> dict[str, NDArray[np.float64]]:
    _require_columns(
        embeddings_df,
        {"word_instance_id", "embedding"},
        "word-instance embeddings",
    )

    embedding_ids = embeddings_df["word_instance_id"].astype(str)
    duplicate_ids = embedding_ids[embedding_ids.duplicated(keep=False)]
    if not duplicate_ids.empty:
        raise ValueError(
            "Duplicate word-instance embeddings: "
            f"{duplicate_ids.drop_duplicates().head(5).tolist()}"
        )

    relevant = embeddings_df.loc[embedding_ids.isin(required_word_ids)].copy()
    relevant["word_instance_id"] = relevant["word_instance_id"].astype(str)
    available_ids = set(relevant["word_instance_id"])
    missing_ids = sorted(required_word_ids - available_ids)
    if missing_ids:
        raise ValueError(
            f"Missing semantic embeddings for {len(missing_ids):,} reviewed Japanese "
            f"word instances; examples: {missing_ids[:5]}. Re-run the embedding stage "
            "against the current word-instance dataset."
        )

    word_instance_ids: list[str] = []
    embeddings: list[NDArray[np.float64]] = []
    expected_dimensions: int | None = None
    for row in relevant.itertuples(index=False):
        word_instance_id = str(row.word_instance_id)
        embedding = np.asarray(row.embedding, dtype=np.float64)
        if embedding.ndim != 1 or embedding.size == 0:
            raise ValueError(
                f"Embedding for word instance {word_instance_id} is not a non-empty vector."
            )
        if not np.isfinite(embedding).all():
            raise ValueError(
                f"Embedding for word instance {word_instance_id} contains non-finite values."
            )

        if expected_dimensions is None:
            expected_dimensions = int(embedding.size)
        elif embedding.size != expected_dimensions:
            raise ValueError(
                f"Embedding for word instance {word_instance_id} has {embedding.size} "
                f"dimensions; expected {expected_dimensions}."
            )

        word_instance_ids.append(word_instance_id)
        embeddings.append(embedding)

    matrix = _mean_center(np.vstack(embeddings), word_instance_ids)
    matrix.setflags(write=False)
    return dict(zip(word_instance_ids, matrix, strict=True))


def _mean_center(
    matrix: NDArray[np.float64],
    word_instance_ids: list[str],
) -> NDArray[np.float64]:
    centered = matrix - matrix.mean(axis=0)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)

    degenerate = np.flatnonzero(norms[:, 0] == 0.0)
    if degenerate.size:
        examples = [word_instance_ids[index] for index in degenerate[:5]]
        raise ValueError(
            f"{degenerate.size:,} embedding(s) equal the corpus centroid exactly, so "
            f"mean-centering leaves a zero vector; examples: {examples}. This means "
            "the embeddings are degenerate (identical, or only one word instance)."
        )

    return centered / norms


def _build_review_histories(
    reviews_df: pd.DataFrame,
    relevant_word_ids: set[str],
) -> dict[str, _ReviewHistory]:
    _require_columns(
        reviews_df,
        {
            "review_id",
            "word_instance_id",
            "timestamp_ms",
            "fsrs_stability",
            "fsrs_difficulty",
        },
        "FSRS-replayed reviews",
    )

    review_ids = reviews_df["word_instance_id"].astype(str)
    reviews = reviews_df.loc[review_ids.isin(relevant_word_ids)].copy()
    if reviews.empty:
        return {}

    reviews["word_instance_id"] = reviews["word_instance_id"].astype(str)
    reviews["review_id"] = reviews["review_id"].astype(str)
    reviews["_timestamp"] = _parse_utc_timestamps(reviews["timestamp_ms"])
    if reviews["_timestamp"].isna().any():
        bad_ids = reviews.loc[
            reviews["_timestamp"].isna(), "review_id"
        ].head(5).tolist()
        raise ValueError(f"FSRS-replayed reviews have missing timestamps: {bad_ids}")

    for column in ("fsrs_stability", "fsrs_difficulty"):
        reviews[column] = pd.to_numeric(reviews[column], errors="raise")
        if reviews[column].isna().any() or not np.isfinite(reviews[column]).all():
            bad_ids = reviews.loc[
                reviews[column].isna() | ~np.isfinite(reviews[column]), "review_id"
            ].head(5).tolist()
            raise ValueError(
                f"FSRS-replayed reviews have invalid {column} values: {bad_ids}"
            )

    reviews = reviews.sort_values(
        ["word_instance_id", "_timestamp", "review_id"],
        kind="stable",
    )
    histories: dict[str, _ReviewHistory] = {}
    for word_instance_id, group in reviews.groupby(
        "word_instance_id",
        sort=False,
    ):
        histories[str(word_instance_id)] = _ReviewHistory(
            # Kept as datetime64 rather than reduced to int64. pandas parses these at
            # microsecond resolution while pd.Timestamp.value is always nanoseconds,
            # so comparing raw integers silently placed every cutoff past the end of
            # the history and returned the word's final state for every pair.
            # Timezone is dropped (the series is already UTC) because a tz-aware
            # Series converts to an object array, which compares by Python objects.
            timestamps=group["_timestamp"].dt.tz_localize(None).to_numpy(),
            stabilities=group["fsrs_stability"].to_numpy(dtype=np.float64),
            difficulties=group["fsrs_difficulty"].to_numpy(dtype=np.float64),
        )

    return histories


def _as_naive_utc(timestamp: pd.Timestamp) -> np.datetime64:
    """UTC-naive datetime64, so numpy compares by value and reconciles resolutions."""
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp.to_datetime64()


def _state_before(
    history: _ReviewHistory | None,
    cutoff: pd.Timestamp,
) -> tuple[float | None, float | None, int]:
    """The word's FSRS state strictly before `cutoff`, and how many reviews it had.

    Returns `(None, None, 0)` when the learner had not reviewed the word yet.
    """
    if history is None:
        return None, None, 0

    review_count = int(
        np.searchsorted(history.timestamps, _as_naive_utc(cutoff), side="left")
    )
    if review_count == 0:
        return None, None, 0

    latest_index = review_count - 1
    return (
        float(history.stabilities[latest_index]),
        float(history.difficulties[latest_index]),
        review_count,
    )


def _compute_pair_scores(
    target: _Word,
    candidate: _Word,
    get_embedding: Callable[[str], NDArray[np.float64]],
    invalid_phonological_pairs: set[tuple[str, str]],
) -> _PairScores:
    semantic = compute_semantic_similarity(
        target.entry,
        candidate.entry,
        get_embedding,
    )

    phonological: float | None = None
    if (
        target.phonological_entry is not None
        and candidate.phonological_entry is not None
    ):
        try:
            phonological = compute_phonological_similarity(
                target.phonological_entry,
                candidate.phonological_entry,
            )
        except PhonologicalSimilarityComputationError:
            invalid_phonological_pairs.add(
                tuple(
                    sorted(
                        (
                            target.phonological_entry.reading,
                            candidate.phonological_entry.reading,
                        )
                    )
                )
            )

    morphological = compute_morphological_similarity(target.entry, candidate.entry)
    orthographic = compute_orthographic_similarity(target.entry, candidate.entry)
    return _PairScores(
        sem_sim=semantic,
        phon_sim=phonological,
        mor_sim=morphological,
        orthographic=orthographic,
    )


def _channel_percentiles(
    scores: _PairScores,
    calibration: NullCalibration,
) -> dict[str, float | None]:
    return {
        channel: calibration.percentile(channel, value)
        for channel, value in channel_values(scores).items()
    }


def _group_words_by_user(words: list[_Word]) -> dict[str, list[_Word]]:
    words_by_user: dict[str, list[_Word]] = defaultdict(list)
    for word in words:
        words_by_user[word.user_id].append(word)

    for user_words in words_by_user.values():
        user_words.sort(
            key=lambda word: (
                word.first_reviewed_at,
                word.word_instance_id,
            )
        )

    return dict(words_by_user)


def _candidate_ends(user_words: list[_Word]) -> list[int]:
    """Returns the strict earlier-word prefix length for every target."""
    ends: list[int] = []
    batch_start = 0
    previous_timestamp: pd.Timestamp | None = None

    for index, word in enumerate(user_words):
        if previous_timestamp is None or word.first_reviewed_at != previous_timestamp:
            batch_start = index
            previous_timestamp = word.first_reviewed_at
        ends.append(batch_start)

    return ends


def _candidate_pair_count(words_by_user: dict[str, list[_Word]]) -> int:
    return sum(
        sum(_candidate_ends(user_words))
        for user_words in words_by_user.values()
    )


def _censored_burden(candidate_state: _CandidateState) -> int | None:
    """
        The neighbour's PLB as known at the target's first review, else `None`.
    """
    burden = candidate_state.candidate.plb_14
    if burden is None or burden > candidate_state.review_count:
        return None
    return burden


def _neighbor_result(
    candidate_state: _CandidateState,
    scores: _PairScores,
    percentiles: dict[str, float | None],
    selected_by: tuple[str, ...],
) -> NeighborSimilarityResult:
    orthographic = scores.orthographic
    burden = _censored_burden(candidate_state)
    return NeighborSimilarityResult(
        sem_pct=percentiles[SEMANTIC],
        phon_pct=percentiles[PHONOLOGICAL],
        mor_pct=percentiles[MORPHOLOGICAL],
        ortho_stroke_pct=percentiles[ORTHO_STROKE],
        ortho_kana_pct=percentiles[ORTHO_KANA],
        selected_by=selected_by,
        word_instance_id=candidate_state.candidate.word_instance_id,
        s=candidate_state.stability,
        d=candidate_state.difficulty,
        reviews=candidate_state.review_count,
        plb_14=burden,
        log_plb_14=(
            None
            if burden is None
            else candidate_state.candidate.log_plb_14
        ),
        is_censored=burden is None,
        sem_sim=scores.sem_sim,
        phon_sim=scores.phon_sim,
        mor_sim=scores.mor_sim,
        ortho_stroke_sim=orthographic.ortho_stroke_sim,
        ortho_stroke_coverage=orthographic.ortho_stroke_coverage,
        ortho_kana_confusion=orthographic.ortho_kana_confusion,
        ortho_kana_coverage=orthographic.ortho_kana_coverage,
    )


def _compute_target_row(
    target: _Word,
    user_words: list[_Word],
    candidate_end: int,
    get_embedding: Callable[[str], NDArray[np.float64]],
    review_histories: dict[str, _ReviewHistory],
    calibration: NullCalibration,
    percentiles: Mapping[str, float],
    top_k: int,
    invalid_phonological_pairs: set[tuple[str, str]],
) -> tuple[WordSimilaritiesResult, int, dict[str, int]]:
    """
        Scores every eligible candidate, then keeps the top K per channel.
    """
    scored: list[tuple[_CandidateState, _PairScores, dict[str, float | None]]] = []
    for candidate_index in range(candidate_end):
        candidate = user_words[candidate_index]
        stability, difficulty, review_count = _state_before(
            review_histories.get(candidate.word_instance_id),
            target.first_reviewed_at,
        )
        scores = _compute_pair_scores(
            target,
            candidate,
            get_embedding,
            invalid_phonological_pairs,
        )
        scored.append(
            (
                _CandidateState(
                    candidate=candidate,
                    stability=stability,
                    difficulty=difficulty,
                    review_count=review_count,
                ),
                scores,
                _channel_percentiles(scores, calibration),
            )
        )

    # Candidate order is already the deterministic tie-break: earlier
    # first_reviewed_at, then word_instance_id.
    selected_by: dict[int, list[str]] = defaultdict(list)
    channel_counts: dict[str, int] = {}
    for channel in CHANNELS:
        threshold = percentiles[channel]
        qualifying = [
            (position, pair_percentiles[channel])
            for position, (_, _, pair_percentiles) in enumerate(scored)
            if pair_percentiles[channel] is not None
            and pair_percentiles[channel] >= threshold
        ]
        qualifying.sort(key=lambda item: (-item[1], item[0]))
        chosen = qualifying[:top_k]
        channel_counts[channel] = len(chosen)
        for position, _ in chosen:
            selected_by[position].append(channel)

    neighbors = [
        _neighbor_result(
            scored[position][0],
            scored[position][1],
            scored[position][2],
            tuple(channels),
        )
        for position, channels in sorted(selected_by.items())
    ]
    return (
        WordSimilaritiesResult(
            word_instance_id=target.word_instance_id,
            neighbors=neighbors,
        ),
        len(scored),
        channel_counts,
    )


def _process_target_chunk_with_context(
    chunk: _TargetChunk,
    context: _WorkerContext,
) -> _ChunkResult:
    user_words = context.words_by_user[chunk.user_id]
    candidate_ends = context.candidate_ends_by_user[chunk.user_id]
    get_embedding = context.embeddings.__getitem__
    invalid_phonological_pairs: set[tuple[str, str]] = set()
    rows: list[WordSimilaritiesResult] = []
    candidate_pairs = 0
    scored_pairs = 0
    channel_counts = dict.fromkeys(CHANNELS, 0)
    saturated_targets = dict.fromkeys(CHANNELS, 0)

    for target_index in range(chunk.start, chunk.stop):
        candidate_end = candidate_ends[target_index]
        row, target_scored_pairs, target_channel_counts = _compute_target_row(
            target=user_words[target_index],
            user_words=user_words,
            candidate_end=candidate_end,
            get_embedding=get_embedding,
            review_histories=context.review_histories,
            calibration=context.calibration,
            percentiles=context.percentiles,
            top_k=context.top_k,
            invalid_phonological_pairs=invalid_phonological_pairs,
        )
        rows.append(row)
        candidate_pairs += candidate_end
        scored_pairs += target_scored_pairs
        for channel, count in target_channel_counts.items():
            channel_counts[channel] += count
            if count == context.top_k:
                saturated_targets[channel] += 1

    return _ChunkResult(
        rows=tuple(rows),
        candidate_pairs=candidate_pairs,
        scored_pairs=scored_pairs,
        channel_counts=channel_counts,
        saturated_targets=saturated_targets,
        invalid_phonological_pairs=frozenset(invalid_phonological_pairs),
    )


def _initialize_worker(context: _WorkerContext) -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = context


def _process_target_chunk(chunk: _TargetChunk) -> _ChunkResult:
    if _WORKER_CONTEXT is None:
        raise RuntimeError("Similarity worker was not initialized.")
    return _process_target_chunk_with_context(chunk, _WORKER_CONTEXT)


def _target_chunks(
    words_by_user: dict[str, list[_Word]],
    chunk_size: int,
) -> list[_TargetChunk]:
    return [
        _TargetChunk(
            user_id=user_id,
            start=start,
            stop=min(start + chunk_size, len(user_words)),
        )
        for user_id, user_words in words_by_user.items()
        for start in range(0, len(user_words), chunk_size)
    ]


def _worker_context(
    words_by_user: dict[str, list[_Word]],
    embeddings: dict[str, NDArray[np.float64]],
    review_histories: dict[str, _ReviewHistory],
    config: SimilaritiesConfig,
    calibration: NullCalibration,
) -> _WorkerContext:
    return _WorkerContext(
        words_by_user=words_by_user,
        candidate_ends_by_user={
            user_id: _candidate_ends(user_words)
            for user_id, user_words in words_by_user.items()
        },
        embeddings=embeddings,
        review_histories=review_histories,
        calibration=calibration,
        percentiles=config.percentiles.by_channel(),
        top_k=config.top_k,
    )


def _record_chunk_result(
    result: _ChunkResult,
    invalid_phonological_pairs: set[tuple[str, str]],
    stats: _SimilarityRunStats,
    progress: tqdm,
) -> tuple[WordSimilaritiesResult, ...]:
    invalid_phonological_pairs.update(result.invalid_phonological_pairs)
    stats.candidate_pairs += result.candidate_pairs
    stats.scored_pairs += result.scored_pairs
    stats.targets += len(result.rows)
    for channel, count in result.channel_counts.items():
        stats.channel_counts[channel] += count
        stats.saturated_targets[channel] += result.saturated_targets[channel]
    progress.update(len(result.rows))
    progress.set_postfix(
        scored_pairs=f"{stats.scored_pairs:,}",
        refresh=False,
    )
    return result.rows


def _iter_similarity_rows(
    words: list[_Word],
    embeddings: dict[str, NDArray[np.float64]],
    review_histories: dict[str, _ReviewHistory],
    config: SimilaritiesConfig,
    invalid_phonological_pairs: set[tuple[str, str]],
    calibration: NullCalibration,
    *,
    show_progress: bool = True,
    stats: _SimilarityRunStats | None = None,
) -> Iterator[WordSimilaritiesResult]:
    words_by_user = _group_words_by_user(words)
    context = _worker_context(
        words_by_user,
        embeddings,
        review_histories,
        config,
        calibration,
    )
    chunks = _target_chunks(words_by_user, config.target_chunk_size)
    run_stats = stats if stats is not None else _SimilarityRunStats()
    progress = tqdm(
        total=len(words),
        desc="Computing neighbor lists",
        unit="word",
        disable=not show_progress,
    )

    try:
        if config.workers == 1:
            for chunk in chunks:
                result = _process_target_chunk_with_context(chunk, context)
                yield from _record_chunk_result(
                    result,
                    invalid_phonological_pairs,
                    run_stats,
                    progress,
                )
        else:
            forking = "fork" in get_all_start_methods()
            multiprocessing_context = get_context("fork" if forking else None)
            if forking:
                _initialize_worker(context)
                pool_arguments = {}
            else:
                pool_arguments = {
                    "initializer": _initialize_worker,
                    "initargs": (context,),
                }
            with ProcessPoolExecutor(
                max_workers=config.workers,
                mp_context=multiprocessing_context,
                **pool_arguments,
            ) as executor:
                results = executor.map(
                    _process_target_chunk,
                    chunks,
                    chunksize=1,
                )
                for result in results:
                    yield from _record_chunk_result(
                        result,
                        invalid_phonological_pairs,
                        run_stats,
                        progress,
                    )
    finally:
        progress.close()


def main() -> None:
    config = load_similarities_config()
    lexemes_df = pd.read_json(config.lexemes_path, lines=True)
    decks_df = pd.read_json(config.decks_path, lines=True)
    word_instances_df = pd.read_json(config.word_instances_path, lines=True)
    words = _prepare_words(lexemes_df, word_instances_df, decks_df)
    del lexemes_df, decks_df, word_instances_df

    # Scoped here rather than inside _prepare_words so that stays cohort-agnostic.
    # Everything downstream — the null calibration, the candidate prefixes, the
    # output — inherits the cohort from this one filter.
    cohort_user_ids = load_cohort().user_ids()
    scoped = [word for word in words if word.user_id in cohort_user_ids]
    if not scoped:
        raise ValueError(
            "No reviewed Japanese word instances belong to the configured cohort; "
            "check cohort.learners in params.yaml."
        )
    print(
        f"Cohort: {len({word.user_id for word in scoped})} learners, "
        f"{len(scoped):,} of {len(words):,} word instances."
    )
    words = scoped

    required_word_ids = {word.word_instance_id for word in words}
    embeddings_df = pd.read_json(config.embeddings_path, lines=True)
    embeddings = _build_embedding_lookup(embeddings_df, required_word_ids)
    del embeddings_df

    reviews_df = pd.read_json(config.reviews_path, lines=True)
    review_histories = _build_review_histories(reviews_df, required_word_ids)
    del reviews_df

    missing_readings = sum(word.phonological_entry is None for word in words)
    if missing_readings:
        print(
            f"{missing_readings:,} reviewed Japanese word instances have no reading; "
            "phon_sim will be null for their pairs.",
            file=sys.stderr,
        )

    missing_burden_targets = sum(
        word.plb_14 is None for word in words
    )
    if missing_burden_targets:
        print(
            f"{missing_burden_targets:,} reviewed Japanese word instances never "
            "reached the stability target; their plb_14 and log_plb_14 are null wherever "
            "they occur as neighbors, on top of the per-pair censoring.",
            file=sys.stderr,
        )

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    invalid_phonological_pairs: set[tuple[str, str]] = set()

    # The null has to be built before scoring starts: it defines what every channel's
    # percentile means, and it is written out so a deployment scores new users against
    # the same boundaries rather than re-deriving its own.
    calibration_pairs: set[tuple[str, str]] = set()
    calibration = build_null_calibration(
        _group_words_by_user(words),
        lambda left, right: _compute_pair_scores(
            left,
            right,
            embeddings.__getitem__,
            calibration_pairs,
        ),
        sample_pairs=config.calibration.sample_pairs,
        seed=config.calibration.seed,
    )
    config.calibration.output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(config.calibration.output_path, calibration.model_dump(mode="json"))
    print(
        f"Calibrated {len(calibration.channels)} channels on "
        f"{config.calibration.sample_pairs:,} random pairs -> "
        f"{config.calibration.output_path}"
    )

    run_stats = _SimilarityRunStats()
    row_count = 0
    with config.output_path.open("w", encoding="utf-8") as output:
        for row in _iter_similarity_rows(
            words,
            embeddings,
            review_histories,
            config,
            invalid_phonological_pairs,
            calibration,
            stats=run_stats,
        ):
            output.write(row.model_dump_json())
            output.write("\n")
            row_count += 1

    if invalid_phonological_pairs:
        examples = sorted(invalid_phonological_pairs)[:5]
        print(
            "Phonological similarity could not be computed for "
            f"{len(invalid_phonological_pairs):,} unique reading pairs; "
            f"phon_sim was written as null. Examples: {examples}",
            file=sys.stderr,
        )

    print(f"Scored {run_stats.scored_pairs:,}/{run_stats.candidate_pairs:,} eligible pairs.")
    if run_stats.targets:
        percentiles = config.percentiles.by_channel()
        print(f"Per-channel selection (top_k={config.top_k}):")
        for channel in CHANNELS:
            selected = run_stats.channel_counts[channel]
            saturated = run_stats.saturated_targets[channel]
            print(
                f"  {channel:<14} p>={percentiles[channel]:<7} "
                f"{selected:>12,} neighbours "
                f"({selected / run_stats.targets:6.2f} per target, "
                f"{saturated / run_stats.targets:6.1%} of targets at top_k)"
            )
    print(f"Wrote {row_count:,} Japanese word rows to {config.output_path}")


if __name__ == "__main__":
    main()
