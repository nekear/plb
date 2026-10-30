"""Stage A: score every candidate pair and cache the raw channel values.

This is the expensive half — ~340 us per pair across four channels, roughly 1.5 h for
the cohort's 75.9M pairs. Nothing here depends on selection settings, so the result is
reusable across every percentile and top_k a later run might want.

Resumable at chunk granularity: each learner's completed chunk starts are appended to
a progress file after the memmap write is flushed, so a crash costs at most one chunk.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_all_start_methods, get_context
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from tqdm import tqdm

from plb.cohort import load_cohort
from plb.similarities import cache
from plb.similarities.config import SimilaritiesConfig, load_similarities_config
from plb.similarities.pipeline import (
    _build_embedding_lookup,
    _candidate_ends,
    _compute_pair_scores,
    _group_words_by_user,
    _prepare_words,
    _TargetChunk,
    _Word,
)


@dataclass(frozen=True)
class _ScoringContext:
    words_by_user: dict[str, list[_Word]]
    candidate_ends_by_user: dict[str, list[int]]
    embeddings: dict[str, NDArray[np.float64]]


@dataclass(frozen=True)
class _ScoredChunk:
    chunk: _TargetChunk
    scores: NDArray[np.float32]
    invalid_phonological_pairs: frozenset[tuple[str, str]]


_SCORING_CONTEXT: _ScoringContext | None = None


def _initialize_worker(context: _ScoringContext) -> None:
    global _SCORING_CONTEXT
    _SCORING_CONTEXT = context


def _score_chunk_with_context(
    chunk: _TargetChunk,
    context: _ScoringContext,
) -> _ScoredChunk:
    user_words = context.words_by_user[chunk.user_id]
    candidate_ends = context.candidate_ends_by_user[chunk.user_id]
    get_embedding = context.embeddings.__getitem__
    invalid_phonological_pairs: set[tuple[str, str]] = set()

    pair_count = sum(
        candidate_ends[index] for index in range(chunk.start, chunk.stop)
    )
    scores = np.empty((pair_count, len(cache.SCORE_COLUMNS)), dtype=cache.SCORE_DTYPE)

    row = 0
    for target_index in range(chunk.start, chunk.stop):
        target = user_words[target_index]
        for candidate_index in range(candidate_ends[target_index]):
            pair = _compute_pair_scores(
                target,
                user_words[candidate_index],
                get_embedding,
                invalid_phonological_pairs,
            )
            orthographic = pair.orthographic
            scores[row] = (
                pair.sem_sim,
                # NaN rather than 0: the channel does not apply to this pair, and a
                # zero would be indistinguishable from a genuinely dissimilar reading.
                np.nan if pair.phon_sim is None else pair.phon_sim,
                pair.mor_sim,
                orthographic.ortho_stroke_sim,
                orthographic.ortho_stroke_coverage,
                orthographic.ortho_kana_confusion,
                orthographic.ortho_kana_coverage,
            )
            row += 1

    return _ScoredChunk(
        chunk=chunk,
        scores=scores,
        invalid_phonological_pairs=frozenset(invalid_phonological_pairs),
    )


def _score_chunk(chunk: _TargetChunk) -> _ScoredChunk:
    if _SCORING_CONTEXT is None:
        raise RuntimeError("Scoring worker was not initialized.")
    return _score_chunk_with_context(chunk, _SCORING_CONTEXT)


def _pending_chunks(
    words_by_user: dict[str, list[_Word]],
    chunk_size: int,
    cache_dir: Path,
) -> tuple[list[_TargetChunk], int]:
    """Chunks still to score, plus how many were already done."""
    pending: list[_TargetChunk] = []
    completed = 0
    for user_id, user_words in words_by_user.items():
        done = cache.load_completed_chunks(cache_dir, user_id)
        for start in range(0, len(user_words), chunk_size):
            if start in done:
                completed += 1
                continue
            pending.append(
                _TargetChunk(
                    user_id=user_id,
                    start=start,
                    stop=min(start + chunk_size, len(user_words)),
                )
            )
    return pending, completed


def _write_chunk(
    result: _ScoredChunk,
    memmaps: dict[str, np.memmap],
    offsets_by_user: dict[str, NDArray[np.int64]],
    cache_dir: Path,
) -> None:
    chunk = result.chunk
    offsets = offsets_by_user[chunk.user_id]
    start_row = int(offsets[chunk.start])
    stop_row = int(offsets[chunk.stop])
    if stop_row - start_row != len(result.scores):
        raise AssertionError(
            f"Chunk {chunk.user_id}[{chunk.start}:{chunk.stop}] produced "
            f"{len(result.scores):,} rows but occupies {stop_row - start_row:,}."
        )
    scores = memmaps[chunk.user_id]
    scores[start_row:stop_row] = result.scores
    scores.flush()
    # Recorded only after the flush, so a crash re-does the chunk rather than
    # trusting a half-written block.
    cache.record_completed_chunk(cache_dir, chunk.user_id, chunk.start)


def score_all_pairs(
    words: Sequence[_Word],
    embeddings: dict[str, NDArray[np.float64]],
    config: SimilaritiesConfig,
    *,
    show_progress: bool = True,
) -> cache.CacheManifest:
    words_by_user = _group_words_by_user(list(words))
    candidate_ends_by_user = {
        user_id: _candidate_ends(user_words)
        for user_id, user_words in words_by_user.items()
    }
    offsets_by_user = {
        user_id: cache.pair_offsets(ends)
        for user_id, ends in candidate_ends_by_user.items()
    }
    cache_dir = config.cache.path
    cache_dir.mkdir(parents=True, exist_ok=True)

    manifest = cache.CacheManifest(
        learners={
            user_id: cache.LearnerCacheEntry(
                pair_count=int(offsets_by_user[user_id][-1]),
                word_count=len(user_words),
                words_fingerprint=cache.fingerprint_words(
                    [word.word_instance_id for word in user_words]
                ),
            )
            for user_id, user_words in words_by_user.items()
        }
    )

    # A fingerprint mismatch means the existing blocks describe different pairs, so
    # resuming into them would interleave two datasets.
    existing = None
    if cache.manifest_path(cache_dir).exists():
        try:
            existing = cache.load_manifest(cache_dir)
            existing.verify_against(words_by_user)
        except ValueError as error:
            print(f"Discarding stale score cache: {error}", file=sys.stderr)
            existing = None
    if existing is None:
        for user_id in words_by_user:
            cache.clear_progress(cache_dir, user_id)
    cache.write_manifest(cache_dir, manifest)

    pending, already_done = _pending_chunks(
        words_by_user, config.target_chunk_size, cache_dir
    )
    total_chunks = len(pending) + already_done
    if already_done:
        print(
            f"Resuming: {already_done:,} of {total_chunks:,} chunks already scored."
        )
    if not pending:
        print("Score cache is already complete.")
        return manifest

    memmaps = {
        user_id: cache.open_scores(
            cache_dir,
            user_id,
            manifest.learners[user_id].pair_count,
            mode="r+" if cache.scores_path(cache_dir, user_id).exists() else "w+",
        )
        for user_id in words_by_user
    }
    context = _ScoringContext(
        words_by_user=words_by_user,
        candidate_ends_by_user=candidate_ends_by_user,
        embeddings=embeddings,
    )
    invalid_phonological_pairs: set[tuple[str, str]] = set()
    pending_pairs = sum(
        int(offsets_by_user[chunk.user_id][chunk.stop])
        - int(offsets_by_user[chunk.user_id][chunk.start])
        for chunk in pending
    )
    progress = tqdm(
        total=pending_pairs,
        desc="Scoring pairs",
        unit="pair",
        unit_scale=True,
        disable=not show_progress,
    )

    def handle(result: _ScoredChunk) -> None:
        _write_chunk(result, memmaps, offsets_by_user, cache_dir)
        invalid_phonological_pairs.update(result.invalid_phonological_pairs)
        progress.update(len(result.scores))

    try:
        if config.workers == 1:
            for chunk in pending:
                handle(_score_chunk_with_context(chunk, context))
        else:
            forking = "fork" in get_all_start_methods()
            multiprocessing_context = get_context("fork" if forking else None)
            if forking:
                # Inherited copy-on-write instead of pickled per worker; the embedding
                # matrix alone is ~300MB.
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
                for result in executor.map(_score_chunk, pending, chunksize=1):
                    handle(result)
    finally:
        progress.close()
        for scores in memmaps.values():
            scores.flush()

    if invalid_phonological_pairs:
        examples = sorted(invalid_phonological_pairs)[:5]
        print(
            "Phonological similarity could not be computed for "
            f"{len(invalid_phonological_pairs):,} unique reading pairs; those pairs "
            f"are cached as NaN. Examples: {examples}",
            file=sys.stderr,
        )
    return manifest


def main() -> None:
    config = load_similarities_config()
    cohort_user_ids = load_cohort().user_ids()

    words = _prepare_words(
        pd.read_json(config.lexemes_path, lines=True),
        pd.read_json(config.word_instances_path, lines=True),
        pd.read_json(config.decks_path, lines=True),
    )
    words = [word for word in words if word.user_id in cohort_user_ids]
    if not words:
        raise ValueError(
            "No reviewed Japanese word instances belong to the configured cohort; "
            "check cohort.learners in params.yaml."
        )

    embeddings = _build_embedding_lookup(
        pd.read_json(config.embeddings_path, lines=True),
        {word.word_instance_id for word in words},
    )

    manifest = score_all_pairs(words, embeddings, config)
    total_pairs = sum(entry.pair_count for entry in manifest.learners.values())
    print(
        f"Cached {total_pairs:,} pair scores for {len(manifest.learners)} learners "
        f"in {config.cache.path}"
    )


if __name__ == "__main__":
    main()
