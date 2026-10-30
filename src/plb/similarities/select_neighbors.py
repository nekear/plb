"""Stage B: turn cached pair scores into neighbour lists.

Everything here is cheap — a percentile lookup and a sort per target — so retuning
`similarities.percentiles` or `top_k` costs a couple of minutes rather than the 1.5 h
the scoring stage takes.

The null calibration is drawn from the cache too, so changing `sample_pairs` or the
seed likewise avoids re-scoring. Note this samples random *candidate* pairs rather
than random word pairs: a closer match to the population actually scored, and the
distribution is the same since similarity does not depend on review order.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator, Sequence
from typing import NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from tqdm import tqdm

from plb.cohort import load_cohort
from plb.similarities import cache
from plb.similarities.calibration import (
    CHANNELS,
    MORPHOLOGICAL,
    ORTHO_KANA,
    ORTHO_STROKE,
    PHONOLOGICAL,
    SEMANTIC,
    NullCalibration,
    build_null_from_samples,
)
from plb.similarities.config import SimilaritiesConfig, load_similarities_config
from plb.similarities.pipeline import (
    _build_review_histories,
    _candidate_ends,
    _as_naive_utc,
    _CandidateState,
    _censored_burden,
    _group_words_by_user,
    _neighbor_result,
    _PairScores,
    _prepare_words,
    _ReviewHistory,
    _SimilarityRunStats,
    _state_before,
    _Word,
)
from plb.similarities.types import (
    NeighborSimilarityResult,
    OrthographicSimilarityResult,
    WordSimilaritiesResult,
)
from plb.utils import write_json

# Column positions in the cached score block.
_COLUMN = {name: index for index, name in enumerate(cache.SCORE_COLUMNS)}

# Which cached column carries each channel's score, and which carries the coverage
# that decides whether the channel applies at all.
_CHANNEL_COLUMNS: dict[str, tuple[str, str | None]] = {
    SEMANTIC: ("sem_sim", None),
    PHONOLOGICAL: ("phon_sim", None),  # inapplicability is encoded as NaN
    MORPHOLOGICAL: ("mor_sim", None),
    ORTHO_STROKE: ("ortho_stroke_sim", "ortho_stroke_coverage"),
    ORTHO_KANA: ("ortho_kana_confusion", "ortho_kana_coverage"),
}


def channel_matrix(block: NDArray[np.float32]) -> dict[str, NDArray[np.float64]]:
    """Per-channel score vectors for a target's candidates, NaN where inapplicable."""
    columns: dict[str, NDArray[np.float64]] = {}
    for channel, (score_name, coverage_name) in _CHANNEL_COLUMNS.items():
        values = block[:, _COLUMN[score_name]].astype(np.float64)
        if coverage_name is not None:
            values = np.where(
                block[:, _COLUMN[coverage_name]] > 0.0, values, np.nan
            )
        columns[channel] = values
    return columns


def percentiles_of(
    values: NDArray[np.float64],
    calibration: NullCalibration,
    channel: str,
) -> NDArray[np.float64]:
    """Vectorised `P(null < value)`, NaN preserved.

    Same strictly-less convention as `NullCalibration.percentile`: the morphological
    null is a point mass at zero, so `<=` would rate a zero-overlap pair at the top.
    """
    grid = np.asarray(calibration.channels[channel].quantiles, dtype=np.float64)
    positions = np.searchsorted(grid, values, side="left")
    result = positions / len(grid)
    return np.where(np.isnan(values), np.nan, result)


def sample_null_from_cache(
    words_by_user: dict[str, list[_Word]],
    manifest: cache.CacheManifest,
    config: SimilaritiesConfig,
) -> NullCalibration:
    generator = np.random.default_rng(config.calibration.seed)
    pair_counts = np.array(
        [manifest.learners[user_id].pair_count for user_id in sorted(words_by_user)],
        dtype=np.float64,
    )
    if pair_counts.sum() == 0:
        raise ValueError("The score cache holds no pairs, so it cannot be calibrated.")

    user_ids = sorted(words_by_user)
    draws = generator.multinomial(
        config.calibration.sample_pairs, pair_counts / pair_counts.sum()
    )
    samples: dict[str, list[float]] = {channel: [] for channel in CHANNELS}
    for user_id, draw in zip(user_ids, draws, strict=True):
        if draw == 0:
            continue
        pair_count = manifest.learners[user_id].pair_count
        scores = cache.open_scores(config.cache.path, user_id, pair_count, mode="r")
        rows = generator.integers(0, pair_count, size=int(draw))
        for channel, values in channel_matrix(np.asarray(scores[rows])).items():
            samples[channel].extend(values[~np.isnan(values)].tolist())
        del scores

    return build_null_from_samples(
        samples,
        sample_pairs=config.calibration.sample_pairs,
        seed=config.calibration.seed,
    )


def _pair_scores(row: NDArray[np.float64]) -> _PairScores:
    phon = row[_COLUMN["phon_sim"]]
    return _PairScores(
        sem_sim=float(row[_COLUMN["sem_sim"]]),
        phon_sim=None if np.isnan(phon) else float(phon),
        mor_sim=float(row[_COLUMN["mor_sim"]]),
        orthographic=OrthographicSimilarityResult(
            ortho_stroke_sim=float(row[_COLUMN["ortho_stroke_sim"]]),
            ortho_stroke_coverage=float(row[_COLUMN["ortho_stroke_coverage"]]),
            ortho_kana_confusion=float(row[_COLUMN["ortho_kana_confusion"]]),
            ortho_kana_coverage=float(row[_COLUMN["ortho_kana_coverage"]]),
        ),
    )


def select_for_target(
    target: _Word,
    user_words: Sequence[_Word],
    block: NDArray[np.float32],
    review_histories: dict[str, _ReviewHistory],
    calibration: NullCalibration,
    thresholds: dict[str, float],
    top_k: int,
) -> tuple[WordSimilaritiesResult, dict[str, int]]:
    channel_scores = channel_matrix(block)
    channel_percentiles = {
        channel: percentiles_of(values, calibration, channel)
        for channel, values in channel_scores.items()
    }

    selected_by: dict[int, list[str]] = {}
    channel_counts: dict[str, int] = {}
    for channel in CHANNELS:
        values = channel_percentiles[channel]
        qualifying = np.flatnonzero(values >= thresholds[channel])
        if qualifying.size:
            # Descending score, then candidate order — which is already the
            # deterministic tie-break (earlier first_reviewed_at, then id).
            order = np.lexsort((qualifying, -values[qualifying]))
            qualifying = qualifying[order][:top_k]
        channel_counts[channel] = int(qualifying.size)
        for position in qualifying.tolist():
            selected_by.setdefault(position, []).append(channel)

    neighbors: list[NeighborSimilarityResult] = []
    for position in sorted(selected_by):
        candidate = user_words[position]
        stability, difficulty, review_count = _state_before(
            review_histories.get(candidate.word_instance_id),
            target.first_reviewed_at,
        )
        state = _CandidateState(
            candidate=candidate,
            stability=stability,
            difficulty=difficulty,
            review_count=review_count,
        )
        percentiles = {
            channel: (
                None
                if np.isnan(channel_percentiles[channel][position])
                else float(channel_percentiles[channel][position])
            )
            for channel in CHANNELS
        }
        neighbors.append(
            _neighbor_result(
                state,
                _pair_scores(block[position].astype(np.float64)),
                percentiles,
                tuple(selected_by[position]),
            )
        )

    return (
        WordSimilaritiesResult(
            word_instance_id=target.word_instance_id,
            neighbors=neighbors,
        ),
        channel_counts,
    )


class SelectionContext(NamedTuple):
    """What the randomised twin of a row needs, computed once per target anyway."""

    target: _Word
    user_words: Sequence[_Word]
    block: NDArray[np.float32]
    candidate_end: int
    candidate_index: dict[str, int]


def iter_selected_rows(
    words_by_user: dict[str, list[_Word]],
    manifest: cache.CacheManifest,
    review_histories: dict[str, _ReviewHistory],
    calibration: NullCalibration,
    config: SimilaritiesConfig,
    *,
    stats: _SimilarityRunStats | None = None,
    show_progress: bool = True,
) -> Iterator[tuple[WordSimilaritiesResult, SelectionContext]]:
    thresholds = config.percentiles.by_channel()
    run_stats = stats if stats is not None else _SimilarityRunStats()
    total = sum(len(user_words) for user_words in words_by_user.values())
    progress = tqdm(
        total=total, desc="Selecting neighbors", unit="word", disable=not show_progress
    )

    try:
        for user_id, user_words in words_by_user.items():
            ends = _candidate_ends(user_words)
            offsets = cache.pair_offsets(ends)
            candidate_index = {
                word.word_instance_id: index
                for index, word in enumerate(user_words)
            }
            scores = cache.open_scores(
                config.cache.path,
                user_id,
                manifest.learners[user_id].pair_count,
                mode="r",
            )
            for index, target in enumerate(user_words):
                block = np.asarray(scores[offsets[index] : offsets[index + 1]])
                row, channel_counts = select_for_target(
                    target,
                    user_words,
                    block,
                    review_histories,
                    calibration,
                    thresholds,
                    config.top_k,
                )
                run_stats.candidate_pairs += ends[index]
                run_stats.scored_pairs += ends[index]
                run_stats.targets += 1
                for channel, count in channel_counts.items():
                    run_stats.channel_counts[channel] += count
                    if count == config.top_k:
                        run_stats.saturated_targets[channel] += 1
                progress.update(1)
                yield row, SelectionContext(
                    target=target,
                    user_words=user_words,
                    block=block,
                    candidate_end=ends[index],
                    candidate_index=candidate_index,
                )
            del scores
    finally:
        progress.close()


# ponytail: +/-30d window, uniform draw. A propensity match on more covariates would
# be tighter, but time and learner already carry the drift this controls for.
RANDOM_WINDOW = np.timedelta64(30, "D")


def completion_pool(
    word_instances: pd.DataFrame,
    words_by_user: dict[str, list[_Word]],
) -> dict[str, tuple[NDArray[np.datetime64], list[str]]]:
    """Per learner, its completed words sorted by when they reached the PLB target.

    Sorted so the +/-30d window is two `searchsorted` calls rather than a scan.
    """
    user_of = {
        word.word_instance_id: user_id
        for user_id, user_words in words_by_user.items()
        for word in user_words
    }
    frame = word_instances[["word_instance_id", "plb_14_achieved_at"]].copy()
    frame["word_instance_id"] = frame["word_instance_id"].astype(str)
    frame = frame[frame["word_instance_id"].isin(user_of)]
    frame = frame[frame["plb_14_achieved_at"].notna()].copy()
    frame["achieved_at"] = (
        pd.to_datetime(frame["plb_14_achieved_at"], utc=True)
        .dt.tz_localize(None)
    )
    frame["user_id"] = frame["word_instance_id"].map(user_of)

    pool: dict[str, tuple[NDArray[np.datetime64], list[str]]] = {}
    for user_id, group in frame.sort_values("achieved_at", kind="stable").groupby(
        "user_id", sort=False
    ):
        pool[user_id] = (
            group["achieved_at"].to_numpy(),
            group["word_instance_id"].tolist(),
        )
    return pool


def draw_replacement(
    pool: tuple[NDArray[np.datetime64], list[str]],
    achieved_at: np.datetime64,
    cutoff: np.datetime64,
    exclude: str,
    candidate_index: dict[str, int],
    candidate_end: int,
    generator: np.random.Generator,
) -> int | None:
    """Row index of a random word the learner completed within the window.

    Bounded above by the target's own first review, which is both the leakage guard
    and what keeps the draw uncensored at the cutoff: a word completed before then has
    already passed its PLB-th review. Returns None when nothing qualifies.
    """
    times, ids = pool
    low = int(np.searchsorted(times, achieved_at - RANDOM_WINDOW, side="left"))
    high = int(
        np.searchsorted(times, min(achieved_at + RANDOM_WINDOW, cutoff), side="right")
    )
    eligible = [
        index
        for position in range(low, high)
        if (word_instance_id := ids[position]) != exclude
        and (index := candidate_index.get(word_instance_id, -1)) >= 0
        and index < candidate_end
    ]
    if not eligible:
        return None
    return int(generator.choice(eligible))


def randomise_row(
    row: WordSimilaritiesResult,
    target: _Word,
    user_words: Sequence[_Word],
    block: NDArray[np.float32],
    candidate_end: int,
    candidate_index: dict[str, int],
    pool: tuple[NDArray[np.datetime64], list[str]] | None,
    achieved_at: dict[str, np.datetime64],
    review_histories: dict[str, _ReviewHistory],
    calibration: NullCalibration,
    generator: np.random.Generator,
) -> tuple[WordSimilaritiesResult, int]:
    """Swap every uncensored neighbour for a time-matched random word.

    The slot survives - it exists, and it keeps the channel that selected it - while
    identity, similarity scores, state and burden all come from the drawn word. Only
    uncensored neighbours are swapped, since the ablation this feeds drops censored
    ones anyway; counts therefore match the similar file exactly.
    """
    cutoff = _as_naive_utc(target.first_reviewed_at)
    channel_percentiles = {
        channel: percentiles_of(values, calibration, channel)
        for channel, values in channel_matrix(block).items()
    }
    neighbors: list[NeighborSimilarityResult] = []
    fallbacks = 0

    for neighbor in row.neighbors:
        if neighbor.is_censored:
            continue
        position = candidate_index[neighbor.word_instance_id]
        drawn = (
            None
            if pool is None
            else draw_replacement(
                pool,
                achieved_at[neighbor.word_instance_id],
                cutoff,
                neighbor.word_instance_id,
                candidate_index,
                candidate_end,
                generator,
            )
        )
        if drawn is None:
            fallbacks += 1
            drawn = position

        candidate = user_words[drawn]
        stability, difficulty, review_count = _state_before(
            review_histories.get(candidate.word_instance_id),
            target.first_reviewed_at,
        )
        state = _CandidateState(
            candidate=candidate,
            stability=stability,
            difficulty=difficulty,
            review_count=review_count,
        )
        percentiles = {
            channel: (None if np.isnan(value) else float(value))
            for channel, values in channel_percentiles.items()
            if (value := values[drawn]) is not None
        }
        neighbors.append(
            _neighbor_result(
                state,
                _pair_scores(block[drawn].astype(np.float64)),
                percentiles,
                neighbor.selected_by,
            )
        )

    return (
        WordSimilaritiesResult(
            word_instance_id=row.word_instance_id,
            neighbors=neighbors,
        ),
        fallbacks,
    )


def main() -> None:
    config = load_similarities_config()
    cohort_user_ids = load_cohort().user_ids()

    word_instances = pd.read_json(config.word_instances_path, lines=True)
    words = _prepare_words(
        pd.read_json(config.lexemes_path, lines=True),
        word_instances,
        pd.read_json(config.decks_path, lines=True),
    )
    words = [word for word in words if word.user_id in cohort_user_ids]
    if not words:
        raise ValueError(
            "No reviewed Japanese word instances belong to the configured cohort; "
            "check cohort.learners in params.yaml."
        )
    words_by_user = _group_words_by_user(words)

    manifest = cache.load_manifest(config.cache.path)
    # The cache is positional: a reordered word list would resolve every offset to a
    # different pair without failing, so this check is load-bearing.
    manifest.verify_against(words_by_user)

    review_histories = _build_review_histories(
        pd.read_json(config.reviews_path, lines=True),
        {word.word_instance_id for word in words},
    )

    calibration = sample_null_from_cache(words_by_user, manifest, config)
    config.calibration.output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(config.calibration.output_path, calibration.model_dump(mode="json"))
    print(
        f"Calibrated {len(calibration.channels)} channels on "
        f"{config.calibration.sample_pairs:,} cached pairs -> "
        f"{config.calibration.output_path}"
    )

    pools = completion_pool(word_instances, words_by_user)
    achieved_at = {
        word_instance_id: times[position]
        for times, ids in pools.values()
        for position, word_instance_id in enumerate(ids)
    }
    generator = np.random.default_rng(config.calibration.seed)

    stats = _SimilarityRunStats()
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    randomised = 0
    fallbacks = 0
    with (
        config.output_path.open("w", encoding="utf-8") as output,
        config.random_output_path.open("w", encoding="utf-8") as random_output,
    ):
        for row, context in iter_selected_rows(
            words_by_user, manifest, review_histories, calibration, config, stats=stats
        ):
            output.write(row.model_dump_json())
            output.write("\n")
            row_count += 1

            random_row, row_fallbacks = randomise_row(
                row,
                context.target,
                context.user_words,
                context.block,
                context.candidate_end,
                context.candidate_index,
                pools.get(context.target.user_id),
                achieved_at,
                review_histories,
                calibration,
                generator,
            )
            random_output.write(random_row.model_dump_json())
            random_output.write("\n")
            randomised += len(random_row.neighbors)
            fallbacks += row_fallbacks

    if stats.targets:
        thresholds = config.percentiles.by_channel()
        print(f"Per-channel selection (top_k={config.top_k}):")
        for channel in CHANNELS:
            selected = stats.channel_counts[channel]
            saturated = stats.saturated_targets[channel]
            print(
                f"  {channel:<14} p>={thresholds[channel]:<7} "
                f"{selected:>12,} neighbours "
                f"({selected / stats.targets:6.2f} per target, "
                f"{saturated / stats.targets:6.1%} of targets at top_k)"
            )
    print(f"Wrote {row_count:,} Japanese word rows to {config.output_path}")
    print(
        f"Wrote {randomised:,} time-matched random neighbours to "
        f"{config.random_output_path} "
        f"({fallbacks:,} kept their original word for want of an alternative)"
    )


if __name__ == "__main__":
    main()
