"""Calibrate every similarity channel against a shared null distribution.

The channels are not on comparable scales. Measured over random within-learner pairs,
the phonological channel averages 0.633 — Japanese readings share articulatory
features by construction — while mean-centered semantic similarity averages 0.057 and
morphological similarity is a point mass at 0. A single raw threshold therefore means
something different in every channel: at the old settings phonological admitted 35% of
unrelated pairs and the rest admitted almost none.

Calibration replaces each raw score with `P(random pair scores below this)`, so one
percentile threshold means the same thing everywhere: "more similar than 99.9% of
pairs drawn at random".

The null is pooled across learners rather than fitted per learner. A deployment
scoring a new user cannot derive that user's null, so the boundaries have to be a
fixed artifact that training and inference share — hence `NullCalibration` is written
to disk alongside the similarity output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters for type checkers
    from .__main__ import _PairScores, _Word

SEMANTIC = "semantic"
PHONOLOGICAL = "phonological"
MORPHOLOGICAL = "morphological"
ORTHO_STROKE = "ortho_stroke"
ORTHO_KANA = "ortho_kana"

# Stroke and kana stay separate: different mechanisms, different coverage.
CHANNELS: tuple[str, ...] = (
    SEMANTIC,
    PHONOLOGICAL,
    MORPHOLOGICAL,
    ORTHO_STROKE,
    ORTHO_KANA,
)

# Resolution of the stored null. 10k points put the quantile grid an order of
# magnitude finer than the 0.999 threshold it has to resolve.
QUANTILE_POINTS = 10_001


def channel_values(scores: _PairScores) -> dict[str, float | None]:
    """Per-channel score for one pair, `None` where the channel does not apply.

    A channel that cannot be computed is absent, not zero: `phon_sim` is `None` when
    either word lacks a reading, and the orthographic channels are meaningless when
    their coverage is 0 (no comparable characters). Scoring those as 0 would pull the
    null down and make the percentile of a real score too generous.
    """
    orthographic = scores.orthographic
    return {
        SEMANTIC: scores.sem_sim,
        PHONOLOGICAL: scores.phon_sim,
        MORPHOLOGICAL: scores.mor_sim,
        ORTHO_STROKE: (
            orthographic.ortho_stroke_sim
            if orthographic.ortho_stroke_coverage > 0.0
            else None
        ),
        ORTHO_KANA: (
            orthographic.ortho_kana_confusion
            if orthographic.ortho_kana_coverage > 0.0
            else None
        ),
    }


class ChannelNull(BaseModel):
    model_config = ConfigDict(frozen=True)

    sample_size: int = Field(ge=1)
    # Ascending values at evenly spaced quantiles of the null sample.
    quantiles: list[float]


class NullCalibration(BaseModel):
    """Empirical null distribution per channel, with the sampling that produced it."""

    model_config = ConfigDict(frozen=True)

    sample_pairs: int = Field(ge=1)
    seed: int
    channels: dict[str, ChannelNull]

    def percentile(self, channel: str, value: float | None) -> float | None:
        """`P(null < value)` — the fraction of random pairs this score beats.

        Strictly less, deliberately. The morphological null is a point mass at exactly
        0, so a `<=` comparison would rate a zero-overlap pair at the 99.9th
        percentile; `<` maps it to 0.0, while any genuine overlap clears the threshold
        on its own.
        """
        if value is None:
            return None

        quantiles = self._quantile_array(channel)
        position = int(np.searchsorted(quantiles, value, side="left"))
        return position / len(quantiles)

    def _quantile_array(self, channel: str) -> NDArray[np.float64]:
        try:
            null = self.channels[channel]
        except KeyError as error:
            raise KeyError(
                f"No null distribution for channel {channel!r}; "
                f"calibrated channels are {sorted(self.channels)}."
            ) from error
        return np.asarray(null.quantiles, dtype=np.float64)


def _sample_pair_indices(
    words_by_user: Mapping[str, Sequence[_Word]],
    sample_pairs: int,
    seed: int,
) -> list[tuple[str, int, int]]:
    """Draw random within-learner pairs, pooled across learners.

    Learners are drawn in proportion to how many candidate pairs they contribute, so
    the sample matches the population the pipeline actually scores rather than
    weighting a 928-word learner equally with an 11,291-word one.
    """
    eligible = {
        user_id: words for user_id, words in words_by_user.items() if len(words) >= 2
    }
    if not eligible:
        raise ValueError(
            "Null calibration needs at least one learner with two or more words."
        )

    user_ids = sorted(eligible)
    pair_counts = np.array(
        [len(eligible[user_id]) * (len(eligible[user_id]) - 1) for user_id in user_ids],
        dtype=np.float64,
    )
    weights = pair_counts / pair_counts.sum()

    generator = np.random.default_rng(seed)
    drawn_users = generator.choice(len(user_ids), size=sample_pairs, p=weights)

    pairs: list[tuple[str, int, int]] = []
    for user_index in drawn_users:
        user_id = user_ids[int(user_index)]
        size = len(eligible[user_id])
        left = int(generator.integers(0, size))
        right = int(generator.integers(0, size - 1))
        if right >= left:  # sample a distinct partner without rejection looping
            right += 1
        pairs.append((user_id, left, right))
    return pairs


def build_null_calibration(
    words_by_user: Mapping[str, Sequence[_Word]],
    score_pair: Callable[[_Word, _Word], _PairScores],
    sample_pairs: int,
    seed: int,
) -> NullCalibration:
    """Score random pairs and reduce each channel's sample to a quantile grid."""
    samples: dict[str, list[float]] = {channel: [] for channel in CHANNELS}
    for user_id, left, right in _sample_pair_indices(
        words_by_user, sample_pairs, seed
    ):
        user_words = words_by_user[user_id]
        scores = score_pair(user_words[left], user_words[right])
        for channel, value in channel_values(scores).items():
            if value is not None:
                samples[channel].append(value)

    return build_null_from_samples(samples, sample_pairs=sample_pairs, seed=seed)


def build_null_from_samples(
    samples: Mapping[str, Sequence[float]],
    *,
    sample_pairs: int,
    seed: int,
) -> NullCalibration:
    """Reduce per-channel null samples to a quantile grid.

    Split out so the scored-pair cache can calibrate from stored rows instead of
    re-scoring pairs, without duplicating the grid construction.
    """
    channels: dict[str, ChannelNull] = {}
    probabilities = np.linspace(0.0, 1.0, QUANTILE_POINTS)
    for channel in CHANNELS:
        values = samples.get(channel, ())
        if len(values) == 0:
            raise ValueError(
                f"Channel {channel!r} produced no comparable values across "
                f"{sample_pairs:,} sampled pairs, so it cannot be calibrated."
            )
        quantiles = np.quantile(np.asarray(values, dtype=np.float64), probabilities)
        channels[channel] = ChannelNull(
            sample_size=len(values),
            quantiles=[float(value) for value in quantiles],
        )

    return NullCalibration(
        sample_pairs=sample_pairs,
        seed=seed,
        channels=channels,
    )
