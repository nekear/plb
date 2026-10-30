"""On-disk cache of raw channel scores for every candidate pair.

Scoring a pair is expensive (~340 us, dominated by the phonological and orthographic
channels) while selecting neighbours from those scores is microseconds. Caching the
raw scores lets thresholds be retuned without re-scoring: the 1.5 h stage runs once,
and every percentile sweep afterwards reads from here.

Layout is positional, which is what keeps it small. A learner's candidate pairs are
already a deterministic sequence — target `i` pairs with candidates
`[0, candidate_ends[i])` of the same word list — so the pair a row describes is
implied by its offset and nothing about pair identity is stored.

That is also the format's one danger: if the word list changes order, every offset
still resolves to a valid-looking row describing a different pair. The manifest
therefore fingerprints each learner's word order, and readers must call
`verify_against` before trusting an offset.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

# Column order is part of the on-disk format; the manifest records it so a reordering
# is caught rather than silently reinterpreted.
SCORE_COLUMNS: tuple[str, ...] = (
    "sem_sim",
    "phon_sim",
    "mor_sim",
    "ortho_stroke_sim",
    "ortho_stroke_coverage",
    "ortho_kana_confusion",
    "ortho_kana_coverage",
)

# float32 halves the cache (1.98 GiB rather than 3.96) and costs ~3e-8 on a score in
# [0, 1]. That rounding can still move a *percentile* by a few points, because the
# null grid has plateaus and searchsorted jumps across them — but every plateau sits
# at the bottom of its channel (semantic 0.00-0.26, ortho_stroke 0.00-0.05,
# ortho_kana 0.13-0.15), i.e. among pairs no threshold would ever select. Verified
# against a pre-refactor run: identical neighbours and channels, raw scores within
# 3e-8. Revisit if thresholds ever drop near those plateaus.
SCORE_DTYPE = np.float32
CACHE_VERSION = 1
MANIFEST_NAME = "manifest.json"


def pair_offsets(candidate_ends: Sequence[int]) -> NDArray[np.int64]:
    """Row where each target's block starts — an exclusive prefix sum.

    The last element is the total pair count, so `offsets[i]:offsets[i + 1]` is
    target `i`'s block.
    """
    offsets = np.zeros(len(candidate_ends) + 1, dtype=np.int64)
    np.cumsum(np.asarray(candidate_ends, dtype=np.int64), out=offsets[1:])
    return offsets


def fingerprint_words(word_instance_ids: Sequence[str]) -> str:
    """Hash of a learner's ordered word instances — the key the layout depends on."""
    digest = hashlib.sha256()
    for word_instance_id in word_instance_ids:
        digest.update(word_instance_id.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


class LearnerCacheEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    pair_count: int = Field(ge=0)
    word_count: int = Field(ge=0)
    words_fingerprint: str


class CacheManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int = CACHE_VERSION
    columns: tuple[str, ...] = SCORE_COLUMNS
    learners: dict[str, LearnerCacheEntry]

    def verify_against(self, words_by_user: Mapping[str, Sequence]) -> None:
        """Raise unless the cache was built from exactly this word list.

        Positional layout means a stale cache does not fail loudly on its own — it
        returns scores for whichever pair now sits at that offset. This is the only
        thing standing between a reordered input and silently wrong neighbours.
        """
        if self.version != CACHE_VERSION:
            raise ValueError(
                f"Score cache is version {self.version}, this code writes "
                f"{CACHE_VERSION}; re-run the scoring stage."
            )
        if tuple(self.columns) != SCORE_COLUMNS:
            raise ValueError(
                f"Score cache columns {list(self.columns)} do not match "
                f"{list(SCORE_COLUMNS)}; re-run the scoring stage."
            )

        missing = sorted(set(words_by_user) - set(self.learners))
        if missing:
            raise ValueError(
                f"Score cache has no entry for learner(s) {missing}; re-run the "
                "scoring stage for the current cohort."
            )
        for user_id, words in words_by_user.items():
            entry = self.learners[user_id]
            actual = fingerprint_words([word.word_instance_id for word in words])
            if actual != entry.words_fingerprint:
                raise ValueError(
                    f"Score cache for learner {user_id} was built from a different "
                    f"word list ({entry.word_count:,} words); the current input has "
                    f"{len(words):,}. Re-run the scoring stage."
                )


def manifest_path(cache_dir: Path) -> Path:
    return cache_dir / MANIFEST_NAME


def scores_path(cache_dir: Path, user_id: str) -> Path:
    return cache_dir / f"{user_id}.npy"


def progress_path(cache_dir: Path, user_id: str) -> Path:
    return cache_dir / f"{user_id}.progress"


def write_manifest(cache_dir: Path, manifest: CacheManifest) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path(cache_dir).write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )


def load_manifest(cache_dir: Path) -> CacheManifest:
    path = manifest_path(cache_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"No score cache manifest at {path}; run the scoring stage first."
        )
    return CacheManifest.model_validate_json(path.read_text(encoding="utf-8"))


def open_scores(
    cache_dir: Path,
    user_id: str,
    pair_count: int,
    mode: str,
) -> np.memmap:
    """Memory-map a learner's score block, shape `(pair_count, len(SCORE_COLUMNS))`."""
    path = scores_path(cache_dir, user_id)
    if mode in {"r", "r+"} and not path.exists():
        raise FileNotFoundError(
            f"No cached scores for learner {user_id} at {path}; run the scoring stage."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.memmap(
        path,
        dtype=SCORE_DTYPE,
        mode=mode,
        shape=(pair_count, len(SCORE_COLUMNS)),
    )


def load_completed_chunks(cache_dir: Path, user_id: str) -> set[int]:
    """Chunk start indices already written, so a crashed run resumes where it left off."""
    path = progress_path(cache_dir, user_id)
    if not path.exists():
        return set()
    completed: set[int] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            # A hard kill can leave a torn final line; a partial number is simply a
            # chunk we redo.
            try:
                completed.add(int(line))
            except ValueError:
                continue
    return completed


def record_completed_chunk(cache_dir: Path, user_id: str, start: int) -> None:
    with progress_path(cache_dir, user_id).open("a", encoding="utf-8") as handle:
        handle.write(f"{start}\n")
        handle.flush()


def clear_progress(cache_dir: Path, user_id: str) -> None:
    progress_path(cache_dir, user_id).unlink(missing_ok=True)
