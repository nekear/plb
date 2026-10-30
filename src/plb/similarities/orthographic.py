from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

from plb.esm.kana_confusability import KanaConfusability
from plb.niteru.niteru import compute_niteru_similarity
from plb.utils import PROJECT_ROOT

from .types import OrthographicSimilarityResult, SimilarityEntry

_ESM_PATH = PROJECT_ROOT / "data" / "raw" / "esm" / "12144_2021_2664_MOESM1_ESM.xlsx"


@dataclass(frozen=True)
class _WeightedEditResult:
    similarity: float
    coverage: float


@dataclass(frozen=True)
class _AlignmentState:
    distance: float
    aligned_pairs: int = 0
    covered_pairs: int = 0


@lru_cache(maxsize=1)
def _kana_confusability() -> KanaConfusability:
    return KanaConfusability(_ESM_PATH)


def _kana_character_similarity(character_1: str, character_2: str) -> float | None:
    try:
        return _kana_confusability().compare(character_1, character_2)["score"]
    except KeyError:
        return None


def _weighted_edit_similarity(
    form_1: str,
    form_2: str,
    character_similarity: Callable[[str, str], float | None],
) -> _WeightedEditResult:
    """Computes normalized weighted edit similarity and substitution coverage"""
    previous = [_AlignmentState(distance=float(j)) for j in range(len(form_2) + 1)]

    for i, character_1 in enumerate(form_1, start=1):
        current = [_AlignmentState(distance=float(i))]

        for j, character_2 in enumerate(form_2, start=1):
            similarity = character_similarity(character_1, character_2)
            is_covered = similarity is not None
            substitution_cost = 1.0 if similarity is None else 1.0 - similarity

            diagonal = _AlignmentState(
                distance=previous[j - 1].distance + substitution_cost,
                aligned_pairs=previous[j - 1].aligned_pairs + 1,
                covered_pairs=previous[j - 1].covered_pairs + int(is_covered),
            )
            deletion = _AlignmentState(
                distance=previous[j].distance + 1.0,
                aligned_pairs=previous[j].aligned_pairs,
                covered_pairs=previous[j].covered_pairs,
            )
            insertion = _AlignmentState(
                distance=current[j - 1].distance + 1.0,
                aligned_pairs=current[j - 1].aligned_pairs,
                covered_pairs=current[j - 1].covered_pairs,
            )

            # Resolving equal-distance paths with symmetric, additive criteria
            current.append(
                min(
                    (diagonal, deletion, insertion),
                    key=lambda state: (
                        state.distance,
                        -state.aligned_pairs,
                        -state.covered_pairs,
                    ),
                )
            )

        previous = current

    final = previous[-1]
    normalizer = max(len(form_1), len(form_2))
    if normalizer == 0:
        similarity = 1.0
    else:
        similarity = 1.0 - final.distance / normalizer

    coverage = final.covered_pairs / final.aligned_pairs if final.aligned_pairs else 0.0
    return _WeightedEditResult(
        similarity=max(0.0, min(1.0, similarity)),
        coverage=max(0.0, min(1.0, coverage)),
    )


def compute_orthographic_similarity(
    w1: SimilarityEntry, w2: SimilarityEntry
) -> OrthographicSimilarityResult:
    """Computes stroke and kana-confusion similarities between lexemes."""
    stroke = _weighted_edit_similarity(
        w1.form,
        w2.form,
        compute_niteru_similarity,
    )
    kana = _weighted_edit_similarity(
        w1.form,
        w2.form,
        _kana_character_similarity,
    )

    # Affine pixel similarity is intentionally deferred until a later Model B
    # iteration. Its implementation remains in similarities/pixel.py.

    return OrthographicSimilarityResult(
        ortho_stroke_sim=stroke.similarity,
        ortho_stroke_coverage=stroke.coverage,
        ortho_kana_confusion=kana.similarity,
        ortho_kana_coverage=kana.coverage,
    )
