from typing import Callable

import numpy as np
from numpy.linalg import norm
from numpy.typing import ArrayLike

from .types import SimilarityEntry


def compute_semantic_similarity(
    w1: SimilarityEntry,
    w2: SimilarityEntry,
    get_embedding: Callable[[str], ArrayLike],
) -> float:
    e1 = np.asarray(get_embedding(w1.id), dtype=np.float64)
    e2 = np.asarray(get_embedding(w2.id), dtype=np.float64)
    if e1.ndim != 1 or e2.ndim != 1:
        raise ValueError("Semantic embeddings must be one-dimensional vectors.")

    denominator = norm(e1) * norm(e2)
    if denominator == 0.0:
        raise ValueError("Semantic similarity is undefined for a zero-norm embedding.")

    similarity = float(np.dot(e1, e2) / denominator)
    if not np.isfinite(similarity):
        raise ValueError("Semantic similarity is not finite.")

    return max(0.0, min(1.0, similarity))
