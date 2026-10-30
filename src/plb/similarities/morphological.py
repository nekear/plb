from functools import lru_cache
from plb.similarities.types import SimilarityEntry

from fugashi import Tagger
from collections import Counter

tagger = Tagger()


def multiset_dice(
        items_1: tuple[str, ...],
        items_2: tuple[str, ...],
) -> float:
    if not items_1 and not items_2:
        return 1.0

    if not items_1 or not items_2:
        return 0.0

    counts_1 = Counter(items_1)
    counts_2 = Counter(items_2)

    overlap = sum((counts_1 & counts_2).values())

    return (2.0 * overlap) / (len(items_1) + len(items_2))


@lru_cache(maxsize=50_000)
def extract_ja_lemmas(form: str) -> tuple[str, ...]:
    """Returns the normalized lemma sequence for a Japanese form."""
    lemmas: list[str] = []

    for token in tagger(form):
        lemma = token.feature.lemma

        if not lemma or lemma == "*":
            lemma = token.surface

        lemmas.append(lemma)

    return tuple(lemmas)


def compute_morphological_similarity(w1: SimilarityEntry, w2: SimilarityEntry) -> float:
    w1_lemmas = extract_ja_lemmas(w1.form)
    w2_lemmas = extract_ja_lemmas(w2.form)

    return multiset_dice(w1_lemmas, w2_lemmas)
