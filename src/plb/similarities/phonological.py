from functools import lru_cache

import epitran
import panphon
from panphon.distance import Distance

from plb.similarities.types import PhonologicalSimilarityEntry

ipa_transcriber = epitran.Epitran("jpn-Hira")
feature_table = panphon.FeatureTable()
feature_distance = Distance()


class PhonologicalSimilarityComputationError(ValueError):
    """Raised when a phonological similarity score cannot be computed."""


@lru_cache(maxsize=100_000)
def kana_to_ipa(reading: str) -> str | None:
    ipa = ipa_transcriber.transliterate(reading)
    if not ipa or not feature_table.validate_word(ipa):
        return None

    return ipa


@lru_cache(maxsize=200_000)
def _compute_reading_similarity(reading_1: str, reading_2: str) -> float | None:
    """Returns cached similarity for a canonicalized pair of readings."""
    ipa1 = kana_to_ipa(reading_1)
    ipa2 = kana_to_ipa(reading_2)
    if not ipa1 or not ipa2:
        return None

    distance = feature_distance.feature_edit_distance_div_maxlen(str(ipa1), str(ipa2))
    return max(0.0, min(1.0, 1.0 - distance))


def compute_phonological_similarity(
    w1: PhonologicalSimilarityEntry, w2: PhonologicalSimilarityEntry
) -> float:
    w1 = PhonologicalSimilarityEntry.model_validate(w1)
    w2 = PhonologicalSimilarityEntry.model_validate(w2)

    reading_1, reading_2 = sorted((w1.reading, w2.reading))
    similarity = _compute_reading_similarity(reading_1, reading_2)
    if similarity is None:
        raise PhonologicalSimilarityComputationError(
            "Failed to convert readings to valid IPA: "
            f"reading1={w1.reading!r}, reading2={w2.reading!r}."
        )

    return similarity
