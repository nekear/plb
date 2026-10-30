import csv
from functools import lru_cache
from pathlib import Path


_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "niteru"
_STROKE_PATH = _DATA_DIR / "stroke_ulrich.txt"


def _load_strokes(path: Path = _STROKE_PATH) -> dict[str, tuple[str, ...]]:
    """Loads Ulrich stroke sequences keyed by character."""
    with path.open(encoding="utf-8", newline="") as file:
        return {
            row[0]: tuple(row[1:])
            for row in csv.reader(file)
            if row and row[0]
        }


@lru_cache(maxsize=1)
def _all_strokes() -> dict[str, tuple[str, ...]]:
    return _load_strokes()


def edit_distance(from_strokes: tuple[str, ...], to_strokes: tuple[str, ...]) -> int:
    """Returns Levenshtein distance between two stroke sequences."""
    if from_strokes == to_strokes:
        return 0
    if not from_strokes:
        return len(to_strokes)
    if not to_strokes:
        return len(from_strokes)

    previous = list(range(len(to_strokes) + 1))
    for i, from_stroke in enumerate(from_strokes, start=1):
        current = [i]
        for j, to_stroke in enumerate(to_strokes, start=1):
            substitution_cost = from_stroke != to_stroke
            current.append(
                min(
                    current[j - 1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + substitution_cost,
                )
            )
        previous = current

    return previous[-1]


def compute_niteru_similarity(from_character: str, to_character: str) -> float | None:
    """Computes symmetric, bounded Niteru similarity for two characters.

    ``None`` indicates that either character is absent from the stroke
    vocabulary. Niteru's original implementation returns a normalized
    distance, while this function returns its inverted similarity instead.
    """
    strokes = _all_strokes()
    from_strokes = strokes.get(from_character)
    to_strokes = strokes.get(to_character)
    if from_strokes is None or to_strokes is None:
        return None

    distance = edit_distance(from_strokes, to_strokes)
    normalizer = max(len(from_strokes), len(to_strokes))
    return max(0.0, min(1.0, 1.0 - distance / normalizer))
