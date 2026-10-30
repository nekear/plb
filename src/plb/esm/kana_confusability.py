from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


class KanaConfusability:
    """
    Lookup based on the Higuchi-Kobayashi kana perceptual-distance matrix.

    Returned score:
        1.0 = among the most confusable kana pairs
        0.0 = among the easiest kana pairs to distinguish

    The score is a percentile, not a probability.
    """

    def __init__(self, xlsx_path: str | Path):
        df = pd.read_excel(xlsx_path, sheet_name="List")

        required = {
            "Letter1",
            "Letter2",
            "Value",
            "Type1",
            "Type2",
        }

        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        # Type 1 = hiragana, type 2 = katakana, type 3 = blank
        # Excluding blank-character comparisons from percentile calibration
        kana_pairs = df[df["Type1"].isin([1, 2]) & df["Type2"].isin([1, 2])].copy()

        kana_pairs["Letter1"] = kana_pairs["Letter1"].astype(str)
        kana_pairs["Letter2"] = kana_pairs["Letter2"].astype(str)
        kana_pairs["Value"] = kana_pairs["Value"].astype(float)

        self._distances: dict[tuple[str, str], float] = {}

        for row in kana_pairs.itertuples(index=False):
            a = row.Letter1
            b = row.Letter2
            distance = float(row.Value)

            # The published comparison is symmetric
            self._distances[(a, b)] = distance
            self._distances[(b, a)] = distance

        self._characters = frozenset(
            character for pair in self._distances for character in pair
        )
        self._all_distances = kana_pairs["Value"].to_numpy(dtype=np.float64)

    def supports(self, character: str) -> bool:
        return len(character) == 1 and character in self._characters

    def compare(self, char1: str, char2: str) -> dict[str, float]:
        if len(char1) != 1 or len(char2) != 1:
            raise ValueError("Both inputs must be single characters.")

        if not self.supports(char1) or not self.supports(char2):
            raise KeyError(f"No empirical kana comparison for {char1!r} and {char2!r}.")

        if char1 == char2:
            return {
                "score": 1.0,
                "percentile": 100.0,
                "distance": 0.0,
            }

        key = (char1, char2)

        if key not in self._distances:
            raise KeyError(f"No empirical kana comparison for {char1!r} and {char2!r}.")

        distance = self._distances[key]

        # Midrank empirical percentile
        #
        # Because lower distance means greater confusability,
        # subtracting the ordinary distance percentile from 1
        less = np.mean(self._all_distances < distance)
        equal = np.mean(self._all_distances == distance)

        score = 1.0 - (less + 0.5 * equal)

        return {
            "score": float(score),
            "percentile": float(100.0 * score),
            "distance": float(distance),
        }
