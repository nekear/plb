# %% [markdown]
# Reading Coverage Analysis
# Informs: docs/decisions/0001-g2p-approach.md
# Questions:
# 1. what fraction of lexemes have a non-null `reading`, per language?
# 2. which approach is better for phonological similarity?

# %%
import pandas as pd


def get_reading_coverage(lexemes_df: pd.DataFrame):
    return lexemes_df.groupby("language")["reading"].agg(
        reading_coverage=lambda s: s.notna().mean(),
        reading_count="size",
    )


lexemes_df = pd.read_json("data/interim/lexemes.jsonl", lines=True)
get_reading_coverage(lexemes_df)

# %%

japanese_lexemes_df = lexemes_df.loc[lexemes_df["language"] == "ja"]
get_reading_coverage(japanese_lexemes_df)

# %%
# Using fugashi to extract readings
from fugashi import Tagger

tagger = Tagger()


def to_hiragana(text: str) -> str:
    return "".join(
        chr(ord(char) - 0x60) if "ァ" <= char <= "ヶ" else char for char in text
    )


def fugashi_reading(form: str) -> str | None:
    tokens = list(tagger(form))

    readings = []
    for token in tokens:
        pronunciation = token.feature.kana

        if not pronunciation or pronunciation == "*":
            return None

        readings.append(pronunciation)

    return to_hiragana("".join(readings)) or None


# %% Checking accuracy
from rapidfuzz.distance import Levenshtein

present = japanese_lexemes_df["reading"].notna()
real_readings = japanese_lexemes_df.loc[present, "reading"].map(to_hiragana)
predicted_readings = japanese_lexemes_df.loc[present, "form"].map(fugashi_reading)

valid = real_readings.notna() & predicted_readings.notna()

distances = [
    Levenshtein.distance(str(a), str(p))
    for a, p in zip(real_readings[valid], predicted_readings[valid])
]

normalized = [
    d / max(len(str(a)), len(str(p)), 1)
    for a, p, d in zip(real_readings[valid], predicted_readings[valid], distances)
]

print("Mean normalized distance:", sum(normalized) / len(normalized))
print("Mean Levenshtein distance:", sum(distances) / len(distances))
print("Exact matches:", sum(d == 0 for d in distances))
print("Compared:", len(distances))
print("Accuracy:", sum(d == 0 for d in distances) / len(distances))
# %%

# Comparing the same rows at the phonological-feature level
from functools import lru_cache

import epitran
import panphon
from panphon.distance import Distance

ipa_transcriber = epitran.Epitran("jpn-Hira")
feature_table = panphon.FeatureTable()
feature_distance = Distance()


@lru_cache(maxsize=100_000)
def kana_to_ipa(reading: str) -> str | None:
    if not reading or not reading.strip():
        return None

    ipa = ipa_transcriber.transliterate(reading)
    if not ipa or not feature_table.validate_word(ipa):
        return None

    return ipa


ipa_input = real_readings.notna() & predicted_readings.notna()

ipa_real_input = real_readings[ipa_input]
ipa_predicted_input = predicted_readings[ipa_input]

real_ipa = pd.Series(
    (kana_to_ipa(reading) for reading in ipa_real_input),
    index=ipa_real_input.index,
)
predicted_ipa = pd.Series(
    (kana_to_ipa(reading) for reading in ipa_predicted_input),
    index=ipa_predicted_input.index,
)
ipa_valid = real_ipa.notna() & predicted_ipa.notna()

ipa_distances = [
    feature_distance.feature_edit_distance(str(real), str(predicted))
    for real, predicted in zip(real_ipa[ipa_valid], predicted_ipa[ipa_valid])
]

ipa_normalized = [
    feature_distance.feature_edit_distance_div_maxlen(str(real), str(predicted))
    for real, predicted in zip(real_ipa[ipa_valid], predicted_ipa[ipa_valid])
]
ipa_similarities = [1.0 - distance for distance in ipa_normalized]

if ipa_distances:
    print("\nPanPhon feature comparison")
    print("Mean similarity:", sum(ipa_similarities) / len(ipa_similarities))
    print("Mean normalized distance:", sum(ipa_normalized) / len(ipa_normalized))
    print("Mean feature edit distance:", sum(ipa_distances) / len(ipa_distances))
    print("Zero-distance pairs:", sum(distance == 0 for distance in ipa_distances))
    print("Compared:", len(ipa_distances))
    print(
        "Zero-distance rate:",
        sum(distance == 0 for distance in ipa_distances) / len(ipa_distances),
    )
else:
    print("No Japanese readings produced valid PanPhon IPA segments.")

# %%

missing = lexemes_df["reading"].isna()
lexemes_df.loc[missing, "reading"] = lexemes_df.loc[missing, "form"].map(
    fugashi_reading
)

get_reading_coverage(lexemes_df)

# %%

still_missing = lexemes_df.loc[
    (lexemes_df["language"] == "ja") & (lexemes_df["reading"].isna())
]
print(f"Total: {len(still_missing)}")
display(still_missing)
