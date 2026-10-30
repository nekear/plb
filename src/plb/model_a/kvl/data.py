from pathlib import Path
from textwrap import dedent
from typing import Callable, Sequence

import pandas as pd
import torch
from torch.utils.data import Dataset

DEFAULT_LANGUAGES = ("de", "es", "zh")

# The data directory uses ISO 639-1 codes ("zh"), but the L1 column inside the
# CSVs uses the BEA task's own codes ("cn" for Chinese) — the two do not match.
# make_prompt needs an actual language name, not either code, for the prompt to
# read naturally in English.
L1_LANGUAGE_NAMES = {
    "es": "Spanish",
    "de": "German",
    "cn": "Chinese",
}


def load_kvl_split(
    data_dir: Path,
    split: str,
    languages: Sequence[str] = DEFAULT_LANGUAGES,
) -> pd.DataFrame:
    """Loads and concatenates one KVL split across the given L1 languages.

    Layout: {data_dir}/{split}/{lang}/kvl_shared_task_{lang}_{split}.csv
    """
    frames = []
    for lang in languages:
        matches = list((data_dir / split / lang).glob("kvl_shared_task_*.csv"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly one CSV for split={split!r} lang={lang!r} "
                f"under {data_dir / split / lang}, found {len(matches)}"
            )
        frames.append(pd.read_csv(matches[0]))

    df = pd.concat(frames, ignore_index=True)

    unknown = set(df["L1"]) - set(L1_LANGUAGE_NAMES)
    if unknown:
        raise ValueError(
            f"No language name mapping for L1 code(s) {unknown} — add them to "
            "L1_LANGUAGE_NAMES before training/evaluating on this data."
        )
    df["L1_name"] = df["L1"].map(L1_LANGUAGE_NAMES)
    return df


def fit_score_range(df: pd.DataFrame, column: str = "GLMM_score") -> tuple[float, float]:
    """Fits on the training split only — reuse the returned bounds for dev/test,
    never refits per-split, or the 1-5 scale stops meaning the same thing across splits."""
    return float(df[column].min()), float(df[column].max())


def remap_scores(
    df: pd.DataFrame,
    score_min: float,
    score_max: float,
    column: str = "GLMM_score",
) -> pd.Series:
    return 5 - 4 * (df[column] - score_min) / (score_max - score_min)


def invert_remap(remapped: float, score_min: float, score_max: float) -> float:
    """Inverse of remap_scores — maps a 1-5 prediction back to raw GLMM_score scale."""
    return score_min + (5 - remapped) / 4 * (score_max - score_min)


# mmBERT's mask token has lstrip=True (standard for MLM special tokens): when the
# literal string "Difficulty: <mask>" is tokenized, the tokenizer swallows the
# preceding space, giving 3 tokens (Difficulty, :, <mask>). A real digit answer
# tokenizes to 4 (Difficulty, :, <space>, 1) — the mask slot is structurally
# different from what the model actually needs to reconstruct. So we never put
# the mask token into the string: we put a real placeholder digit in its place,
# tokenize normally (producing the correct token structure), then swap that
# digit's token id for the mask token id afterwards. See make_collate_fn.
PLACEHOLDER_DIGIT = "1"


def make_prompt(
    l1_word: str,
    l1_name: str,
    l1_context: str,
    clue: str,
    en_word: str,
) -> str:
    return dedent(f"""
    Rate how easy it is for learners to guess the English word based on the {l1_name} word, context and clue on a scale from 1 to 5 (1=very easy, 5=very difficult).
    {l1_name} word: {l1_word}
    {l1_name} context: {l1_context}
    Clue: {clue}
    English word: {en_word}
    Difficulty: {PLACEHOLDER_DIGIT}
    """).strip()


class KVLDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.rows = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows.iloc[idx]
        text = make_prompt(
            l1_word=row["L1_source_word"],
            l1_name=row["L1_name"],
            l1_context=row["L1_context"],
            clue=row["en_target_clue"],
            en_word=row["en_target_word"],
        )
        return {"text": text, "score": float(row["remapped_score"])}


def make_collate_fn(tokenizer, placeholder_token_id: int) -> Callable[[list[dict]], dict]:
    def collate_batch(batch: list[dict]) -> dict:
        texts = [item["text"] for item in batch]
        tokenized = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

        input_ids = tokenized["input_ids"]
        is_placeholder = input_ids == placeholder_token_id
        # Takes the *last* match per row: the placeholder is the answer slot we
        # inserted, but the digit "1" could coincidentally also appear earlier
        # in L1_context (e.g. a number in the example sentence).
        if not torch.all(is_placeholder.any(dim=1)):
            raise ValueError(
                "Placeholder answer digit not found in one or more rows — a long "
                "L1_context is likely truncating away the answer slot."
            )
        last_positions = is_placeholder.size(1) - 1 - is_placeholder.flip(dims=[1]).long().argmax(dim=1)
        input_ids[torch.arange(input_ids.size(0)), last_positions] = tokenizer.mask_token_id

        # "score" (the gold label) is only present for datasets that have one —
        # KVLDataset does, inference-only datasets like MultiLSDataset don't.
        if all("score" in item for item in batch):
            tokenized["score"] = torch.tensor(
                [item["score"] for item in batch], dtype=torch.float32
            )
        return tokenized

    return collate_batch
