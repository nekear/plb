"""Precompute the fine-tuned mmBERT (Model A) difficulty score for every word instance.

Model B carries the KVL prior as a word-level feature. The score depends only on the
lexeme form, the deck's target language, the learner's L1, and the stored meaning —
all fixed per word instance — so it is computed once here instead of on every
training or inference pass. The output is a `word_instance_id` -> `kvl_score` map,
where `kvl_score` is the raw 1-5 prediction (1=very easy, 5=very difficult); the log
transform belongs to Model B's feature extractor.

The prompt is the `full_l2_deploy` condition used in the MultiLS evaluation, with the
deck's target language substituted for its hardcoded "Japanese" — see
`make_deploy_prompt`. Two approximations are inherited from that evaluation and kept
deliberately, so the precomputed scores stay comparable with the reported results:

- only the first of a learner's native languages is used, even when several are known;
- `meaning` is the deck's own gloss, which is not always written in that L1 (some
  Japanese decks carry monolingual Japanese definitions), yet the prompt labels it as
  the L1 gloss regardless.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from textwrap import dedent

import pandas as pd
from transformers import AutoModelForMaskedLM, AutoTokenizer

from plb.model_a.common.config import load_model_a_config
from plb.model_a.common.inference import predict_prompts
from plb.model_a.kvl.data import PLACEHOLDER_DIGIT
from plb.utils import get_device

# Prompts read naturally only with a language name, never a code. Unknown codes fail
# rather than falling back to the raw code, which would silently change the prompt
# wording for a whole deck; add new codes here as new decks enter the pool.
LANGUAGE_NAMES = {
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "nl": "Dutch",
    "pt": "Portuguese",
    "zh": "Chinese",
    "zh-Hans": "Simplified Chinese",
    "zh-Hant": "Traditional Chinese",
}


def language_name(code: str) -> str:
    try:
        return LANGUAGE_NAMES[code]
    except KeyError as error:
        raise ValueError(
            f"No language name mapping for code {code!r} — add it to LANGUAGE_NAMES "
            "before precomputing predictions for this data."
        ) from error


def primary_native_language(native_languages: Sequence[str]) -> str:
    """The learner's first L1 — the one the deploy prompt is conditioned on."""
    if native_languages is None or len(native_languages) == 0:
        raise ValueError("Learner has no native language recorded")
    return native_languages[0]


def make_deploy_prompt(
    *,
    l2_name: str,
    target: str,
    l1_name: str,
    l1_gloss: str,
) -> str:
    """The `full_l2_deploy` prompt, generalised over the target language.

    With `l2_name="Japanese"` this reproduces
    `make_multils_prompt(MultiLSCondition.FULL_L2_DEPLOY, ...)` verbatim — the test
    suite pins that equality, since the precomputed scores are only comparable with
    the reported MultiLS results while the two prompts agree.
    """
    return dedent(
        f"""
            Rate how easy it is for learners to guess the {l1_name} word based on the {l2_name} word on a scale from 1 to 5 (1=very easy, 5=very difficult).
            {l2_name} word: {target}
            {l1_name} word: {l1_gloss}
            Difficulty: {PLACEHOLDER_DIGIT}
        """
    ).strip()


def _require_columns(
    dataframe: pd.DataFrame,
    required: Iterable[str],
    dataset_name: str,
) -> None:
    missing = sorted(set(required) - set(dataframe.columns))
    if missing:
        raise ValueError(f"{dataset_name} is missing required columns: {missing}")


def _merge_or_raise(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    on: str,
    left_name: str,
    right_name: str,
) -> pd.DataFrame:
    merged = left.merge(
        right,
        on=on,
        how="left",
        validate="many_to_one",
        indicator="_merge_indicator",
    )
    unmatched = merged["_merge_indicator"] != "both"
    if unmatched.any():
        examples = (
            merged.loc[unmatched, on].astype(str).drop_duplicates().head(5).tolist()
        )
        raise ValueError(
            f"{left_name} reference {on} values missing from {right_name}; "
            f"examples: {examples}"
        )
    return merged.drop(columns="_merge_indicator")


def build_prompt_frame(
    word_instances_df: pd.DataFrame,
    lexemes_df: pd.DataFrame,
    decks_df: pd.DataFrame,
    users_df: pd.DataFrame,
) -> pd.DataFrame:
    """Joins every word instance to its lexeme form, deck language and learner L1.

    Returns `word_instance_id` and the rendered prompt `text`, in input order.
    """
    _require_columns(
        word_instances_df,
        {"word_instance_id", "lexeme_id", "deck_id", "meaning"},
        "word instances",
    )
    _require_columns(lexemes_df, {"lexeme_id", "form"}, "lexemes")
    _require_columns(decks_df, {"deck_id", "user_id", "target_language"}, "decks")
    _require_columns(users_df, {"user_id", "native_languages"}, "users")

    rows = word_instances_df[["word_instance_id", "lexeme_id", "deck_id", "meaning"]]
    rows = _merge_or_raise(
        rows,
        lexemes_df[["lexeme_id", "form"]],
        on="lexeme_id",
        left_name="Word instances",
        right_name="lexemes",
    )
    rows = _merge_or_raise(
        rows,
        decks_df[["deck_id", "user_id", "target_language"]],
        on="deck_id",
        left_name="Word instances",
        right_name="decks",
    )
    rows = _merge_or_raise(
        rows,
        users_df[["user_id", "native_languages"]],
        on="user_id",
        left_name="Decks",
        right_name="users",
    )

    l1_codes = rows["native_languages"].map(primary_native_language)
    unknown_codes = sorted(
        set(l1_codes).union(rows["target_language"]) - set(LANGUAGE_NAMES)
    )
    if unknown_codes:
        raise ValueError(
            f"No language name mapping for code(s) {unknown_codes} — add them to "
            "LANGUAGE_NAMES before precomputing predictions for this data."
        )

    l1_names = l1_codes.map(language_name)
    l2_names = rows["target_language"].map(language_name)

    texts = [
        make_deploy_prompt(
            l2_name=l2_name,
            target=form,
            l1_name=l1_name,
            l1_gloss=meaning,
        )
        for l2_name, form, l1_name, meaning in zip(
            l2_names, rows["form"], l1_names, rows["meaning"], strict=True
        )
    ]
    return pd.DataFrame({"word_instance_id": rows["word_instance_id"], "text": texts})


def main() -> None:
    config = load_model_a_config()
    precompute = config.precompute

    prompts = build_prompt_frame(
        pd.read_json(precompute.word_instances_path, lines=True),
        pd.read_json(precompute.lexemes_path, lines=True),
        pd.read_json(precompute.decks_path, lines=True),
        pd.read_json(precompute.users_path, lines=True),
    )

    device = get_device()
    run_dir = config.run_dir_models
    tokenizer = AutoTokenizer.from_pretrained(run_dir)
    model = AutoModelForMaskedLM.from_pretrained(run_dir).to(device)

    print(f"Device: {device}")
    print(f"Scoring {len(prompts):,} word instances with run {config.run_name!r}")

    scores = predict_prompts(
        prompts,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=precompute.batch_size,
        desc="Precomputing KVL scores",
        max_batch_tokens=precompute.max_batch_tokens,
    )

    predictions = pd.DataFrame(
        {"word_instance_id": prompts["word_instance_id"], "kvl_score": scores}
    )
    precompute.output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_json(precompute.output_path, orient="records", lines=True)
    print(f"Wrote {len(predictions):,} KVL scores to {precompute.output_path}")


if __name__ == "__main__":
    main()
