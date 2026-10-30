from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plb.model_a.common.inference import predict_prompts
from plb.model_a.multils.cues import (
    AlignmentKind,
    EvaluationBucket,
    ExpressionType,
    GlossKind,
    SourceWordClass,
    cue_warning_codes,
    validate_complete_sidecar,
)
from plb.model_a.multils.data import (
    load_multils_items,
    load_multils_long,
    load_original_profiles,
    original_l1_names,
)
from plb.model_a.multils.metrics import evaluate_prediction_frame
from plb.model_a.multils.prompts import MultiLSCondition, make_multils_prompt
from plb.utils import write_json

CUE_CONDITIONS = {
    MultiLSCondition.FULL_L1,
    MultiLSCondition.FULL_L2,
    MultiLSCondition.FULL_L2_NO_CONTEXT,
    MultiLSCondition.FULL_L2_DEPLOY,
}

CUE_METADATA_COLUMNS = [
    "id",
    "l1_name",
    "l1_gloss",
    "l1_context",
    "gloss_kind",
    "l1_target_surface",
    "alignment_kind",
    "alignment_note",
    "l2_clue",
    "l1_clue",
    "word_origin",
    "expression_type",
    "source_word_class",
]


def evaluation_bucket(row: pd.Series) -> str:
    warnings = cue_warning_codes(row.to_dict())
    if row["expression_type"] == ExpressionType.MWE:
        return EvaluationBucket.MWES.value
    clean_single_word = (
        row["source_word_class"] == SourceWordClass.CONTENT_WORD
        and row["gloss_kind"] == GlossKind.TRANSLATION
        and row["alignment_kind"] in {AlignmentKind.EXACT, AlignmentKind.INFLECTED}
        and not warnings
    )
    return (
        EvaluationBucket.SINGLE_WORDS.value
        if clean_single_word
        else EvaluationBucket.OTHERS.value
    )


def load_validated_cues(data_dir: Path, cues_path: Path) -> pd.DataFrame:
    if not cues_path.exists():
        raise FileNotFoundError(
            f"MultiLS condition requires generated L1 cues, but {cues_path} does not exist; "
            "run the augmentation pilot and full generator first"
        )
    items = load_multils_items(data_dir)
    profiles = load_original_profiles(data_dir)
    cues = pd.read_csv(cues_path, sep="\t", keep_default_na=False)
    validate_complete_sidecar(cues, items, original_l1_names(profiles))
    return cues


def prepare_inference_prompts(
    data_dir: Path,
    condition: MultiLSCondition | str,
    cues_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    condition = MultiLSCondition(condition)
    per_annotator = load_multils_long(data_dir)
    if condition is MultiLSCondition.JA_WORD_ONLY:
        key_columns = ["id"]
        prompts = per_annotator.drop_duplicates("id")[
            ["id", "target", "context", "l2_clue"]
        ].copy()
    else:
        key_columns = ["id", "l1_name"]
        prompts = per_annotator.drop_duplicates(key_columns)[
            [
                "id",
                "l1_name",
                "target",
                "context",
                "l2_clue",
            ]
        ].copy()

    if condition in CUE_CONDITIONS:
        cues = load_validated_cues(data_dir, cues_path)
        prompts = prompts.merge(
            cues[["id", "l1_name", "l1_gloss", "l1_context", "l1_clue"]],
            on=["id", "l1_name"],
            how="left",
            validate="one_to_one",
        )
        if prompts[["l1_gloss", "l1_context", "l1_clue"]].isna().any().any():
            raise ValueError("Generated cue join left one or more prompts incomplete")

    def render(row: pd.Series) -> str:
        return make_multils_prompt(
            condition,
            target=row["target"],
            l1_name=row.get("l1_name"),
            l1_gloss=row.get("l1_gloss"),
            l1_context=row.get("l1_context"),
            japanese_context=row["context"],
            l2_clue=row.get("l2_clue"),
            l1_clue=row.get("l1_clue"),
        )

    prompts["text"] = prompts.apply(render, axis=1)
    return prompts, per_annotator, key_columns


def enrich_evaluation_rows(
    per_annotator: pd.DataFrame,
    cues: pd.DataFrame,
) -> pd.DataFrame:
    source_metadata = [
        "l2_clue",
        "word_origin",
        "expression_type",
        "source_word_class",
    ]
    enriched = per_annotator.drop(columns=source_metadata).merge(
        cues[CUE_METADATA_COLUMNS],
        on=["id", "l1_name"],
        how="left",
        validate="many_to_one",
    )
    if enriched[CUE_METADATA_COLUMNS[2:]].isna().any().any():
        raise ValueError("Cue metadata join left one or more annotator rows incomplete")
    enriched["cue_warnings"] = enriched.apply(
        lambda row: json.dumps(list(cue_warning_codes(row.to_dict())), ensure_ascii=False),
        axis=1,
    )
    enriched["evaluation_bucket"] = enriched.apply(evaluation_bucket, axis=1)
    output_columns = [
        "id",
        "annotator",
        "l1_name",
        "human_rating",
        "predicted_rating",
        "source_split",
        "target",
        "expression_type",
        "source_word_class",
        "word_origin",
        "gloss_kind",
        "alignment_kind",
        "cue_warnings",
        "evaluation_bucket",
    ]
    return enriched[output_columns].copy()


def attach_predictions(
    prompts: pd.DataFrame,
    per_annotator: pd.DataFrame,
    key_columns: list[str],
    predictions: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    inference = prompts[[*key_columns, "text"]].copy()
    inference["predicted_rating"] = predictions
    annotated = per_annotator.merge(
        inference[[*key_columns, "predicted_rating"]],
        on=key_columns,
        how="left",
        validate="many_to_one",
    )
    if annotated["predicted_rating"].isna().any():
        raise ValueError("Prediction join left one or more annotator rows without a score")
    return inference, annotated


def _save_calibration_plot(item_macro: pd.DataFrame, output_path: Path, model_label: str) -> None:
    test = item_macro[item_macro["source_split"] == "test"]
    plt.scatter(test["human_rating"], test["predicted_rating"], alpha=0.3, s=10)
    plt.plot([1, 5], [1, 5], color="black", linestyle="--", linewidth=1)
    plt.xlabel("Mean human rating (1-5) per item")
    plt.ylabel("Mean L1-conditioned prediction (1-5) per item")
    plt.title(f"Model A on MultiLS-Japanese — {model_label}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def evaluate_multils(
    *,
    model,
    tokenizer,
    device: torch.device,
    data_dir: Path,
    cues_path: Path,
    condition: MultiLSCondition | str,
    batch_size: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
    output_dir: Path,
    model_kind: str,
    model_identifier: str,
) -> dict:
    condition = MultiLSCondition(condition)
    prompts, source_rows, key_columns = prepare_inference_prompts(
        data_dir, condition, cues_path
    )
    predictions = predict_prompts(
        prompts,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
        desc="Evaluating on MultiLS-Japanese",
    )
    inference, per_annotator = attach_predictions(
        prompts, source_rows, key_columns, predictions
    )
    cues = load_validated_cues(data_dir, cues_path)
    per_annotator = enrich_evaluation_rows(per_annotator, cues)
    result_metrics, item_macro, l1_items = evaluate_prediction_frame(
        per_annotator,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    metrics = {
        "metadata": {
            "condition": condition.value,
            "primary_condition": condition is MultiLSCondition.FULL_L1,
            "model_kind": model_kind,
            "model_identifier": model_identifier,
            "gold_scale": "individual_lcp_rating_1_to_5",
            "primary_split": "test",
            "trial_role": "affine_error_calibration_only",
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
        },
        **result_metrics,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "metrics.json", metrics)
    inference.to_csv(output_dir / "predictions_inference.csv", index=False)
    per_annotator.to_csv(output_dir / "predictions_per_annotator.csv", index=False)
    item_macro.to_csv(output_dir / "predictions_item_macro.csv", index=False)
    l1_items.to_csv(output_dir / "predictions_per_l1_item.csv", index=False)
    _save_calibration_plot(item_macro, output_dir / "calibration.png", model_kind)
    return metrics
