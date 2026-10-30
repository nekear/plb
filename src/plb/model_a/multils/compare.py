from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from plb.model_a.common.config import load_model_a_config
from plb.model_a.multils.metrics import (
    aggregate_l1_items,
    merge_prediction_frames,
    paired_deltas_by_view,
    paired_item_macro_deltas,
)
from plb.utils import write_json


def validate_comparable_predictions(fine_tuned: pd.DataFrame, base: pd.DataFrame) -> None:
    key = ["id", "annotator", "l1_name"]
    comparison_columns = [
        *key,
        "human_rating",
        "source_split",
        "expression_type",
        "source_word_class",
        "word_origin",
        "gloss_kind",
        "alignment_kind",
        "cue_warnings",
        "evaluation_bucket",
        "target",
    ]
    fine = fine_tuned[comparison_columns].sort_values(key).reset_index(drop=True)
    base_view = base[comparison_columns].sort_values(key).reset_index(drop=True)
    if not fine.equals(base_view):
        raise ValueError(
            "Fine-tuned and base prediction artifacts do not describe identical gold rows"
        )


def compare_prediction_artifacts(
    fine_tuned: pd.DataFrame,
    base: pd.DataFrame,
    *,
    condition: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[dict, pd.DataFrame]:
    validate_comparable_predictions(fine_tuned, base)

    fine_test = fine_tuned[fine_tuned["source_split"] == "test"]
    base_test = base[base["source_split"] == "test"]
    item_macro_metrics, item_macro = paired_item_macro_deltas(
        fine_test,
        base_test,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    direct = merge_prediction_frames(fine_tuned, base, view_builder=lambda frame: frame.copy())
    direct_test = direct[direct["source_split"] == "test"]
    l1_items = merge_prediction_frames(
        fine_tuned, base, view_builder=aggregate_l1_items
    )
    l1_test = l1_items[l1_items["source_split"] == "test"]

    per_annotator = {
        annotator: paired_deltas_by_view(
            group,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + index * 10,
        )
        for index, (annotator, group) in enumerate(
            direct_test.groupby("annotator", sort=True), start=1
        )
    }
    per_l1 = {
        l1_name: paired_deltas_by_view(
            group,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + index * 10,
        )
        for index, (l1_name, group) in enumerate(
            l1_test.groupby("l1_name", sort=True), start=1
        )
    }
    metrics = {
        "metadata": {
            "condition": condition,
            "primary_condition": condition == "full_l1",
            "comparison": "kvl_fine_tuned_minus_untouched_base",
            "primary_hypothesis_metric": (
                "test.item_macro.single_words.delta_fine_tuned_minus_base.spearman"
            ),
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
        },
        "test": {
            "item_macro": item_macro_metrics,
            "per_annotator": per_annotator,
            "per_l1": per_l1,
        },
    }
    return metrics, item_macro


def main() -> None:
    config = load_model_a_config()
    fine_path = config.multils_fine_tuned_dir / "predictions_per_annotator.csv"
    base_path = config.multils_base_dir / "predictions_per_annotator.csv"
    fine_tuned = pd.read_csv(fine_path)
    base = pd.read_csv(base_path)
    metrics, item_macro = compare_prediction_artifacts(
        fine_tuned,
        base,
        condition=config.multils.condition,
        bootstrap_samples=config.multils.bootstrap_samples,
        bootstrap_seed=config.multils.bootstrap_seed,
    )
    output_dir: Path = config.multils_comparison_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "metrics.json", metrics)
    item_macro.to_csv(output_dir / "predictions_item_macro_comparison.csv", index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
