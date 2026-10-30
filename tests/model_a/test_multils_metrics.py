import pandas as pd

from plb.model_a.multils.metrics import (
    aggregate_item_macro,
    aggregate_l1_items,
    compute_metrics_with_bootstrap,
    evaluate_prediction_frame,
    paired_delta_metrics,
)


def prediction_frame() -> pd.DataFrame:
    rows = []
    for item_id, expression, bucket, ratings, predictions in [
        ("one", "single_word", "single_words", (1.0, 3.0), (2.0, 4.0)),
        ("two", "single_word", "single_words", (2.0, 4.0), (2.5, 4.5)),
        ("three", "mwe", "mwes", (4.0, 5.0), (3.5, 4.5)),
        ("four", "single_word", "others", (3.0, 4.0), (2.0, 3.0)),
    ]:
        for annotator, l1_name, rating, prediction in zip(
            ("a1", "a2"), ("English", "French"), ratings, predictions, strict=True
        ):
            rows.append(
                {
                    "id": item_id,
                    "target": item_id,
                    "annotator": annotator,
                    "l1_name": l1_name,
                    "human_rating": rating,
                    "predicted_rating": prediction,
                    "source_split": "test",
                    "expression_type": expression,
                    "source_word_class": (
                        "function_word" if bucket == "others" else "content_word"
                    ),
                    "word_origin": "wago",
                    "gloss_kind": (
                        "metalinguistic" if bucket == "others" else "translation"
                    ),
                    "alignment_kind": (
                        "not_applicable" if bucket == "others" else "exact"
                    ),
                    "cue_warnings": "[]",
                    "evaluation_bucket": bucket,
                }
            )
    trial = pd.DataFrame(rows[:4]).copy()
    trial["id"] = ["trial-one", "trial-one", "trial-two", "trial-two"]
    trial["source_split"] = "trial"
    trial["human_rating"] = [1.0, 2.0, 2.0, 3.0]
    trial["predicted_rating"] = [1.5, 2.5, 2.5, 3.5]
    return pd.concat([pd.DataFrame(rows), trial], ignore_index=True)


def test_item_macro_averages_predictions_with_annotator_weights() -> None:
    items = aggregate_item_macro(prediction_frame())
    first = items[items["id"] == "one"].iloc[0]
    assert first["human_rating"] == 2.0
    assert first["predicted_rating"] == 3.0


def test_l1_items_preserve_annotator_counts() -> None:
    rows = prediction_frame()
    duplicate_english = rows[(rows["id"] == "one") & (rows["l1_name"] == "English")].copy()
    duplicate_english["annotator"] = "a3"
    grouped = aggregate_l1_items(pd.concat([rows, duplicate_english], ignore_index=True))
    english = grouped[(grouped["id"] == "one") & (grouped["l1_name"] == "English")].iloc[0]
    assert english["annotator_count"] == 2


def test_constant_prediction_correlation_is_json_safe_null() -> None:
    frame = pd.DataFrame({"human_rating": [1.0, 2.0], "predicted_rating": [3.0, 3.0]})
    metrics = compute_metrics_with_bootstrap(
        frame, bootstrap_samples=5, bootstrap_seed=42
    )
    assert metrics["spearman"] is None
    assert metrics["pearson"] is None
    assert metrics["ci95"]["spearman"] is None


def test_evaluation_reports_all_expression_views_and_annotators() -> None:
    metrics, item_macro, l1_items = evaluate_prediction_frame(
        prediction_frame(), bootstrap_samples=5, bootstrap_seed=42
    )
    assert set(metrics["test"]["item_macro"]) == {
        "all_items",
        "single_words",
        "mwes",
        "others",
    }
    assert set(metrics["test"]["per_annotator"]["annotators"]) == {"a1", "a2"}
    assert set(metrics["test"]["per_l1"]) == {"English", "French"}
    assert len(item_macro) == 6
    assert len(l1_items) == 12
    assert metrics["test"]["bucket_composition"]["all_items"]["unique_items"] == 4
    assert metrics["test"]["calibrated_on_trial"]["fit"]["slope"] > 0


def test_paired_delta_uses_fine_tuned_minus_base_direction() -> None:
    frame = pd.DataFrame(
        {
            "human_rating": [1.0, 2.0, 3.0, 4.0],
            "fine_tuned_rating": [1.0, 2.0, 3.0, 4.0],
            "base_rating": [4.0, 3.0, 2.0, 1.0],
        }
    )
    result = paired_delta_metrics(frame, bootstrap_samples=5, bootstrap_seed=42)
    assert result["delta_fine_tuned_minus_base"]["spearman"] == 2.0
    assert result["delta_fine_tuned_minus_base"]["mae"] < 0


def test_item_metrics_filter_l1_bucket_before_item_aggregation() -> None:
    rows = prediction_frame()
    rows = rows[rows["source_split"] == "test"].iloc[:2].copy()
    rows.loc[rows.index[0], ["human_rating", "predicted_rating"]] = [1.0, 1.0]
    rows.loc[rows.index[1], ["human_rating", "predicted_rating"]] = [5.0, 5.0]
    rows.loc[rows.index[0], "evaluation_bucket"] = "single_words"
    rows.loc[rows.index[1], "evaluation_bucket"] = "others"
    metrics, _, _ = evaluate_prediction_frame(
        rows, bootstrap_samples=0, bootstrap_seed=42
    )
    assert metrics["test"]["item_macro"]["all_items"]["n"] == 1
    assert metrics["test"]["item_macro"]["single_words"]["n"] == 1
    assert metrics["test"]["item_macro"]["others"]["n"] == 1
    assert metrics["test"]["item_macro"]["single_words"]["mae"] == 0.0


def test_non_positive_trial_calibration_is_reported_but_not_applied() -> None:
    rows = prediction_frame()
    trial_mask = rows["source_split"] == "trial"
    rows.loc[trial_mask, "predicted_rating"] = [4.0, 4.0, 2.0, 2.0]
    metrics, _, _ = evaluate_prediction_frame(
        rows, bootstrap_samples=0, bootstrap_seed=42
    )
    calibration = metrics["test"]["calibrated_on_trial"]
    assert calibration["fit"]["slope"] < 0
    assert calibration["warning"] == "non_positive_calibration_slope"
    assert "test_error_metrics" not in calibration
