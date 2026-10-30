import pandas as pd
import pytest

from plb.model_a.multils.compare import compare_prediction_artifacts


def frame(predictions: list[float]) -> pd.DataFrame:
    rows = []
    index = 0
    for item_id, expression in [("one", "single_word"), ("two", "mwe")]:
        for annotator, l1_name, rating in [
            ("a1", "English", 1.0 + index),
            ("a2", "French", 2.0 + index),
        ]:
            rows.append(
                {
                    "id": item_id,
                    "target": item_id,
                    "annotator": annotator,
                    "l1_name": l1_name,
                    "human_rating": rating,
                    "predicted_rating": predictions[len(rows)],
                    "source_split": "test",
                    "expression_type": expression,
                    "source_word_class": "content_word",
                    "word_origin": "wago",
                    "gloss_kind": "translation",
                    "alignment_kind": "exact",
                    "cue_warnings": "[]",
                    "evaluation_bucket": (
                        "mwes" if expression == "mwe" else "single_words"
                    ),
                }
            )
        index += 1
    return pd.DataFrame(rows)


def test_compare_prediction_artifacts_reports_paired_views() -> None:
    fine = frame([1.0, 2.0, 2.0, 3.0])
    base = frame([3.0, 3.0, 2.0, 2.0])
    metrics, item_macro = compare_prediction_artifacts(
        fine,
        base,
        condition="full_l1",
        bootstrap_samples=5,
        bootstrap_seed=42,
    )
    assert metrics["metadata"]["primary_condition"] is True
    assert set(metrics["test"]["item_macro"]) == {
        "all_items",
        "single_words",
        "mwes",
        "others",
    }
    assert metrics["metadata"]["primary_hypothesis_metric"].startswith(
        "test.item_macro.single_words"
    )
    assert set(metrics["test"]["per_annotator"]) == {"a1", "a2"}
    assert set(metrics["test"]["per_l1"]) == {"English", "French"}
    assert len(item_macro) == 2


def test_compare_rejects_different_gold_rows() -> None:
    fine = frame([1.0, 2.0, 2.0, 3.0])
    base = frame([1.0, 2.0, 2.0, 3.0])
    base.loc[0, "human_rating"] = 5.0
    with pytest.raises(ValueError, match="identical gold rows"):
        compare_prediction_artifacts(
            fine,
            base,
            condition="full_l1",
            bootstrap_samples=0,
            bootstrap_seed=42,
        )
