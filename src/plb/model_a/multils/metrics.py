from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Callable, Iterator

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr

METRIC_NAMES = ("spearman", "pearson", "mae", "rmse")
PERFORMANCE_VIEWS = ("all_items", "single_words", "mwes", "others")
ITEM_STABLE_COLUMNS = (
    "source_split",
    "target",
    "expression_type",
    "source_word_class",
    "word_origin",
)
L1_STABLE_COLUMNS = (
    *ITEM_STABLE_COLUMNS,
    "gloss_kind",
    "alignment_kind",
    "cue_warnings",
    "evaluation_bucket",
)


def iter_performance_views(frame: pd.DataFrame) -> Iterator[tuple[str, pd.DataFrame]]:
    yield "all_items", frame
    for bucket in PERFORMANCE_VIEWS[1:]:
        yield bucket, frame[frame["evaluation_bucket"] == bucket]


def _finite_float(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def compute_point_metrics(
    frame: pd.DataFrame,
    *,
    gold_col: str = "human_rating",
    pred_col: str = "predicted_rating",
) -> dict:
    gold = frame[gold_col].to_numpy(dtype=float)
    pred = frame[pred_col].to_numpy(dtype=float)
    warnings: list[str] = []
    if len(frame) < 2:
        spearman = pearson = None
        warnings.append("correlation_requires_at_least_two_items")
    elif np.unique(gold).size < 2 or np.unique(pred).size < 2:
        spearman = pearson = None
        warnings.append("correlation_undefined_for_constant_values")
    else:
        spearman = _finite_float(spearmanr(gold, pred).statistic)
        pearson = _finite_float(pearsonr(gold, pred).statistic)
    result = {
        "n": int(len(frame)),
        "spearman": spearman,
        "pearson": pearson,
        "mae": _finite_float(np.mean(np.abs(gold - pred))) if len(frame) else None,
        "rmse": _finite_float(np.sqrt(np.mean((gold - pred) ** 2))) if len(frame) else None,
    }
    if warnings:
        result["warnings"] = warnings
    return result


def _rowwise_correlation(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.shape[1] < 2:
        return np.full(left.shape[0], np.nan)
    left_centered = left - left.mean(axis=1, keepdims=True)
    right_centered = right - right.mean(axis=1, keepdims=True)
    numerator = np.sum(left_centered * right_centered, axis=1)
    denominator = np.sqrt(
        np.sum(left_centered**2, axis=1) * np.sum(right_centered**2, axis=1)
    )
    return np.divide(
        numerator,
        denominator,
        out=np.full(left.shape[0], np.nan),
        where=denominator != 0,
    )


def _matrix_metrics(gold: np.ndarray, pred: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "spearman": _rowwise_correlation(rankdata(gold, axis=1), rankdata(pred, axis=1)),
        "pearson": _rowwise_correlation(gold, pred),
        "mae": np.mean(np.abs(gold - pred), axis=1),
        "rmse": np.sqrt(np.mean((gold - pred) ** 2, axis=1)),
    }


def _bootstrap_metric_samples(
    frame: pd.DataFrame,
    *,
    samples: int,
    seed: int,
    gold_col: str,
    pred_col: str,
) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}
    if samples <= 0 or frame.empty:
        return values
    rng = np.random.default_rng(seed)
    gold = frame[gold_col].to_numpy(dtype=float)
    pred = frame[pred_col].to_numpy(dtype=float)
    for start in range(0, samples, 256):
        count = min(256, samples - start)
        positions = rng.integers(0, len(frame), size=(count, len(frame)))
        chunk = _matrix_metrics(gold[positions], pred[positions])
        for name in METRIC_NAMES:
            finite = chunk[name][np.isfinite(chunk[name])]
            values[name].extend(finite.astype(float).tolist())
    return values


def _confidence_intervals(samples: dict[str, list[float]]) -> dict[str, dict | None]:
    intervals: dict[str, dict | None] = {}
    for name, values in samples.items():
        if not values:
            intervals[name] = None
        else:
            low, high = np.quantile(np.asarray(values), [0.025, 0.975])
            intervals[name] = {"low": float(low), "high": float(high)}
    return intervals


def compute_metrics_with_bootstrap(
    frame: pd.DataFrame,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
    gold_col: str = "human_rating",
    pred_col: str = "predicted_rating",
) -> dict:
    result = compute_point_metrics(frame, gold_col=gold_col, pred_col=pred_col)
    samples = _bootstrap_metric_samples(
        frame,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
        gold_col=gold_col,
        pred_col=pred_col,
    )
    result["ci95"] = _confidence_intervals(samples)
    result["bootstrap_samples"] = int(bootstrap_samples)
    return result


def metrics_by_view(
    frame: pd.DataFrame,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
    gold_col: str = "human_rating",
    pred_col: str = "predicted_rating",
) -> dict:
    return {
        name: compute_metrics_with_bootstrap(
            group,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + offset,
            gold_col=gold_col,
            pred_col=pred_col,
        )
        for offset, (name, group) in enumerate(iter_performance_views(frame))
    }


def _first_aggregation(columns: tuple[str, ...], frame: pd.DataFrame) -> dict:
    return {column: (column, "first") for column in columns if column in frame.columns}


def aggregate_item_macro(per_annotator: pd.DataFrame) -> pd.DataFrame:
    aggregations = {
        "human_rating": ("human_rating", "mean"),
        "predicted_rating": ("predicted_rating", "mean"),
        "rating_count": ("human_rating", "size"),
        "l1_item_count": ("l1_name", "nunique"),
        **_first_aggregation(ITEM_STABLE_COLUMNS, per_annotator),
    }
    return (
        per_annotator.groupby("id", as_index=False)
        .agg(**aggregations)
        .sort_values("id")
        .reset_index(drop=True)
    )


def aggregate_l1_items(per_annotator: pd.DataFrame) -> pd.DataFrame:
    aggregations = {
        "human_rating": ("human_rating", "mean"),
        "predicted_rating": ("predicted_rating", "first"),
        "annotator_count": ("annotator", "nunique"),
        **_first_aggregation(L1_STABLE_COLUMNS, per_annotator),
    }
    return (
        per_annotator.groupby(["id", "l1_name"], as_index=False)
        .agg(**aggregations)
        .sort_values(["l1_name", "id"])
        .reset_index(drop=True)
    )


def item_macro_metrics_by_view(
    frame: pd.DataFrame, *, bootstrap_samples: int, bootstrap_seed: int
) -> dict:
    results = {}
    for offset, (name, rows) in enumerate(iter_performance_views(frame)):
        items = aggregate_item_macro(rows)
        results[name] = compute_metrics_with_bootstrap(
            items,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + offset,
        )
    return results


def _macro_summary(per_annotator_metrics: dict[str, dict]) -> dict:
    result: dict[str, dict] = {}
    for view in PERFORMANCE_VIEWS:
        result[view] = {}
        for metric in METRIC_NAMES:
            values = [
                metrics[view][metric]
                for metrics in per_annotator_metrics.values()
                if metrics[view][metric] is not None
            ]
            result[view][metric] = {
                "mean": float(np.mean(values)) if values else None,
                "median": float(np.median(values)) if values else None,
                "annotator_count": len(values),
            }
    return result


def _parse_warnings(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    if pd.isna(value) or not str(value).strip():
        return []
    parsed = json.loads(str(value))
    if not isinstance(parsed, list):
        raise ValueError("cue_warnings must be a JSON array")
    return [str(item) for item in parsed]


def bucket_composition(frame: pd.DataFrame) -> dict:
    def counts(series: pd.Series) -> dict[str, int]:
        return {
            str(key): int(value)
            for key, value in series.value_counts().sort_index().items()
        }

    result: dict[str, dict] = {}
    for name, group in iter_performance_views(frame):
        warning_counts: Counter[str] = Counter()
        for value in group.get("cue_warnings", pd.Series(dtype=object)):
            warning_counts.update(_parse_warnings(value))
        result[name] = {
            "annotator_rows": int(len(group)),
            "l1_item_pairs": int(group[["id", "l1_name"]].drop_duplicates().shape[0]),
            "unique_items": int(group["id"].nunique()),
            "by_alignment_kind": counts(group["alignment_kind"]),
            "by_gloss_kind": counts(group["gloss_kind"]),
            "by_source_word_class": counts(group["source_word_class"]),
            "warning_counts": dict(sorted(warning_counts.items())),
        }
    return result


def _fit_trial_calibration(per_annotator: pd.DataFrame) -> tuple[dict, float | None, float | None]:
    trial_rows = per_annotator[per_annotator["source_split"] == "trial"]
    trial_items = aggregate_item_macro(trial_rows)
    if len(trial_items) < 2 or trial_items["predicted_rating"].nunique() < 2:
        return (
            {"fit": None, "warning": "trial_predictions_are_constant_or_insufficient"},
            None,
            None,
        )
    slope, intercept = np.polyfit(
        trial_items["predicted_rating"].to_numpy(),
        trial_items["human_rating"].to_numpy(),
        deg=1,
    )
    fit = {"n": int(len(trial_items)), "slope": float(slope), "intercept": float(intercept)}
    if slope <= 0:
        return ({"fit": fit, "warning": "non_positive_calibration_slope"}, None, None)
    return ({"fit": fit}, float(slope), float(intercept))


def _calibrated_error_metrics(
    test_rows: pd.DataFrame, *, slope: float, intercept: float
) -> dict:
    result = {}
    for name, rows in iter_performance_views(test_rows):
        items = aggregate_item_macro(rows)
        calibrated = np.clip(slope * items["predicted_rating"] + intercept, 1.0, 5.0)
        gold = items["human_rating"].to_numpy(dtype=float)
        pred = calibrated.to_numpy(dtype=float)
        result[name] = {
            "n": int(len(items)),
            "mae": float(np.mean(np.abs(gold - pred))) if len(items) else None,
            "rmse": float(np.sqrt(np.mean((gold - pred) ** 2))) if len(items) else None,
        }
    return result


def evaluate_prediction_frame(
    per_annotator: pd.DataFrame,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    test = per_annotator[per_annotator["source_split"] == "test"].copy()
    item_macro = aggregate_item_macro(per_annotator)
    l1_items = aggregate_l1_items(per_annotator)
    test_l1_items = l1_items[l1_items["source_split"] == "test"]

    per_annotator_metrics = {
        annotator: metrics_by_view(
            group,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + index * 10,
        )
        for index, (annotator, group) in enumerate(test.groupby("annotator", sort=True), 1)
    }
    per_l1_metrics = {
        l1_name: {
            "annotator_count": int(group["annotator_count"].iloc[0]),
            "metrics": metrics_by_view(
                group,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed + index * 10,
            ),
        }
        for index, (l1_name, group) in enumerate(
            test_l1_items.groupby("l1_name", sort=True), 1
        )
    }
    calibration, slope, intercept = _fit_trial_calibration(per_annotator)
    if slope is not None and intercept is not None:
        calibration["test_error_metrics"] = _calibrated_error_metrics(
            test, slope=slope, intercept=intercept
        )

    metrics = {
        "test": {
            "item_macro": item_macro_metrics_by_view(
                test,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            ),
            "per_annotator": {
                "summary": _macro_summary(per_annotator_metrics),
                "annotators": per_annotator_metrics,
            },
            "per_l1": per_l1_metrics,
            "bucket_composition": bucket_composition(test),
            "calibrated_on_trial": calibration,
        }
    }
    return metrics, item_macro, l1_items


def merge_prediction_frames(
    fine_tuned: pd.DataFrame,
    base: pd.DataFrame,
    *,
    view_builder: Callable[[pd.DataFrame], pd.DataFrame],
) -> pd.DataFrame:
    fine_view = view_builder(fine_tuned)
    base_view = view_builder(base)
    key_columns = [column for column in ("id", "annotator", "l1_name") if column in fine_view]
    merged = fine_view.merge(
        base_view[[*key_columns, "predicted_rating"]],
        on=key_columns,
        how="inner",
        validate="one_to_one",
        suffixes=("_fine_tuned", "_base"),
    )
    if len(merged) != len(fine_view) or len(merged) != len(base_view):
        raise ValueError("Fine-tuned and base prediction views lack identical comparison keys")
    return merged.rename(
        columns={
            "predicted_rating_fine_tuned": "fine_tuned_rating",
            "predicted_rating_base": "base_rating",
        }
    )


def paired_delta_metrics(
    frame: pd.DataFrame, *, bootstrap_samples: int, bootstrap_seed: int
) -> dict:
    fine = compute_point_metrics(frame, pred_col="fine_tuned_rating")
    base = compute_point_metrics(frame, pred_col="base_rating")
    delta = {
        metric: None
        if fine[metric] is None or base[metric] is None
        else float(fine[metric] - base[metric])
        for metric in METRIC_NAMES
    }
    bootstrap_values: dict[str, list[float]] = {metric: [] for metric in METRIC_NAMES}
    if bootstrap_samples > 0 and not frame.empty:
        rng = np.random.default_rng(bootstrap_seed)
        gold = frame["human_rating"].to_numpy(dtype=float)
        fine_pred = frame["fine_tuned_rating"].to_numpy(dtype=float)
        base_pred = frame["base_rating"].to_numpy(dtype=float)
        for start in range(0, bootstrap_samples, 256):
            count = min(256, bootstrap_samples - start)
            positions = rng.integers(0, len(frame), size=(count, len(frame)))
            fine_sample = _matrix_metrics(gold[positions], fine_pred[positions])
            base_sample = _matrix_metrics(gold[positions], base_pred[positions])
            for metric in METRIC_NAMES:
                deltas = fine_sample[metric] - base_sample[metric]
                finite = deltas[np.isfinite(deltas)]
                bootstrap_values[metric].extend(finite.astype(float).tolist())
    return {
        "n": int(len(frame)),
        "fine_tuned": fine,
        "base": base,
        "delta_fine_tuned_minus_base": delta,
        "delta_ci95": _confidence_intervals(bootstrap_values),
        "bootstrap_samples": int(bootstrap_samples),
        "error_delta_direction": "negative MAE/RMSE deltas favor fine-tuning",
    }


def paired_deltas_by_view(
    frame: pd.DataFrame, *, bootstrap_samples: int, bootstrap_seed: int
) -> dict:
    return {
        name: paired_delta_metrics(
            group,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + offset,
        )
        for offset, (name, group) in enumerate(iter_performance_views(frame))
    }


def paired_item_macro_deltas(
    fine_tuned: pd.DataFrame,
    base: pd.DataFrame,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[dict, pd.DataFrame]:
    results = {}
    all_items_comparison: pd.DataFrame | None = None
    for offset, (name, fine_rows) in enumerate(iter_performance_views(fine_tuned)):
        base_rows = base if name == "all_items" else base[base["evaluation_bucket"] == name]
        comparison = merge_prediction_frames(
            fine_rows, base_rows, view_builder=aggregate_item_macro
        )
        if name == "all_items":
            all_items_comparison = comparison
        results[name] = paired_delta_metrics(
            comparison,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + offset,
        )
    if all_items_comparison is None:
        raise AssertionError("all_items comparison was not produced")
    return results, all_items_comparison
