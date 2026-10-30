"""Per-learner evaluation with paired bootstrap confidence intervals.

Every metric is computed within learner and never pooled: the model's job is to rank
words for one specific learner, so a pooled number would mostly reflect between-learner
scale differences. The bootstrap is *paired* — one index matrix per learner is applied
identically to the targets and to every model's predictions — so the pairwise ΔRER
intervals measure the difference between models on the same resampled items, and the
baseline's own relative error reduction is identically zero in every replicate.

Medians predict a constant within a learner, so their Spearman correlation is
undefined; that is reported as null, never raised.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import spearmanr

from plb.learners import LEARNER_CODE_ORDER, get_learner_code
from plb.model_b.preparation.create_splits import LOG_TARGET_COLUMN, TARGET_COLUMN

METRICS = ("spearman", "log_mae", "mae", "mean_error", "rer")


def _none_if_nan(value: float | None) -> float | None:
    """
    Convert to `None` if values are `NaN` or `inf` to make them clean and JSON-compatible.
    """

    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _metric_values(
    y_log: NDArray[np.float64],
    plb: NDArray[np.float64],
    pred: NDArray[np.float64],
    baseline_log_mae: float,
) -> dict[str, float]:
    """
    Expand the metrics to a convenient form.

    Args: All inputs are respective NumPy vectors of values.
    """

    log_mae = float(np.mean(np.abs(pred - y_log)))
    # If peak-to-peak is zero, then there's no variation in the input, and hence correlation is undefined for a constant input
    if np.ptp(pred) == 0.0 or np.ptp(y_log) == 0.0:
        spearman = float("nan")
    else:
        spearman = float(spearmanr(y_log, pred).statistic)

    return {
        "spearman": spearman,
        "log_mae": log_mae,
        # Back-transformed to the review-count axis for interpretability.
        "mae": float(np.mean(np.abs(np.exp(pred) - plb))),
        "mean_error": float(np.mean(pred - y_log)),
        "rer": 1.0 - log_mae / baseline_log_mae
        if baseline_log_mae > 0
        else float("nan"),
    }


def _compute_cis(point: float, replicates: NDArray[np.float64]) -> dict[str, float | None]:
    """
    Compute CIs from bootstrapped replicates. Metrics from which `replicates` were created don't affect the calculations.
    """

    finite = replicates[np.isfinite(replicates)]
    if finite.size:
        ci_low, ci_high = np.percentile(finite, [2.5, 97.5])
    else:
        ci_low = ci_high = None


    return {
        "value": _none_if_nan(point),
        "ci_low": _none_if_nan(ci_low),
        "ci_high": _none_if_nan(ci_high),
    }


def evaluate_models(
    frame: pd.DataFrame,
    predictions: Mapping[str, NDArray[np.float64]],
    *,
    baseline: str,
    delta_pairs: Sequence[tuple[str, str]] = (),
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 42,
) -> tuple[dict, dict]:
    """Return ``(per_model_report, pairwise_delta_report)``.

    ``per_model_report[model][learner_code]`` holds ``n_rows`` plus a
    ``{value, ci_low, ci_high}`` triple per metric; the pairwise report holds the same
    triple for each pair's ΔRER. Relative error reduction is read against the
    ``baseline`` model's log-MAE, recomputed inside every bootstrap replicate.
    """
    if baseline not in predictions:
        raise ValueError(f"baseline {baseline!r} is not among the evaluated models")

    # Ensuring that predictions for compared pairs are in place, and that the pairs are not the same model
    for first, second in delta_pairs:
        unknown = {first, second} - set(predictions)
        if unknown:
            raise ValueError(
                f"delta_pairs references unknown models: {sorted(unknown)}"
            )
        if first == second:
            raise ValueError(
                f"delta_pairs must compare two different models; got {first!r} twice"
            )

    # Ensuring that all models have predictions for every item
    for name, pred in predictions.items():
        if len(pred) != len(frame):
            raise ValueError(
                f"{name} has {len(pred)} predictions for {len(frame)} rows"
            )

    y_log = frame[LOG_TARGET_COLUMN].to_numpy(dtype=np.float64)
    plb = frame[TARGET_COLUMN].to_numpy(dtype=np.float64)
    codes = frame["user_id"].map(get_learner_code)

    # Defining containers for each of the models
    per_model: dict[str, dict] = {name: {} for name in predictions}
    pairwise: dict[str, dict] = {f"{a}_vs_{b}": {} for a, b in delta_pairs}

    # The seed offset is tied to the canonical learner order, so a learner's replicates
    # do not change when another learner enters or leaves the test set.
    for offset, code in enumerate(LEARNER_CODE_ORDER):
        # Selecting rows belonging to the currently iterated learner
        rows = np.flatnonzero((codes == code).to_numpy())
        if rows.size == 0:
            continue
        learner_y = y_log[rows]
        learner_plb = plb[rows]
        learner_preds = {
            name: np.asarray(pred, dtype=np.float64)[rows]
            for name, pred in predictions.items()
        }

        # Computing the constant baseline log MAE
        baseline_log_mae = float(np.mean(np.abs(learner_preds[baseline] - learner_y)))

        # Computing metrics for a single prediction iteration per model (this is the "real" prediction on our current data without bootstrapping)
        points = {
            name: _metric_values(learner_y, learner_plb, pred, baseline_log_mae)
            for name, pred in learner_preds.items()
        }

        # Generating a [number of bootstrap samples x number of rows] matrix determining selected rows for each bootstrap iteration
        rng = np.random.default_rng(bootstrap_seed + offset)
        samples = rng.integers(0, rows.size, size=(bootstrap_samples, rows.size))

        # Forming a dictionary where each ablation (by name), for each metric we're interested in (by name), gets a list of computed values by sample
        replicates = {
            name: {metric: np.empty(bootstrap_samples) for metric in METRICS}
            for name in predictions
        }

        # Deltas are similar to `replicates`, but since we're interested only in RER, we're keying by the pair name, and using a single list of samples for RER deltas only
        delta_replicates = {pair: np.empty(bootstrap_samples) for pair in pairwise}

        for index in range(bootstrap_samples):
            take = samples[index]  # ensuring that paired models use the same samples

            # Extracting predictions for the sample
            sample_y = learner_y[take]
            sample_plb = learner_plb[take]
            sample_baseline_log_mae = float(
                np.mean(np.abs(learner_preds[baseline][take] - sample_y))
            )
            sample_rer = {}

            # Looping through the (model, predictions) pairs
            for name, pred in learner_preds.items():
                # Summarizing the metrics for that specific sample
                values = _metric_values(
                    sample_y, sample_plb, pred[take], sample_baseline_log_mae
                )
                # Accumulating metrics by name in `replicates` to then compute CIs
                for metric in METRICS:
                    replicates[name][metric][index] = values[metric]

                sample_rer[name] = values["rer"]

            # Looping through delta pairs to calculate the differences in RER between the pairs of interest
            for first, second in delta_pairs:
                delta_replicates[f"{first}_vs_{second}"][index] = (
                    sample_rer[first] - sample_rer[second]
                )

        # Computing CIs for all metrics
        for name in predictions:
            per_model[name][code] = {
                "n_rows": int(rows.size),
                **{
                    metric: _compute_cis(points[name][metric], replicates[name][metric])
                    for metric in METRICS
                },
            }
        for first, second in delta_pairs:
            pairwise[f"{first}_vs_{second}"][code] = {
                "delta_rer": _compute_cis(
                    points[first]["rer"] - points[second]["rer"],
                    delta_replicates[f"{first}_vs_{second}"],
                )
            }

    return per_model, pairwise
