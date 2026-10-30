import json

import numpy as np
import pandas as pd
import pytest

from plb.learners import get_user_id
from plb.model_b.core.evaluate import evaluate_models

L01 = get_user_id("L01")
L02 = get_user_id("L02")

RNG = np.random.default_rng(0)


def _frame(rows_per_learner: int = 40) -> pd.DataFrame:
    plb = np.tile(np.arange(1, rows_per_learner + 1), 2)
    return pd.DataFrame(
        {
            "user_id": [L01] * rows_per_learner + [L02] * rows_per_learner,
            "plb_14": plb,
            "log_plb_14": np.log(plb),
        }
    )


def _predictions(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    y = frame["log_plb_14"].to_numpy()
    return {
        "good": y + RNG.normal(0, 0.1, len(frame)),
        "noisy": y + RNG.normal(0, 1.0, len(frame)),
        "baseline": np.where(
            frame["user_id"] == L01,
            np.median(y[: len(y) // 2]),
            np.median(y[len(y) // 2 :]),
        ),
    }


def test_metrics_are_computed_per_learner_never_pooled() -> None:
    frame = _frame()
    predictions = _predictions(frame)
    # Make the model much worse for L02 only; L01's numbers must not move.
    skewed = dict(predictions)
    skewed["good"] = np.where(
        frame["user_id"] == L02, predictions["good"] + 5.0, predictions["good"]
    )

    original, _ = evaluate_models(frame, predictions, baseline="baseline")
    reskewed, _ = evaluate_models(frame, skewed, baseline="baseline")

    assert reskewed["good"]["L01"] == original["good"]["L01"]
    assert (
        reskewed["good"]["L02"]["log_mae"]["value"]
        > original["good"]["L02"]["log_mae"]["value"]
    )


def test_constant_predictions_report_null_spearman_instead_of_crashing() -> None:
    frame = _frame()
    per_model, _ = evaluate_models(frame, _predictions(frame), baseline="baseline")

    for code in ("L01", "L02"):
        assert per_model["baseline"][code]["spearman"]["value"] is None
        assert per_model["baseline"][code]["spearman"]["ci_low"] is None
        assert per_model["good"][code]["spearman"]["value"] is not None


def test_the_baselines_own_rer_is_exactly_zero_with_a_degenerate_ci() -> None:
    frame = _frame()
    per_model, _ = evaluate_models(frame, _predictions(frame), baseline="baseline")

    for code in ("L01", "L02"):
        rer = per_model["baseline"][code]["rer"]
        assert rer == {"value": 0.0, "ci_low": 0.0, "ci_high": 0.0}


def test_delta_rer_of_a_model_against_itself_is_zero_proving_pairing() -> None:
    frame = _frame()
    predictions = _predictions(frame)
    predictions["good_copy"] = predictions["good"].copy()

    _, pairwise = evaluate_models(
        frame, predictions, baseline="baseline", delta_pairs=[("good", "good_copy")]
    )

    for code in ("L01", "L02"):
        delta = pairwise["good_vs_good_copy"][code]["delta_rer"]
        assert delta == {"value": 0.0, "ci_low": 0.0, "ci_high": 0.0}


def test_the_seed_alone_determines_the_confidence_intervals() -> None:
    frame = _frame()
    predictions = _predictions(frame)

    first, _ = evaluate_models(
        frame, predictions, baseline="baseline", bootstrap_seed=1
    )
    same, _ = evaluate_models(frame, predictions, baseline="baseline", bootstrap_seed=1)
    other, _ = evaluate_models(
        frame, predictions, baseline="baseline", bootstrap_seed=2
    )

    assert first == same
    assert first["good"]["L01"]["log_mae"] != other["good"]["L01"]["log_mae"]


def test_an_unknown_baseline_is_rejected() -> None:
    frame = _frame()
    with pytest.raises(ValueError, match="baseline 'missing' is not among"):
        evaluate_models(frame, _predictions(frame), baseline="missing")


def test_delta_pairs_must_name_two_distinct_known_models() -> None:
    frame = _frame()
    predictions = _predictions(frame)

    with pytest.raises(ValueError, match="unknown models.*missing"):
        evaluate_models(
            frame, predictions, baseline="baseline", delta_pairs=[("good", "missing")]
        )
    with pytest.raises(ValueError, match="two different models"):
        evaluate_models(
            frame, predictions, baseline="baseline", delta_pairs=[("good", "good")]
        )


def test_reports_are_json_safe_with_nan_forbidden() -> None:
    frame = _frame()
    per_model, pairwise = evaluate_models(
        frame,
        _predictions(frame),
        baseline="baseline",
        delta_pairs=[("good", "baseline")],
    )

    # write_json uses allow_nan=False; a stray NaN anywhere would raise here.
    json.dumps({"models": per_model, "pairwise": pairwise}, allow_nan=False)
