import math

import numpy as np
import pandas as pd

from plb.model_b.ablations.gbdt import (
    BASE_COLUMNS,
    CENSORED_COLUMNS,
    aggregate_neighbors,
    neighbor_feature_frame,
)


def _nbr(**overrides) -> dict:
    base = {
        "word_instance_id": "n",
        "s": None,
        "d": None,
        "reviews": 0,
        "plb_14": None,
        "log_plb_14": None,
        "is_censored": False,
        "sem_pct": None,
        "phon_pct": None,
        "mor_pct": None,
        "ortho_stroke_pct": None,
        "ortho_kana_pct": None,
        "selected_by": ["semantic"],
    }
    base.update(overrides)
    return base


# Two uncensored + one censored neighbor with hand-checkable numbers.
A = _nbr(sem_pct=0.9, log_plb_14=1.0, plb_14=3, s=1.0, d=5.0)
B = _nbr(
    sem_pct=0.8,
    phon_pct=0.95,
    log_plb_14=2.0,
    plb_14=7,
    s=3.0,
    d=6.0,
    selected_by=["semantic", "phonological"],
)
C = _nbr(sem_pct=0.7, s=0.5, d=7.0, is_censored=True)


def test_the_completed_variant_drops_censored_neighbors_everywhere() -> None:
    features = aggregate_neighbors([A, B, C], include_censored=False)

    assert features["n_neighbors"] == 2.0
    assert features["n_nbr_semantic"] == 2.0
    assert features["mean_pct_semantic"] == (0.9 + 0.8) / 2
    assert features["mean_d"] == (5.0 + 6.0) / 2  # C's d=7.0 is excluded
    assert "cen_n_neighbors" not in features


def test_the_censored_variant_keeps_censored_neighbors_in_the_base_aggregates() -> None:
    features = aggregate_neighbors([A, B, C], include_censored=True)

    assert features["n_neighbors"] == 3.0
    assert features["n_nbr_semantic"] == 3.0
    assert features["mean_pct_semantic"] == (0.9 + 0.8 + 0.7) / 3
    assert features["mean_d"] == (5.0 + 6.0 + 7.0) / 3
    assert (
        features["mean_log1p_s"]
        == (math.log1p(1.0) + math.log1p(3.0) + math.log1p(0.5)) / 3
    )


def test_burden_means_stay_uncensored_only_in_both_variants() -> None:
    completed = aggregate_neighbors([A, B, C], include_censored=False)
    censored = aggregate_neighbors([A, B, C], include_censored=True)

    for features in (completed, censored):
        assert features["mean_log_plb"] == (1.0 + 2.0) / 2
        assert features["mean_log_plb_semantic"] == (1.0 + 2.0) / 2


def test_per_channel_features_only_cover_neighbors_selected_by_that_channel() -> None:
    features = aggregate_neighbors([A, B], include_censored=False)

    assert features["n_nbr_phonological"] == 1.0  # only B
    assert features["mean_pct_phonological"] == 0.95
    assert features["n_nbr_morphological"] == 0.0
    assert math.isnan(features["mean_pct_morphological"])


def test_union_mean_pct_uses_each_neighbors_best_channel_percentile() -> None:
    features = aggregate_neighbors([A, B], include_censored=False)

    # A's best is sem 0.9; B's best is phon 0.95 (> its sem 0.8).
    assert features["mean_pct"] == (0.9 + 0.95) / 2


def test_null_values_are_skipped_not_treated_as_zero() -> None:
    no_stability = _nbr(sem_pct=0.6, log_plb_14=1.0, plb_14=3)  # s and d null

    features = aggregate_neighbors([A, no_stability], include_censored=False)

    assert features["mean_log1p_s"] == math.log1p(1.0)  # only A contributes
    assert features["mean_d"] == 5.0


def test_an_empty_neighbor_list_yields_zero_counts_and_nan_means() -> None:
    features = aggregate_neighbors([], include_censored=True)

    assert features["n_neighbors"] == 0.0
    assert features["n_nbr_semantic"] == 0.0
    assert math.isnan(features["mean_pct"])
    assert math.isnan(features["mean_log_plb"])
    assert math.isnan(features["cen_share"])


def test_cen_share_is_the_censored_fraction() -> None:
    features = aggregate_neighbors([A, B, C], include_censored=True)

    assert features["cen_share"] == 1 / 3
    assert features["cen_n_neighbors"] == 1.0
    assert features["cen_mean_pct_semantic"] == 0.7
    assert features["cen_mean_d"] == 7.0


def test_frame_column_order_matches_the_exported_constants() -> None:
    series = pd.Series([[A, B, C], []])

    completed = neighbor_feature_frame(series, include_censored=False)
    censored = neighbor_feature_frame(series, include_censored=True)

    assert list(completed.columns) == list(BASE_COLUMNS)
    assert list(censored.columns) == list(BASE_COLUMNS + CENSORED_COLUMNS)
    assert len(completed) == 2
    assert np.isnan(censored.loc[1, "mean_pct"])
