"""GBDT ablations: intralexical features, optionally with neighbor aggregates.

One ``XGBoostModel`` class covers five ablations via params:

- ``include_nbrs: false`` — target word features only (do word features beat the
  learner median?);
- ``include_nbrs: true`` — plus per-channel and union aggregates over the target's
  UNCENSORED similarity neighbors (does selecting similar words help?);
- ``include_censored: true`` — censored neighbors additionally count in the base
  aggregates and get their own ``cen_*`` block (do words that resisted mastery carry
  signal about a harder tail?);
- ``nbr_features: count`` — keeps only ``n_neighbors`` out of the aggregates, so the
  model sees how far into the learner's history the target sits but nothing about what
  the neighborhood contains (is neighborhood size the whole neighbor signal?);
- ``include_days: true`` — replaces the neighborhood with the target's own position on
  the learner's timeline, in days since their first review (is the count just a noisy
  clock?).

Neighbor aggregation lives here rather than in ``core`` because it is GBDT-specific:
the SetTransformer consumes per-neighbor rows, not means. Aggregation conventions:
counts of empty sets are 0 (a real observation), means of empty sets are NaN (XGBoost
handles missing values natively), and the union ``mean_pct`` takes each neighbor's
best (max) non-null channel percentile before averaging.

The objective is pinned to ``reg:absoluteerror`` so the training loss estimates the
same quantity as the reported MAE metrics.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from xgboost import XGBRegressor

from plb.model_b.core.base import BaseModel
from plb.model_b.core.features import (
    INTRALEXICAL_COLUMNS,
    add_intralexical_features,
    load_intralexical_inputs,
)
from plb.utils import write_json

CHANNELS = ("semantic", "phonological", "morphological", "ortho_stroke", "ortho_kana")
PCT_FIELDS = {
    "semantic": "sem_pct",
    "phonological": "phon_pct",
    "morphological": "mor_pct",
    "ortho_stroke": "ortho_stroke_pct",
    "ortho_kana": "ortho_kana_pct",
}

BASE_COLUMNS = tuple(
    f"{prefix}_{channel}"
    for channel in CHANNELS
    for prefix in ("n_nbr", "mean_pct", "mean_log_plb")
) + ("n_neighbors", "mean_pct", "mean_log1p_s", "mean_d", "mean_log_plb")

CENSORED_COLUMNS = tuple(
    f"cen_{prefix}_{channel}"
    for channel in CHANNELS
    for prefix in ("n_nbr", "mean_pct")
) + ("cen_n_neighbors", "cen_share", "cen_mean_pct", "cen_mean_log1p_s", "cen_mean_d")


def _is_null(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _channel_features(
    neighbors: list[dict], channel: str, prefix: str = ""
) -> dict[str, float]:
    selected = [n for n in neighbors if channel in n["selected_by"]]
    pct_field = PCT_FIELDS[channel]
    features = {
        f"{prefix}n_nbr_{channel}": float(len(selected)),
        f"{prefix}mean_pct_{channel}": _mean(
            [n[pct_field] for n in selected if not _is_null(n[pct_field])]
        ),
    }
    if not prefix:  # burden means exist only in the base block, uncensored-only
        features[f"mean_log_plb_{channel}"] = _mean(
            [
                n["log_plb_14"]
                for n in selected
                if not n["is_censored"] and not _is_null(n["log_plb_14"])
            ]
        )
    return features


def _union_features(neighbors: list[dict], prefix: str = "") -> dict[str, float]:
    best_pcts = []
    for neighbor in neighbors:
        pcts = [
            neighbor[field]
            for field in PCT_FIELDS.values()
            if not _is_null(neighbor[field])
        ]
        if pcts:
            best_pcts.append(max(pcts))
    features = {
        f"{prefix}n_neighbors": float(len(neighbors)),
        f"{prefix}mean_pct": _mean(best_pcts),
        f"{prefix}mean_log1p_s": _mean(
            [math.log1p(n["s"]) for n in neighbors if not _is_null(n["s"])]
        ),
        f"{prefix}mean_d": _mean([n["d"] for n in neighbors if not _is_null(n["d"])]),
    }
    if not prefix:
        features["mean_log_plb"] = _mean(
            [
                n["log_plb_14"]
                for n in neighbors
                if not n["is_censored"] and not _is_null(n["log_plb_14"])
            ]
        )
    return features


def aggregate_neighbors(
    neighbors: list[dict], *, include_censored: bool
) -> dict[str, float]:
    """Aggregate one target's neighbor list into a flat feature dict."""
    base = (
        list(neighbors)
        if include_censored
        else [n for n in neighbors if not n["is_censored"]]
    )

    features: dict[str, float] = {}
    for channel in CHANNELS:
        features.update(_channel_features(base, channel))
    features.update(_union_features(base))

    if include_censored:
        censored = [n for n in neighbors if n["is_censored"]]
        for channel in CHANNELS:
            features.update(_channel_features(censored, channel, prefix="cen_"))
        union = _union_features(censored, prefix="cen_")
        union["cen_share"] = (
            len(censored) / len(neighbors) if neighbors else float("nan")
        )
        features.update(union)

    return features


def neighbor_feature_frame(
    neighbors: pd.Series, *, include_censored: bool
) -> pd.DataFrame:
    """One aggregated feature row per entry of a ``neighbors`` list column."""
    columns = BASE_COLUMNS + (CENSORED_COLUMNS if include_censored else ())
    rows = [
        aggregate_neighbors(neighbor_list, include_censored=include_censored)
        for neighbor_list in neighbors
    ]
    return pd.DataFrame(rows, index=neighbors.index, columns=list(columns))


FORBIDDEN_XGB_KEYS = (
    "objective",
    "eval_metric",
    "random_state",
    "seed",
    "early_stopping_rounds",
)


class XGBoostModel(BaseModel[pd.DataFrame]):
    def __init__(self, params: dict) -> None:
        super().__init__(params)

        allowed = {
            "include_nbrs",
            "include_censored",
            "nbr_features",
            "include_days",
            "neighbors_source",
            "seed",
            "early_stopping_rounds",
            "xgb",
        }
        unexpected = set(self.params) - allowed
        if unexpected:
            raise ValueError(f"Unexpected XGBoostModel params: {sorted(unexpected)}")

        self.include_nbrs = self.params.get("include_nbrs")
        if not isinstance(self.include_nbrs, bool):
            raise ValueError("include_nbrs is required and must be a bool")
        self.include_censored = self.params.get("include_censored", False)
        if not isinstance(self.include_censored, bool):
            raise ValueError("include_censored must be a bool")
        if self.include_censored and not self.include_nbrs:
            raise ValueError("include_censored requires include_nbrs")
        self.nbr_features = self.params.get("nbr_features", "full")
        if self.nbr_features not in ("full", "count"):
            raise ValueError(
                f"nbr_features must be 'full' or 'count'; got {self.nbr_features!r}"
            )
        if self.nbr_features == "count":
            if not self.include_nbrs:
                raise ValueError("nbr_features='count' requires include_nbrs")
            if self.include_censored:
                raise ValueError(
                    "nbr_features='count' is incompatible with include_censored"
                )
        self.include_days = self.params.get("include_days", False)
        if not isinstance(self.include_days, bool):
            raise ValueError("include_days must be a bool")
        self.neighbors_source = self.params.get("neighbors_source", "similar")
        if self.neighbors_source not in ("similar", "random"):
            raise ValueError(
                "neighbors_source must be 'similar' or 'random'; got "
                f"{self.neighbors_source!r}"
            )
        if self.neighbors_source == "random" and not self.include_nbrs:
            raise ValueError("neighbors_source='random' requires include_nbrs")

        self.seed = self.params.get("seed")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed is required and must be an int")
        self.early_stopping_rounds = self.params.get("early_stopping_rounds")
        if (
            not isinstance(self.early_stopping_rounds, int)
            or isinstance(self.early_stopping_rounds, bool)
            or self.early_stopping_rounds < 1
        ):
            raise ValueError(
                "early_stopping_rounds is required and must be an int >= 1"
            )

        self.xgb_params = dict(self.params.get("xgb", {}))
        forbidden = sorted(set(self.xgb_params) & set(FORBIDDEN_XGB_KEYS))
        if forbidden:
            raise ValueError(
                f"xgb params must not set {forbidden}; these are fixed by the "
                "ablation design (objective/eval_metric) or lifted to top level"
            )

        self.requires_neighbors = self.include_nbrs

        self.regressor: XGBRegressor | None = None
        self.feature_columns: list[str] | None = None
        self.learner_medians: dict[str, float] = {}
        self.global_median: float | None = None
        self.learner_starts: dict[str, float] = {}
        self.global_start: float | None = None
        self.train_rows: int | None = None
        self.best_iteration: int | None = None

    @staticmethod
    def _epoch_days(values: pd.Series) -> pd.Series:
        """First-review timestamps as a float number of days since the epoch.

        Divided by a Timedelta rather than cast to int64, so the result is days
        whatever datetime resolution pandas hands back.
        """
        stamps = pd.to_datetime(values, utc=True)
        return (stamps - pd.Timestamp(0, tz="UTC")) / pd.Timedelta(days=1)

    def _build_matrix(
        self, X: pd.DataFrame, inputs: dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        enriched = add_intralexical_features(
            X[["word_instance_id", "lexeme_id"]], **inputs
        )
        matrix = enriched[list(INTRALEXICAL_COLUMNS)].copy()
        matrix.index = X.index
        matrix["learner_median_plb"] = (
            X["user_id"].map(self.learner_medians).fillna(self.global_median)
        )
        if self.include_days:
            # Position on the learner's timeline, known at the target's first review.
            # The origin comes from the training split only, so an unseen learner falls
            # back to the cohort's earliest start rather than peeking at their own.
            starts = X["user_id"].map(self.learner_starts).fillna(self.global_start)
            matrix["days_elapsed"] = (
                self._epoch_days(X["first_reviewed_at"]).to_numpy() - starts.to_numpy()
            )
        if self.include_nbrs:
            nbrs = neighbor_feature_frame(
                X["neighbors"], include_censored=self.include_censored
            )
            if self.nbr_features == "count":
                nbrs = nbrs[["n_neighbors"]]
            matrix = pd.concat([matrix, nbrs], axis=1)

        if self.feature_columns is None:
            self.feature_columns = list(matrix.columns)
        elif list(matrix.columns) != self.feature_columns:
            raise ValueError(
                f"Feature columns drifted from the fitted model: expected "
                f"{self.feature_columns}, got {list(matrix.columns)}"
            )
        return matrix

    def fit(
        self,
        X: pd.DataFrame,
        y: NDArray[np.float64],
        eval_X: pd.DataFrame | None = None,
        eval_y: NDArray[np.float64] | None = None,
    ) -> "XGBoostModel":
        if X is None or y is None:
            raise ValueError("X and y cannot be None")
        if len(X) != len(y):
            raise ValueError(f"X has {len(X)} rows but y has {len(y)} values")
        if (eval_X is None) != (eval_y is None):
            raise ValueError("eval_X and eval_y must be provided together")
        y = np.asarray(y, dtype=np.float64)
        if not np.isfinite(y).all():
            raise ValueError("y contains non-finite values")

        required = {"user_id", "word_instance_id", "lexeme_id"}
        if self.include_nbrs:
            required.add("neighbors")
        if self.include_days:
            required.add("first_reviewed_at")
        missing = sorted(required - set(X.columns))
        if missing:
            raise ValueError(f"X is missing required columns: {missing}")

        # Learner scale comes from the training targets only, so the model can be
        # compared with the within_learner_median baseline without leakage.
        targets = pd.Series(y, index=X.index)
        self.global_median = float(np.median(y))
        self.learner_medians = {
            user_id: float(np.median(group.to_numpy()))
            for user_id, group in targets.groupby(X["user_id"], sort=True)
        }
        if self.include_days:
            days = self._epoch_days(X["first_reviewed_at"])
            self.global_start = float(days.min())
            self.learner_starts = {
                user_id: float(group.min())
                for user_id, group in days.groupby(X["user_id"], sort=True)
            }
        self.train_rows = len(X)

        inputs = load_intralexical_inputs()
        train_matrix = self._build_matrix(X, inputs)

        regressor = XGBRegressor(
            objective="reg:absoluteerror",
            eval_metric="mae",
            random_state=self.seed,
            early_stopping_rounds=self.early_stopping_rounds
            if eval_X is not None
            else None,
            **self.xgb_params,
        )
        if eval_X is not None:
            eval_matrix = self._build_matrix(eval_X, inputs)
            regressor.fit(
                train_matrix,
                y,
                eval_set=[(eval_matrix, np.asarray(eval_y, dtype=np.float64))],
                verbose=False,
            )
            self.best_iteration = int(regressor.best_iteration)
        else:
            regressor.fit(train_matrix, y)
            self.best_iteration = None
        self.regressor = regressor

        stopped = (
            f"best_iteration={self.best_iteration}"
            if self.best_iteration is not None
            else "no early stopping"
        )
        print(
            f"[XGBoostModel] fitted on {len(X):,} rows x "
            f"{len(self.feature_columns)} features ({stopped})"
        )
        return self

    def predict(self, X: pd.DataFrame) -> NDArray[np.float64]:
        if self.regressor is None:
            raise ValueError(
                "XGBoostModel must be fitted (or loaded) before predicting"
            )
        matrix = self._build_matrix(X, load_intralexical_inputs())
        return self.regressor.predict(matrix).astype(np.float64)

    def save(self, path: Path) -> None:
        path = Path(path)
        self.regressor.save_model(path)  # native XGBoost JSON
        write_json(
            path.parent / "state.json",
            {
                "params": self.params,
                "feature_columns": self.feature_columns,
                "learner_medians": self.learner_medians,
                "global_median": self.global_median,
                "learner_starts": self.learner_starts,
                "global_start": self.global_start,
                "train_rows": self.train_rows,
                "best_iteration": self.best_iteration,
            },
        )

    @classmethod
    def load(cls, path: Path) -> "XGBoostModel":
        path = Path(path)
        state = json.loads((path.parent / "state.json").read_text())
        instance = cls(state["params"])
        instance.feature_columns = state["feature_columns"]
        instance.learner_medians = state["learner_medians"]
        instance.global_median = state["global_median"]
        instance.learner_starts = state.get("learner_starts", {})
        instance.global_start = state.get("global_start")
        instance.train_rows = state["train_rows"]
        instance.best_iteration = state["best_iteration"]
        regressor = XGBRegressor()
        regressor.load_model(path)
        instance.regressor = regressor
        return instance
