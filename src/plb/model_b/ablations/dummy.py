"""Median-based ablations: the cheapest models the learned ones must beat.

One class covers three ablations, selected by ``params["type"]``:

- ``global``: one median over every training row.
- ``within``: a per-learner median — the personalization floor and the baseline the
  relative-error-reduction metric is read against.
- ``recent``: ability drift — for each target, the median over the learner's ``k``
  most recently *resolved* training items: those whose burden label was achieved
  (``plb_14_achieved_at``, the stability-14 crossing) strictly before the target's
  ``first_reviewed_at``. Windowing on when items merely *started* would read labels
  from the future — the crossing lands a median ~23 days after the start. Splits are
  lexeme-based rather than chronological, so train history interleaves in time with
  the targets and each target gets its own window.

The fallback chain (empty recent window -> learner median -> global median) is always
fitted in full, so every type predicts something for every row.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from plb.model_b.core.base import BaseModel
from plb.utils import write_json

TYPES = ("global", "within", "recent")


def _epoch_ns(values: Any) -> NDArray[np.int64]:
    """Convert timestamps (tz-aware, naive, or ISO strings) to UTC epoch nanoseconds.

    Pandas 2.x may parse to a coarser resolution (e.g. microseconds), so the unit is
    forced to nanoseconds before taking the integer view.
    """
    index = pd.DatetimeIndex(pd.to_datetime(values, utc=True, format="ISO8601"))
    return index.as_unit("ns").astype("int64").to_numpy()


class DummyModel(BaseModel[pd.DataFrame]):
    def __init__(self, params: dict) -> None:
        """
        Args:
            - ``params`` accepts:
                1. `type` - `global`, `within`, `recent`
                2. `k` - "k" as in "k-recent" when the `recent` type is selected
        """

        super().__init__(params)

        # Checking incoming params for unexpected values
        unexpected = set(self.params) - {"type", "k"}
        if unexpected:
            raise ValueError(f"Unexpected DummyModel params: {sorted(unexpected)}")

        # Validating the "type" parameter
        self.type = self.params.get("type")
        if self.type not in TYPES:
            raise ValueError(
                f"DummyModel type must be one of {TYPES}; got {self.type!r}"
            )

        # Validating the "k" parameter
        self.k = self.params.get("k")
        if self.type == "recent":
            if not isinstance(self.k, int) or isinstance(self.k, bool) or self.k < 1:
                raise ValueError(
                    f"type 'recent' requires an integer k >= 1; got {self.k!r}"
                )
        elif self.k is not None:
            raise ValueError(f"k is only valid for type 'recent', not {self.type!r}")

        self.train_rows: int | None = None
        self.global_median: float | None = None
        self.learner_medians: dict[str, float] = {}
        # Per learner: (plb_14_achieved_at as epoch ns, log burden), sorted ascending.
        self.history: dict[str, tuple[NDArray[np.int64], NDArray[np.float64]]] = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: NDArray[np.float64],
        eval_X: pd.DataFrame | None = None,
        eval_y: NDArray[np.float64] | None = None,
    ) -> "DummyModel":
        # Medians have nothing to early-stop; the eval pair is accepted and ignored.
        if X is None or y is None:
            raise ValueError("X and y cannot be None")
        if len(X) != len(y):
            raise ValueError(f"X has {len(X)} rows but y has {len(y)} values")
        y = np.asarray(y, dtype=np.float64)
        if not np.isfinite(y).all():
            raise ValueError("y contains non-finite values")

        required = {"user_id"}
        if self.type == "recent":
            required |= {"plb_14_achieved_at", "word_instance_id"}
        missing = sorted(required - set(X.columns))
        if missing:
            raise ValueError(f"X is missing required columns: {missing}")

        self.train_rows = len(X)

        # Computing for the `global_median` ablation
        self.global_median = float(np.median(y))

        # Computing for the `within_learner_median` ablation
        frame = X[sorted(required)].copy()
        frame["y"] = y
        self.learner_medians = {
            user_id: float(np.median(group["y"].to_numpy()))
            for user_id, group in frame.groupby("user_id", sort=True)
        }

        # Computing only if the ablation is `learner_recent_k_median`
        if self.type == "recent":
            # History is keyed on when each item's label became knowable (the
            # stability crossing), never on when the item started. A stable sort key
            # on (timestamp, word_instance_id) pins the order of same-timestamp rows,
            # so serialization round-trips deterministically.
            frame["_ts"] = _epoch_ns(frame["plb_14_achieved_at"])
            frame = frame.sort_values(["_ts", "word_instance_id"], kind="stable")
            self.history = {
                user_id: (
                    group["_ts"].to_numpy(dtype=np.int64),
                    group["y"].to_numpy(dtype=np.float64),
                )
                for user_id, group in frame.groupby("user_id", sort=True)
            }

        return self

    def predict(self, X: pd.DataFrame) -> NDArray[np.float64]:
        if self.global_median is None:
            raise ValueError("DummyModel must be fitted (or loaded) before predicting")

        if self.type == "global":
            return np.full(len(X), self.global_median, dtype=np.float64)

        learner_medians = X["user_id"].map(self.learner_medians)

        # For unseen learners (just in case) replacing the within-learner median with the global median
        within = learner_medians.fillna(self.global_median).to_numpy(dtype=np.float64)

        if self.type == "within":
            return within

        # type == "recent": per-target temporal window with strict precedence
        predictions = within.copy()
        target_ts = _epoch_ns(X["first_reviewed_at"])

        # Iterating over each test row: (row_index, user_id, target_start_timestamp)
        for row, (user_id, ts) in enumerate(zip(X["user_id"].to_numpy(), target_ts)):
            hist = self.history.get(user_id) # hist = (sorted_resolution_timestamps, sorted_burdens)
            if hist is None:
                continue  # unseen learner: keep the global-median fallback

            hist_ts, hist_y = hist

            # Finding how many training items finished BEFORE the target started.
            # side="left" excludes items whose label was achieved at exactly the target's start timestamp to avoid including the target's PLB value itself.
            end = int(np.searchsorted(hist_ts, ts, side="left"))

            if end == 0:
                continue  # nothing precedes the target: keep the learner median

            # Getting the k most recent resolved items
            window = hist_y[max(0, end - self.k) : end]

            predictions[row] = float(np.median(window))
        return predictions

    def save(self, path: Path) -> None:
        history = None
        if self.type == "recent":
            history = {
                user_id: {
                    "plb_14_achieved_at": [
                        pd.Timestamp(ts, unit="ns", tz="UTC").isoformat()
                        for ts in hist_ts
                    ],
                    "log_plb_14": hist_y.tolist(),
                }
                for user_id, (hist_ts, hist_y) in self.history.items()
            }
        state = {
            "params": self.params,
            "train_rows": self.train_rows,
            "global_median": self.global_median,
            "learner_medians": self.learner_medians,
            "history": history,
        }
        write_json(Path(path), state)

    @classmethod
    def load(cls, path: Path) -> "DummyModel":
        state = json.loads(Path(path).read_text())
        instance = cls(state["params"])
        instance.train_rows = state["train_rows"]
        instance.global_median = state["global_median"]
        instance.learner_medians = state["learner_medians"]
        if state["history"] is not None:
            instance.history = {
                user_id: (
                    _epoch_ns(entry["plb_14_achieved_at"]),
                    np.asarray(entry["log_plb_14"], dtype=np.float64),
                )
                for user_id, entry in state["history"].items()
            }
        return instance
