"""The contract every Model B ablation implements.

Kept deliberately minimal: fit/predict plus disk round-tripping, so each ablation can
train in its own DVC stage and later be reloaded by the shared evaluation runner.
Metrics live in `core.evaluate`, not here — the paired bootstrap must apply the same
resample indices across every model's predictions, which a per-model method cannot do.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar

import numpy as np
from numpy.typing import NDArray

XType = TypeVar("XType")


class BaseModel(ABC, Generic[XType]):
    # Data-loader hint: neighbor-aware models set this (or an instance attribute)
    # to True so the trainer/runner merge the similarity neighbors column.
    requires_neighbors: bool = False
    # Which neighbor file to merge: "similar" (the selected neighbors) or "random"
    # (each slot refilled with a time-matched random word, the drift control).
    neighbors_source: str = "similar"
    # Name of the primary artifact each model persists; the registry builds model
    # paths from it (JSON state for dummies/GBDT, a torch checkpoint for the ST).
    artifact_filename: str = "model.json"

    def __init__(self, params: dict) -> None:
        self.params = dict(params)

    @abstractmethod
    def fit(
        self,
        X: XType,
        y: NDArray[np.float64],
        eval_X: XType | None = None,
        eval_y: NDArray[np.float64] | None = None,
    ) -> "BaseModel[XType]":
        """Fit on the log-burden target and return self.

        The optional eval pair is early-stopping material; models may ignore it.
        """

    @abstractmethod
    def predict(self, X: XType) -> NDArray[np.float64]:
        """Return log-burden predictions aligned with X's rows."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Serialize the fitted model to disk."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseModel[XType]":
        """Reconstruct a fitted model from disk."""
