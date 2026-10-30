"""SetTransformer ablations: learned per-neighbor weighting via attention.

One ``SetTransformerModel`` class covers three ablations via params:

- ``include_nbrs: false`` — architecture control: the target's word features pass
  through the same model family with an empty neighbor set (the head sees a zero
  pooled vector and log1p(0)), so any gap to the neighbor variants is attributable
  to the neighbors, not to the switch of model family;
- ``include_nbrs: true`` — each UNCENSORED neighbor becomes a feature vector and the
  set is pooled with attention (PMA), letting the model weight individual neighbors;
- ``include_censored: true`` — censored neighbors join the visible set, each neighbor
  gains an ``is_censored`` flag, and censored burdens are zero-filled (the flag
  carries the missingness).

The set size is passed explicitly as log1p(|N|) into the regression head, and the
loss is L1 so training estimates the same quantity as the reported MAE metrics.

Torch tensors cannot carry NaN, so nullable fields follow a fill-plus-flag pattern:
``s`` (and ``d``, which is null exactly when ``s`` is — flagged by ``s_missing``
alone; revisit if a future data version decouples them), ``phon_pct``
(``phon_missing``), and the ortho percentiles, whose coverage features already carry
the missingness.
"""

from __future__ import annotations

import math
import random
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch.utils.data import DataLoader
from tqdm import tqdm

from plb.model_b.ablations.set_transformer_modules import PMA, SAB
from plb.model_b.core.base import BaseModel
from plb.model_b.core.features import (
    add_intralexical_features,
    load_intralexical_inputs,
    load_word_instance_index,
)
from plb.utils import autocast_context, get_device

TARGET_FEATURES = ("kvl_score", "log_freq", "is_oov", "learner_median_plb")
TARGET_STANDARDIZED = (0, 1, 3)  # is_oov is binary

NEIGHBOR_FEATURES = (
    "kvl_score",
    "log_freq",
    "is_oov",
    "log_plb_14",  # zero-filled when censored; is_censored carries it
    "log1p_reviews",
    "log1p_s",
    "s_missing",
    "d",  # null exactly when s is null; s_missing covers both
    "sem_pct",
    "mor_pct",
    "phon_pct",
    "phon_missing",
    "ortho_stroke_pct",
    "ortho_kana_pct",
    "ortho_stroke_coverage",
    "ortho_kana_coverage",
)
# include_censored appends "is_censored" as the final neighbor feature.
NEIGHBOR_STANDARDIZED = (0, 1, 3, 4, 5, 7)  # the unbounded ones


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _is_null(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _fill(value: object) -> float:
    return 0.0 if _is_null(value) else float(value)


@dataclass(frozen=True)
class StExample:
    target_features: NDArray[np.float32]  # [T]
    neighbor_features: NDArray[np.float32]  # [N, F], N may be 0
    target: float


@dataclass(frozen=True)
class FeatureNormalizer:
    """Standardizes only the unbounded feature indices; fitted on train only."""

    target_mean: NDArray[np.float32]
    target_scale: NDArray[np.float32]
    neighbor_mean: NDArray[np.float32]
    neighbor_scale: NDArray[np.float32]

    @classmethod
    def fit(
        cls, examples: list[StExample], neighbor_feature_count: int
    ) -> "FeatureNormalizer":
        target_mean = np.zeros(len(TARGET_FEATURES), dtype=np.float32)
        target_scale = np.ones(len(TARGET_FEATURES), dtype=np.float32)
        targets = np.stack([example.target_features for example in examples])
        for index in TARGET_STANDARDIZED:
            target_mean[index] = targets[:, index].mean()
            target_scale[index] = targets[:, index].std() or 1.0

        neighbor_mean = np.zeros(neighbor_feature_count, dtype=np.float32)
        neighbor_scale = np.ones(neighbor_feature_count, dtype=np.float32)
        nonempty = [
            example.neighbor_features
            for example in examples
            if len(example.neighbor_features)
        ]
        if nonempty:
            neighbors = np.concatenate(nonempty, axis=0)
            for index in NEIGHBOR_STANDARDIZED:
                neighbor_mean[index] = neighbors[:, index].mean()
                neighbor_scale[index] = neighbors[:, index].std() or 1.0

        return cls(target_mean, target_scale, neighbor_mean, neighbor_scale)

    def transform(self, example: StExample) -> StExample:
        return StExample(
            target_features=(example.target_features - self.target_mean)
            / self.target_scale,
            neighbor_features=(example.neighbor_features - self.neighbor_mean)
            / self.neighbor_scale,
            target=example.target,
        )

    def to_dict(self) -> dict:
        # Plain lists keep the checkpoint loadable with weights_only=True.
        return {
            "target_mean": self.target_mean.tolist(),
            "target_scale": self.target_scale.tolist(),
            "neighbor_mean": self.neighbor_mean.tolist(),
            "neighbor_scale": self.neighbor_scale.tolist(),
        }

    @classmethod
    def from_dict(cls, state: dict) -> "FeatureNormalizer":
        return cls(
            **{key: np.asarray(value, dtype=np.float32) for key, value in state.items()}
        )


def _collate(examples: list[StExample]) -> dict[str, torch.Tensor]:
    batch_size = len(examples)
    feature_count = examples[0].neighbor_features.shape[1]
    counts = torch.tensor(
        [len(example.neighbor_features) for example in examples], dtype=torch.int64
    )
    max_count = int(counts.max()) if batch_size else 0

    neighbors = torch.zeros(batch_size, max_count, feature_count)
    mask = torch.ones(batch_size, max_count, dtype=torch.bool)  # True = padded
    for row, example in enumerate(examples):
        count = len(example.neighbor_features)
        if count:
            neighbors[row, :count] = torch.from_numpy(example.neighbor_features)
            mask[row, :count] = False

    return {
        "targets": torch.from_numpy(
            np.stack([example.target_features for example in examples])
        ),
        "neighbors": neighbors,
        "mask": mask,
        "counts": counts,
        "y": torch.tensor(
            [example.target for example in examples], dtype=torch.float32
        ),
    }


class _SetTransformerNet(nn.Module):
    def __init__(
        self, target_features: int, neighbor_features: int, dim: int, num_heads: int
    ) -> None:
        super().__init__()
        self.dim = dim
        self.embed = nn.Linear(neighbor_features, dim)
        self.sab1 = SAB(dim, dim, num_heads, ln=True)
        self.sab2 = SAB(dim, dim, num_heads, ln=True)
        self.pma = PMA(dim, num_heads, num_seeds=1, ln=True)
        self.head = nn.Sequential(
            nn.Linear(target_features + dim + 1, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
        )

    def forward(
        self,
        targets: torch.Tensor,
        neighbors: torch.Tensor,
        mask: torch.Tensor,
        counts: torch.Tensor,
    ) -> torch.Tensor:
        if neighbors.size(1) == 0:
            pooled = targets.new_zeros(targets.size(0), self.dim)
        else:
            hidden = self.embed(neighbors)
            hidden = self.sab2(self.sab1(hidden, mask), mask)
            pooled = self.pma(hidden, mask).squeeze(1)
            # Zero-neighbor rows carry garbage uniform attention (see the vendored
            # modules' mask fill); zero their pooled vector explicitly.
            pooled = pooled * (counts > 0).float().unsqueeze(1)
        count_feature = torch.log1p(counts.float()).unsqueeze(1)
        combined = torch.cat((targets, pooled, count_feature), dim=1)
        return self.head(combined).squeeze(1)


class SetTransformerModel(BaseModel[pd.DataFrame]):
    artifact_filename = "model.pt"

    def __init__(self, params: dict) -> None:
        super().__init__(params)

        allowed = {
            "include_nbrs",
            "include_censored",
            "seed",
            "epochs",
            "batch_size",
            "learning_rate",
            "weight_decay",
            "patience",
            "dim",
            "num_heads",
        }
        unexpected = set(self.params) - allowed
        if unexpected:
            raise ValueError(
                f"Unexpected SetTransformerModel params: {sorted(unexpected)}"
            )

        self.include_nbrs = self.params.get("include_nbrs")
        if not isinstance(self.include_nbrs, bool):
            raise ValueError("include_nbrs is required and must be a bool")
        self.include_censored = self.params.get("include_censored", False)
        if not isinstance(self.include_censored, bool):
            raise ValueError("include_censored must be a bool")
        if self.include_censored and not self.include_nbrs:
            raise ValueError("include_censored requires include_nbrs")

        def _require_int(key: str, minimum: int) -> int:
            value = self.params.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"{key} is required and must be an int >= {minimum}")
            return value

        self.seed = _require_int("seed", 0)
        self.epochs = _require_int("epochs", 1)
        self.batch_size = _require_int("batch_size", 1)
        self.patience = _require_int("patience", 1)
        self.dim = _require_int("dim", 1)
        self.num_heads = _require_int("num_heads", 1)
        if self.dim % self.num_heads != 0:
            raise ValueError(
                f"num_heads must divide dim; got dim={self.dim}, "
                f"num_heads={self.num_heads}"
            )

        self.learning_rate = float(self.params.get("learning_rate", 0))
        if self.learning_rate <= 0:
            raise ValueError("learning_rate is required and must be > 0")
        self.weight_decay = float(self.params.get("weight_decay", -1))
        if self.weight_decay < 0:
            raise ValueError("weight_decay is required and must be >= 0")

        self.requires_neighbors = self.include_nbrs
        self.neighbor_feature_names = list(NEIGHBOR_FEATURES) + (
            ["is_censored"] if self.include_censored else []
        )

        self.net: _SetTransformerNet | None = None
        self.normalizer: FeatureNormalizer | None = None
        self.learner_medians: dict[str, float] = {}
        self.global_median: float | None = None
        self.train_rows: int | None = None
        self.best_epoch: int | None = None
        self.history: list[dict[str, float]] = []

    # ---------------------- features

    def _neighbor_vector(self, neighbor: dict, word_rep: dict) -> list[float]:
        kvl_score, log_freq, is_oov = word_rep[neighbor["word_instance_id"]]
        s_missing = 1.0 if _is_null(neighbor["s"]) else 0.0
        phon_missing = 1.0 if _is_null(neighbor["phon_pct"]) else 0.0
        vector = [
            kvl_score,
            log_freq,
            is_oov,
            _fill(neighbor["log_plb_14"]),
            math.log1p(neighbor["reviews"]),
            math.log1p(neighbor["s"]) if not s_missing else 0.0,
            s_missing,
            _fill(neighbor["d"]),
            float(neighbor["sem_pct"]),
            float(neighbor["mor_pct"]),
            _fill(neighbor["phon_pct"]),
            phon_missing,
            _fill(neighbor["ortho_stroke_pct"]),
            _fill(neighbor["ortho_kana_pct"]),
            float(neighbor["ortho_stroke_coverage"]),
            float(neighbor["ortho_kana_coverage"]),
        ]
        if self.include_censored:
            vector.append(1.0 if neighbor["is_censored"] else 0.0)
        return vector

    def _build_examples(
        self,
        X: pd.DataFrame,
        y: NDArray[np.float64] | None,
        inputs: dict[str, pd.DataFrame],
        word_index: pd.DataFrame | None,
    ) -> list[StExample]:
        enriched = add_intralexical_features(
            X[["word_instance_id", "lexeme_id"]], **inputs
        )
        learner_median = (
            X["user_id"].map(self.learner_medians).fillna(self.global_median)
        )
        target_matrix = np.column_stack(
            [
                enriched["kvl_score"].to_numpy(dtype=np.float32),
                enriched["log_freq"].to_numpy(dtype=np.float32),
                enriched["is_oov"].to_numpy(dtype=np.float32),
                learner_median.to_numpy(dtype=np.float32),
            ]
        )

        word_rep: dict[str, tuple[float, float, float]] = {}
        neighbor_lists: list[list[dict]] = [[] for _ in range(len(X))]
        if self.include_nbrs:
            neighbor_lists = [
                [
                    neighbor
                    for neighbor in row
                    if self.include_censored or not neighbor["is_censored"]
                ]
                for row in X["neighbors"]
            ]
            unique_ids = sorted(
                {n["word_instance_id"] for row in neighbor_lists for n in row}
            )
            if unique_ids:
                rep_frame = add_intralexical_features(
                    pd.DataFrame({"word_instance_id": unique_ids}),
                    **inputs,
                    word_instances=word_index,
                )
                word_rep = {
                    wid: (kvl, freq, oov)
                    for wid, kvl, freq, oov in zip(
                        rep_frame["word_instance_id"],
                        rep_frame["kvl_score"],
                        rep_frame["log_freq"],
                        rep_frame["is_oov"],
                    )
                }

        feature_count = len(self.neighbor_feature_names)
        targets = y if y is not None else np.zeros(len(X))
        examples = []
        for row_index, neighbors in enumerate(neighbor_lists):
            if neighbors:
                neighbor_matrix = np.asarray(
                    [self._neighbor_vector(n, word_rep) for n in neighbors],
                    dtype=np.float32,
                )
            else:
                neighbor_matrix = np.empty((0, feature_count), dtype=np.float32)
            examples.append(
                StExample(
                    target_features=target_matrix[row_index],
                    neighbor_features=neighbor_matrix,
                    target=float(targets[row_index]),
                )
            )
        return examples

    def _make_loader(self, examples: list[StExample], *, shuffle: bool) -> DataLoader:
        generator = None
        if shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed)
        return DataLoader(
            examples,
            batch_size=self.batch_size,
            shuffle=shuffle,
            generator=generator,
            collate_fn=_collate,
        )

    # ---------------------- training

    def fit(
        self,
        X: pd.DataFrame,
        y: NDArray[np.float64],
        eval_X: pd.DataFrame | None = None,
        eval_y: NDArray[np.float64] | None = None,
    ) -> "SetTransformerModel":
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
        missing = sorted(required - set(X.columns))
        if missing:
            raise ValueError(f"X is missing required columns: {missing}")

        # Learner scale from train targets only (same recipe as XGBoostModel).
        targets = pd.Series(y, index=X.index)
        self.global_median = float(np.median(y))
        self.learner_medians = {
            user_id: float(np.median(group.to_numpy()))
            for user_id, group in targets.groupby(X["user_id"], sort=True)
        }
        self.train_rows = len(X)

        set_seed(self.seed)
        device = get_device()
        inputs = load_intralexical_inputs()
        word_index = load_word_instance_index() if self.include_nbrs else None

        train_examples = self._build_examples(X, y, inputs, word_index)
        self.normalizer = FeatureNormalizer.fit(
            train_examples, len(self.neighbor_feature_names)
        )
        train_examples = [self.normalizer.transform(e) for e in train_examples]

        eval_loader = None
        if eval_X is not None:
            eval_examples = [
                self.normalizer.transform(e)
                for e in self._build_examples(
                    eval_X, np.asarray(eval_y, dtype=np.float64), inputs, word_index
                )
            ]
            eval_loader = self._make_loader(eval_examples, shuffle=False)

        net = _SetTransformerNet(
            len(TARGET_FEATURES),
            len(self.neighbor_feature_names),
            self.dim,
            self.num_heads,
        ).to(device)
        criterion = nn.L1Loss()
        optimizer = torch.optim.AdamW(
            net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        train_loader = self._make_loader(train_examples, shuffle=True)

        best_eval_loss = float("inf")
        best_state: dict | None = None
        epochs_without_improvement = 0
        self.history = []
        progress = tqdm(range(1, self.epochs + 1), desc="Training SetTransformer")
        for epoch in progress:
            net.train()
            train_loss = 0.0
            for batch in train_loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                with autocast_context(device):
                    predictions = net(
                        batch["targets"],
                        batch["neighbors"],
                        batch["mask"],
                        batch["counts"],
                    )
                    loss = criterion(predictions, batch["y"])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            record: dict[str, float] = {"epoch": epoch, "train_l1": train_loss}
            if eval_loader is not None:
                eval_loss = self._evaluate_loss(net, eval_loader, criterion, device)
                record["eval_l1"] = eval_loss
                progress.set_postfix(train_l1=train_loss, eval_l1=eval_loss)
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    self.best_epoch = epoch
                    best_state = deepcopy(
                        {k: v.detach().cpu() for k, v in net.state_dict().items()}
                    )
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= self.patience:
                        self.history.append(record)
                        break
            else:
                progress.set_postfix(train_l1=train_loss)
            self.history.append(record)

        if eval_loader is not None:
            if best_state is None:
                raise RuntimeError("No best state recorded during training")
            net.load_state_dict(best_state)
        else:
            self.best_epoch = None
        net.eval()
        self.net = net.cpu()

        stopped = (
            f"best_epoch={self.best_epoch}"
            if self.best_epoch is not None
            else "no early stopping"
        )
        print(
            f"[SetTransformerModel] fitted on {len(X):,} rows "
            f"(target={len(TARGET_FEATURES)} dims, "
            f"neighbor={len(self.neighbor_feature_names)} dims, {stopped})"
        )
        return self

    @staticmethod
    def _evaluate_loss(
        net: _SetTransformerNet,
        loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
    ) -> float:
        net.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for batch in loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                predictions = net(
                    batch["targets"], batch["neighbors"], batch["mask"], batch["counts"]
                )
                total += criterion(predictions, batch["y"]).item() * len(batch["y"])
                count += len(batch["y"])
        return total / count

    # ---------------------- inference

    def predict(self, X: pd.DataFrame) -> NDArray[np.float64]:
        if self.net is None or self.normalizer is None:
            raise ValueError(
                "SetTransformerModel must be fitted (or loaded) before predicting"
            )
        inputs = load_intralexical_inputs()
        word_index = load_word_instance_index() if self.include_nbrs else None
        examples = [
            self.normalizer.transform(e)
            for e in self._build_examples(X, None, inputs, word_index)
        ]
        loader = self._make_loader(examples, shuffle=False)

        self.net.eval()
        outputs = []
        with torch.no_grad():
            for batch in loader:
                outputs.append(
                    self.net(
                        batch["targets"],
                        batch["neighbors"],
                        batch["mask"],
                        batch["counts"],
                    ).numpy()
                )
        return np.concatenate(outputs).astype(np.float64)

    # -------------------- persistence

    def save(self, path: Path) -> None:
        torch.save(
            {
                "params": self.params,
                "model_state_dict": self.net.state_dict(),
                "normalizer": self.normalizer.to_dict(),
                "target_features": list(TARGET_FEATURES),
                "neighbor_features": self.neighbor_feature_names,
                "learner_medians": self.learner_medians,
                "global_median": self.global_median,
                "train_rows": self.train_rows,
                "best_epoch": self.best_epoch,
                "history": self.history,
            },
            Path(path),
        )

    @classmethod
    def load(cls, path: Path) -> "SetTransformerModel":
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
        instance = cls(checkpoint["params"])
        instance.normalizer = FeatureNormalizer.from_dict(checkpoint["normalizer"])
        instance.learner_medians = checkpoint["learner_medians"]
        instance.global_median = checkpoint["global_median"]
        instance.train_rows = checkpoint["train_rows"]
        instance.best_epoch = checkpoint["best_epoch"]
        instance.history = checkpoint["history"]
        net = _SetTransformerNet(
            len(TARGET_FEATURES),
            len(instance.neighbor_feature_names),
            instance.dim,
            instance.num_heads,
        )
        net.load_state_dict(checkpoint["model_state_dict"])
        net.eval()
        instance.net = net
        return instance
