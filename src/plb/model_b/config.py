from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from plb.utils import PROJECT_ROOT


class SplitsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # Lexemes studied by every test-eligible learner, held out for all of them so
    # cross-learner comparison runs on identical items.
    shared_test_lexemes: int = Field(default=250, ge=1)
    # Total test size per learner, shared core included; the remainder is filled from
    # that learner's own vocabulary.
    test_fraction: float = Field(default=0.15, gt=0.0, lt=1.0)
    eval_fraction: float = Field(default=0.15, gt=0.0, lt=1.0)
    seed: int = 42

    @model_validator(mode="after")
    def validate_fractions(self) -> "SplitsConfig":
        if self.test_fraction + self.eval_fraction >= 1.0:
            raise ValueError(
                "test_fraction + eval_fraction must leave room for training data; "
                f"got {self.test_fraction} + {self.eval_fraction}."
            )
        return self


class AblationsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    bootstrap_samples: int = Field(default=1000, ge=1)
    bootstrap_seed: int = 42
    # Relative error reduction (and delta-RER) is read against this model's log-MAE.
    baseline: str = "within_learner_median"
    delta_pairs: tuple[tuple[str, str], ...] = ()
    # Raw per-ablation param dicts; each model class validates its own shape.
    models: dict[str, dict] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_names(self) -> "AblationsConfig":
        if not self.models:
            return self  # a default-constructed config carries no ablations yet
        if self.baseline not in self.models:
            raise ValueError(
                f"baseline {self.baseline!r} is not a configured model: "
                f"{sorted(self.models)}"
            )
        for first, second in self.delta_pairs:
            unknown = {first, second} - set(self.models)
            if unknown:
                raise ValueError(
                    f"delta_pairs references unconfigured models: {sorted(unknown)}"
                )
            if first == second:
                raise ValueError(
                    f"delta_pairs must compare two different models; got {first!r} twice"
                )
        return self


class ModelBConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    splits: SplitsConfig = Field(default_factory=SplitsConfig)
    ablations: AblationsConfig = Field(default_factory=AblationsConfig)


def load_model_b_config(
    params_path: Path = PROJECT_ROOT / "params.yaml",
) -> ModelBConfig:
    raw = yaml.safe_load(params_path.read_text()) or {}
    return ModelBConfig(**raw.get("model_b", {}))
