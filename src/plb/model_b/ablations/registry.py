"""Ablation name -> model class.

Each class owns its serialization format, so the evaluation runner can reload any
persisted ablation with ``ABLATIONS[name].load(path)`` without knowing whether the
artifact is a JSON state file, an XGBoost booster, or a torch checkpoint. Later
iterations add the GBDT and SetTransformer ablations here.
"""

from pathlib import Path

from plb.model_b.ablations.dummy import DummyModel
from plb.model_b.ablations.gbdt import XGBoostModel
from plb.model_b.ablations.st import SetTransformerModel
from plb.model_b.core.base import BaseModel
from plb.utils import PROJECT_ROOT

ABLATIONS: dict[str, type[BaseModel]] = {
    "global_median": DummyModel,
    "within_learner_median": DummyModel,
    "learner_recent_k_median": DummyModel,
    "gbdt_no_nbrs": XGBoostModel,
    "gbdt_w_nbrs": XGBoostModel,
    "gbdt_w_nbrs_censored": XGBoostModel,
    "gbdt_days": XGBoostModel,
    "gbdt_w_nbrs_random": XGBoostModel,
    "gbdt_w_nbrs_count_only": XGBoostModel,
    "st_no_nbrs": SetTransformerModel,
    "st_w_nbrs": SetTransformerModel,
    "st_w_nbrs_censored": SetTransformerModel,
}

MODELS_DIR = PROJECT_ROOT / "models" / "model_b" / "ablations"
EVALUATION_DIR = PROJECT_ROOT / "evaluations" / "model_b" / "ablations"


def model_path(name: str) -> Path:
    return MODELS_DIR / name / ABLATIONS[name].artifact_filename
