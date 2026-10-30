from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from plb.utils import PROJECT_ROOT, get_dataset_as_path


class MultiLSAugmentationConfig(BaseModel):
    model: str = "gpt-5.6-terra"
    prompt_version: str = "v2"
    reasoning_effort: Literal[
        "minimal",
        "low",
        "medium",
        "high",
        "extra_high",
        "max",
        "ultra",
    ] = "low"


class MultiLSConfig(BaseModel):
    condition: Literal[
        "full_l1",
        "full_l2",
        "full_l2_no_context",
        "full_l2_deploy",
        "ja_word_only",
        "l1_name_only",
    ] = "full_l1"
    cues_path: Path = get_dataset_as_path("interim/model_a/multils_l1_cues.tsv")
    bootstrap_samples: int = Field(default=2_000, ge=0)
    bootstrap_seed: int = 42
    augmentation: MultiLSAugmentationConfig = Field(default_factory=MultiLSAugmentationConfig)

class PrecomputeConfig(BaseModel):
    word_instances_path: Path = get_dataset_as_path("interim/word_instances.jsonl")
    lexemes_path: Path = get_dataset_as_path("interim/lexemes.jsonl")
    decks_path: Path = get_dataset_as_path("interim/decks.jsonl")
    users_path: Path = get_dataset_as_path("interim/users.jsonl")
    output_path: Path = get_dataset_as_path("processed/word_instances_kvl_predictions.jsonl")
    batch_size: int = Field(default=32, gt=0)
    # Caps padded tokens per batch. mmBERT's MLM head allocates
    # rows x padded_length x 256k vocab logits, so glosses long enough to be a
    # whole dictionary entry exhaust GPU memory at a fixed batch size.
    max_batch_tokens: int = Field(default=4096, gt=0)


class ModelAConfig(BaseModel):
    model_id: str = "jhu-clsp/mmBERT-base"
    model_revision: str
    languages: list[str] = ["de", "es", "zh"]
    epochs: int = 3
    batch_size: int = 8
    lr: float = 2e-5
    seed: int = 42
    run_name: str = Field(default="baseline", pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    multils: MultiLSConfig = Field(default_factory=MultiLSConfig)
    precompute: PrecomputeConfig = Field(default_factory=PrecomputeConfig)

    data_dir: Path = get_dataset_as_path("raw/kvl_bea2026")
    multils_data_dir: Path = get_dataset_as_path("raw/multils_japanese")
    models_dir: Path = PROJECT_ROOT / "models" / "model_a"
    evaluations_dir: Path = PROJECT_ROOT / "evaluations" / "model_a"

    @property
    def run_dir_models(self) -> Path:
        return self.models_dir / self.run_name

    @property
    def run_dir_evaluations(self) -> Path:
        return self.evaluations_dir / self.run_name

    @property
    def multils_evaluations_dir(self) -> Path:
        return self.run_dir_evaluations / "multils_japanese"

    @property
    def multils_fine_tuned_dir(self) -> Path:
        return self.multils_evaluations_dir / "fine_tuned"

    @property
    def multils_base_dir(self) -> Path:
        return self.multils_evaluations_dir / "base"

    @property
    def multils_comparison_dir(self) -> Path:
        return self.multils_evaluations_dir / "comparison"


def load_model_a_config(
    params_path: Path = PROJECT_ROOT / "params.yaml",
) -> ModelAConfig:
    raw = yaml.safe_load(params_path.read_text()) or {}
    return ModelAConfig(**raw.get("model_a", {}))
