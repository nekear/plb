from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

from plb.utils import PROJECT_ROOT, get_dataset_as_path


class DirectEmbeddingsConfig(BaseModel):
    model_id: str
    model_revision: str
    batch_size: int = Field(gt=0)
    max_length: int = Field(gt=0)
    word_instances_path: Path
    output_path: Path

    @field_validator("model_id", "model_revision")
    @classmethod
    def validate_model_reference(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Embedding model identifier and revision cannot be empty")
        return value


def load_direct_embeddings_config(
        params_path: Path = PROJECT_ROOT / "params.yaml",
) -> DirectEmbeddingsConfig:
    raw = yaml.safe_load(params_path.read_text()) or {}
    return DirectEmbeddingsConfig(**raw.get("embeddings", {}).get("direct", {}))
