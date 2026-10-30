from pathlib import Path

import yaml
from pydantic import BaseModel, field_validator

from plb.utils import PROJECT_ROOT, get_dataset_as_path


class TubelexConfig(BaseModel):
    file_name: str
    evaluations_dir: Path = PROJECT_ROOT / "evaluations" / "tubelex"

    tubelex_data_dir: Path = get_dataset_as_path("raw/tubelex")
    multils_data_dir: Path = get_dataset_as_path("raw/multils_japanese")

    @field_validator("file_name")
    @classmethod
    def validate_file_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Tubelex file_name must be specified")
        return value

    def get_frequencies_file_path(self) -> Path:
        return self.tubelex_data_dir / self.file_name

    def get_multils_split_file_path(self, split: str) -> Path:
        """
        Returns the path to the specified split of the multils annotations.

        Args:
            split (str): The split to retrieve (e.g., "trial", "test").
        """
        return self.multils_data_dir / "annotations" / f"agg_original_{split}.tsv"

def load_tubelex_config(
    params_path: Path = PROJECT_ROOT / "params.yaml",
) -> TubelexConfig:
    raw = yaml.safe_load(params_path.read_text()) or {}
    return TubelexConfig(**raw.get("tubelex", {}))
