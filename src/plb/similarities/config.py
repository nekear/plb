from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from plb.utils import PROJECT_ROOT, get_dataset_as_path

from .calibration import CHANNELS


class ChannelPercentilesConfig(BaseModel):
    """The selection threshold for each channel, stated explicitly.

    Every value is a percentile of that channel's null distribution, so they mean the
    same thing across channels: 0.999 is "more similar than 99.9% of random pairs".
    All five are required rather than defaulted from a shared value, so the params
    file always shows what each channel is actually set to.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    semantic: float = Field(ge=0.0, lt=1.0)
    phonological: float = Field(ge=0.0, lt=1.0)
    morphological: float = Field(ge=0.0, lt=1.0)
    ortho_stroke: float = Field(ge=0.0, lt=1.0)
    ortho_kana: float = Field(ge=0.0, lt=1.0)

    def by_channel(self) -> dict[str, float]:
        return {channel: getattr(self, channel) for channel in CHANNELS}


class CalibrationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_pairs: int = Field(default=50_000, ge=100)
    seed: int = 42
    output_path: Path = get_dataset_as_path("processed/similarity_null_calibration.json")


class CacheConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # Raw per-pair channel scores, written once by the scoring stage and read by every
    # selection run, so retuning thresholds never re-scores.
    path: Path = get_dataset_as_path("processed/similarity_scores")


class SimilaritiesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    percentiles: ChannelPercentilesConfig
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    top_k: int = Field(default=30, ge=1)
    workers: int = Field(default=8, ge=1)
    target_chunk_size: int = Field(default=32, ge=1)

    lexemes_path: Path = get_dataset_as_path("interim/lexemes.jsonl")
    decks_path: Path = get_dataset_as_path("interim/decks.jsonl")
    word_instances_path: Path = get_dataset_as_path(
        "processed/word_instances_w_stability_targets.jsonl"
    )
    reviews_path: Path = get_dataset_as_path("processed/clean_reviews_w_fsrs.jsonl")
    embeddings_path: Path
    output_path: Path = get_dataset_as_path("processed/similarities.jsonl")
    # Same schema, but every neighbour swapped for a time-matched random word of
    # the same learner: the control that separates similarity from temporal drift.
    random_output_path: Path = get_dataset_as_path("processed/randomities.jsonl")


def load_similarities_config(
        params_path: Path = PROJECT_ROOT / "params.yaml",
) -> SimilaritiesConfig:
    raw = yaml.safe_load(params_path.read_text()) or {}
    return SimilaritiesConfig(**raw.get("similarities", {}))
