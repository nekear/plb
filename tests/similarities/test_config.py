from pathlib import Path

import pytest
from pydantic import ValidationError

from plb.similarities.calibration import CHANNELS
from plb.similarities.config import SimilaritiesConfig, load_similarities_config
from plb.utils import get_dataset_as_path

EMBEDDINGS_PATH = get_dataset_as_path(
    "processed/embeddings/word_instance_embeddings_direct.jsonl"
)


def test_load_similarities_config_reads_percentile_and_calibration(
    tmp_path: Path,
) -> None:
    params_path = tmp_path / "params.yaml"
    params_path.write_text(
        """
similarities:
  percentiles:
    semantic: 0.995
    phonological: 0.9999
    morphological: 0.99
    ortho_stroke: 0.995
    ortho_kana: 0.995
  calibration:
    sample_pairs: 1000
    seed: 7
  top_k: 12
  embeddings_path: data/processed/embeddings/word_instance_embeddings_direct.jsonl
""".strip()
    )

    config = load_similarities_config(params_path)

    assert config.percentiles.semantic == 0.995
    assert config.percentiles.phonological == 0.9999
    assert config.percentiles.by_channel() == {
        "semantic": 0.995,
        "phonological": 0.9999,
        "morphological": 0.99,
        "ortho_stroke": 0.995,
        "ortho_kana": 0.995,
    }
    assert config.calibration.sample_pairs == 1000
    assert config.calibration.seed == 7
    assert config.calibration.output_path == get_dataset_as_path(
        "processed/similarity_null_calibration.json"
    )
    assert config.top_k == 12
    assert config.workers == 8
    assert config.target_chunk_size == 32
    assert config.embeddings_path == Path(
        "data/processed/embeddings/word_instance_embeddings_direct.jsonl"
    )
    assert config.reviews_path == get_dataset_as_path(
        "processed/clean_reviews_w_fsrs.jsonl"
    )
    assert config.output_path == get_dataset_as_path("processed/similarities.jsonl")


def _percentiles(**overrides: float) -> dict[str, float]:
    return {**dict.fromkeys(CHANNELS, 0.999), **overrides}


def test_percentile_must_stay_below_one() -> None:
    # A percentile of exactly 1.0 could never be reached — no score beats 100% of the
    # null — so it would silently disable that channel.
    with pytest.raises(ValidationError, match="less than 1"):
        SimilaritiesConfig(
            percentiles=_percentiles(semantic=1.0),
            embeddings_path=EMBEDDINGS_PATH,
        )


def test_every_channel_must_be_configured() -> None:
    """No shared fallback: a channel missing from params is an error, not a silent
    default, so the file always shows what each channel is set to."""
    incomplete = _percentiles()
    del incomplete["ortho_kana"]

    with pytest.raises(ValidationError, match="ortho_kana"):
        SimilaritiesConfig(percentiles=incomplete, embeddings_path=EMBEDDINGS_PATH)


def test_percentile_fields_match_the_calibrated_channels() -> None:
    """Guards against drift between the config model and CHANNELS."""
    assert set(SimilaritiesConfig.model_fields["percentiles"].annotation.model_fields) == set(
        CHANNELS
    )


def test_unknown_channel_is_rejected() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden|Extra inputs"):
        SimilaritiesConfig(
            percentiles=_percentiles(phonetic=0.99),
            embeddings_path=EMBEDDINGS_PATH,
        )


def test_channels_are_tuned_independently() -> None:
    config = SimilaritiesConfig(
        percentiles=_percentiles(morphological=0.99, ortho_kana=0.95),
        embeddings_path=EMBEDDINGS_PATH,
    )

    resolved = config.percentiles.by_channel()

    assert resolved["morphological"] == 0.99
    assert resolved["ortho_kana"] == 0.95
    assert resolved["semantic"] == 0.999
    assert set(resolved) == set(CHANNELS)


def test_raw_thresholds_are_no_longer_accepted() -> None:
    """Raw per-channel cut-offs are not comparable across channels; selection is by
    calibrated percentile. A stale params block must fail loudly rather than be
    ignored."""
    with pytest.raises(ValidationError, match="extra_forbidden|Extra inputs"):
        SimilaritiesConfig(
            percentiles=_percentiles(),
            thresholds={"semantic": 0.9},
            embeddings_path=EMBEDDINGS_PATH,
        )


@pytest.mark.parametrize("field", ["workers", "target_chunk_size"])
def test_parallelism_configuration_must_be_positive(field: str) -> None:
    values = {
        "percentiles": _percentiles(),
        "embeddings_path": EMBEDDINGS_PATH,
        field: 0,
    }

    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        SimilaritiesConfig(**values)


def test_embeddings_path_must_be_configured_explicitly() -> None:
    """It has no default on purpose: the embeddings file varies by model, so
    similarities must be pointed at the one its neighbours were built from."""
    with pytest.raises(ValidationError, match="embeddings_path"):
        SimilaritiesConfig()
