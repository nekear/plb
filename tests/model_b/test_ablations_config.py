import pytest
from pydantic import ValidationError

from plb.model_b.config import AblationsConfig, ModelBConfig

MODELS = {
    "global_median": {"type": "global"},
    "within_learner_median": {"type": "within"},
}


def test_defaults_construct_cleanly_with_no_models_configured() -> None:
    config = ModelBConfig()

    assert config.ablations.models == {}
    assert config.ablations.baseline == "within_learner_median"


def test_the_baseline_must_be_a_configured_model() -> None:
    with pytest.raises(
        ValidationError, match="'off_the_list' is not a configured model"
    ):
        AblationsConfig(baseline="off_the_list", models=MODELS)


def test_delta_pairs_must_reference_configured_models() -> None:
    with pytest.raises(ValidationError, match="unconfigured models.*missing"):
        AblationsConfig(models=MODELS, delta_pairs=(("global_median", "missing"),))


def test_a_delta_pair_cannot_compare_a_model_with_itself() -> None:
    with pytest.raises(ValidationError, match="two different models"):
        AblationsConfig(
            models=MODELS, delta_pairs=(("global_median", "global_median"),)
        )


def test_a_valid_config_passes() -> None:
    config = AblationsConfig(
        models=MODELS, delta_pairs=(("global_median", "within_learner_median"),)
    )

    assert config.delta_pairs == (("global_median", "within_learner_median"),)
