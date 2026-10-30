import pytest
from pydantic import ValidationError

from plb.model_a.common.config import ModelAConfig, MultiLSConfig, load_model_a_config


def test_repository_config_is_valid() -> None:
    config = load_model_a_config()
    assert config.model_revision


def test_multils_defaults_remain_prespecified_primary_condition() -> None:
    config = MultiLSConfig()
    assert config.condition == "full_l1"
    assert config.augmentation.prompt_version == "v2"
    assert config.augmentation.reasoning_effort == "low"


@pytest.mark.parametrize(
    "condition",
    [
        "full_l1",
        "full_l2",
        "full_l2_no_context",
        "full_l2_deploy",
        "ja_word_only",
        "l1_name_only",
    ],
)
def test_supported_multils_conditions_are_accepted(condition: str) -> None:
    assert MultiLSConfig(condition=condition).condition == condition


def test_unknown_multils_condition_is_rejected() -> None:
    with pytest.raises(ValidationError):
        MultiLSConfig(condition="best_condition")


@pytest.mark.parametrize("run_name", ["baseline", "mmbert_seed42", "kvl-v2", "a.b"])
def test_safe_run_names_are_accepted(run_name: str) -> None:
    config = ModelAConfig(model_revision="revision", run_name=run_name)
    assert config.run_name == run_name


@pytest.mark.parametrize("run_name", ["../escape", "with space", "/absolute", "-leading"])
def test_unsafe_run_names_are_rejected(run_name: str) -> None:
    with pytest.raises(ValidationError):
        ModelAConfig(model_revision="revision", run_name=run_name)


def test_unknown_reasoning_effort_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ModelAConfig(
            model_revision="revision",
            multils={"augmentation": {"reasoning_effort": "automatic"}},
        )
