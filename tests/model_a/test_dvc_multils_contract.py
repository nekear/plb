import yaml

from plb.utils import PROJECT_ROOT


def load_dvc() -> dict:
    return yaml.safe_load((PROJECT_ROOT / "dvc.yaml").read_text())


def stage_params(stage: dict) -> set[str]:
    result: set[str] = set()
    for entry in stage.get("params", []):
        for values in entry.values():
            result.update(values)
    return result


def test_multils_condition_does_not_invalidate_training_or_kvl_evaluation() -> None:
    stages = load_dvc()["stages"]
    for stage_name in ("train_model_a", "evaluate_model_a", "evaluate_model_a_test"):
        assert "model_a.multils.condition" not in stage_params(stages[stage_name])
        assert all("multils" not in dependency for dependency in stages[stage_name]["deps"])


def test_both_models_and_comparison_depend_on_condition() -> None:
    stages = load_dvc()["stages"]
    for stage_name in (
        "evaluate_model_a_multils_fine_tuned",
        "evaluate_model_a_multils_base",
        "compare_model_a_multils",
    ):
        assert "model_a.multils.condition" in stage_params(stages[stage_name])


def test_cue_domain_changes_invalidate_evaluation_but_api_runtime_changes_do_not() -> None:
    stages = load_dvc()["stages"]
    augmentation_module = "src/plb/model_a/multils/augmentation.py"
    cue_module = "src/plb/model_a/multils/cues.py"
    assert augmentation_module in stages["augment_model_a_multils"]["deps"]
    for stage_name in (
        "evaluate_model_a_multils_fine_tuned",
        "evaluate_model_a_multils_base",
    ):
        assert cue_module in stages[stage_name]["deps"]
        assert augmentation_module not in stages[stage_name]["deps"]


def test_all_run_scoped_paths_use_dvc_run_name_interpolation() -> None:
    stages = load_dvc()["stages"]
    metric_paths = []
    for stage_name in (
        "evaluate_model_a_multils_fine_tuned",
        "evaluate_model_a_multils_base",
        "compare_model_a_multils",
    ):
        metric_entry = stages[stage_name]["metrics"][0]
        metric_paths.append(next(iter(metric_entry)))
    assert metric_paths == [
        "evaluations/model_a/${model_a.run_name}/multils_japanese/fine_tuned/metrics.json",
        "evaluations/model_a/${model_a.run_name}/multils_japanese/base/metrics.json",
        "evaluations/model_a/${model_a.run_name}/multils_japanese/comparison/metrics.json",
    ]
    assert stages["train_model_a"]["outs"] == [
        "models/model_a/${model_a.run_name}"
    ]
    dvc_text = (PROJECT_ROOT / "dvc.yaml").read_text()
    assert "evaluations/model_a/baseline/" not in dvc_text
