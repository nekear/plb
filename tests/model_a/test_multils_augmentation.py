from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from pydantic import ValidationError

from plb.model_a.multils.augment import main as augment_main
from plb.model_a.common.config import load_model_a_config
from plb.utils import get_dataset_as_path
from plb.model_a.multils.augmentation import (
    OpenAICueClient,
    TRANSLATION_INSTRUCTIONS,
    append_checkpoint,
    augmentation_prompt_hash,
    call_with_retries,
    generate_item_cues,
    is_transient_api_error,
    load_checkpoint,
    select_items_by_id,
    select_pilot_items,
)
from plb.model_a.multils.cues import (
    GeneratedItem,
    ItemCueOutput,
    MetalinguisticCue,
    TranslationCue,
    build_quarantine,
    flatten_generated_items,
    sort_sidecar,
    validate_complete_sidecar,
    validate_sidecar_coverage,
)
from plb.model_a.multils.data import (
    load_multils_items,
    load_original_profiles,
    original_l1_names,
    required_l1_names,
)


class FakeCueClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def translate(self, **kwargs) -> GeneratedItem:
        self.calls.append(kwargs["item_id"])
        cues = [
            TranslationCue(
                l1_name=language,
                l1_gloss=f"{language} gloss",
                l1_context=f"A {language} context with surface.",
                gloss_kind="translation",
                l1_target_surface="surface",
                alignment_kind="exact",
                alignment_note="",
            )
            for language in kwargs["l1_names"]
        ]
        return GeneratedItem(
            id=kwargs["item_id"],
            input_hash=kwargs["input_hash"],
            prompt_hash=kwargs["prompt_hash"],
            generation_model=kwargs["model"],
            response_model=kwargs["model"],
            reasoning_effort=kwargs["reasoning_effort"],
            prompt_version=kwargs["prompt_version"],
            response_id=f"response-{kwargs['item_id']}",
            cues=cues,
        )


def sample_items() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": "test-1",
                "source_split": "test",
                "target": "奴",
                "context": "奴だ",
                "position": "0:1",
                "marked_context": "<TARGET>奴</TARGET>だ",
                "main_pos": "N",
                "l2_clue": "奴",
                "word_origin": "wago",
                "expression_type": "single_word",
                "source_word_class": "content_word",
            },
            {
                "id": "trial-1",
                "source_split": "trial",
                "target": "掲載した",
                "context": "掲載した。",
                "position": "0:4",
                "marked_context": "<TARGET>掲載した</TARGET>。",
                "main_pos": "V",
                "l2_clue": "掲___",
                "word_origin": "kango",
                "expression_type": "single_word",
                "source_word_class": "content_word",
            },
        ]
    )


def make_record(
    *,
    item_id: str = "test-1",
    cue: TranslationCue | MetalinguisticCue | None = None,
    reasoning_effort: str = "medium",
) -> GeneratedItem:
    return GeneratedItem(
        id=item_id,
        input_hash="input",
        prompt_hash=augmentation_prompt_hash("v2", reasoning_effort),
        generation_model="model",
        response_model="model-snapshot",
        reasoning_effort=reasoning_effort,
        prompt_version="v2",
        response_id="response",
        cues=[
            cue
            or TranslationCue(
                l1_name="English",
                l1_gloss="person",
                l1_context="That person is annoying.",
                gloss_kind="translation",
                l1_target_surface="person",
                alignment_kind="exact",
                alignment_note="",
            )
        ],
    )


def test_cue_union_enforces_metalinguistic_contract() -> None:
    cue = MetalinguisticCue(
        l1_name="English",
        l1_gloss="emphatic sentence-final particle",
        l1_context="Watch out!",
        gloss_kind="metalinguistic",
        l1_target_surface="",
        alignment_kind="not_applicable",
        alignment_note="English expresses the force through the utterance.",
    )
    assert cue.l1_target_surface == ""
    assert cue.alignment_kind == "not_applicable"
    with pytest.raises(ValidationError):
        MetalinguisticCue(
            l1_name="English",
            l1_gloss="particle",
            l1_context="Watch out!",
            gloss_kind="metalinguistic",
            l1_target_surface="Watch out!",
            alignment_kind="not_applicable",
            alignment_note="",
        )


def test_structured_output_schema_requires_every_cue_field() -> None:
    schema = ItemCueOutput.model_json_schema()
    for definition in ("TranslationCue", "MetalinguisticCue"):
        cue_schema = schema["$defs"][definition]
        assert set(cue_schema["required"]) == set(cue_schema["properties"])
    assert "anyOf" in schema["properties"]["cues"]["items"]
    assert "oneOf" not in schema["properties"]["cues"]["items"]
    with pytest.raises(ValidationError, match="require l1_target_surface"):
        TranslationCue(
            l1_name="English",
            l1_gloss="person",
            l1_context="A person.",
            gloss_kind="translation",
            l1_target_surface="",
            alignment_kind="exact",
            alignment_note="",
        )


def test_openai_client_passes_reasoning_and_marked_request() -> None:
    captured: dict = {}

    class Responses:
        def parse(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                id="response",
                model="snapshot",
                output_parsed=ItemCueOutput(cues=[make_record().cues[0]]),
            )

    result = OpenAICueClient(SimpleNamespace(responses=Responses())).translate(
        item_id="test-1",
        target="奴",
        marked_context="<TARGET>奴</TARGET>だ",
        main_pos="N",
        source_word_class="content_word",
        l1_names=("English",),
        model="model",
        reasoning_effort="medium",
        prompt_version="v2",
        prompt_hash="prompt",
        input_hash="input",
    )
    assert captured["reasoning"] == {"effort": "medium"}
    assert "<TARGET>奴</TARGET>" in captured["input"]
    assert captured["store"] is False
    assert result.reasoning_effort == "medium"


def test_translation_instructions_make_marked_boundary_strict() -> None:
    assert "complete target boundary" in TRANSLATION_INSTRUCTIONS
    assert "outside the tags is context only" in TRANSLATION_INSTRUCTIONS
    assert "Never absorb outside material" in TRANSLATION_INSTRUCTIONS
    assert "Never label that joint realization exact or inflected" in (
        TRANSLATION_INSTRUCTIONS
    )


def test_checkpoint_ignores_only_truncated_final_record(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.jsonl"
    valid = make_record()
    append_checkpoint(path, valid)
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"id":')
    assert load_checkpoint(path) == {"test-1": valid}


def test_legacy_schema_is_stale_even_when_it_is_the_final_record(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.jsonl"
    path.write_text(
        '{"id":"legacy-v1","input_hash":"input","prompt_hash":"prompt",'
        '"generation_model":"model","response_model":"model",'
        '"prompt_version":"v1","response_id":"response","cues":[]}\n'
    )
    with pytest.raises(ValueError, match="Incompatible or stale"):
        load_checkpoint(path)


def test_generation_resumes_without_repeating_completed_calls(tmp_path: Path) -> None:
    items = sample_items()
    client = FakeCueClient()
    checkpoint = tmp_path / "checkpoint.jsonl"
    kwargs = dict(
        original_l1s=("English", "French"),
        client=client,
        generation_model="model",
        reasoning_effort="medium",
        prompt_version="v2",
        checkpoint_path=checkpoint,
        retry_base_seconds=0,
    )
    first = generate_item_cues(items, **kwargs)
    assert client.calls == ["test-1", "trial-1"]
    client.calls.clear()
    second = generate_item_cues(items, **kwargs)
    assert client.calls == []
    assert second == first


def test_generation_rejects_changed_input_or_reasoning(tmp_path: Path) -> None:
    items = sample_items().iloc[:1].copy()
    checkpoint = tmp_path / "checkpoint.jsonl"
    client = FakeCueClient()
    base = dict(
        original_l1s=("English",),
        client=client,
        generation_model="model",
        reasoning_effort="medium",
        prompt_version="v2",
        checkpoint_path=checkpoint,
        retry_base_seconds=0,
    )
    generate_item_cues(items, **base)
    changed = items.copy()
    changed.loc[changed.index[0], "context"] = "奴です"
    with pytest.raises(ValueError, match="stale"):
        generate_item_cues(changed, **base)
    with pytest.raises(ValueError, match="stale"):
        generate_item_cues(items, **{**base, "reasoning_effort": "low"})


def test_flatten_and_quarantine_keep_paraphrastic_rows() -> None:
    cue = TranslationCue(
        l1_name="English",
        l1_gloss="a person",
        l1_context="That guy is annoying.",
        gloss_kind="translation",
        l1_target_surface="person",
        alignment_kind="paraphrastic",
        alignment_note="No literal span",
    )
    sidecar = flatten_generated_items(sample_items().iloc[:1], [make_record(cue=cue)])
    validate_complete_sidecar(sidecar, sample_items().iloc[:1], ("English",))
    quarantine = build_quarantine(sidecar)
    assert len(sidecar) == 1
    assert len(quarantine) == 1
    assert "alignment:paraphrastic" in quarantine.iloc[0]["warnings"]
    assert "surface_not_literal_substring" in quarantine.iloc[0]["warnings"]
    assert sidecar.iloc[0]["l1_clue"] == "a ______"


def test_metalinguistic_function_word_is_valid_and_not_quarantined() -> None:
    items = sample_items().iloc[:1].copy()
    items.loc[:, "source_word_class"] = "function_word"
    cue = MetalinguisticCue(
        l1_name="English",
        l1_gloss="emphatic sentence-final particle",
        l1_context="Watch out!",
        gloss_kind="metalinguistic",
        l1_target_surface="",
        alignment_kind="not_applicable",
        alignment_note="English has no standalone equivalent.",
    )
    sidecar = flatten_generated_items(items, [make_record(cue=cue)])
    validate_complete_sidecar(sidecar, items, ("English",))
    assert build_quarantine(sidecar).empty


def test_trial_coverage_requires_chinese() -> None:
    items = sample_items().iloc[1:]
    sidecar = pd.DataFrame(
        {"id": ["trial-1"], "l1_name": ["English"]}
    )
    with pytest.raises(ValueError, match="Chinese"):
        validate_sidecar_coverage(sidecar, items, ("English",))


def test_pilot_selection_is_deterministic_and_covers_shaky_categories() -> None:
    items = load_multils_items(get_dataset_as_path("raw/multils_japanese"))
    first = select_pilot_items(items, sample_size=12, seed=42)
    second = select_pilot_items(items, sample_size=12, seed=42)
    assert first["id"].tolist() == second["id"].tolist()
    assert set(first["source_split"]) == {"test", "trial"}
    assert set(first["expression_type"]) == {"single_word", "mwe"}
    assert "function_word" in set(first["source_word_class"])


@pytest.mark.parametrize("sample_size", [1, 2, 3, 4])
def test_small_pilot_selection_returns_exact_requested_size(sample_size: int) -> None:
    items = load_multils_items(get_dataset_as_path("raw/multils_japanese"))
    selected = select_pilot_items(items, sample_size=sample_size, seed=42)
    assert len(selected) == sample_size


def test_exact_item_selection_preserves_requested_order() -> None:
    items = load_multils_items(get_dataset_as_path("raw/multils_japanese"))
    selected = select_items_by_id(items, ["ja_298", "ja_128", "ja_436"])
    assert selected["id"].tolist() == ["ja_298", "ja_128", "ja_436"]


def test_exact_item_selection_rejects_duplicates_and_unknown_ids() -> None:
    items = load_multils_items(get_dataset_as_path("raw/multils_japanese"))
    with pytest.raises(ValueError, match="Duplicate --item-id"):
        select_items_by_id(items, ["ja_298", "ja_298"])
    with pytest.raises(ValueError, match="Unknown --item-id"):
        select_items_by_id(items, ["ja_missing"])


def test_full_expected_coverage_is_3030_rows() -> None:
    data_dir = get_dataset_as_path("raw/multils_japanese")
    items = load_multils_items(data_dir)
    original_l1s = original_l1_names(load_original_profiles(data_dir))
    rows = [
        {"id": item.id, "l1_name": l1_name}
        for item in items.itertuples(index=False)
        for l1_name in required_l1_names(item.source_split, original_l1s)
    ]
    sidecar = pd.DataFrame(rows)
    assert len(sidecar) == 3030
    validate_sidecar_coverage(sidecar, items, original_l1s)


def test_complete_sidecar_rejects_stale_deterministic_features() -> None:
    items = sample_items().iloc[:1]
    sidecar = flatten_generated_items(items, [make_record()])
    sidecar.loc[0, "l2_clue"] = "wrong"
    with pytest.raises(ValueError, match="l2_clue"):
        validate_complete_sidecar(sidecar, items, ("English",))


def test_complete_sidecar_rejects_legacy_ambiguous_clue_column() -> None:
    items = sample_items().iloc[:1]
    legacy = flatten_generated_items(items, [make_record()]).rename(
        columns={"l2_clue": "clue"}
    )
    with pytest.raises(ValueError, match="l2_clue"):
        validate_complete_sidecar(legacy, items, ("English",))


def test_retry_is_bounded_and_uses_exponential_delays() -> None:
    attempts = 0
    delays: list[float] = []

    def fail_twice():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError("transient")
        return "done"

    assert call_with_retries(
        fail_twice,
        max_retries=2,
        retry_base_seconds=0.5,
        sleep=delays.append,
    ) == "done"
    assert delays == [0.5, 1.0]


def test_permanent_errors_are_not_retried() -> None:
    attempts = 0

    def invalid_response():
        nonlocal attempts
        attempts += 1
        raise ValueError("schema mismatch")

    with pytest.raises(ValueError, match="schema mismatch"):
        call_with_retries(
            invalid_response,
            max_retries=4,
            retry_base_seconds=0,
            is_retryable=is_transient_api_error,
        )
    assert attempts == 1


def test_chinese_han_characters_are_not_japanese_residue() -> None:
    items = sample_items().iloc[:1]
    cue = TranslationCue(
        l1_name="Chinese",
        l1_gloss="刊登",
        l1_context="回答也刊登了。",
        gloss_kind="translation",
        l1_target_surface="刊登",
        alignment_kind="exact",
        alignment_note="Direct realization.",
    )
    record = make_record(cue=cue)
    sidecar = flatten_generated_items(items, [record])
    assert build_quarantine(sidecar).empty


def test_sidecar_sort_uses_source_item_and_profile_language_order() -> None:
    items = sample_items()
    sidecar = pd.DataFrame(
        {
            "id": ["trial-1", "test-1", "test-1"],
            "l1_name": ["Chinese", "French", "English"],
        }
    )
    ordered = sort_sidecar(sidecar, items, ("English", "French"))
    assert ordered[["id", "l1_name"]].values.tolist() == [
        ["test-1", "English"],
        ["test-1", "French"],
        ["trial-1", "Chinese"],
    ]


def test_pilot_cli_with_fake_client_writes_isolated_valid_artifacts(tmp_path: Path) -> None:
    expected_reasoning_effort = (
        load_model_a_config().multils.augmentation.reasoning_effort
    )
    output = tmp_path / "pilot.tsv"
    checkpoint = tmp_path / "pilot.jsonl"
    quarantine = tmp_path / "pilot_quarantine.tsv"
    result = augment_main(
        [
            "--output",
            str(output),
            "--checkpoint",
            str(checkpoint),
            "--quarantine",
            str(quarantine),
            "--sample-size",
            "12",
            "--seed",
            "42",
            "--retry-base-seconds",
            "0",
            "--no-progress",
        ],
        client=FakeCueClient(),
    )
    assert result == 0
    cues = pd.read_csv(output, sep="\t")
    assert cues["id"].nunique() == 12
    assert "Chinese" in set(cues["l1_name"])
    assert "l2_clue" in cues.columns
    assert "clue" not in cues.columns
    assert set(cues["reasoning_effort"]) == {expected_reasoning_effort}
    assert checkpoint.exists()
    assert quarantine.exists()


def test_targeted_cli_preserves_item_order_and_writes_only_requested_rows(
    tmp_path: Path,
) -> None:
    output = tmp_path / "targeted.tsv"
    checkpoint = tmp_path / "targeted.jsonl"
    item_ids = ["ja_128", "ja_298", "ja_436", "ja_196", "ja_256"]
    arguments = [
        "--output",
        str(output),
        "--checkpoint",
        str(checkpoint),
        "--retry-base-seconds",
        "0",
        "--no-progress",
    ]
    for item_id in item_ids:
        arguments.extend(["--item-id", item_id])
    result = augment_main(arguments, client=FakeCueClient())
    assert result == 0
    cues = pd.read_csv(output, sep="\t")
    assert cues["id"].drop_duplicates().tolist() == item_ids
    assert len(cues) == 25
    assert "Chinese" not in set(cues["l1_name"])


def test_cli_rejects_combined_random_and_exact_selection() -> None:
    with pytest.raises(SystemExit):
        augment_main(
            ["--sample-size", "5", "--item-id", "ja_298", "--dry-run"]
        )
