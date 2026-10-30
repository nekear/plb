from pathlib import Path

import pandas as pd
import pytest

from plb.utils import get_dataset_as_path
from plb.model_a.multils.cues import make_l1_clue
from plb.model_a.multils.evaluation import enrich_evaluation_rows, prepare_inference_prompts
from plb.model_a.multils.data import (
    load_multils_items,
    load_original_profiles,
    original_l1_names,
    required_l1_names,
)

DATA_DIR = get_dataset_as_path("raw/multils_japanese")
MISSING_CUES = get_dataset_as_path("interim/model_a/not-generated.tsv")


def test_japanese_word_only_deduplicates_by_item_without_cues() -> None:
    prompts, per_annotator, keys = prepare_inference_prompts(
        DATA_DIR, "ja_word_only", MISSING_CUES
    )
    assert keys == ["id"]
    assert len(prompts) == 600
    assert len(per_annotator) == 6000
    assert prompts["text"].str.contains("native language").sum() == 0


def test_l1_name_only_deduplicates_by_item_and_l1_without_cues() -> None:
    prompts, per_annotator, keys = prepare_inference_prompts(
        DATA_DIR, "l1_name_only", MISSING_CUES
    )
    assert keys == ["id", "l1_name"]
    assert len(prompts) == 3000
    assert len(per_annotator) == 6000
    assert prompts["text"].str.contains("native language is").all()


def write_complete_synthetic_cues(path: Path) -> None:
    items = load_multils_items(DATA_DIR)
    l1_names = original_l1_names(load_original_profiles(DATA_DIR))
    rows = []
    for item in items.itertuples(index=False):
        for l1_name in required_l1_names(item.source_split, l1_names):
            rows.append(
                {
                    "id": item.id,
                    "l1_name": l1_name,
                    "l1_gloss": f"{l1_name} gloss",
                    "l1_context": f"{l1_name} translated context with surface",
                    "gloss_kind": "translation",
                    "l1_target_surface": "surface",
                    "alignment_kind": "exact",
                    "alignment_note": "",
                    "l2_clue": item.l2_clue,
                    "l1_clue": make_l1_clue(f"{l1_name} gloss"),
                    "word_origin": item.word_origin,
                    "expression_type": item.expression_type,
                    "source_word_class": item.source_word_class,
                    "generation_model": "model",
                    "response_model": "model",
                    "reasoning_effort": "medium",
                    "prompt_version": "v2",
                    "prompt_hash": "prompt",
                    "input_hash": "input",
                    "response_id": f"response-{item.id}",
                }
            )
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


@pytest.mark.parametrize(
    "condition",
    [
        "full_l1",
        "full_l2",
        "full_l2_no_context",
        "full_l2_deploy",
    ],
)
def test_cue_conditions_join_complete_sidecar(
    condition: str, tmp_path: Path
) -> None:
    cues_path = tmp_path / "cues.tsv"
    write_complete_synthetic_cues(cues_path)
    prompts, per_annotator, keys = prepare_inference_prompts(
        DATA_DIR, condition, cues_path
    )
    assert keys == ["id", "l1_name"]
    assert len(prompts) == 3000
    assert len(per_annotator) == 6000
    assert "Chinese" not in set(prompts["l1_name"])
    if condition == "full_l2_deploy":
        assert not prompts["text"].str.contains("Clue:").any()
    else:
        assert prompts["text"].str.contains("Clue:").all()


def test_enrichment_assigns_l1_specific_clean_and_other_buckets(tmp_path: Path) -> None:
    cues_path = tmp_path / "cues.tsv"
    write_complete_synthetic_cues(cues_path)
    cues = pd.read_csv(cues_path, sep="\t", keep_default_na=False)
    _, rows, _ = prepare_inference_prompts(DATA_DIR, "ja_word_only", MISSING_CUES)
    rows["predicted_rating"] = 3.0
    enriched = enrich_evaluation_rows(rows, cues)
    assert set(enriched["evaluation_bucket"]) == {"single_words", "mwes", "others"}
    assert enriched["cue_warnings"].eq("[]").all()
    assert list(enriched.columns) == [
        "id",
        "annotator",
        "l1_name",
        "human_rating",
        "predicted_rating",
        "source_split",
        "target",
        "expression_type",
        "source_word_class",
        "word_origin",
        "gloss_kind",
        "alignment_kind",
        "cue_warnings",
        "evaluation_bucket",
    ]
