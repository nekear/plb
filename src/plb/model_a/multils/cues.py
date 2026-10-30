from __future__ import annotations

import os
import re
import unicodedata
from collections.abc import Iterable, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, TypeAdapter, field_validator, model_validator

CHINESE_L1 = "Chinese"
KANA_RE = re.compile(r"[\u3040-\u30ff]")
JAPANESE_SCRIPT_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
MULTIPLE_GLOSS_CANDIDATES_RE = re.compile(r"[;|/\n]")


class GlossKind(StrEnum):
    TRANSLATION = "translation"
    METALINGUISTIC = "metalinguistic"


class AlignmentKind(StrEnum):
    EXACT = "exact"
    INFLECTED = "inflected"
    DISCONTINUOUS = "discontinuous"
    PARAPHRASTIC = "paraphrastic"
    UNCERTAIN = "uncertain"
    NOT_APPLICABLE = "not_applicable"


class SourceWordClass(StrEnum):
    CONTENT_WORD = "content_word"
    FUNCTION_WORD = "function_word"


class ExpressionType(StrEnum):
    SINGLE_WORD = "single_word"
    MWE = "mwe"


class EvaluationBucket(StrEnum):
    SINGLE_WORDS = "single_words"
    MWES = "mwes"
    OTHERS = "others"


class CueBase(BaseModel):
    l1_name: str
    l1_gloss: str
    l1_context: str
    alignment_note: str

    @field_validator("l1_name", "l1_gloss", "l1_context")
    @classmethod
    def require_nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("cue language, gloss, and context must be nonblank")
        return value


class TranslationCue(CueBase):
    gloss_kind: Literal[GlossKind.TRANSLATION]
    l1_target_surface: str
    alignment_kind: Literal[
        AlignmentKind.EXACT,
        AlignmentKind.INFLECTED,
        AlignmentKind.DISCONTINUOUS,
        AlignmentKind.PARAPHRASTIC,
        AlignmentKind.UNCERTAIN,
    ]

    @model_validator(mode="after")
    def validate_surface(self) -> TranslationCue:
        if self.alignment_kind in {AlignmentKind.EXACT, AlignmentKind.INFLECTED}:
            if not self.l1_target_surface.strip():
                raise ValueError("exact and inflected alignments require l1_target_surface")
        return self


class MetalinguisticCue(CueBase):
    gloss_kind: Literal[GlossKind.METALINGUISTIC]
    l1_target_surface: Literal[""]
    alignment_kind: Literal[AlignmentKind.NOT_APPLICABLE]


L1Cue = TranslationCue | MetalinguisticCue
L1_CUE_ADAPTER = TypeAdapter(L1Cue)


class ItemCueOutput(BaseModel):
    cues: list[L1Cue]


class GeneratedItem(BaseModel):
    id: str
    input_hash: str
    prompt_hash: str
    generation_model: str
    response_model: str
    reasoning_effort: str
    prompt_version: str
    response_id: str
    cues: list[L1Cue]


SIDECAR_COLUMNS = [
    "id",
    "l1_name",
    "l1_gloss",
    "l1_context",
    "gloss_kind",
    "l1_target_surface",
    "alignment_kind",
    "alignment_note",
    "l2_clue",
    "l1_clue",
    "word_origin",
    "expression_type",
    "source_word_class",
    "generation_model",
    "response_model",
    "reasoning_effort",
    "prompt_version",
    "prompt_hash",
    "input_hash",
    "response_id",
]


def required_l1_names(source_split: str, original_l1s: Iterable[str]) -> tuple[str, ...]:
    languages = tuple(original_l1s)
    if source_split == "trial":
        return (*languages, CHINESE_L1)
    if source_split == "test":
        return languages
    raise ValueError(f"Unsupported MultiLS source split: {source_split!r}")


def make_l2_clue(target: str) -> str:
    """Keep the first target character and mask the remaining characters."""
    if not target:
        raise ValueError("Cannot construct an L2 clue for an empty target")
    return target[0] + "_" * (len(target) - 1)


def make_l1_clue(gloss: str) -> str:
    """Mask every Unicode letter/number after the first, preserving separators."""
    normalized = unicodedata.normalize("NFC", gloss.strip())
    if not normalized:
        raise ValueError("Cannot construct an L1 clue for an empty gloss")

    retained = False
    result: list[str] = []
    for character in normalized:
        is_word_character = unicodedata.category(character)[0] in {"L", "N"}
        if not is_word_character:
            result.append(character)
        elif not retained:
            result.append(character)
            retained = True
        else:
            result.append("_")
    if not retained:
        raise ValueError(f"Gloss contains no Unicode letter or number: {gloss!r}")
    return "".join(result)


def validate_returned_languages(cues: Sequence[L1Cue], expected: Sequence[str]) -> None:
    returned = [cue.l1_name for cue in cues]
    if len(returned) != len(set(returned)):
        raise ValueError(f"API response contains duplicate L1 names: {returned}")
    if set(returned) != set(expected) or len(returned) != len(expected):
        raise ValueError(
            f"API response language mismatch: expected {list(expected)}, returned {returned}"
        )


def flatten_generated_items(
    items: pd.DataFrame, generated: Iterable[GeneratedItem]
) -> pd.DataFrame:
    item_lookup = items.set_index("id")
    rows: list[dict] = []
    for record in generated:
        if record.id not in item_lookup.index:
            raise ValueError(f"Generated record references unknown item {record.id!r}")
        item = item_lookup.loc[record.id]
        for cue in record.cues:
            rows.append(
                {
                    "id": record.id,
                    **cue.model_dump(mode="json"),
                    "l2_clue": item["l2_clue"],
                    "l1_clue": make_l1_clue(cue.l1_gloss),
                    "word_origin": item["word_origin"],
                    "expression_type": item["expression_type"],
                    "source_word_class": item["source_word_class"],
                    "generation_model": record.generation_model,
                    "response_model": record.response_model,
                    "reasoning_effort": record.reasoning_effort,
                    "prompt_version": record.prompt_version,
                    "prompt_hash": record.prompt_hash,
                    "input_hash": record.input_hash,
                    "response_id": record.response_id,
                }
            )
    return pd.DataFrame(rows, columns=SIDECAR_COLUMNS)


def sort_sidecar(
    sidecar: pd.DataFrame, items: pd.DataFrame, original_l1s: Sequence[str]
) -> pd.DataFrame:
    item_order = {item_id: index for index, item_id in enumerate(items["id"])}
    language_order = {
        l1_name: index for index, l1_name in enumerate((*original_l1s, CHINESE_L1))
    }
    ordered = sidecar.assign(
        _item_order=sidecar["id"].map(item_order),
        _language_order=sidecar["l1_name"].map(language_order),
    )
    if ordered[["_item_order", "_language_order"]].isna().any().any():
        raise ValueError("Cannot order sidecar containing unknown item IDs or L1 names")
    return (
        ordered.sort_values(["_item_order", "_language_order"])
        .drop(columns=["_item_order", "_language_order"])
        .reset_index(drop=True)
    )


def validate_sidecar_coverage(
    sidecar: pd.DataFrame,
    items: pd.DataFrame,
    original_l1s: Sequence[str],
) -> None:
    duplicates = sidecar.duplicated(subset=["id", "l1_name"], keep=False)
    if duplicates.any():
        keys = sidecar.loc[duplicates, ["id", "l1_name"]].to_dict("records")
        raise ValueError(f"Duplicate sidecar keys: {keys[:10]}")
    actual = {
        item_id: set(group["l1_name"])
        for item_id, group in sidecar.groupby("id", sort=False)
    }
    expected_ids = set(items["id"])
    if set(actual) != expected_ids:
        missing = sorted(expected_ids - set(actual))
        extra = sorted(set(actual) - expected_ids)
        raise ValueError(f"Sidecar ID mismatch; missing={missing[:10]}, extra={extra[:10]}")
    split_by_id = items.set_index("id")["source_split"].to_dict()
    for item_id in sorted(expected_ids):
        expected_l1s = set(required_l1_names(split_by_id[item_id], original_l1s))
        if actual[item_id] != expected_l1s:
            raise ValueError(
                f"Sidecar languages for {item_id} are {sorted(actual[item_id])}; "
                f"expected {sorted(expected_l1s)}"
            )


def validate_complete_sidecar(
    sidecar: pd.DataFrame,
    items: pd.DataFrame,
    original_l1s: Sequence[str],
) -> None:
    missing_columns = set(SIDECAR_COLUMNS) - set(sidecar.columns)
    if missing_columns:
        raise ValueError(f"Sidecar is missing required columns: {sorted(missing_columns)}")
    validate_sidecar_coverage(sidecar, items, original_l1s)

    required_nonblank = [
        column
        for column in SIDECAR_COLUMNS
        if column not in {"l1_target_surface", "alignment_note"}
    ]
    if sidecar[required_nonblank].isna().any().any():
        raise ValueError("Sidecar contains missing required values")
    for column in required_nonblank:
        if (sidecar[column].astype(str).str.strip() == "").any():
            raise ValueError(f"Sidecar contains blank values in {column!r}")

    for row in sidecar.itertuples(index=False):
        L1_CUE_ADAPTER.validate_python(
            {
                "l1_name": row.l1_name,
                "l1_gloss": row.l1_gloss,
                "l1_context": row.l1_context,
                "gloss_kind": row.gloss_kind,
                "l1_target_surface": row.l1_target_surface,
                "alignment_kind": row.alignment_kind,
                "alignment_note": row.alignment_note,
            }
        )

    deterministic_columns = [
        "l2_clue",
        "word_origin",
        "expression_type",
        "source_word_class",
    ]
    expected = items.set_index("id")[deterministic_columns]
    actual = sidecar[["id", *deterministic_columns]].merge(
        expected,
        left_on="id",
        right_index=True,
        validate="many_to_one",
        suffixes=("_sidecar", "_source"),
    )
    for column in deterministic_columns:
        if not actual[f"{column}_sidecar"].equals(actual[f"{column}_source"]):
            raise ValueError(
                f"Sidecar {column!r} does not match its deterministic source derivation"
            )
    expected_l1_clues = sidecar["l1_gloss"].map(make_l1_clue)
    if not sidecar["l1_clue"].equals(expected_l1_clues):
        raise ValueError("Sidecar 'l1_clue' does not match its deterministic gloss derivation")


def cue_warning_codes(row: object) -> tuple[str, ...]:
    get = row.get if isinstance(row, dict) else lambda name: getattr(row, name)
    warnings: list[str] = []
    alignment_kind = str(get("alignment_kind"))
    if alignment_kind in {
        AlignmentKind.DISCONTINUOUS,
        AlignmentKind.PARAPHRASTIC,
        AlignmentKind.UNCERTAIN,
    }:
        warnings.append(f"alignment:{alignment_kind}")
    surface = str(get("l1_target_surface") or "")
    context = str(get("l1_context") or "")
    if surface and surface not in context:
        warnings.append("surface_not_literal_substring")
    gloss = str(get("l1_gloss") or "")
    if MULTIPLE_GLOSS_CANDIDATES_RE.search(gloss):
        warnings.append("multiple_gloss_candidates")
    if (
        str(get("source_word_class")) == SourceWordClass.FUNCTION_WORD
        and str(get("gloss_kind")) == GlossKind.TRANSLATION
        and alignment_kind in {AlignmentKind.PARAPHRASTIC, AlignmentKind.UNCERTAIN}
    ):
        warnings.append("functional_target_paraphrase")
    residue_pattern = KANA_RE if str(get("l1_name")) == CHINESE_L1 else JAPANESE_SCRIPT_RE
    if residue_pattern.search(gloss) or residue_pattern.search(context):
        warnings.append("japanese_script_residue")
    return tuple(warnings)


def build_quarantine(sidecar: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for row in sidecar.itertuples(index=False):
        warnings = cue_warning_codes(row)
        if warnings:
            item = row._asdict()
            item["warnings"] = ",".join(warnings)
            rows.append(item)
    return pd.DataFrame(rows, columns=[*SIDECAR_COLUMNS, "warnings"])


def write_tsv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, sep="\t", index=False)
    os.replace(temporary, path)
