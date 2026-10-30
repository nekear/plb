from __future__ import annotations

from pathlib import Path

import pandas as pd

from plb.model_a.multils.cues import (
    ExpressionType,
    SourceWordClass,
    make_l2_clue,
    required_l1_names,
)

ANNOTATOR_COLUMNS = tuple(f"lcp_annotator_{i}" for i in range(1, 11))
ORIGINAL_FILES = (
    ("test", "agg_original_test.tsv"),
    ("trial", "agg_original_trial.tsv"),
)

ORIGIN_CODES = {
    "和": "wago",
    "漢": "kango",
    "外": "gairaigo",
    "混": "mixed",
}
IGNORED_POS_PREFIXES = ("助詞", "助動詞", "補助記号")
LIGHT_VERBS = {"する", "為る"}
FUNCTION_MAIN_POS = {"PART", "AUX", "CONJ", "SUF", "PREF"}


def split_pipe(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return str(value).split("|")


def derive_word_origin(target_base: object, target_type: object, target_pos: object) -> str:
    bases = split_pipe(target_base)
    types = split_pipe(target_type)
    parts_of_speech = split_pipe(target_pos)
    if not (len(bases) == len(types) == len(parts_of_speech)):
        raise ValueError(
            "target_base, target_type and target_pos must contain the same number "
            f"of pipe-separated tokens; got {len(bases)}, {len(types)}, {len(parts_of_speech)}"
        )

    origins: set[str] = set()
    for base, origin_code, part_of_speech in zip(bases, types, parts_of_speech, strict=True):
        if part_of_speech.startswith(IGNORED_POS_PREFIXES) or base in LIGHT_VERBS:
            continue
        mapped = ORIGIN_CODES.get(origin_code)
        if mapped == "mixed":
            return "mixed"
        if mapped is not None:
            origins.add(mapped)

    if not origins:
        return "unknown"
    if len(origins) > 1:
        return "mixed"
    return origins.pop()


def derive_expression_type(main_pos: object) -> str:
    return (
        ExpressionType.MWE.value
        if not pd.isna(main_pos) and main_pos == "MWE"
        else ExpressionType.SINGLE_WORD.value
    )


def derive_source_word_class(main_pos: object) -> str:
    return (
        SourceWordClass.FUNCTION_WORD.value
        if not pd.isna(main_pos) and str(main_pos) in FUNCTION_MAIN_POS
        else SourceWordClass.CONTENT_WORD.value
    )


def parse_target_position(position: object) -> tuple[int, int]:
    if pd.isna(position):
        raise ValueError("Target position is missing")
    parts = str(position).split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid target position {position!r}; expected START:END")
    try:
        start, end = (int(part) for part in parts)
    except ValueError as error:
        raise ValueError(f"Invalid target position {position!r}; expected integers") from error
    if start < 0 or end <= start:
        raise ValueError(f"Invalid target position bounds {position!r}")
    return start, end


def mark_target_span(context: str, target: str, position: object) -> str:
    start, end = parse_target_position(position)
    if end > len(context):
        raise ValueError(
            f"Target position {position!r} exceeds context length {len(context)}"
        )
    observed = context[start:end]
    if observed != target:
        raise ValueError(
            f"Target span mismatch at {position!r}: expected {target!r}, found {observed!r}"
        )
    return f"{context[:start]}<TARGET>{target}</TARGET>{context[end:]}"


def primary_l1(native_languages: object) -> str:
    if pd.isna(native_languages):
        raise ValueError("native_languages is missing")
    language = str(native_languages).split(",", maxsplit=1)[0].strip()
    if not language:
        raise ValueError("native_languages does not contain a primary language")
    return language


def load_original_profiles(data_dir: Path) -> pd.DataFrame:
    profiles = pd.read_csv(data_dir / "profiles" / "original_profiles.tsv", sep="\t")
    missing = set(ANNOTATOR_COLUMNS) - set(profiles["annotator"])
    if missing:
        raise ValueError(f"Missing original annotator profiles: {sorted(missing)}")
    profiles = profiles.copy()
    profiles["l1_name"] = profiles["native_languages"].map(primary_l1)
    return profiles


def original_l1_names(profiles: pd.DataFrame) -> tuple[str, ...]:
    return tuple(dict.fromkeys(profiles["l1_name"].tolist()))


def load_multils_items(data_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for source_split, filename in ORIGINAL_FILES:
        frame = pd.read_csv(data_dir / "annotations" / filename, sep="\t")
        frame = frame.copy()
        frame["source_split"] = source_split
        frames.append(frame)
    items = pd.concat(frames, ignore_index=True)
    if items["id"].duplicated().any():
        duplicates = items.loc[items["id"].duplicated(keep=False), "id"].unique()
        raise ValueError(f"MultiLS item IDs must be unique: {duplicates.tolist()}")
    items["l2_clue"] = items["target"].map(make_l2_clue)
    items["word_origin"] = items.apply(
        lambda row: derive_word_origin(row["target_base"], row["target_type"], row["target_pos"]),
        axis=1,
    )
    items["expression_type"] = items["main_pos"].map(derive_expression_type)
    items["source_word_class"] = items["main_pos"].map(derive_source_word_class)
    items["marked_context"] = items.apply(
        lambda row: mark_target_span(row["context"], row["target"], row["position"]),
        axis=1,
    )
    return items


def load_multils_long(data_dir: Path) -> pd.DataFrame:
    items = load_multils_items(data_dir)
    profiles = load_original_profiles(data_dir)
    long = items.melt(
        id_vars=[
            "id",
            "context",
            "target",
            "source_split",
            "l2_clue",
            "word_origin",
            "expression_type",
            "source_word_class",
            "main_pos",
        ],
        value_vars=list(ANNOTATOR_COLUMNS),
        var_name="annotator",
        value_name="human_rating",
    )
    long = long.merge(profiles[["annotator", "l1_name"]], on="annotator", how="left")
    if long["l1_name"].isna().any():
        missing = long.loc[long["l1_name"].isna(), "annotator"].unique()
        raise ValueError(f"No L1 profile found for annotators: {missing.tolist()}")
    return long
