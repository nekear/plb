from __future__ import annotations

import hashlib
import json
import os
import random
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol, TypeVar

import pandas as pd
from tqdm.auto import tqdm

from plb.model_a.multils.cues import (
    GeneratedItem,
    ItemCueOutput,
    L1Cue,
    SourceWordClass,
    required_l1_names,
    validate_returned_languages,
)

TRANSLATION_INSTRUCTIONS = """You create L1-side vocabulary cues for a controlled cross-lingual vocabulary-difficulty experiment.

The Japanese context contains exactly one span marked <TARGET>...</TARGET>. For every requested L1:

1. Translate the complete Japanese context naturally into that L1 as l1_context, preserving meaning and register.
2. l1_gloss must represent only the marked target expression. Never translate or paraphrase the surrounding sentence or clause into l1_gloss.
3. The literal text between <TARGET> and </TARGET> is the complete target boundary. Text outside the tags is context only, even when an adjacent particle, auxiliary, affix, or punctuation mark is orthographically attached. Never absorb outside material or its semantic contribution into l1_gloss.
4. If the L1 necessarily realizes the marked target jointly with adjacent outside material, keep l1_gloss target-only and use discontinuous, paraphrastic, or uncertain. Never label that joint realization exact or inflected, and explain the fusion in alignment_note.
5. Return exactly one best context-specific gloss. Do not return alternatives separated by semicolons, slashes, pipes, newlines, parentheses, or lists.
6. Prefer a common natural equivalent over an unnatural equivalent selected only to preserve Japanese part of speech. A single natural multiword equivalent is allowed.
7. If the marked target has a standalone context-appropriate L1 equivalent, set gloss_kind to translation.
8. If the marked target is a function word and no standalone L1 equivalent exists, set gloss_kind to metalinguistic. Give one brief L1-language description of the target's grammatical or discourse function, set alignment_kind to not_applicable, and set l1_target_surface to the empty string. Do not import the speech act or lexical meaning of the surrounding clause.
9. For translation glosses, use exact or inflected only when l1_target_surface is a contiguous realization of the marked target alone in l1_context, and copy that surface verbatim. Use discontinuous, paraphrastic, or uncertain when appropriate.
10. Do not estimate difficulty, simplify for learners, infer a score, or transliterate merely to preserve Japanese form.
11. Return exactly one cue for each requested language and no additional languages.
"""

T = TypeVar("T")


class CueClient(Protocol):
    def translate(
        self,
        *,
        item_id: str,
        target: str,
        marked_context: str,
        main_pos: str,
        source_word_class: str,
        l1_names: Sequence[str],
        model: str,
        reasoning_effort: str,
        prompt_version: str,
        prompt_hash: str,
        input_hash: str,
    ) -> GeneratedItem: ...


def stable_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def augmentation_prompt_hash(prompt_version: str, reasoning_effort: str) -> str:
    return stable_hash(
        {
            "prompt_version": prompt_version,
            "reasoning_effort": reasoning_effort,
            "instructions": TRANSLATION_INSTRUCTIONS,
            "schema": ItemCueOutput.model_json_schema(),
        }
    )


def item_input_hash(
    *,
    target: str,
    context: str,
    position: str,
    marked_context: str,
    main_pos: str,
    source_word_class: str,
    l1_names: Sequence[str],
) -> str:
    return stable_hash(
        {
            "target": target,
            "context": context,
            "position": position,
            "marked_context": marked_context,
            "main_pos": main_pos,
            "source_word_class": source_word_class,
            "l1_names": list(l1_names),
        }
    )


def render_translation_request(
    *,
    item_id: str,
    target: str,
    marked_context: str,
    main_pos: str,
    source_word_class: str,
    l1_names: Sequence[str],
) -> str:
    return (
        f"Item ID: {item_id}\n"
        f"Japanese target: {target}\n"
        f"Author main POS: {main_pos}\n"
        f"Source word class: {source_word_class}\n"
        "Japanese context with target marked:\n"
        f"{marked_context}\n"
        f"Requested L1s: {', '.join(l1_names)}"
    )


def validate_generated_record(
    record: GeneratedItem,
    *,
    item_id: str,
    generation_model: str,
    reasoning_effort: str,
    prompt_version: str,
    prompt_hash: str,
    input_hash: str,
) -> None:
    expected = {
        "id": item_id,
        "generation_model": generation_model,
        "reasoning_effort": reasoning_effort,
        "prompt_version": prompt_version,
        "prompt_hash": prompt_hash,
        "input_hash": input_hash,
    }
    mismatches = {
        field: {"expected": value, "returned": getattr(record, field)}
        for field, value in expected.items()
        if getattr(record, field) != value
    }
    if mismatches:
        raise ValueError(f"Cue client returned inconsistent generation provenance: {mismatches}")


class OpenAICueClient:
    def __init__(self, client=None):
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self.client = client

    def translate(
        self,
        *,
        item_id: str,
        target: str,
        marked_context: str,
        main_pos: str,
        source_word_class: str,
        l1_names: Sequence[str],
        model: str,
        reasoning_effort: str,
        prompt_version: str,
        prompt_hash: str,
        input_hash: str,
    ) -> GeneratedItem:
        response = self.client.responses.parse(
            model=model,
            reasoning={"effort": reasoning_effort},
            instructions=TRANSLATION_INSTRUCTIONS,
            input=render_translation_request(
                item_id=item_id,
                target=target,
                marked_context=marked_context,
                main_pos=main_pos,
                source_word_class=source_word_class,
                l1_names=l1_names,
            ),
            text_format=ItemCueOutput,
            store=False,
        )
        parsed = response.output_parsed
        if parsed is None:
            raise ValueError(f"OpenAI response {response.id} did not contain parsed cue output")
        validate_returned_languages(parsed.cues, l1_names)
        return GeneratedItem(
            id=item_id,
            input_hash=input_hash,
            prompt_hash=prompt_hash,
            generation_model=model,
            response_model=str(response.model),
            reasoning_effort=reasoning_effort,
            prompt_version=prompt_version,
            response_id=str(response.id),
            cues=parsed.cues,
        )


def load_checkpoint(path: Path) -> dict[str, GeneratedItem]:
    if not path.exists():
        return {}
    lines = path.read_text(encoding="utf-8").splitlines()
    records: dict[str, GeneratedItem] = {}
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            if index == len(lines) - 1:
                break
            raise ValueError(
                f"Malformed checkpoint JSON at line {index + 1} in {path}"
            ) from error
        try:
            record = GeneratedItem.model_validate(payload)
        except Exception as error:
            raise ValueError(
                f"Incompatible or stale checkpoint record at line {index + 1} in {path}"
            ) from error
        if record.id in records:
            raise ValueError(f"Duplicate item {record.id!r} in checkpoint {path}")
        records[record.id] = record
    return records


def append_checkpoint(path: Path, record: GeneratedItem) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(record.model_dump_json() + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def call_with_retries(
    operation: Callable[[], T],
    *,
    max_retries: int,
    retry_base_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
    is_retryable: Callable[[Exception], bool] | None = None,
    on_retry: Callable[[int, float, Exception], None] | None = None,
) -> T:
    if max_retries < 0:
        raise ValueError("max_retries cannot be negative")
    for attempt in range(max_retries + 1):
        try:
            return operation()
        except Exception as error:
            retryable = is_retryable(error) if is_retryable is not None else True
            if attempt == max_retries or not retryable:
                raise
            delay = retry_base_seconds * (2**attempt)
            if on_retry is not None:
                on_retry(attempt + 1, delay, error)
            sleep(delay)
    raise AssertionError("retry loop must return or raise")


def is_transient_api_error(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None)
    if status_code in {408, 409, 429} or (
        isinstance(status_code, int) and status_code >= 500
    ):
        return True
    return type(error).__name__ in {
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "InternalServerError",
    }


def select_pilot_items(items: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    if not 1 <= sample_size <= len(items):
        raise ValueError(f"sample_size must be between 1 and {len(items)}")
    work = items.copy()
    work["length_bucket"] = pd.cut(
        work["target"].str.len(),
        bins=[0, 2, 4, float("inf")],
        labels=["short", "medium", "long"],
    )
    work["pilot_stratum"] = (
        work["source_split"].astype(str)
        + "|"
        + work["word_origin"].astype(str)
        + "|"
        + work["expression_type"].astype(str)
        + "|"
        + work["source_word_class"].astype(str)
        + "|"
        + work["length_bucket"].astype(str)
    )
    rng = random.Random(seed)
    chosen: list[int] = []

    coverage_filters = (
        work["source_word_class"] == SourceWordClass.FUNCTION_WORD,
        work["expression_type"] == "mwe",
        work["source_split"] == "trial",
        work["source_split"] == "test",
    )
    for selector in coverage_filters:
        candidates = [index for index in work.index[selector] if index not in chosen]
        if candidates and len(chosen) < sample_size:
            chosen.append(rng.choice(candidates))
        if len(chosen) >= sample_size:
            return items.loc[chosen[:sample_size]].sort_values("id").reset_index(drop=True)

    strata = list(work.groupby("pilot_stratum", sort=True).groups)
    rng.shuffle(strata)
    for stratum in strata:
        candidates = [
            index
            for index in work.index[work["pilot_stratum"] == stratum]
            if index not in chosen
        ]
        if candidates:
            chosen.append(rng.choice(candidates))
        if len(chosen) >= sample_size:
            break
    if len(chosen) < sample_size:
        remaining = list(work.index.difference(chosen))
        rng.shuffle(remaining)
        chosen.extend(remaining[: sample_size - len(chosen)])
    return items.loc[chosen[:sample_size]].sort_values("id").reset_index(drop=True)


def select_items_by_id(items: pd.DataFrame, item_ids: Sequence[str]) -> pd.DataFrame:
    requested = list(item_ids)
    if not requested:
        raise ValueError("At least one --item-id is required")

    seen: set[str] = set()
    duplicates: list[str] = []
    for item_id in requested:
        if item_id in seen and item_id not in duplicates:
            duplicates.append(item_id)
        seen.add(item_id)
    if duplicates:
        raise ValueError(f"Duplicate --item-id values: {duplicates}")

    indexed = items.set_index("id")
    if not indexed.index.is_unique:
        raise ValueError("Source items contain duplicate IDs")
    unknown = [item_id for item_id in requested if item_id not in indexed.index]
    if unknown:
        raise ValueError(f"Unknown --item-id values: {unknown}")
    return indexed.loc[requested].reset_index()


def _checkpoint_record_matches(
    record: GeneratedItem,
    *,
    generation_model: str,
    reasoning_effort: str,
    prompt_hash: str,
    input_hash: str,
) -> bool:
    return (
        record.generation_model == generation_model
        and record.reasoning_effort == reasoning_effort
        and record.prompt_hash == prompt_hash
        and record.input_hash == input_hash
    )


def generate_item_cues(
    items: pd.DataFrame,
    *,
    original_l1s: Sequence[str],
    client: CueClient,
    generation_model: str,
    reasoning_effort: str,
    prompt_version: str,
    checkpoint_path: Path,
    max_retries: int = 4,
    retry_base_seconds: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
    show_progress: bool = False,
) -> list[GeneratedItem]:
    prompt_hash = augmentation_prompt_hash(prompt_version, reasoning_effort)
    checkpoint = load_checkpoint(checkpoint_path)
    records: list[GeneratedItem] = []
    api_calls = 0
    cache_hits = 0
    progress = tqdm(
        items.itertuples(index=False),
        total=len(items),
        desc="Generating MultiLS L1 cues",
        disable=not show_progress,
    )
    for row in progress:
        progress.set_postfix(item=row.id, api_calls=api_calls, cache_hits=cache_hits)
        l1_names = required_l1_names(row.source_split, original_l1s)
        main_pos = str(row.main_pos) if not pd.isna(row.main_pos) else "UNKNOWN"
        input_hash = item_input_hash(
            target=row.target,
            context=row.context,
            position=str(row.position),
            marked_context=row.marked_context,
            main_pos=main_pos,
            source_word_class=row.source_word_class,
            l1_names=l1_names,
        )
        existing = checkpoint.get(row.id)
        if existing is not None:
            if not _checkpoint_record_matches(
                existing,
                generation_model=generation_model,
                reasoning_effort=reasoning_effort,
                prompt_hash=prompt_hash,
                input_hash=input_hash,
            ):
                raise ValueError(
                    f"Checkpoint record for {row.id!r} was generated with stale model, "
                    "reasoning effort, prompt, or input data; use a new checkpoint or "
                    "remove it explicitly"
                )
            cache_hits += 1
            records.append(existing)
            continue

        def report_retry(attempt: int, delay: float, error: Exception) -> None:
            if show_progress:
                tqdm.write(
                    f"Retry {attempt}/{max_retries} for {row.id} in {delay:g}s "
                    f"after {type(error).__name__}: {error}"
                )

        record = call_with_retries(
            lambda: client.translate(
                item_id=row.id,
                target=row.target,
                marked_context=row.marked_context,
                main_pos=main_pos,
                source_word_class=row.source_word_class,
                l1_names=l1_names,
                model=generation_model,
                reasoning_effort=reasoning_effort,
                prompt_version=prompt_version,
                prompt_hash=prompt_hash,
                input_hash=input_hash,
            ),
            max_retries=max_retries,
            retry_base_seconds=retry_base_seconds,
            sleep=sleep,
            is_retryable=is_transient_api_error,
            on_retry=report_retry,
        )
        validate_generated_record(
            record,
            item_id=row.id,
            generation_model=generation_model,
            reasoning_effort=reasoning_effort,
            prompt_version=prompt_version,
            prompt_hash=prompt_hash,
            input_hash=input_hash,
        )
        validate_returned_languages(record.cues, l1_names)
        append_checkpoint(checkpoint_path, record)
        checkpoint[row.id] = record
        records.append(record)
        api_calls += 1
    progress.set_postfix(api_calls=api_calls, cache_hits=cache_hits)
    return records
