from __future__ import annotations

import argparse
import json
from pathlib import Path

from plb.model_a.common.config import load_model_a_config
from plb.model_a.multils.augmentation import (
    OpenAICueClient,
    TRANSLATION_INSTRUCTIONS,
    generate_item_cues,
    render_translation_request,
    select_items_by_id,
    select_pilot_items,
)
from plb.model_a.multils.cues import (
    ItemCueOutput,
    build_quarantine,
    flatten_generated_items,
    required_l1_names,
    sort_sidecar,
    validate_complete_sidecar,
    write_tsv_atomic,
)
from plb.model_a.multils.data import (
    load_multils_items,
    load_original_profiles,
    original_l1_names,
)
from plb.utils import PROJECT_ROOT


def pilot_path(path: Path, label: str) -> Path:
    return path.with_name(f"{path.stem}.{label}{path.suffix}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate L1 cues for MultiLS-Japanese")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--quarantine", type=Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--sample-size", type=int)
    selection.add_argument(
        "--item-id",
        dest="item_ids",
        action="append",
        help="Select one exact item ID; repeat for a targeted pilot",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-base-seconds", type=float, default=1.0)
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the source-item progress bar (useful for CI and redirected logs)",
    )
    return parser


def main(argv: list[str] | None = None, *, client=None) -> int:
    args = build_parser().parse_args(argv)
    config = load_model_a_config()

    print(f"Using effort: {config.multils.augmentation.reasoning_effort}")

    data_dir = args.data_dir or config.multils_data_dir
    output = args.output or config.multils.cues_path
    if not output.is_absolute():
        output = PROJECT_ROOT / output

    items = load_multils_items(data_dir)
    profiles = load_original_profiles(data_dir)
    original_l1s = original_l1_names(profiles)
    pilot_label: str | None = None
    if args.sample_size is not None:
        items = select_pilot_items(items, args.sample_size, args.seed)
        pilot_label = "pilot"
    elif args.item_ids is not None:
        items = select_items_by_id(items, args.item_ids)
        pilot_label = "targeted_pilot"

    if pilot_label is not None:
        if args.output is None:
            output = pilot_path(output, pilot_label)
        elif output == (PROJECT_ROOT / config.multils.cues_path):
            raise ValueError("A pilot cannot write to the production cue sidecar")

    checkpoint = args.checkpoint or output.with_suffix(".checkpoint.jsonl")
    quarantine = args.quarantine or output.with_name(f"{output.stem}_quarantine.tsv")

    if args.dry_run:
        first = items.iloc[0]
        languages = required_l1_names(first["source_split"], original_l1s)
        print(
            json.dumps(
                {
                    "instructions": TRANSLATION_INSTRUCTIONS,
                    "request": render_translation_request(
                        item_id=first["id"],
                        target=first["target"],
                        marked_context=first["marked_context"],
                        main_pos=str(first["main_pos"]),
                        source_word_class=first["source_word_class"],
                        l1_names=languages,
                    ),
                    "schema": ItemCueOutput.model_json_schema(),
                    "model": config.multils.augmentation.model,
                    "reasoning_effort": config.multils.augmentation.reasoning_effort,
                    "prompt_version": config.multils.augmentation.prompt_version,
                    "selected_item_ids": items["id"].tolist(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    cue_client = client or OpenAICueClient()
    generated = generate_item_cues(
        items,
        original_l1s=original_l1s,
        client=cue_client,
        generation_model=config.multils.augmentation.model,
        reasoning_effort=config.multils.augmentation.reasoning_effort,
        prompt_version=config.multils.augmentation.prompt_version,
        checkpoint_path=checkpoint,
        max_retries=args.max_retries,
        retry_base_seconds=args.retry_base_seconds,
        show_progress=not args.no_progress,
    )
    sidecar = sort_sidecar(
        flatten_generated_items(items, generated), items, original_l1s
    )
    validate_complete_sidecar(sidecar, items, original_l1s)
    write_tsv_atomic(sidecar, output)
    write_tsv_atomic(build_quarantine(sidecar), quarantine)
    print(f"Saved {len(sidecar)} cue rows to {output}")
    print(f"Saved quarantine report to {quarantine}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
