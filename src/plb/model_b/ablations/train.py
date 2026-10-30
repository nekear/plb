"""Train one ablation and persist it for the shared evaluation runner.

Usage: ``uv run -m plb.model_b.ablations.train <ablation_name>``

Each ablation is its own DVC stage, so the name arrives as a positional argument and
must be both registered in the code and configured in ``params.yaml``.
"""

from __future__ import annotations

import sys

from plb.model_b.ablations.registry import ABLATIONS, model_path
from plb.model_b.config import load_model_b_config
from plb.model_b.core.data import load_split
from plb.model_b.preparation.create_splits import EVAL, LOG_TARGET_COLUMN, TRAIN


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            f"Usage: {sys.argv[0]} <ablation_name>; choices: {sorted(ABLATIONS)}"
        )
    name = sys.argv[1]
    if name not in ABLATIONS:
        raise SystemExit(f"Unknown ablation {name!r}; registered: {sorted(ABLATIONS)}")

    config = load_model_b_config().ablations
    if name not in config.models:
        raise SystemExit(
            f"Ablation {name!r} has no params under model_b.ablations.models; "
            f"configured: {sorted(config.models)}"
        )

    # The model is constructed first so the loader knows whether to merge neighbors.
    model = ABLATIONS[name](config.models[name])
    load = dict(
        with_neighbors=model.requires_neighbors,
        neighbors_source=model.neighbors_source,
    )
    frame = load_split(TRAIN, **load)
    eval_frame = load_split(EVAL, **load)
    model.fit(
        frame,
        frame[LOG_TARGET_COLUMN].to_numpy(),
        eval_X=eval_frame,
        eval_y=eval_frame[LOG_TARGET_COLUMN].to_numpy(),
    )

    output_path = model_path(name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    print(f"Trained {name} on {len(frame):,} train rows -> {output_path}")


if __name__ == "__main__":
    raise SystemExit(main())
