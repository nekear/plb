"""Evaluate every persisted ablation on the held-out test split.

Loads the models trained by ``plb.model_b.ablations.train`` (one DVC stage each),
scores them per learner with paired-bootstrap confidence intervals, and writes one
``metrics.json`` plus a ``predictions.csv`` for downstream analysis.

The eval split is deliberately untouched here: it is reserved for hyperparameter
tuning (e.g. the recent-median ``k``); the test split is the reporting target.
"""

from __future__ import annotations


from plb.learners import get_learner_code
from plb.model_b.ablations.registry import ABLATIONS, EVALUATION_DIR, model_path
from plb.model_b.config import load_model_b_config
from plb.model_b.core.data import load_split
from plb.model_b.core.evaluate import evaluate_models
from plb.model_b.preparation.create_splits import TEST
from plb.utils import write_json

PREDICTION_COLUMNS = [
    "user_id",
    "learner",
    "word_instance_id",
    "is_shared_test",
    "plb_14",
    "log_plb_14",
]


def main() -> None:
    config = load_model_b_config().ablations
    models = {name: ABLATIONS[name].load(model_path(name)) for name in config.models}

    # Checking if any models need neighbors
    with_neighbors = any(model.requires_neighbors for model in models.values())

    # Collecting the neighbor sources ("similar" or "random") models may reference
    sources = {
        model.neighbors_source if model.requires_neighbors else "similar"
        for model in models.values()
    }

    # Loading TEST splits for different neighbor sources
    frames = {
        source: load_split(TEST, with_neighbors=with_neighbors, neighbors_source=source)
        for source in sources
    }

    # Computing predictions for every model and storing by name
    predictions = {
        name: model.predict(
            frames[model.neighbors_source if model.requires_neighbors else "similar"]
        )
        for name, model in models.items()
    }

    # One frame per neighbor source (at most two, given "random" and "similar" sources): every model predicts on the frame
    # carrying its own neighbors. Row order is identical because both merge onto the same split file, so any of them serves for the metrics (since neighbors don't matter there).
    frame = frames["similar"] if "similar" in frames else next(iter(frames.values()))

    per_model, pairwise = evaluate_models(
        frame,
        predictions,
        baseline=config.baseline,
        delta_pairs=config.delta_pairs,
        bootstrap_samples=config.bootstrap_samples,
        bootstrap_seed=config.bootstrap_seed,
    )

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    write_json(
        EVALUATION_DIR / "metrics.json",
        {
            "test_rows": len(frame),
            "bootstrap": {
                "samples": config.bootstrap_samples,
                "seed": config.bootstrap_seed,
            },
            "baseline": config.baseline,
            "models": per_model,
            "pairwise": pairwise,
        },
    )

    output = frame.copy()
    output["learner"] = output["user_id"].map(get_learner_code)
    for name, values in predictions.items():
        output[f"pred_{name}"] = values
    prediction_columns = PREDICTION_COLUMNS + [f"pred_{name}" for name in predictions]
    output[prediction_columns].to_csv(EVALUATION_DIR / "predictions.csv", index=False)

    _print_summary(per_model, pairwise)


def _print_summary(per_model: dict, pairwise: dict) -> None:
    def _fmt(entry: dict) -> str:
        value = entry["value"]
        return "  null" if value is None else f"{value:6.3f}"

    print(f"\n{'model':<28} {'learner':<8} {'spearman':>8} {'log_mae':>8} {'rer':>7}")
    for name, learners in per_model.items():
        for code, metrics in learners.items():
            print(
                f"{name:<28} {code:<8} {_fmt(metrics['spearman']):>8} "
                f"{_fmt(metrics['log_mae']):>8} {_fmt(metrics['rer']):>7}"
            )

    if pairwise:
        print(f"\n{'pair':<50} {'learner':<8} {'delta_rer':>9}")
        for pair, learners in pairwise.items():
            for code, metrics in learners.items():
                print(f"{pair:<50} {code:<8} {_fmt(metrics['delta_rer']):>9}")


if __name__ == "__main__":
    raise SystemExit(main())
