import numpy as np
import pandas as pd

from plb.utils import write_json

from .config import load_tubelex_config


def main():
    config = load_tubelex_config()

    df_ja_lemma_pos = pd.read_csv(
        config.get_frequencies_file_path(), sep="\t", header=0
    )
    df_multils_test_split = pd.read_csv(
        config.get_multils_split_file_path("test"), sep="\t", header=0
    )

    df_merged = pd.merge(
        df_ja_lemma_pos,
        df_multils_test_split,
        left_on="word",
        right_on="target_lemma",
        how="right",
    )

    annotators_cols = [f"lcp_annotator_{i + 1}" for i in range(0, 10)]

    df_minimized = df_merged[
        ["word", "target_lemma", "count", "complexity", *annotators_cols]
    ]

    df_minimized_filled_only = df_minimized[df_minimized["word"].notna()].copy()

    df_minimized_filled_only["log_frequency"] = np.log10(
        df_minimized_filled_only["count"] + 1
    )

    metrics = {
        "meta": {
            "file_name": config.file_name,
            "processed_items": df_minimized_filled_only.shape[0],
            "all_items": df_merged.shape[0],
        },
        "group_group_corr": df_minimized_filled_only[["log_frequency", "complexity"]]
        .corr()
        .iloc[0, 1],
        "indiv_group_corr": df_minimized_filled_only[annotators_cols]
        .corrwith(df_minimized_filled_only["count"])
        .mean(),
    }

    config.evaluations_dir.mkdir(parents=True, exist_ok=True)
    write_json(config.evaluations_dir / "metrics.json", metrics)


if __name__ == "__main__":
    raise SystemExit(main())
