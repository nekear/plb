"""Run the whole similarity pipeline: score every pair, then select neighbours.

The two halves are separate stages because only selection depends on the percentile
thresholds — see `score_pairs` and `select_neighbors`. This entry point exists so a
one-shot `uv run -m plb.similarities` still does the whole job; DVC invokes the two
stages individually so retuning thresholds skips the expensive one.
"""

from plb.similarities import score_pairs, select_neighbors


def main() -> None:
    score_pairs.main()
    select_neighbors.main()


if __name__ == "__main__":
    main()
