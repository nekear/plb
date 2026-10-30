import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Is E5 anisotropy inflating the semantic similarities?

    The issue is that `compute_semantic_similarity` is a plain cosine over the direct E5 embeddings of
    each word instance's `meaning`. If the embedding space is anisotropic, every
    vector sits in a narrow cone, so even unrelated meanings score high and the
    `similarities.thresholds.semantic` cut stops separating anything.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from plb.embeddings.direct.config import load_direct_embeddings_config
    from plb.utils import PROJECT_ROOT, get_dataset_as_path

    return (
        PROJECT_ROOT,
        get_dataset_as_path,
        load_direct_embeddings_config,
        np,
        pd,
        plt,
    )


@app.cell
def _(
    PROJECT_ROOT,
    get_dataset_as_path,
    load_direct_embeddings_config,
    np,
    pd,
):
    _config = load_direct_embeddings_config()
    # params.yaml stores this relative to the repo root, but a notebook's working
    # directory is wherever marimo was started.
    _embeddings_path = _config.output_path
    if not _embeddings_path.is_absolute():
        _embeddings_path = PROJECT_ROOT / _embeddings_path

    _embeddings = pd.read_json(_embeddings_path, lines=True)
    _word_instances = pd.read_json(
        get_dataset_as_path("interim/word_instances.jsonl"), lines=True
    )
    _lexemes = pd.read_json(get_dataset_as_path("interim/lexemes.jsonl"), lines=True)

    words = (
        _word_instances[["word_instance_id", "lexeme_id", "meaning"]]
        .merge(_lexemes[["lexeme_id", "form", "reading"]], on="lexeme_id")
        .merge(_embeddings, on="word_instance_id")
        .reset_index(drop=True)
    )

    # Already L2-normalised by the embedding stage, so cosine is a plain dot product here
    vectors = np.asarray(words.pop("embedding").tolist(), dtype=np.float32)

    # Performing the mean-centering
    centered = vectors - vectors.mean(axis=0)
    centered /= np.linalg.norm(centered, axis=1, keepdims=True)

    embedding_coverage = (
        f"{len(words):,} word instances embedded "
        f"({len(_embeddings):,} rows in the embeddings file, "
        f"{len(_word_instances):,} word instances on disk)"
    )
    return centered, embedding_coverage, vectors, words


@app.cell
def _(embedding_coverage, mo):
    mo.md(f"""
    ## Baseline: how similar are two random word instances?

    {embedding_coverage}
    """)
    return


@app.cell
def _(centered, np, pd, vectors):
    _rng = np.random.default_rng(0)
    _left = _rng.integers(0, len(vectors), 50_000)
    _right = _rng.integers(0, len(vectors), 50_000)
    _distinct = _left != _right
    left_index, right_index = _left[_distinct], _right[_distinct]

    random_pairs = pd.DataFrame(
        {
            "raw_cosine": np.einsum(
                "ij,ij->i", vectors[left_index], vectors[right_index]
            ),
            "centered_cosine": np.einsum(
                "ij,ij->i", centered[left_index], centered[right_index]
            ),
        }
    )
    random_pairs.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).round(3)
    return left_index, random_pairs, right_index


@app.cell
def _(mo, random_pairs):
    _above = (random_pairs["raw_cosine"] >= 0.90).mean()
    mo.md(f"""
    Random pairs average **{random_pairs["raw_cosine"].mean():.3f}** raw cosine, and
    **{_above:.2%}** of them clear the current `semantic: 0.90` threshold.
    After centering they average
    **{random_pairs["centered_cosine"].mean():.3f}**.
    """)
    return


@app.cell
def _(plt, random_pairs):
    _figure, _axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
    for _axis, _column in zip(_axes, ["raw_cosine", "centered_cosine"]):
        _axis.hist(random_pairs[_column], bins=80)
        _axis.axvline(0.90, color="crimson", linestyle="--", linewidth=1)
        _axis.set_title(_column)
        _axis.set_xlabel("cosine similarity")
        _axis.set_xlim(-1, 1)
    _axes[0].set_ylabel("random pairs")
    _figure.tight_layout()
    _figure
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Visualization of the pairs
    """)
    return


@app.cell
def _(left_index, pd, random_pairs, right_index, words):
    _sample = random_pairs.head(25)
    random_pair_examples = pd.DataFrame(
        {
            "form_a": words["form"].to_numpy()[left_index[:25]],
            "meaning_a": words["meaning"].to_numpy()[left_index[:25]].astype(str),
            "form_b": words["form"].to_numpy()[right_index[:25]],
            "meaning_b": words["meaning"].to_numpy()[right_index[:25]].astype(str),
            "raw_cosine": _sample["raw_cosine"].round(3).to_numpy(),
            "centered_cosine": _sample["centered_cosine"].round(3).to_numpy(),
        }
    )
    random_pair_examples
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Nearest neighbours of one word

    Search for a word, pick a row, and compare its neighbours under both scorings.
    """)
    return


@app.cell
def _(centered, np, pd, vectors, words):
    def neighbours_of(index: int, k: int = 15) -> pd.DataFrame:
        """Top-k most similar word instances to `index`, by raw cosine."""
        raw = vectors @ vectors[index]
        adjusted = centered @ centered[index]
        order = np.argsort(-raw)[: k + 1]
        order = order[order != index][:k]
        return pd.DataFrame(
            {
                "form": words["form"].to_numpy()[order],
                "meaning": words["meaning"].to_numpy()[order].astype(str),
                "raw_cosine": raw[order].round(3),
                "centered_cosine": adjusted[order].round(3),
            }
        )

    return (neighbours_of,)


@app.cell
def _(mo):
    query = mo.ui.text(value="cat", label="Search form or meaning")
    query
    return (query,)


@app.cell
def _(query, words):
    _mask = words["form"].str.contains(
        query.value, case=False, na=False, regex=False
    ) | words["meaning"].str.contains(query.value, case=False, na=False, regex=False)
    # The index is each word's row in `words`, which is also its row in `vectors`.
    matches = words.loc[_mask, ["form", "reading", "meaning"]].head(50)
    matches
    return (matches,)


@app.cell
def _(matches, mo):
    _options = {
        f"{_row.form} — {str(_row.meaning)[:60]}": _index
        for _index, _row in matches.iterrows()
    }
    chosen = mo.ui.dropdown(
        options=_options,
        value=next(iter(_options), None),  # first match, so the cell below is never blank
        label="Word instance",
    )
    chosen
    return (chosen,)


@app.cell
def _(chosen, mo, neighbours_of):
    _view = (
        mo.md("Nothing selected — search above.")
        if chosen.value is None
        else neighbours_of(int(chosen.value))
    )
    _view
    return


if __name__ == "__main__":
    app.run()
