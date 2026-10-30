import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # PLB analysis
    ***
    > Question: should I residualise the PLB targets taking inspiration from Koren's baseline predictors and helping the models to reduce the prediction scale, or not?

    `L01`–`L06` are arbitrary stable learner codes; they encode neither
    learner characteristics nor outcome ordering.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1. Calculating the median PLB, and censored/uncesored  coverage to understand how much learners fluctuate
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.ticker import MultipleLocator
    from IPython.display import display

    from plb.learners import (
        LEARNER_CODE_ORDER,
        LEARNER_COLOR_BY_CODE,
        get_learner_code,
        validate_learner_ids,
    )
    from plb.utils import get_dataset_as_path

    users_df = pd.read_json(get_dataset_as_path("interim/users.jsonl"), lines=True)
    decks_df = pd.read_json(get_dataset_as_path("interim/decks.jsonl"), lines=True)
    lexemes_df = pd.read_json(get_dataset_as_path("interim/lexemes.jsonl"), lines=True)
    word_instances_df = pd.read_json(get_dataset_as_path("interim/word_instances.jsonl"), lines=True)
    clean_reviews_w_fsrs_df = pd.read_json(get_dataset_as_path("processed/clean_reviews_w_fsrs.jsonl"), lines=True)

    validate_learner_ids(users_df["user_id"])
    _present_learner_codes = {
        get_learner_code(_user_id) for _user_id in users_df["user_id"]
    }
    learner_code_order = [
        _learner_code
        for _learner_code in LEARNER_CODE_ORDER
        if _learner_code in _present_learner_codes
    ]
    learner_color_by_code = {
        _learner_code: LEARNER_COLOR_BY_CODE[_learner_code]
        for _learner_code in learner_code_order
    }

    display(lexemes_df["language"].value_counts())

    _wis_enriched = word_instances_df.merge(
        decks_df[["deck_id", "user_id"]],
        on="deck_id",
        how="left",
        validate="many_to_one",
    )

    _cr_enriched = clean_reviews_w_fsrs_df.copy().merge(
        _wis_enriched[["word_instance_id", "lexeme_id", "user_id"]],
        on="word_instance_id",
        how="left",
        validate="many_to_one",
    )

    _cr_enriched["learner_code"] = _cr_enriched["user_id"].map(
        get_learner_code
    )

    _cr_enriched.head()


    plt.figure(figsize=(18, 8))
    _sp_ax = plt.gca()

    _ax = sns.histplot(
        data=_cr_enriched[_cr_enriched["fsrs_stability"] < 60],
        x="fsrs_stability",
        hue="learner_code",
        hue_order=learner_code_order,
        palette=learner_color_by_code,
        bins=30,
        ax=_sp_ax,
        multiple="dodge",
        shrink=.8,
    )

    _ax.xaxis.get_major_locator()
    _ax.xaxis.get_major_formatter()

    _ax.xaxis.set_major_locator(MultipleLocator(5))

    plt.show()

    print(f"Minimum stability: {np.min(_cr_enriched['fsrs_stability'])}")
    print(f"Maximum stability: {np.max(_cr_enriched['fsrs_stability'])}")

    def get_wis_with_targets(tsh):
        _reviews = _cr_enriched.sort_values(["word_instance_id", "timestamp_ms", "review_id"], kind="stable").copy()

        _reviews["review_number"] = _reviews.groupby("word_instance_id").cumcount() + 1

        _first_target_reviews = (
            _reviews.loc[
                _reviews["fsrs_stability"] >= tsh,
                ["word_instance_id", "review_number"]
            ].groupby("word_instance_id", as_index=False)["review_number"]
            .agg(reviews_until_stability_tsh="min")
        )

        _review_counts = (
            _reviews.groupby("word_instance_id", as_index=False)
            .size()
            .rename(columns={"size": "current_review_count"})
        )

        _lapses_counts = (
            _reviews.assign(is_lapse=_reviews["rating"] == 1)
            .groupby("word_instance_id", as_index=False)
            .agg(lapses_count=("is_lapse", "sum"))
        )

        _wis_enriched_with_targets = _wis_enriched.merge(
             _wis_enriched
            .merge(
                _review_counts,
                on="word_instance_id",
                how="left",
                validate="one_to_one",
            )
             .merge(
                 _lapses_counts,
                on="word_instance_id",
                how="left",
                validate="one_to_one",
             )
             .loc[lambda df: df["current_review_count"].fillna(0).gt(0)]
            .merge(
                _first_target_reviews[
                    ["word_instance_id", "reviews_until_stability_tsh"]
                ],
                on="word_instance_id",
                how="left",
                validate="one_to_one",
            )
        )

        _wis_enriched_with_targets["learner_code"] = (
            _wis_enriched_with_targets["user_id"].map(get_learner_code)
        )

        return _wis_enriched_with_targets

    plb_thresholds = [14, 30, 60]

    for i, tsh in enumerate(plb_thresholds):
        _wis_enriched_with_targets = get_wis_with_targets(tsh)

        def _get_iqr(s: pd.Series) -> str:
            Q1 = s.quantile(0.25)
            Q3 = s.quantile(0.75)
            IQR = Q3-Q1

            return f"Q1: {Q1}, Q3: {Q3}, IQR: {IQR}"

        summary = (
            _wis_enriched_with_targets
            .groupby("learner_code")
            .agg(
                median_plb=("reviews_until_stability_tsh", "median"),
                started_uncensored=("reviews_until_stability_tsh", "count"),
                started_censored=("reviews_until_stability_tsh", lambda s: s.isna().sum()),
                total=("reviews_until_stability_tsh", "size"),
                censored_share=("reviews_until_stability_tsh", lambda s: s.isna().mean() * 100),
                iqr=("reviews_until_stability_tsh", _get_iqr)
            )
        )

        display(f"#{i+1} - Threshold: {tsh}")
        display(
            summary.style.format({
                "median_plb": "{:.1f}",
                "censored_share": "{:.2f}%"
            })
        )

        sns.boxplot(
            _wis_enriched_with_targets,
            x="learner_code",
            y="reviews_until_stability_tsh",
            hue="learner_code",
            order=learner_code_order,
            hue_order=learner_code_order,
            palette=learner_color_by_code,
        )
        plt.title(f"#{i+1} - Threshold: {tsh}")
        plt.show()
    return (
        clean_reviews_w_fsrs_df,
        get_dataset_as_path,
        get_learner_code,
        get_wis_with_targets,
        learner_code_order,
        learner_color_by_code,
        np,
        pd,
        plt,
        sns,
        users_df,
    )


@app.cell
def _(get_learner_code, mo, users_df):
    _unique_user_ids = users_df["user_id"].unique().tolist()
    _learner_options = {
        get_learner_code(_user_id): _user_id
        for _user_id in _unique_user_ids
    }

    user_selector = mo.ui.multiselect(
        options=_learner_options,
        value=list(_learner_options),
        label="Select learners to show:"
    )
    return (user_selector,)


@app.cell
def _(user_selector):
    user_selector
    return


@app.cell
def _(
    clean_reviews_w_fsrs_df,
    get_dataset_as_path,
    get_wis_with_targets,
    learner_code_order,
    learner_color_by_code,
    mo,
    np,
    pd,
    plt,
    sns,
    user_selector,
):
    import seaborn.objects as so

    mo.stop(
        not user_selector.value,
        mo.md("Select at least one learner to display the analysis."),
    )

    _wis_enriched_with_targets = get_wis_with_targets(14)

    _wis_enriched_with_targets = _wis_enriched_with_targets[_wis_enriched_with_targets["user_id"].isin(user_selector.value)]

    _unique_wis = _wis_enriched_with_targets['word_instance_id'].unique()

    _reviews_dirty = pd.read_json(get_dataset_as_path("interim/reviews.jsonl"), lines=True)

    print(f"Unique lexeme forms: {_wis_enriched_with_targets['lexeme_id'].nunique()}")
    print(f"Unique word instances: {len(_unique_wis)}")
    print(f"Unique reviews: {len(_reviews_dirty[_reviews_dirty['word_instance_id'].isin(_unique_wis)].index)}")
    print(f"Clean unique reviews: {len(clean_reviews_w_fsrs_df[clean_reviews_w_fsrs_df['word_instance_id'].isin(_unique_wis)].index)}")

    _selected_learner_codes = set(
        _wis_enriched_with_targets["learner_code"]
    )
    selected_learner_code_order = [
        _learner_code
        for _learner_code in learner_code_order
        if _learner_code in _selected_learner_codes
    ]
    selected_learner_color_by_code = {
        _learner_code: learner_color_by_code[_learner_code]
        for _learner_code in selected_learner_code_order
    }

    _wis_enriched_with_targets["learner_median"] = (
        _wis_enriched_with_targets
        .groupby("learner_code")["reviews_until_stability_tsh"]
        .transform("median")
    )

    _wis_enriched_with_targets["residual"] = (
        _wis_enriched_with_targets["reviews_until_stability_tsh"]
        - _wis_enriched_with_targets["learner_median"]
    )

    _plot_df = _wis_enriched_with_targets.melt(
        id_vars="learner_code",
        value_vars=["reviews_until_stability_tsh", "residual"],
        var_name="distribution",
        value_name="value",
    )

    _plot_df["distribution"] = _plot_df["distribution"].replace({
        "reviews_until_stability_tsh": "Before",
        "residual": "After",
    })

    _fig, _ax = plt.subplots(figsize=(14, 6))

    sns.violinplot(
        data=_plot_df,
        x="learner_code",
        y="value",
        hue="distribution",
        order=selected_learner_code_order,
        split=True,
        inner="quart",
        cut=0,
        ax=_ax,
    )

    _ax.set_ylim(
        _plot_df["value"].quantile(0.01),
        _plot_df["value"].quantile(0.99),
    )

    plt.tight_layout()
    plt.show()

    def get_other_learners_median(group):
        # For each row, filtering out rows belonging to the same learner.
        full_medians = []
        residual_medians = []
        for uid in group["user_id"]:
            other_vals = group[group["user_id"] != uid][["reviews_until_stability_tsh", "residual"]]
            full_medians.append(other_vals["reviews_until_stability_tsh"].median() if not other_vals.empty else np.nan)
            residual_medians.append(other_vals["residual"].median() if not other_vals.empty else np.nan)
        group["other_learners_median_full"] = full_medians
        group["other_learners_median_residual"] = residual_medians
        return group

    _wis_enriched_with_targets["is_shared"] = _wis_enriched_with_targets.groupby("lexeme_id")["user_id"].transform("nunique") > 1

    wis_shared_only = _wis_enriched_with_targets[_wis_enriched_with_targets["is_shared"]]
    wis_shared_only = wis_shared_only.groupby(
        "lexeme_id", group_keys=False
    ).apply(get_other_learners_median)
    _peer_learner_codes = set(wis_shared_only["learner_code"])
    peer_learner_code_order = [
        _learner_code
        for _learner_code in selected_learner_code_order
        if _learner_code in _peer_learner_codes
    ]
    peer_learner_color_by_code = {
        _learner_code: selected_learner_color_by_code[_learner_code]
        for _learner_code in peer_learner_code_order
    }
    return peer_learner_code_order, peer_learner_color_by_code, wis_shared_only


@app.cell
def _(
    mo,
    peer_learner_code_order,
    peer_learner_color_by_code,
    plt,
    sns,
    wis_shared_only,
):
    mo.stop(
        wis_shared_only.empty,
        mo.md(
            "Select at least two learners with shared lexemes to display "
            "the peer-comparison charts."
        ),
    )

    _fig, _axes = plt.subplots(1, 2, figsize=(15, 6))

    sns.scatterplot(
        data=wis_shared_only,
        x="reviews_until_stability_tsh",
        y="other_learners_median_full",
        hue="learner_code",
        hue_order=peer_learner_code_order,
        palette=peer_learner_color_by_code,
        alpha=0.2,
        edgecolor=None,
        ax=_axes[0]
    )

    _max_full = max(
        wis_shared_only["reviews_until_stability_tsh"].max(), 
        wis_shared_only["other_learners_median_full"].max()
    )
    _axes[0].plot([0, _max_full], [0, _max_full], color="black", linestyle="--", label="1:1 Ratio")
    _axes[0].set_title("Before: Learner PLB vs peer median PLB on shared words only")
    _axes[0].set_xlabel("Learner PLB (full, no residual)")
    _axes[0].set_ylabel("Peer Median PLB (Full, no residual)")

    sns.move_legend(_axes[0], "upper left", bbox_to_anchor=(1.02, 1), title="Learner")

    sns.scatterplot(
        data=wis_shared_only,
        x="residual",
        y="other_learners_median_residual",
        hue="learner_code",
        hue_order=peer_learner_code_order,
        palette=peer_learner_color_by_code,
        alpha=0.2,
        edgecolor=None,
        legend=False,
        ax=_axes[1]
    )

    _min_res = min(
        wis_shared_only["residual"].min(), 
        wis_shared_only["other_learners_median_residual"].min()
    )
    _max_res = max(
        wis_shared_only["residual"].max(), 
        wis_shared_only["other_learners_median_residual"].max()
    )
    _axes[1].plot([_min_res, _max_res], [_min_res, _max_res], color="black", linestyle="--")
    _axes[1].set_title("After: Residuals vs Peer Residual Median")
    _axes[1].set_xlabel("Learner residual")
    _axes[1].set_ylabel("Peer Median (Residual)")

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo, plt, sns, wis_shared_only):
    mo.stop(
        wis_shared_only.empty,
        mo.md(
            "Select at least two learners with shared lexemes to display "
            "the peer-difference distribution."
        ),
    )

    wis_shared_only["full_difference"] = wis_shared_only["reviews_until_stability_tsh"] - wis_shared_only["other_learners_median_full"]
    wis_shared_only["residual_difference"] = wis_shared_only["residual"] - wis_shared_only["other_learners_median_residual"]

    plt.figure(figsize=(10, 5))

    sns.kdeplot(data=wis_shared_only, x="full_difference", fill=True, label="Before (Full Difference)", color="red", alpha=0.4)
    sns.kdeplot(data=wis_shared_only, x="residual_difference", fill=True, label="After (Residual Difference)", color="blue", alpha=0.4)

    plt.axvline(0, color="black", linestyle="--")
    plt.title("Difference Between Learner and Peer Median (Before vs After Normalization)")
    plt.xlabel("Difference (Learner - Peers)")
    plt.ylabel("Density")
    plt.legend()
    plt.xlim(-50, 50)

    plt.show()
    return


@app.cell
def _(get_wis_with_targets, np, pd):
    _wis_enriched_with_targets = get_wis_with_targets(14)

    print("[PCC] PLB / lapses: ",  _wis_enriched_with_targets["reviews_until_stability_tsh"].corr(_wis_enriched_with_targets["lapses_count"]))
    print("[PCC] log PLB / lapses: ",np.log(_wis_enriched_with_targets["reviews_until_stability_tsh"]).corr(_wis_enriched_with_targets["lapses_count"]))

    _clean_df = _wis_enriched_with_targets.dropna(subset=["lapses_count", "reviews_until_stability_tsh"])
    _plb_decomposition = np.polyfit(_clean_df["lapses_count"], _clean_df["reviews_until_stability_tsh"], 1)

    print(f"PLB ~~ {_plb_decomposition[0]:.3f} * lapses + {_plb_decomposition[1]:.3f} + noise")

    pd.crosstab((_wis_enriched_with_targets["lapses_count"] == 0),
                _wis_enriched_with_targets["reviews_until_stability_tsh"].isna())
    return


if __name__ == "__main__":
    app.run()
