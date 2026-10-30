import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # Model B exploratory analysis
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    from plb.learners import (
        LEARNER_CODE_ORDER as _LEARNER_CODE_ORDER,
        get_learner_code as _get_learner_code,
        validate_learner_ids as _validate_learner_ids,
    )
    from plb.utils import get_dataset_as_path

    df_lexemes = pd.read_json(
        get_dataset_as_path("interim/lexemes.jsonl"),
        lines=True,
    )
    all_word_instances_df = pd.read_json(
        get_dataset_as_path(
            "processed/word_instances_w_stability_targets.jsonl"
        ),
        lines=True,
    )
    # Loading the TubeLex corpus counts, since they are the non-personalized
    # difficulty signal we want to test against the observed review burden.
    df_tubelex_ja = pd.read_csv(
        get_dataset_as_path("raw/tubelex/ja_full.tsv"), sep="\t", header=0
    )
    # Loading deck metadata here rather than further down, because deck
    # ownership is what maps a card to a learner and is therefore needed by
    # every per-learner analysis in the notebook, starting with the colour map.
    all_decks_df = pd.read_json(
        get_dataset_as_path("interim/decks.jsonl"),
        lines=True,
    )

    _all_japanese_user_ids = all_decks_df.loc[
        all_decks_df["target_language"].eq("ja"), "user_id"
    ].unique()
    _validate_learner_ids(_all_japanese_user_ids)
    _user_id_by_learner_code = {
        _get_learner_code(_user_id): _user_id
        for _user_id in _all_japanese_user_ids
    }
    learner_selector_options = {
        _learner_code: _user_id_by_learner_code[_learner_code]
        for _learner_code in _LEARNER_CODE_ORDER
        if _learner_code in _user_id_by_learner_code
    }
    return (
        all_decks_df,
        all_word_instances_df,
        df_lexemes,
        df_tubelex_ja,
        get_dataset_as_path,
        learner_selector_options,
        np,
        pd,
    )


@app.cell
def _(learner_selector_options, mo):
    learner_selector = mo.ui.multiselect(
        options=learner_selector_options,
        value=list(learner_selector_options),
        label="Select learners to include:",
    )
    return (learner_selector,)


@app.cell
def _(learner_selector):
    learner_selector
    return


@app.cell
def _(all_decks_df, all_word_instances_df, learner_selector, mo):
    mo.stop(
        not learner_selector.value,
        mo.md("Select at least one learner to run the analysis."),
    )

    # This is the notebook's filtering boundary. Every downstream table,
    # simulation, statistic, and chart receives only these selected learners.
    decks_df = all_decks_df[
        all_decks_df["target_language"].eq("ja")
        & all_decks_df["user_id"].isin(learner_selector.value)
    ].copy()
    _selected_deck_ids = set(decks_df["deck_id"])
    df_word_instances = all_word_instances_df[
        all_word_instances_df["deck_id"].isin(_selected_deck_ids)
    ].copy()
    return decks_df, df_word_instances


@app.cell
def _(mo):
    mo.md("""
    # Analyzing correlations
    """)
    return


@app.cell
def _(df_lexemes, df_tubelex_ja, df_word_instances, np, pd):
    # Attaching surface forms, because TubeLex is keyed by written form while
    # word instances are keyed by lexeme id.
    _word_instances_w_frequency = pd.merge(
        df_word_instances,
        df_lexemes[["lexeme_id", "form", "reading"]],
        on="lexeme_id",
        how="left",
    )

    _word_instances_w_frequency = pd.merge(
        _word_instances_w_frequency,
        df_tubelex_ja[["word", "count"]],
        left_on="form",
        right_on="word",
        how="left",
    )

    # Log-transforming both sides, because corpus frequency is Zipf-distributed
    # and review counts are right-skewed; correlating them raw would let a
    # handful of extreme values dominate.
    _word_instances_w_frequency["log_tubelex_frequency"] = np.log1p(
        _word_instances_w_frequency["count"]
    )

    _word_instances_w_frequency["log_plb_14"] = np.log1p(
        _word_instances_w_frequency["plb_14"]
    )

    # Reporting coverage first, so a weak correlation cannot be silently caused
    # by most lexemes simply missing a frequency value.
    _empty_frequencies = _word_instances_w_frequency[
        "log_tubelex_frequency"
    ].isna().sum()
    print(f"Empty log frequencies: {_empty_frequencies}")

    _covered_items = _word_instances_w_frequency["count"].notna().sum()
    print(
        f"Coverage: {_covered_items}/"
        f"{len(_word_instances_w_frequency.index)}/"
        f"{len(df_word_instances.index)}"
    )

    _correlation_x = _word_instances_w_frequency["log_tubelex_frequency"]
    _correlation_y = _word_instances_w_frequency[
        "log_plb_14"
    ]

    print(
        "Pearson correlation:  "
        f"{_correlation_x.corr(_correlation_y, method='pearson'):.4f}"
    )
    print(
        "Spearman correlation: "
        f"{_correlation_x.corr(_correlation_y, method='spearman'):.4f}"
    )
    return


@app.cell
def _(mo):
    mo.md("""
    # Analyzing proportions of shared words
    """)
    return


@app.cell
def _(decks_df, df_lexemes, get_dataset_as_path, pd):
    import matplotlib.pyplot as plt
    import seaborn as sns

    reviews_df = pd.read_json(
        get_dataset_as_path("interim/reviews.jsonl"),
        lines=True,
    )
    # Reading the pre-cleaning word instances, because overlap between learners
    # is a property of what they chose to study, not of which reviews survived
    # the cleaning stage.
    _original_word_instances = pd.read_json(
        get_dataset_as_path("interim/word_instances.jsonl"),
        lines=True,
    )

    # Restricting the whole chain to Japanese, since the rest of the notebook
    # only ever looks at the Japanese learners; counting the other languages
    # here would inflate every headline number reported below.
    _japanese_lexeme_ids = df_lexemes.loc[
        df_lexemes["language"] == "ja", "lexeme_id"
    ]
    _original_word_instances = _original_word_instances[
        _original_word_instances["lexeme_id"].isin(_japanese_lexeme_ids)
        & _original_word_instances["deck_id"].isin(decks_df["deck_id"])
    ].copy()
    reviews_df = reviews_df[
        reviews_df["word_instance_id"].isin(
            _original_word_instances["word_instance_id"]
        )
    ].copy()

    # Count distinct selected learners, not cards: one learner may hold the
    # same lexeme in multiple decks and must still contribute exactly once.
    _word_instances_with_owners = _original_word_instances.merge(
        decks_df[["deck_id", "user_id"]],
        on="deck_id",
        how="inner",
        validate="many_to_one",
    )
    shared_lexemes = _word_instances_with_owners.groupby("lexeme_id")[
        "user_id"
    ].nunique()

    _shared_lexemes_count = (
        shared_lexemes.value_counts()
        .rename_axis("learners")
        .rename("lexemes")
        .reset_index()
    )

    plt.figure(figsize=(10, 6))
    _sharing_ax = sns.barplot(
        _shared_lexemes_count,
        x="learners",
        y="lexemes",
        hue="learners",
        palette="colorblind",
        legend=False,
    )
    for _container in _sharing_ax.containers:
        _sharing_ax.bar_label(_container)
    _sharing_ax.set(
        xlabel="Learners per lexeme", ylabel="Number of lexemes shared"
    )
    plt.show()

    print(f"Lexemes: {len(_japanese_lexeme_ids.index)}")
    print(f"Word instances: {len(_original_word_instances.index)}")
    print(f"Reviews: {len(reviews_df.index)}")
    return plt, reviews_df, shared_lexemes, sns


@app.cell
def _(decks_df):
    from plb.learners import (
        LEARNER_CODE_ORDER,
        LEARNER_COLOR_BY_CODE,
        get_learner_code,
        validate_learner_ids,
    )

    # Deriving the learner -> colour mapping once, so that every chart in the
    # notebook binds a colour to a learner rather than to a position in the hue
    # order. Any subset that drops a learner must not silently recolour the
    # remaining ones.
    _japanese_user_ids = sorted(
        decks_df.loc[
            decks_df["target_language"] == "ja", "user_id"
        ].unique()
    )
    validate_learner_ids(_japanese_user_ids)

    learner_code_by_user_id = {
        _user_id: get_learner_code(_user_id)
        for _user_id in _japanese_user_ids
    }
    _present_learner_codes = set(learner_code_by_user_id.values())
    learner_code_order = [
        _learner_code
        for _learner_code in LEARNER_CODE_ORDER
        if _learner_code in _present_learner_codes
    ]

    learner_color_by_code = {
        _learner_code: LEARNER_COLOR_BY_CODE[_learner_code]
        for _learner_code in learner_code_order
    }

    # Keying the same colours by raw `user_id` as well, so charts that plot the
    # full identifier do not have to relabel their data just to stay in sync.
    user_id_palette = {
        _user_id: learner_color_by_code[learner_code_by_user_id[_user_id]]
        for _user_id in _japanese_user_ids
    }
    return (
        learner_code_by_user_id,
        learner_code_order,
        learner_color_by_code,
        user_id_palette,
    )


@app.cell
def _(mo):
    mo.md("""
    # Analyzing review patterns across shared words
    """)
    return


@app.cell
def _(
    decks_df,
    df_lexemes,
    df_word_instances,
    get_dataset_as_path,
    learner_code_by_user_id,
    learner_code_order,
    mo,
    pd,
    plt,
    reviews_df,
    shared_lexemes,
    sns,
    user_id_palette,
):
    mo.stop(
        len(learner_code_order) < 2,
        mo.md(
            "Select at least two learners to display shared-word review "
            "patterns."
        ),
    )

    # Loading the cleaned reviews up front, so the enrichment steps
    # below read top-to-bottom without interleaved I/O.
    _clean_reviews_df = pd.read_json(
        get_dataset_as_path("processed/clean_reviews_w_fsrs.jsonl"),
        lines=True,
    )
    _clean_reviews_df = _clean_reviews_df[
        _clean_reviews_df["word_instance_id"].isin(
            df_word_instances["word_instance_id"]
        )
    ].copy()
    _clean_reviews_df = _clean_reviews_df.sort_values(
        ["word_instance_id", "timestamp_ms", "review_id"],
        kind="stable",
    )

    _selected_learner_count = len(learner_code_order)
    _globally_shared_lexemes = shared_lexemes.eq(
        _selected_learner_count
    ).to_frame(
        "is_globally_shared"
    )



    print(f"The number of globally shared: {_globally_shared_lexemes.sum()}")

    # Selecting word instances for lexemes studied by every selected learner.
    _globally_shared_lexeme_ids = _globally_shared_lexemes[
        _globally_shared_lexemes["is_globally_shared"]
    ].index
    _globally_shared_wi = df_word_instances[
        df_word_instances["lexeme_id"].isin(_globally_shared_lexeme_ids)
    ].copy()

    # Filtering out word instances that hasn't yet reached the PLB target
    _globally_shared_wi = _globally_shared_wi[
        _globally_shared_wi["plb_14"].notna()
    ].copy()

    # Requiring every selected learner to have reached the target for each
    # displayed lexeme. Count learners, not cards, to handle duplicate cards.
    _globally_shared_wi_with_owners = _globally_shared_wi.merge(
        decks_df[["deck_id", "user_id"]],
        on="deck_id",
        how="inner",
        validate="many_to_one",
    )
    _lexeme_counts = _globally_shared_wi_with_owners.groupby("lexeme_id")[
        "user_id"
    ].nunique()
    _fully_observed_lexeme_ids = _lexeme_counts[
        _lexeme_counts.eq(_selected_learner_count)
    ].index

    # Keeping word instances that are related to the lexemes that we kept at the previous stage
    _globally_shared_wi = _globally_shared_wi[
        _globally_shared_wi["lexeme_id"].isin(
            _fully_observed_lexeme_ids[:5]
        )
    ].copy()

    mo.stop(
        _globally_shared_wi.empty,
        mo.md(
            "No lexeme has reached stability 14 for every selected learner."
        ),
    )

    print(
        f"The number of globally shared WIs: {len(_globally_shared_wi.index)}"
    )

    # Keeping reviews assigned only the previously kept word instances
    _shared_reviews = reviews_df[
        reviews_df["word_instance_id"].isin(
            _globally_shared_wi["word_instance_id"]
        )
    ].copy()

    # Stabilizing the reviews order
    _shared_reviews = _shared_reviews.sort_values(
        ["word_instance_id", "timestamp_ms", "review_id"],
        kind="stable",
    )

    _captured_words = _shared_reviews["word_instance_id"].nunique()

    print(f"Reviews capture words: {_captured_words}")
    print(f"Total reviews captured: {len(_shared_reviews.index)}")

    # Calculating the date offset for each review, so we can align reviews from different dates on the same scale and compare the patterns
    def _date_to_offset_day(timestamps: pd.Series) -> pd.Series:
        timestamps = pd.to_datetime(timestamps)
        first_timestamp = timestamps.min()
        return (timestamps - first_timestamp).dt.days

    _shared_reviews["day_offset"] = _shared_reviews.groupby(
        "word_instance_id"
    )["timestamp_ms"].transform(_date_to_offset_day)

    # Several reviews fall after day 400, which unnecessarily stretches the chart horizontally, so we're removing them
    _shared_reviews = _shared_reviews[
        _shared_reviews["day_offset"].lt(400)
    ].copy()

    # Since it's beneficial to show which reviews we considered "clean", we're enriching the original dataset with the `is_clean` flag
    _shared_reviews["is_clean"] = _shared_reviews["review_id"].isin(
        _clean_reviews_df["review_id"]
    )

    # Determining when each learner reached PLB on the given words for the first time
    _first_plb_reviews = _clean_reviews_df[
        _clean_reviews_df["fsrs_stability"].ge(14)
    ].drop_duplicates("word_instance_id", keep="first")
    _shared_reviews["is_plb"] = _shared_reviews["review_id"].isin(
        _first_plb_reviews["review_id"]
    )

    # Determining reviews where they failed
    _shared_reviews["is_lapse"] = _shared_reviews["review_type"] == 2

    # For easier charting, marking reviews which happened after PLB (we will remove them)
    def _check_is_after_plb(group: pd.DataFrame) -> pd.Series:
        plb_rows = group[group["is_plb"]]

        if plb_rows.empty:
            return pd.Series(False, index=group.index)

        first_plb_timestamp = plb_rows["timestamp_ms"].min()
        return group["timestamp_ms"].gt(first_plb_timestamp)

    _shared_reviews["is_after_plb"] = _shared_reviews.groupby(
        "word_instance_id",
        group_keys=False,
    ).apply(_check_is_after_plb)

    _shared_reviews = _shared_reviews[
        ~_shared_reviews["is_after_plb"]
    ].copy()

    print(f"With PLB: {_shared_reviews['is_plb'].sum()}")

    # Preparing a final dataset form for plotting with all required fields
    _plot_df = _shared_reviews.merge(
        _globally_shared_wi[
            [
                "word_instance_id",
                "lexeme_id",
                "deck_id",
                "plb_14",
            ]
        ],
        on="word_instance_id",
        how="inner",
        validate="many_to_one",
    )

    _plot_df = _plot_df.merge(
        df_lexemes[["lexeme_id", "form"]],
        on="lexeme_id",
        how="inner",
        validate="many_to_one",
    )

    _plot_df = _plot_df.merge(
        decks_df[["deck_id", "user_id"]],
        on="deck_id",
        how="inner",
        validate="many_to_one",
    ).sort_values(
        ["form", "user_id", "timestamp_ms"],
        kind="stable",
    )

    _form_order = _plot_df["form"].drop_duplicates().tolist()
    _user_order = _plot_df["user_id"].drop_duplicates().tolist()

    # Reusing the notebook-wide colour map, so a learner keeps the same colour
    # here and in the survival/agreement charts further down.
    _user_palette = {
        _user_id: user_id_palette[_user_id] for _user_id in _user_order
    }

    _lapse_palette = {}
    for _user_id in _user_order:
        _lapse_palette[_user_id] = "black"

    # Extracting datasets for each plot to simplify plotting
    _general_reviews = _plot_df[~_plot_df["is_clean"] & ~_plot_df["is_plb"]]
    _clean_reviews = _plot_df[_plot_df["is_clean"] & ~_plot_df["is_plb"]]
    _plb_reviews = _plot_df[_plot_df["is_plb"]]

    _general_lapse_reviews = _general_reviews[_general_reviews["is_lapse"]]
    _clean_lapse_reviews = _clean_reviews[_clean_reviews["is_lapse"]]

    print(
        "Relearning/lapse reviews (review_type == 2): "
        f"{_plot_df['is_lapse'].sum()}"
    )
    print(
        "Clean relearning/lapse reviews: "
        f"{_clean_lapse_reviews.shape[0]}"
    )

    plt.figure(figsize=(18, 8))
    _sp_ax = plt.gca()

    sns.stripplot(
        data=_general_reviews,
        x="day_offset",
        y="form",
        hue="user_id",
        order=_form_order,
        hue_order=_user_order,
        palette=_user_palette,
        marker="X",
        dodge=True,
        jitter=False,
        alpha=0.6,
        legend=False,
        ax=_sp_ax,
    )

    sns.stripplot(
        data=_clean_reviews,
        x="day_offset",
        y="form",
        hue="user_id",
        order=_form_order,
        hue_order=_user_order,
        palette=_user_palette,
        marker="o",
        dodge=True,
        jitter=False,
        legend=False,
        ax=_sp_ax,
    )

    if not _general_lapse_reviews.empty:
        sns.stripplot(
            data=_general_lapse_reviews,
            x="day_offset",
            y="form",
            hue="user_id",
            order=_form_order,
            hue_order=_user_order,
            palette=_lapse_palette,
            marker="X",
            dodge=True,
            jitter=False,
            alpha=1.0,
            legend=False,
            ax=_sp_ax,
        )

    if not _clean_lapse_reviews.empty:
        sns.stripplot(
            data=_clean_lapse_reviews,
            x="day_offset",
            y="form",
            hue="user_id",
            order=_form_order,
            hue_order=_user_order,
            palette=_lapse_palette,
            marker="o",
            dodge=True,
            jitter=False,
            alpha=1.0,
            legend=False,
            ax=_sp_ax,
        )

    sns.stripplot(
        data=_plb_reviews,
        x="day_offset",
        y="form",
        hue="user_id",
        order=_form_order,
        hue_order=_user_order,
        palette=_user_palette,
        marker="*",
        dodge=True,
        jitter=False,
        size=10,
        legend=False,
        ax=_sp_ax,
    )

    # Applying bounding boxes for better graphics
    from matplotlib import patches as _patches

    _max_day = _plot_df["day_offset"].max() * 1.05

    for _form_index, _form in enumerate(_form_order):
        _rect = _patches.Rectangle(
            (-4, _form_index - 0.45),
            _max_day + 4,
            0.9,
            linewidth=1,
            edgecolor="gray",
            facecolor=(
                "whitesmoke"
                if _form_index % 2 == 0
                else "aliceblue"
            ),
            linestyle="--",
            alpha=0.4,
            zorder=0,
        )
        _sp_ax.add_patch(_rect)

    # Applying annotations with the number of days words required to reach PLB

    from matplotlib import font_manager as _font_manager
    from matplotlib.lines import Line2D as _Line2D

    # Matplotlib's default DejaVu Sans does not contain Japanese glyphs. Resolve
    # a Japanese-capable system font for the lexeme tick labels without
    # changing the font used by unrelated figures in this notebook.
    _japanese_font = None
    for _font_family in (
        "Noto Sans CJK JP",
        "IPAexGothic",
        "IPAGothic",
        "Yu Gothic",
        "Meiryo",
        "Hiragino Sans",
    ):
        _font_properties = _font_manager.FontProperties(
            family=_font_family
        )
        try:
            _font_manager.findfont(
                _font_properties,
                fallback_to_default=False,
            )
        except ValueError:
            continue
        _japanese_font = _font_properties
        break

    if _japanese_font is None:
        raise RuntimeError(
            "No Japanese-capable Matplotlib font is installed. Install "
            "Noto Sans CJK JP or IPA Gothic."
        )

    _user_legend_handles = []
    for _user_id in _user_order:
        _user_legend_handles.append(
            _Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                color=_user_palette[_user_id],
                markerfacecolor=_user_palette[_user_id],
                label=learner_code_by_user_id[_user_id],
            )
        )

    _user_legend = _sp_ax.legend(
        handles=_user_legend_handles,
        title="Learner",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
    )
    _sp_ax.add_artist(_user_legend)

    _review_legend_handles = [
        _Line2D(
            [0],
            [0],
            marker="X",
            linestyle="None",
            color="gray",
            label="Cramming review",
        ),
        _Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            color="gray",
            label="Clean review",
        ),
        _Line2D(
            [0],
            [0],
            marker="*",
            markersize=10,
            linestyle="None",
            color="gray",
            label="Reached target stability",
        ),
        _Line2D(
            [0],
            [0],
            marker="X",
            linestyle="None",
            color="black",
            label="Relearning/lapse",
        ),
    ]
    _sp_ax.legend(
        handles=_review_legend_handles,
        title="Review markers",
        loc="lower left",
        bbox_to_anchor=(1.02, 0.0),
        borderaxespad=0,
    )

    _form_positions = {}
    for _form_index, _form in enumerate(_form_order):
        _form_positions[_form] = _form_index

    # Spreading annotations vertically to match the dodge offsets applied by
    # the strip plots, so each label sits on its own learner's marker
    _user_offsets = {}
    _number_of_users = len(_user_order)
    for _user_index, _user_id in enumerate(_user_order):
        _user_offsets[_user_id] = (
            (_user_index + 0.5) / _number_of_users - 0.5
        ) * 0.8

    for _, _row in _plb_reviews.iterrows():
        _preceding_reviews = int(_row["plb_14"]) - 1
        _annotation_y = (
            _form_positions[_row["form"]]
            + _user_offsets[_row["user_id"]]
        )
        _sp_ax.annotate(
            str(_preceding_reviews),
            xy=(_row["day_offset"], _annotation_y),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "white",
                "alpha": 0.75,
                "edgecolor": "none",
            },
        )

    _sp_ax.set_xlim(-4, _max_day)
    _sp_ax.set_xlabel("Days since the first review")
    _sp_ax.set_ylabel("Lexeme form")
    for _tick_label in _sp_ax.get_yticklabels():
        _tick_label.set_fontproperties(_japanese_font)
    _sp_ax.set_title(
        "Review patterns until stability 14 "
        "(star labels show preceding clean reviews)"
    )

    plt.subplots_adjust(right=0.72)
    plt.show()
    return


@app.cell(hide_code=True)
def _(learner_code_order, mo):
    mo.stop(len(learner_code_order) < 2)

    _l02_example = ""
    if "L02" in learner_code_order:
        _l02_example = """
    Learner `L02` is a useful example: for 伝える, the learner recalled the
    word without a lapse for nearly six months, including intervals longer
    than 14 days, but did not reach FSRS stability 14 until after day 125.
        """

    mo.md(f"""
    ### Takeaways
    ***
    1. **Review patterns are highly idiosyncratic**: among the selected learners, the journey to reaching the target FSRS stability varies substantially. Learners require different timelines and review counts to achieve the same stability threshold for identical words.
    2. **Influence of learner settings**: the big difference in these timelines suggests that individual configurations (such as a learner's desired retention rate and schedule configurations) heavily influence the scheduling algorithm and shape these distinct patterns.

    {_l02_example}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # == [Configuration] ==
    ***
    For further analysis to work, a learner should be selected.
    ***
    """)
    return


@app.cell(hide_code=True)
def _(dac_user_order, learner_code_by_user_id, mo):
    _learner_options = {
        learner_code_by_user_id[_user_id]: _user_id
        for _user_id in dac_user_order
    }
    user_id_dropdown = mo.ui.dropdown(
        options=_learner_options,
        value=learner_code_by_user_id[dac_user_order[0]],
        label="Choose learner:",
    )
    return (user_id_dropdown,)


@app.cell(hide_code=True)
def _(learner_code_by_user_id, mo, user_id_dropdown):
    _selected_learner_code = learner_code_by_user_id[
        user_id_dropdown.value
    ]
    mo.vstack(
        [
            user_id_dropdown,
            mo.md(f"Selected learner: `{_selected_learner_code}`"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ====
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Analyzing the FSRS configurations
    ***
    > Question: Do FSRS configurations affect the **number of reviews** required to reach the **target stability of 14**?
    """)
    return


@app.cell(hide_code=True)
def _(all_decks_df, decks_df):
    from fsrs import Scheduler as _Scheduler

    # Build the configuration catalogue before applying the learner selection,
    # so Cxx identifiers do not change when an earlier learner is deselected.
    # The catalogue supplies labels only; the final frame is filtered below.
    _decks_for_analysis = all_decks_df[
        all_decks_df["target_language"] == "ja"
    ].copy()

    # Flattening the weight vector into a string, because lists are unhashable
    # and therefore cannot be used directly for de-duplication.
    _decks_for_analysis["fsrs_weights_flattened"] = _decks_for_analysis[
        "fsrs_weights"
    ].transform(
        lambda weights: "_".join(
            [str(weight) for weight in weights]
        )
    )

    decks_for_analysis_configurations = _decks_for_analysis[
        [
            "user_id",
            "fsrs_weights",
            "fsrs_weights_flattened",
            "desired_retention",
        ]
    ].drop_duplicates(
        subset=[
            "user_id",
            "fsrs_weights_flattened",
            "desired_retention",
        ]
    )

    # Sorting before assigning ids, so that `config_id` labels stay stable
    # across re-runs and remain comparable between charts.
    decks_for_analysis_configurations = (
        decks_for_analysis_configurations.sort_values(
            [
                "user_id",
                "desired_retention",
                "fsrs_weights_flattened",
            ],
            kind="stable",
        ).reset_index(drop=True)
    )
    decks_for_analysis_configurations["config_id"] = (
        decks_for_analysis_configurations.index + 1
    )
    decks_for_analysis_configurations["uses_default_weights"] = (
        decks_for_analysis_configurations["fsrs_weights"].transform(len) == 0
    )

    # Substituting the library defaults where a deck stores no weights, so that
    # every configuration can be compared on an equal footing.
    _default_fsrs_weights = list(_Scheduler().parameters)
    _effective_fsrs_weights = []
    for _weights in decks_for_analysis_configurations["fsrs_weights"]:
        if len(_weights) == 0:
            _effective_fsrs_weights.append(_default_fsrs_weights)
        else:
            _effective_fsrs_weights.append(_weights)

    decks_for_analysis_configurations["effective_fsrs_weights"] = (
        _effective_fsrs_weights
    )
    decks_for_analysis_configurations["fsrs_weight_count"] = (
        decks_for_analysis_configurations[
            "effective_fsrs_weights"
        ].transform(len)
    )
    decks_for_analysis_configurations["is_default_equivalent"] = (
        decks_for_analysis_configurations[
            "effective_fsrs_weights"
        ].transform(
            lambda weights: list(map(float, weights))
            == list(map(float, _default_fsrs_weights))
        )
    )
    decks_for_analysis_configurations = (
        decks_for_analysis_configurations[
            decks_for_analysis_configurations["user_id"].isin(
                decks_df["user_id"]
            )
        ].copy()
    )

    print(
        "Unique learner x `fsrs_weights` x `desired_retention` "
        "combinations: "
        f"{len(decks_for_analysis_configurations.index)}."
    )
    return (decks_for_analysis_configurations,)


@app.cell(hide_code=True)
def _(decks_for_analysis_configurations, pd):
    from datetime import datetime as _datetime, timezone as _timezone
    from fsrs import Card as _Card, Rating as _Rating, Scheduler as _Scheduler

    stability_target = 14.0

    # Capping the simulation so a pathological configuration cannot loop
    # forever; reaching the cap is treated as an error below.
    _maximum_reviews = 100
    _simulation_start = _datetime(2025, 1, 1, tzinfo=_timezone.utc)

    _simulation_rows = []

    for _configuration in decks_for_analysis_configurations.itertuples():
        # Disabling learning steps and fuzzing, because we want the trajectory
        # implied by the weights alone, without Anki's randomisation.
        _scheduler_arguments = {
            "desired_retention": float(_configuration.desired_retention),
            "learning_steps": (),
            "enable_fuzzing": False,
        }

        if len(_configuration.fsrs_weights) > 0:
            _scheduler_arguments["parameters"] = tuple(
                float(weight)
                for weight in _configuration.fsrs_weights
            )

        _scheduler = _Scheduler(**_scheduler_arguments)
        _card = _Card()
        _review_datetime = _simulation_start

        for _review_number in range(1, _maximum_reviews + 1):
            if _card.last_review is None:
                _interval_days = 0.0
            else:
                _interval_days = (
                    _review_datetime - _card.last_review
                ).total_seconds() / 86_400

            _card, _ = _scheduler.review_card(
                _card,
                _Rating.Good,
                review_datetime=_review_datetime,
            )

            _day_offset = (
                _review_datetime - _simulation_start
            ).total_seconds() / 86_400
            _is_plb = _card.stability >= stability_target

            _weights_name = "default"
            if not _configuration.uses_default_weights:
                _weights_name = "custom"

            _configuration_label = (
                f"C{_configuration.config_id:02d}: "
                f"r={_configuration.desired_retention:.2f}, "
                f"{_weights_name} weights"
            )

            _simulation_rows.append(
                {
                    "user_id": _configuration.user_id,
                    "config_id": _configuration.config_id,
                    "configuration_label": _configuration_label,
                    "desired_retention": _configuration.desired_retention,
                    "uses_default_weights": (
                        _configuration.uses_default_weights
                    ),
                    "review_number": _review_number,
                    "day_offset": _day_offset,
                    "interval_days": _interval_days,
                    "difficulty": _card.difficulty,
                    "stability": _card.stability,
                    "is_plb": _is_plb,
                }
            )

            if _is_plb:
                break

            # Following the schedule exactly, so the simulation stays on-policy
            _review_datetime = _card.due
        else:
            raise RuntimeError(
                "A simulated configuration did not reach stability "
                f"{stability_target:g} within {_maximum_reviews} reviews: "
                f"config_id={_configuration.config_id}."
            )

    fsrs_simulation_df = pd.DataFrame(_simulation_rows)
    dac_user_order = (
        decks_for_analysis_configurations["user_id"]
        .drop_duplicates()
        .tolist()
    )

    print(
        f"Simulated {fsrs_simulation_df['config_id'].nunique()} "
        "configurations with always-Good reviews."
    )
    return dac_user_order, fsrs_simulation_df, stability_target


@app.cell(hide_code=True)
def _(
    fsrs_simulation_df,
    learner_code_by_user_id,
    pd,
    sns,
    stability_target,
    user_id_dropdown,
):
    import seaborn.objects as _so

    _selected_user_simulation = fsrs_simulation_df[
        fsrs_simulation_df["user_id"] == user_id_dropdown.value
    ].copy()

    _configuration_order = (
        _selected_user_simulation["configuration_label"]
        .drop_duplicates()
        .tolist()
    )

    # Building the palette as an explicit label -> colour dict, so a
    # configuration keeps its colour regardless of how many are plotted.
    _configuration_colors = sns.color_palette(
        "tab10",
        n_colors=len(_configuration_order),
    ).as_hex()
    _configuration_palette = {}
    for _config_index, _configuration_label in enumerate(
        _configuration_order
    ):
        _configuration_palette[_configuration_label] = (
            _configuration_colors[_config_index]
        )

    # Isolating the target-reaching review, so it can be marked and labelled
    # on top of the trajectory line.
    _target_reviews = _selected_user_simulation[
        _selected_user_simulation["is_plb"]
    ].copy()
    _target_reviews["target_label"] = _target_reviews[
        "review_number"
    ].transform(
        lambda review_number: f"{review_number} reviews"
    )

    _learner_code = learner_code_by_user_id[user_id_dropdown.value]

    _configuration_plot = (
        _so.Plot(
            _selected_user_simulation,
            x="day_offset",
            y="configuration_label",
            color="configuration_label",
        )
        .add(
            _so.Line(linewidth=1.5),
            group="configuration_label",
        )
        .add(
            _so.Dot(pointsize=7),
        )
        .add(
            _so.Dot(marker="*", pointsize=14),
            data=_target_reviews,
        )
        .add(
            _so.Text(
                color="black",
                fontsize=9,
                halign="left",
                offset=8,
            ),
            data=_target_reviews,
            text="target_label",
        )
        .scale(
            y=_so.Nominal(order=_configuration_order),
            color=_so.Nominal(_configuration_palette),
        )
        .label(
            x="Days since the first review",
            y="FSRS configuration",
            color="Configuration",
            title=(
                f"Always-Good review simulation for learner "
                f"{_learner_code}\n"
                f"Star: first review reaching stability "
                f"{stability_target:g}"
            ),
        )
        .layout(
            size=(
                14,
                max(4, len(_configuration_order) * 0.8),
            )
        )
    )

    _configuration_plot.show()

    # Summarising each trajectory in a table, since the chart alone makes exact
    # review counts and intervals hard to read off.
    _configuration_summary_rows = []
    for _summary_configuration_label in _configuration_order:
        _configuration_reviews = _selected_user_simulation[
            _selected_user_simulation["configuration_label"]
            == _summary_configuration_label
        ]
        _target_review = _configuration_reviews[
            _configuration_reviews["is_plb"]
        ].iloc[0]

        _review_days = []
        for _summary_day_offset in _configuration_reviews["day_offset"]:
            _review_days.append(round(float(_summary_day_offset), 2))

        _configuration_summary_rows.append(
            {
                "configuration": _summary_configuration_label,
                "reviews_to_stability_14": int(
                    _target_review["review_number"]
                ),
                "days_to_stability_14": round(
                    float(_target_review["day_offset"]),
                    2,
                ),
                "stability_at_target": round(
                    float(_target_review["stability"]),
                    2,
                ),
                "review_days": _review_days,
            }
        )

    _configuration_summary_df = pd.DataFrame(_configuration_summary_rows)
    _configuration_summary_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Comparing FSRS weight vectors across learners

    To isolate the weights, the comparison is restricted to one compatible
    stratum: the same desired retention and the same number of effective FSRS
    parameters. Exact duplicate vectors are collapsed even when they appear in
    multiple learners' decks.

    Five vectors are selected deterministically for a readable matrix. The
    selection first covers every observed simulated review count, then adds the
    vectors that are least similar to those already selected. This is an
    illustrative controlled simulation, not a claim that one learner's fitted
    weights are intrinsically better than another's.

    Similarity is the mean relative parameter closeness, or
    `1 - normalized Canberra distance`:

    \[
    \operatorname{sim}(a,b) =
    1 - \frac{1}{n}\sum_i
    \frac{|a_i-b_i|}{|a_i|+|b_i|}
    \]

    Identical vectors score 1. Values closer to 0 indicate larger
    parameter-wise relative differences. Equal parameter count is a necessary
    compatibility check, although count alone does not prove that two vectors
    came from exactly the same FSRS release.
    """)
    return


@app.cell(hide_code=True)
def _(
    decks_for_analysis_configurations,
    fsrs_simulation_df,
    learner_code_by_user_id,
    mo,
    np,
    pd,
    plt,
    sns,
):
    from IPython.display import display as _display

    mo.stop(
        len(learner_code_by_user_id) < 2,
        mo.md(
            "Select at least two learners for the cross-learner FSRS "
            "comparison."
        ),
    )

    # Attaching the simulated targets, so each configuration can be labelled
    # with the review count it needs rather than with its id alone.
    _simulated_targets = fsrs_simulation_df[
        fsrs_simulation_df["is_plb"]
    ][
        [
            "config_id",
            "review_number",
            "day_offset",
        ]
    ].rename(
        columns={
            "review_number": "reviews_to_stability_14",
            "day_offset": "days_to_stability_14",
        }
    )

    _all_configurations = decks_for_analysis_configurations.merge(
        _simulated_targets,
        on="config_id",
        how="left",
        validate="one_to_one",
    )
    _all_configurations["learner_code"] = _all_configurations[
        "user_id"
    ].map(learner_code_by_user_id)
    _all_configurations["effective_weights_key"] = _all_configurations[
        "effective_fsrs_weights"
    ].transform(
        lambda weights: tuple(float(weight) for weight in weights)
    )

    # Choosing the stratum with the broadest learner coverage and at least two
    # distinct simulated review counts. Holding retention and vector length
    # fixed prevents either quantity from masquerading as a weight effect.
    _strata = (
        _all_configurations.groupby(
            ["desired_retention", "fsrs_weight_count"],
            as_index=False,
        )
        .agg(
            learner_count=("learner_code", "nunique"),
            configuration_count=("config_id", "size"),
            review_count_variants=(
                "reviews_to_stability_14",
                "nunique",
            ),
        )
    )
    _eligible_strata = _strata[
        _strata["learner_count"].ge(2)
        & _strata["review_count_variants"].ge(2)
    ].copy()
    mo.stop(
        _eligible_strata.empty,
        mo.md(
            "The selected learners have no FSRS stratum with the same "
            "retention and parameter count but different simulated review "
            "counts."
        ),
    )

    _selected_stratum = _eligible_strata.sort_values(
        [
            "learner_count",
            "review_count_variants",
            "configuration_count",
            "desired_retention",
            "fsrs_weight_count",
        ],
        ascending=[False, False, False, False, False],
        kind="stable",
    ).iloc[0]
    _selected_retention = float(_selected_stratum["desired_retention"])
    _selected_weight_count = int(_selected_stratum["fsrs_weight_count"])

    _compatible_configurations = _all_configurations[
        _all_configurations["desired_retention"].eq(
            _selected_retention
        )
        & _all_configurations["fsrs_weight_count"].eq(
            _selected_weight_count
        )
    ].copy()

    # Collapsing identical effective vectors. With retention, ratings, and
    # review policy fixed, duplicate vectors must produce the same trajectory;
    # disagreement would indicate a simulation or grouping bug.
    _unique_configuration_rows = []
    for _, _duplicate_group in _compatible_configurations.groupby(
        "effective_weights_key",
        sort=False,
    ):
        _duplicate_group = _duplicate_group.sort_values(
            "config_id", kind="stable"
        )
        _review_counts = _duplicate_group[
            "reviews_to_stability_14"
        ].unique()
        if len(_review_counts) != 1:
            raise RuntimeError(
                "Identical FSRS weights produced different review counts "
                "under the same simulation conditions."
            )

        _representative = _duplicate_group.iloc[0]
        _source_config_ids = sorted(
            int(config_id)
            for config_id in _duplicate_group["config_id"]
        )
        _source_learners = sorted(
            _duplicate_group["learner_code"].unique()
        )
        _unique_configuration_rows.append(
            {
                "representative_config_id": int(
                    _representative["config_id"]
                ),
                "source_configurations": "/".join(
                    f"C{config_id:02d}"
                    for config_id in _source_config_ids
                ),
                "source_learners": "/".join(_source_learners),
                "effective_fsrs_weights": list(
                    _representative["effective_fsrs_weights"]
                ),
                "is_default_equivalent": bool(
                    _representative["is_default_equivalent"]
                ),
                "reviews_to_stability_14": int(_review_counts[0]),
                "days_to_stability_14": float(
                    _representative["days_to_stability_14"]
                ),
            }
        )

    _unique_configurations = pd.DataFrame(_unique_configuration_rows)
    mo.stop(
        len(_unique_configurations.index) < 4,
        mo.md(
            "The selected compatible stratum contains fewer than four "
            "unique FSRS weight vectors. Select more learners to build the "
            "comparison matrix."
        ),
    )

    def _get_weight_similarity(first_weights, second_weights):
        _first_weights = np.asarray(first_weights, dtype=float)
        _second_weights = np.asarray(second_weights, dtype=float)
        if _first_weights.shape != _second_weights.shape:
            raise ValueError(
                "FSRS weight similarity requires equal-length vectors."
            )

        _denominators = np.abs(_first_weights) + np.abs(_second_weights)
        _relative_differences = np.zeros_like(
            _denominators,
            dtype=float,
        )
        _nonzero_denominators = _denominators > 0
        _relative_differences[_nonzero_denominators] = (
            np.abs(_first_weights - _second_weights)[
                _nonzero_denominators
            ]
            / _denominators[_nonzero_denominators]
        )
        return float(1.0 - _relative_differences.mean())

    # Seed the comparison with one stable representative for every distinct
    # review-count outcome, then use farthest-point sampling in weight space to
    # fill the remaining slots. This keeps the five-vector cap reproducible.
    _selected_indices = []
    for _review_count in sorted(
        _unique_configurations["reviews_to_stability_14"].unique()
    ):
        _outcome_candidates = _unique_configurations[
            _unique_configurations["reviews_to_stability_14"].eq(
                _review_count
            )
        ].sort_values("representative_config_id", kind="stable")
        _selected_indices.append(_outcome_candidates.index[0])

    _selection_limit = min(5, len(_unique_configurations.index))
    mo.stop(
        len(_selected_indices) > _selection_limit,
        mo.md(
            "The compatible stratum has more review-count outcomes than the "
            "five-configuration comparison can represent."
        ),
    )

    while len(_selected_indices) < _selection_limit:
        _remaining_indices = [
            _candidate_index
            for _candidate_index in _unique_configurations.index
            if _candidate_index not in _selected_indices
        ]

        def _maximum_similarity_to_selection(_candidate_index):
            _candidate_weights = _unique_configurations.loc[
                _candidate_index, "effective_fsrs_weights"
            ]
            return max(
                _get_weight_similarity(
                    _candidate_weights,
                    _unique_configurations.loc[
                        _selected_index, "effective_fsrs_weights"
                    ],
                )
                for _selected_index in _selected_indices
            )

        _next_index = min(
            _remaining_indices,
            key=lambda _candidate_index: (
                _maximum_similarity_to_selection(_candidate_index),
                int(
                    _unique_configurations.loc[
                        _candidate_index, "representative_config_id"
                    ]
                ),
            ),
        )
        _selected_indices.append(_next_index)

    _selected_configurations = (
        _unique_configurations.loc[_selected_indices]
        .sort_values(
            ["reviews_to_stability_14", "representative_config_id"],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    _matrix_labels = []
    for _configuration in _selected_configurations.itertuples():
        if _configuration.is_default_equivalent:
            _configuration_name = (
                "default-equivalent\n"
                f"{_configuration.source_learners}"
            )
        else:
            _configuration_name = (
                f"{_configuration.source_learners} "
                f"{_configuration.source_configurations}"
            )
        _matrix_labels.append(
            f"{_configuration_name}\n"
            f"{_configuration.reviews_to_stability_14} reviews"
        )

    _comparison_count = len(_selected_configurations.index)
    _similarity_matrix = np.zeros(
        (_comparison_count, _comparison_count), dtype=float
    )

    for _first_index, _first_configuration in enumerate(
        _selected_configurations.itertuples()
    ):
        for _second_index, _second_configuration in enumerate(
            _selected_configurations.itertuples()
        ):
            _weight_similarity = _get_weight_similarity(
                _first_configuration.effective_fsrs_weights,
                _second_configuration.effective_fsrs_weights,
            )
            _similarity_matrix[_first_index, _second_index] = (
                _weight_similarity
            )

    _selection_summary = _selected_configurations[
        [
            "source_learners",
            "source_configurations",
            "is_default_equivalent",
            "reviews_to_stability_14",
            "days_to_stability_14",
        ]
    ].copy()
    _selection_summary["days_to_stability_14"] = _selection_summary[
        "days_to_stability_14"
    ].round(2)

    print(
        "Controlled comparison: "
        f"desired retention={_selected_retention:.2f}, "
        f"weights={_selected_weight_count}, "
        f"unique vectors={len(_unique_configurations.index)}, "
        f"selected={_comparison_count}."
    )
    _display(_selection_summary)

    _similarity_matrix_df = pd.DataFrame(
        _similarity_matrix,
        index=_matrix_labels,
        columns=_matrix_labels,
    )

    _figure_size = max(8, _comparison_count * 1.4)
    plt.figure(figsize=(_figure_size, _figure_size))
    _similarity_ax = sns.heatmap(
        _similarity_matrix_df,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        vmin=0,
        vmax=1,
        square=True,
        cbar_kws={"label": "Weight similarity"},
    )
    _similarity_ax.set_title(
        "Across-learner FSRS weight similarity\n"
        f"desired retention={_selected_retention:.2f}; "
        f"{_selected_weight_count} parameters"
    )
    _similarity_ax.set_xlabel("Configuration")
    _similarity_ax.set_ylabel("Configuration")
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # The matrix is outcome-stratified for readability, so calculating the
    # descriptive association from those five vectors would bias it. Use every
    # unique vector in the compatible stratum instead.
    _weight_similarity_pair_rows = []
    for _first_index, _first_configuration in enumerate(
        _unique_configurations.itertuples()
    ):
        for _second_index, _second_configuration in enumerate(
            _unique_configurations.itertuples()
        ):
            if _second_index <= _first_index:
                continue

            _first_reviews = int(
                _first_configuration.reviews_to_stability_14
            )
            _second_reviews = int(
                _second_configuration.reviews_to_stability_14
            )
            _weight_similarity_pair_rows.append(
                {
                    "configuration_a": (
                        _first_configuration.source_configurations
                    ),
                    "configuration_b": (
                        _second_configuration.source_configurations
                    ),
                    "weight_similarity": _get_weight_similarity(
                        _first_configuration.effective_fsrs_weights,
                        _second_configuration.effective_fsrs_weights,
                    ),
                    "reviews_a": _first_reviews,
                    "reviews_b": _second_reviews,
                    "absolute_review_difference": abs(
                        _first_reviews - _second_reviews
                    ),
                }
            )

    _weight_similarity_pairs_df = pd.DataFrame(_weight_similarity_pair_rows)
    if not _weight_similarity_pairs_df.empty:
        _weight_similarity_pairs_df = (
            _weight_similarity_pairs_df.sort_values(
                [
                    "weight_similarity",
                    "absolute_review_difference",
                ],
                ascending=[False, True],
                kind="stable",
            ).reset_index(drop=True)
        )

        _same_review_count = (
            _weight_similarity_pairs_df["absolute_review_difference"] == 0
        )

    fsrs_comparison_summary = {
        "desired_retention": _selected_retention,
        "weight_count": _selected_weight_count,
        "selected_vector_count": _comparison_count,
        "minimum_reviews": int(
            _selected_configurations["reviews_to_stability_14"].min()
        ),
        "maximum_reviews": int(
            _selected_configurations["reviews_to_stability_14"].max()
        ),
    }

    _weight_similarity_pairs_df
    return (fsrs_comparison_summary,)


@app.cell(hide_code=True)
def _(fsrs_comparison_summary, mo):
    _retention = fsrs_comparison_summary["desired_retention"]
    _weight_count = fsrs_comparison_summary["weight_count"]
    _minimum_reviews = fsrs_comparison_summary["minimum_reviews"]
    _maximum_reviews = fsrs_comparison_summary["maximum_reviews"]

    mo.md(f"""
    ### Takeaways
    ***
    The current experiment can answer a narrow question:
    > Under an always-`Good`, on-schedule trajectory, can the FSRS configuration change the number of reviews needed to reach stability 14?

    **Yes.** In the controlled cross-learner stratum, desired retention is fixed
    at `{_retention:.2f}`, every vector has {_weight_count} parameters, and the
    simulated policy is held constant. The selected weight vectors nevertheless
    require between {_minimum_reviews} and {_maximum_reviews} reviews to reach
    stability 14. Within this simulation, that difference is therefore induced
    by the FSRS weights rather than by retention, rating, or review timing.

    This is a **mechanism demonstration**, not an effectiveness result. It does
    not show that one learner's fitted weights are better, that the weights are
    causally personalized, or that the same difference occurs in observed
    review histories. Establishing those claims would require held-out,
    learner-specific predictive evaluation rather than an always-`Good`
    trajectory.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Checking burden distributions
    ***
    > Question: do learners practice vocabulary in the same difficulty range?

    The burden measure is `plb_14`, which is **right-censored**:
    when a word has not reached the target yet the column is `NaN`, but the
    matching `current_review_count` still tells us the answer is *greater than*
    the number of reviews already done. Dropping those rows would silently
    compare survivors only, and the censoring rate differs several-fold between
    learners — so the sections below use survival analysis instead.
    """)
    return


@app.cell
def _(decks_df, df_word_instances, learner_code_by_user_id, np):
    # Attaching deck ownership, so burden can be grouped per learner.
    _word_instances_w_users = df_word_instances.merge(
        decks_df[["deck_id", "user_id", "target_language"]],
        on="deck_id",
        validate="many_to_one",
    )

    _japanese_word_instances = _word_instances_w_users[
        _word_instances_w_users["target_language"] == "ja"
    ].copy()

    # Excluding cards that were never reviewed: they are study backlog rather
    # than evidence about difficulty, and a censoring time of zero reviews
    # carries no information about how much work a word takes.
    _started_word_instances = _japanese_word_instances[
        _japanese_word_instances["current_review_count"] > 0
    ].copy()

    _started_word_instances["learner_code"] = _started_word_instances[
        "user_id"
    ].map(
        learner_code_by_user_id
    )

    # Encoding the event indicator: `True` when the target stability was
    # actually observed, `False` when observation stopped first.
    _started_word_instances["reached_target"] = _started_word_instances[
        "plb_14"
    ].notna()

    # Using the review count at which observation ends as the censoring time,
    # so unfinished words still contribute "not yet reached after N reviews".
    _started_word_instances["reviews_observed"] = np.where(
        _started_word_instances["reached_target"],
        _started_word_instances["plb_14"],
        _started_word_instances["current_review_count"],
    ).astype(int)

    burden_df = _started_word_instances[
        [
            "learner_code",
            "lexeme_id",
            "word_instance_id",
            "reviews_observed",
            "reached_target",
        ]
    ].copy()

    print(
        "Excluded never-reviewed word instances: "
        f"{len(_japanese_word_instances.index) - len(burden_df.index)}"
    )

    _censoring_summary = (
        burden_df.groupby("learner_code")
        .agg(
            word_instances=("reviews_observed", "size"),
            reached_target=("reached_target", "sum"),
        )
        .assign(
            censored_share=lambda frame: (
                1 - frame["reached_target"] / frame["word_instances"]
            ).round(3)
        )
        .reset_index()
    )
    _censoring_summary
    return (burden_df,)


@app.cell
def _(burden_df, learner_code_order, learner_color_by_code, pd, plt):
    from lifelines import KaplanMeierFitter as _KaplanMeierFitter
    from lifelines.statistics import (
        multivariate_logrank_test as _multivariate_logrank_test,
    )

    _survival_fig, _survival_ax = plt.subplots(figsize=(12, 7))

    _label_review = 20
    _curve_endpoints = {}
    _median_rows = []
    for _learner_code in learner_code_order:
        _learner_burden = burden_df[
            burden_df["learner_code"] == _learner_code
        ]

        # Fitting Kaplan-Meier per learner, because it is the estimator that
        # uses censored word instances instead of discarding them: each one
        # stays in the risk set right up to the review where observation ends.
        _fitter = _KaplanMeierFitter(label=_learner_code).fit(
            _learner_burden["reviews_observed"],
            event_observed=_learner_burden["reached_target"],
        )

        # Drawing the confidence band, since sample sizes differ by more than
        # an order of magnitude between learners and a gap between two noisy
        # curves means something very different from a gap between tight ones.
        _fitter.plot_survival_function(
            ax=_survival_ax,
            color=learner_color_by_code[_learner_code],
            ci_alpha=0.15,
        )
        _curve_end_review = min(_label_review, _fitter.timeline[-1])
        _curve_end_y = float(_fitter.predict(_curve_end_review))
        if _curve_end_review < _label_review:
            # A short observed curve must be labeled at its actual endpoint.
            _survival_ax.annotate(
                _learner_code,
                xy=(_curve_end_review, _curve_end_y),
                xytext=(6, 10),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=11,
                color="black",
                arrowprops={"arrowstyle": "-", "color": "0.4", "lw": 0.8},
            )
        else:
            _curve_endpoints[_learner_code] = _curve_end_y

        _median_rows.append(
            {
                "learner_code": _learner_code,
                "word_instances": len(_learner_burden.index),
                "km_median_reviews": _fitter.median_survival_time_,
                # Keeping the survivors-only median alongside, to show how far
                # the naive estimate is pulled by differential censoring.
                "naive_median_reviews": _learner_burden.loc[
                    _learner_burden["reached_target"], "reviews_observed"
                ].median(),
            }
        )

    _survival_ax.set_xlim(0, _label_review)
    _survival_ax.set_ylim(0, 1)
    _survival_ax.set_xlabel("Clean reviews", fontsize=13)
    _survival_ax.set_ylabel(
        "P(stability 14 not yet reached)", fontsize=13
    )
    _survival_ax.set_title(
        "Reviews needed to reach stability 14, corrected for censoring\n"
        "Lower curve = the learner finishes words in fewer reviews"
    )
    # Label curves that reach the right edge. Spread labels where several
    # survival probabilities are nearly zero, and connect them to their curves.
    _next_label_y = 0.03
    for _learner_code, _curve_y in sorted(
        _curve_endpoints.items(), key=lambda item: item[1]
    ):
        _label_y = max(_curve_y, _next_label_y)
        _survival_ax.annotate(
            _learner_code,
            xy=(_label_review, _curve_y),
            xytext=(1.02, _label_y),
            textcoords=_survival_ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=11,
            color="black",
            arrowprops={"arrowstyle": "-", "color": "0.4", "lw": 0.8},
            annotation_clip=False,
        )
        _next_label_y = _label_y + 0.05
    _survival_ax.legend_.remove()
    _survival_ax.grid(alpha=0.2)
    plt.tight_layout(rect=(0, 0, 0.92, 1))
    plt.show()

    # A log-rank comparison is defined only when at least two learners remain.
    if len(learner_code_order) >= 2:
        _logrank_result = _multivariate_logrank_test(
            burden_df["reviews_observed"],
            burden_df["learner_code"],
            burden_df["reached_target"],
        )
        print(
            "Multivariate log-rank test: "
            f"chi2={_logrank_result.test_statistic:.1f}, "
            f"df={_logrank_result.degrees_of_freedom}, "
            f"p={_logrank_result.p_value:.3g}"
        )
    else:
        print("Select at least two learners to run the log-rank test.")

    pd.DataFrame(_median_rows)
    return


@app.cell(hide_code=True)
def _(learner_code_order, mo):
    _selected_learner_count = len(learner_code_order)
    if _selected_learner_count < 2:
        _agreement_description = (
            "Select at least two learners to estimate item-level agreement."
        )
    else:
        _agreement_description = f"""
    Requiring a lexeme to be completed by all {_selected_learner_count} selected
    learners can discard most of the data. The analysis below therefore uses
    every lexeme a learner shares with *any* other selected learner rather than
    only the fully-crossed ones.
        """

    mo.md(f"""
    # Checking item-level agreement
    ***
    > Question: do learners agree on **which lexemes** are hard, and how much of that can non-personalized features explain?

    {_agreement_description}
    """)
    return


@app.cell
def _(burden_df, learner_code_order, mo, np, pd):
    mo.stop(
        len(learner_code_order) < 2,
        mo.md(
            "Select at least two learners to estimate item-level agreement."
        ),
    )

    # Collapsing to one row per (learner, lexeme), because a learner can hold
    # several cards for the same lexeme across decks and would otherwise be
    # weighted by how often they happened to duplicate a word.
    agreement_df = (
        burden_df[burden_df["reached_target"]]
        .groupby(["learner_code", "lexeme_id"])["reviews_observed"]
        .median()
        .rename("reviews")
        .reset_index()
    )

    # Log-transforming, since review counts are right-skewed with a long tail
    # and the variance model fitted below assumes roughly symmetric residuals.
    agreement_df["log_reviews"] = np.log1p(agreement_df["reviews"])

    _lexemes_per_count = (
        agreement_df.groupby("lexeme_id")["learner_code"]
        .nunique()
        .value_counts()
        .sort_index()
        .rename("lexemes")
        .rename_axis("learners_sharing_lexeme")
        .reset_index()
    )

    _learners_per_lexeme = agreement_df.groupby("lexeme_id")[
        "learner_code"
    ].nunique()
    _fully_crossed_lexeme_count = int(
        _learners_per_lexeme.eq(len(learner_code_order)).sum()
    )
    print(
        "Lexemes completed by all selected learners: "
        f"{_fully_crossed_lexeme_count}"
    )
    print(
        "Lexemes usable for pairwise agreement (>= 2 learners): "
        f"{_learners_per_lexeme.ge(2).sum()}"
    )

    pd.DataFrame(_lexemes_per_count)
    return (agreement_df,)


@app.cell
def _(agreement_df, learner_code_order, mo, np, pd, plt, sns):
    import itertools as _itertools
    from scipy import stats as _stats

    # Pivoting to a lexeme x learner matrix, so each pair can be reduced to the
    # lexemes both learners actually completed.
    _burden_matrix = agreement_df.pivot(
        index="lexeme_id",
        columns="learner_code",
        values="reviews",
    )

    _correlations = pd.DataFrame(
        np.nan, index=learner_code_order, columns=learner_code_order
    )
    _pair_counts = pd.DataFrame(
        0, index=learner_code_order, columns=learner_code_order
    )

    for _learner_a, _learner_b in _itertools.combinations(
        learner_code_order, 2
    ):
        _shared = _burden_matrix[[_learner_a, _learner_b]].dropna()

        # Skipping thin pairs, since a correlation over a few dozen lexemes is
        # dominated by noise and would mislead when read off the heatmap.
        if len(_shared.index) < 100:
            continue

        # Using Spearman rather than Pearson, because burden is discrete and
        # heavy-tailed: a single 88-review word would otherwise carry enormous
        # leverage on a product-moment correlation.
        _rho = _stats.spearmanr(
            _shared[_learner_a], _shared[_learner_b]
        ).statistic
        _correlations.loc[_learner_a, _learner_b] = _rho
        _correlations.loc[_learner_b, _learner_a] = _rho
        _pair_counts.loc[_learner_a, _learner_b] = len(_shared.index)
        _pair_counts.loc[_learner_b, _learner_a] = len(_shared.index)

    _upper_triangle = np.triu_indices(len(learner_code_order), k=1)
    _comparable_correlations = _correlations.to_numpy()[_upper_triangle]
    mo.stop(
        not np.isfinite(_comparable_correlations).any(),
        mo.md(
            "No selected learner pair has at least 100 shared completed "
            "lexemes with a defined correlation."
        ),
    )

    _agreement_fig, _agreement_ax = plt.subplots(figsize=(8, 6.5))

    # Annotating with both rho and the pair size, so a strong-looking cell
    # backed by few lexemes cannot be misread as a strong result.
    _annotations = _correlations.copy().astype(object)
    for _row_label in learner_code_order:
        for _column_label in learner_code_order:
            _cell = _correlations.loc[_row_label, _column_label]
            _annotations.loc[_row_label, _column_label] = (
                ""
                if pd.isna(_cell)
                else f"{_cell:.2f}\nn={_pair_counts.loc[_row_label, _column_label]}"
            )

    sns.heatmap(
        _correlations,
        annot=_annotations,
        fmt="",
        cmap="vlag",
        vmin=-0.4,
        vmax=0.4,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Spearman rho"},
        annot_kws={"fontsize": 8},
        ax=_agreement_ax,
    )
    _agreement_ax.set_title(
        "Do learners agree on which lexemes are hard?\n"
        "Spearman correlation of PLB over shared lexemes"
    )
    _agreement_ax.set_xlabel("")
    _agreement_ax.set_ylabel("")
    plt.tight_layout()
    plt.show()

    _observed_rho = _comparable_correlations
    _observed_rho = _observed_rho[~np.isnan(_observed_rho)]
    _mean_rho = _observed_rho.mean()

    print(f"Mean pairwise Spearman rho: {_mean_rho:.3f}")

    # Reporting the equivalent Kendall's W, since it is the concordance
    # statistic usually reached for here -- but it is an algebraic rescaling of
    # the mean pairwise rho, and its usual form would additionally require
    # every learner to have completed every lexeme.
    _rater_count = len(learner_code_order)
    print(
        "Equivalent Kendall's W: "
        f"{(_mean_rho * (_rater_count - 1) + 1) / _rater_count:.3f}"
    )
    return


@app.cell
def _(agreement_df, df_lexemes, df_tubelex_ja, np, pd):
    import statsmodels.formula.api as _smf

    # Attaching corpus frequency, the non-personalized feature whose
    # explanatory power we want to size.
    _lexeme_frequency = df_lexemes[["lexeme_id", "form"]].merge(
        df_tubelex_ja[["word", "count"]],
        left_on="form",
        right_on="word",
        how="left",
    )

    _model_df = agreement_df.merge(
        _lexeme_frequency[["lexeme_id", "count"]],
        on="lexeme_id",
        how="left",
    )
    _model_df["log_frequency"] = np.log1p(_model_df["count"])
    _model_df = _model_df.dropna(subset=["log_frequency"]).copy()

    print(
        f"Rows: {len(_model_df.index)}, "
        f"lexemes: {_model_df['lexeme_id'].nunique()}"
    )

    # Treating the learner as a fixed effect: the selected learners are the
    # specific population being compared, not a sample from which we can
    # estimate a stable learner-level variance. The lexeme is a random effect,
    # because thousands of partially overlapping lexemes are exactly what a
    # random intercept is for. This lets the model use every pairwise overlap
    # instead of only the fully-crossed lexemes.
    _baseline_model = _smf.mixedlm(
        "log_reviews ~ C(learner_code)",
        _model_df,
        groups=_model_df["lexeme_id"],
    ).fit()

    _frequency_model = _smf.mixedlm(
        "log_reviews ~ C(learner_code) + log_frequency",
        _model_df,
        groups=_model_df["lexeme_id"],
    ).fit()

    _variance_rows = []
    for _model_name, _model in (
        ("learner only", _baseline_model),
        ("learner + log frequency", _frequency_model),
    ):
        _lexeme_variance = float(_model.cov_re.iloc[0, 0])
        _residual_variance = float(_model.scale)
        _variance_rows.append(
            {
                "model": _model_name,
                "var_lexeme": round(_lexeme_variance, 4),
                "var_residual": round(_residual_variance, 4),
                # Reporting the ICC, i.e. the share of the remaining variance
                # attributable to *which word it is*. It is the ceiling on what
                # any non-personalized feature set could ever explain, since
                # such features are constant within a lexeme.
                "icc_lexeme": round(
                    _lexeme_variance
                    / (_lexeme_variance + _residual_variance),
                    4,
                ),
            }
        )

    _baseline_lexeme_variance = float(_baseline_model.cov_re.iloc[0, 0])
    _frequency_lexeme_variance = float(_frequency_model.cov_re.iloc[0, 0])
    _absorbed_share = (
        _baseline_lexeme_variance - _frequency_lexeme_variance
    ) / _baseline_lexeme_variance

    print(
        "log_frequency coefficient: "
        f"{_frequency_model.params['log_frequency']:.4f} "
        f"(p={_frequency_model.pvalues['log_frequency']:.3g})"
    )
    print(
        "Share of lexeme-level variance absorbed by log frequency: "
        f"{_absorbed_share:.1%}"
    )

    pd.DataFrame(_variance_rows)
    return


if __name__ == "__main__":
    app.run()
