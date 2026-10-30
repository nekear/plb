import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import numpy as np
    import pandas as pd
    from plb.learners import get_learner_code
    from plb.utils import get_dataset_as_path

    # 1. Load data
    train_df = pd.read_json(get_dataset_as_path("processed/model_b/train_word_instances.jsonl"), lines=True)
    test_df = pd.read_json(get_dataset_as_path("processed/model_b/test_word_instances.jsonl"), lines=True)
    clean_reviews_w_fsrs_df = pd.read_json(get_dataset_as_path("processed/clean_reviews_w_fsrs.jsonl"), lines=True)

    # 2. Convert timestamps ensuring BOTH are UTC (tz-aware)
    test_df["first_reviewed_at_dt"] = pd.to_datetime(test_df["first_reviewed_at"], utc=True)
    train_df["first_reviewed_at_dt"] = pd.to_datetime(train_df["first_reviewed_at"], utc=True)
    test_df["learner_code"] = test_df["user_id"].map(get_learner_code)
    train_df["learner_code"] = train_df["user_id"].map(get_learner_code)

    _reviews_for_train_df = train_df.merge(
        clean_reviews_w_fsrs_df[["word_instance_id", "timestamp_ms", "fsrs_stability", "fsrs_difficulty"]],
        on="word_instance_id",
        how="inner",
        validate="one_to_many"
    )

    _reviews_for_train_df = _reviews_for_train_df.sort_values(["word_instance_id", "timestamp_ms"])
    _reviews_for_train_df["timestamp_dt"] = pd.to_datetime(_reviews_for_train_df["timestamp_ms"], unit="ms", utc=True)
    _reviews_for_train_df["preceding_review_count"] = _reviews_for_train_df.groupby("word_instance_id").cumcount()


    def _compute_recent_k_median_for_user(user_test_df: pd.DataFrame, user_train_reviews_df: pd.DataFrame, user_train_words_df: pd.DataFrame, k: int = 20) -> pd.DataFrame:
        user_test_df = user_test_df.sort_values("first_reviewed_at_dt").copy()
        user_train_reviews_df = user_train_reviews_df.sort_values("timestamp_dt")
        user_train_words_df = user_train_words_df.sort_values("first_reviewed_at_dt")

        medians_raw = []
        medians_log = []
        medians_raw_leaky = []
        medians_log_leaky = []
        recent_k_mean_days_ago = []

        for _, test_row in user_test_df.iterrows():
            cutoff = test_row["first_reviewed_at_dt"]

            # 1. Filtering train reviews strictly before this test word's first review
            prior_reviews = user_train_reviews_df[user_train_reviews_df["timestamp_dt"] < cutoff]

            if prior_reviews.empty:
                medians_raw.append(np.nan)
                medians_log.append(np.nan)
                recent_k_mean_days_ago.append(np.nan)
            else:
                # 2. Locating the first review reaching stability >= 14 per word instance
                reached_threshold = prior_reviews[prior_reviews["fsrs_stability"] >= 14]

                if reached_threshold.empty:
                    medians_raw.append(np.nan)
                    medians_log.append(np.nan)
                    recent_k_mean_days_ago.append(np.nan)
                else:
                    first_plb_per_word = reached_threshold.groupby("word_instance_id").first()
                    first_plb_per_word["plb"] = first_plb_per_word["preceding_review_count"] + 1

                    # 3. Taking the k most recently achieved PLBs
                    recent_window = first_plb_per_word.sort_values("timestamp_dt").tail(k)
                    recent_plbs = recent_window["plb"]

                    medians_raw.append(recent_plbs.median())
                    medians_log.append(np.log1p(recent_plbs).median())

                    # How many days before `cutoff` each window item's PLB was achieved.
                    days_ago = (cutoff - recent_window["timestamp_dt"]).dt.total_seconds() / 86400
                    recent_k_mean_days_ago.append(days_ago.mean())

            # Leaky variant: mirrors the production `recent` ablation (dummy.py) — gates
            # only on item start time (first_reviewed_at), then takes each item's already
            # -resolved plb_14/log_plb_14 even if the stability-14 crossing review that
            # determined it happened after `cutoff`.
            prior_words = user_train_words_df[user_train_words_df["first_reviewed_at_dt"] < cutoff]
            recent_words = prior_words.tail(k)

            if recent_words.empty:
                medians_raw_leaky.append(np.nan)
                medians_log_leaky.append(np.nan)
            else:
                medians_raw_leaky.append(recent_words["plb_14"].median())
                medians_log_leaky.append(recent_words["log_plb_14"].median())

        user_test_df["recent_k_median_plb_raw"] = medians_raw
        user_test_df["recent_k_median_plb_log"] = medians_log
        user_test_df["recent_k_median_plb_raw_leaky"] = medians_raw_leaky
        user_test_df["recent_k_median_plb_log_leaky"] = medians_log_leaky
        user_test_df["recent_k_mean_days_ago"] = recent_k_mean_days_ago
        return user_test_df


    # 3. Processing by user
    _results = []
    _train_grouped = {u: df for u, df in _reviews_for_train_df.groupby("user_id")}
    _train_grouped_words = {u: df for u, df in train_df.groupby("user_id")}
    _empty_train_reviews = _reviews_for_train_df.iloc[0:0]
    _empty_train_words = train_df.iloc[0:0]

    for user_id, user_test in test_df.groupby("user_id"):
        user_train = _train_grouped.get(user_id, _empty_train_reviews)
        user_train_words = _train_grouped_words.get(user_id, _empty_train_words)
        processed = _compute_recent_k_median_for_user(user_test, user_train, user_train_words, k=10)
        _results.append(processed)

    final_test_df = pd.concat(_results, ignore_index=True)
    final_test_df["learner_code"] = final_test_df["user_id"].map(get_learner_code)
    return final_test_df, get_dataset_as_path, np, pd, test_df, train_df


@app.cell
def _(final_test_df, get_dataset_as_path, np, pd):
    similarities_df = pd.read_json(get_dataset_as_path("processed/similarities.jsonl"), lines=True)

    def _get_neighbor_median_plb(neighbors: list, key: str = "plb_14") -> float:
        if not neighbors:
            return np.nan

        # Extracting non-null numeric values
        values = [n[key] for n in neighbors if n.get(key) is not None]
        return float(np.median(values)) if values else np.nan

    # Calculate raw and log median PLB per word instance
    similarities_df["neighbor_median_plb_raw"] = [_get_neighbor_median_plb(n, key="plb_14") for n in similarities_df["neighbors"]]

    similarities_df["neighbor_median_plb_log"] = [_get_neighbor_median_plb(n, key="log_plb_14") for n in similarities_df["neighbors"]]

    def _get_neighbor_mean_plb(neighbors: list, key: str = "plb_14") -> float:
        # Mean instead of median — matches gbdt.py's union `mean_log_plb` feature.
        if not neighbors:
            return np.nan
        values = [n[key] for n in neighbors if n.get(key) is not None]
        return float(np.mean(values)) if values else np.nan

    similarities_df["neighbor_mean_plb_log"] = [_get_neighbor_mean_plb(n, key="log_plb_14") for n in similarities_df["neighbors"]]

    similarities_df["neighborhood_size"] = [len(n) for n in similarities_df["neighbors"]]

    # word_instance_id -> first_reviewed_at for every word in the cohort (neighbors can
    # predate the model_b split, so this can't be limited to train_df/test_df).
    _word_first_reviewed_at = pd.read_json(
        get_dataset_as_path("processed/word_instances_w_stability_targets.jsonl"), lines=True
    )
    _neighbor_ts_by_id = pd.to_datetime(
        _word_first_reviewed_at.set_index("word_instance_id")["first_reviewed_at"], utc=True
    )

    def _neighbor_mean_days_ago(neighbors: list, target_ts) -> float:
        # Only neighbors that actually contribute to neighbor_median_plb_log (i.e. not
        # censored), same set the median above is taken over.
        days = [
            (target_ts - _neighbor_ts_by_id.get(n["word_instance_id"])).total_seconds() / 86400
            for n in neighbors
            if n.get("log_plb_14") is not None
            and pd.notna(_neighbor_ts_by_id.get(n["word_instance_id"]))
        ]
        return float(np.mean(days)) if days else np.nan

    similarities_df = similarities_df.merge(
        final_test_df[["word_instance_id", "first_reviewed_at_dt"]],
        on="word_instance_id",
        how="inner",
    )
    similarities_df["neighbor_mean_days_ago"] = [
        _neighbor_mean_days_ago(n, ts)
        for n, ts in zip(similarities_df["neighbors"], similarities_df["first_reviewed_at_dt"])
    ]

    final_test_with_sims_df = final_test_df.merge(
        similarities_df[[
            "word_instance_id",
            "neighbor_median_plb_raw",
            "neighbor_median_plb_log",
            "neighbor_mean_plb_log",
            "neighbor_mean_days_ago",
            "neighborhood_size"
        ]],
        on="word_instance_id",
        how="left",
        validate="one_to_many"
    )
    return (final_test_with_sims_df,)


@app.cell
def _(final_test_with_sims_df):
    _both_medians_filled = (final_test_with_sims_df["recent_k_median_plb_log"].notna() & final_test_with_sims_df["neighbor_median_plb_log"].notna())

    print(f"[Non-leaky] The proportion of both filled: {_both_medians_filled.sum() / len(final_test_with_sims_df.index)}")

    # Leaky vs. causal recent-k median, against the same rows
    _both_leaky_filled = (final_test_with_sims_df["recent_k_median_plb_log_leaky"].notna() & final_test_with_sims_df["neighbor_median_plb_log"].notna())
    print(f"[Leaky] The proportion of both filled: {_both_leaky_filled.sum() / len(final_test_with_sims_df.index)}")

    _df = final_test_with_sims_df[_both_leaky_filled]

    _corr_cols = ["log_plb_14", "recent_k_median_plb_log", "recent_k_median_plb_log_leaky", "neighbor_median_plb_log", "neighborhood_size"]

    # Within-learner correlations: one corr matrix per learner_code, stacked.
    _per_learner_corr = _df.groupby("learner_code")[_corr_cols].corr(method="spearman")
    _per_learner_corr
    return


@app.cell
def _(final_test_with_sims_df):
    import seaborn as sns

    _y_cols = ["recent_k_median_plb_log", "recent_k_median_plb_log_leaky", "neighbor_median_plb_log"]
    _learner_codes = ["L01", "L02", "L03"]

    _long_df = final_test_with_sims_df[final_test_with_sims_df["learner_code"].isin(_learner_codes)].melt(
        id_vars=["learner_code", "log_plb_14"],
        value_vars=_y_cols,
        var_name="series",
        value_name="predicted_log_plb_14",
    ).dropna(subset=["predicted_log_plb_14"])

    _grid = sns.relplot(
        data=_long_df,
        x="log_plb_14",
        y="predicted_log_plb_14",
        hue="series",
        col="learner_code",
        col_order=_learner_codes,
        alpha=0.6,
        height=5,
    )

    _lims = (final_test_with_sims_df["log_plb_14"].min(), final_test_with_sims_df["log_plb_14"].max())
    for _ax in _grid.axes.flat:
        _ax.plot(_lims, _lims, color="grey", linestyle="--", linewidth=1)

    _grid.figure
    return (sns,)


@app.cell
def _(pd, sns, test_df, train_df):
    _combined_df = pd.concat(
        [train_df.assign(split="train"), test_df.assign(split="test")],
        ignore_index=True,
    )
    _combined_df = _combined_df[~_combined_df["learner_code"].isin(["L06"])].sort_values("learner_code")

    # Anonymize: each learner's own day 0, not a real calendar date.
    _combined_df["days_since_first_review"] = _combined_df.groupby("learner_code")["first_reviewed_at_dt"].transform(
        lambda ts: (ts - ts.min()) / pd.Timedelta(days=1)
    )

    _grid = sns.relplot(
        data=_combined_df,
        x="days_since_first_review",
        y="plb_14",
        # Marker shape carries the split so the figure remains interpretable when
        # printed or photocopied in grayscale.
        hue="split",
        style="split",
        markers={"train": "o", "test": "^"},
        palette={"train": "#0072B2", "test": "#D55E00"},
        col="learner_code",
        col_wrap=3,
        alpha=0.6,
        height=4,
        facet_kws={"sharex": False},
        linewidth=0.3,
    )
    _grid.set_axis_labels("Days Since First Review", "PLB")
    _grid.set_titles("{col_name}")
    _grid.legend.set_title("Split")
    _grid.figure
    return


@app.cell
def _(get_dataset_as_path, pd, sns):
    from plb.learners import get_learner_code as _code

    # Same view as above, but over all six Japanese learners rather than the model_b
    # cohort. The split files only carry L01-L03 and L06, so this reads the
    # pre-cohort stability targets and recovers the learner via the deck owner.
    _words = pd.read_json(
        get_dataset_as_path("processed/word_instances_w_stability_targets.jsonl"),
        lines=True,
    )
    _decks = pd.read_json(get_dataset_as_path("interim/decks.jsonl"), lines=True)
    _all_learners_df = _words.merge(
        _decks[["deck_id", "user_id"]], on="deck_id", how="left", validate="many_to_one"
    )
    _all_learners_df["learner_code"] = _all_learners_df["user_id"].astype(str).map(_code)

    # Table I's population: cards with at least one clean review, uncensored only.
    # Censored items are dropped because they have no PLB to plot, which is a real
    # selection effect on this axis -- later items have had less time to reach the
    # stability target, so they are censored more often, and dropping them removes
    # harder items disproportionately from the right-hand side of each facet.
    _all_learners_df = _all_learners_df[
        _all_learners_df["learner_code"].notna()
        & (_all_learners_df["current_review_count"] > 0)
        & _all_learners_df["plb_14"].notna()
    ].sort_values("learner_code")

    # Anonymize: each learner's own day 0, not a real calendar date.
    _all_learners_df["first_reviewed_at_dt"] = pd.to_datetime(
        _all_learners_df["first_reviewed_at"], utc=True
    )
    _all_learners_df["days_since_first_review"] = _all_learners_df.groupby(
        "learner_code"
    )["first_reviewed_at_dt"].transform(lambda ts: (ts - ts.min()) / pd.Timedelta(days=1))

    _grid = sns.relplot(
        data=_all_learners_df,
        x="days_since_first_review",
        y="plb_14",
        col="learner_code",
        col_wrap=3,
        alpha=0.6,
        height=4,
        facet_kws={"sharex": False},
    )
    # A median trend per facet: L04 alone contributes 10,612 points, so the cloud is
    # unreadable on its own and the drift is what the plot is for.
    for _code_name, _ax in _grid.axes_dict.items():
        _sub = _all_learners_df[_all_learners_df["learner_code"] == _code_name]
        _bins = _sub.assign(_b=_sub["days_since_first_review"].rank(pct=True).mul(20).astype(int))
        _trend = _bins.groupby("_b").agg(
            x=("days_since_first_review", "median"), y=("plb_14", "median")
        )
        _ax.plot(_trend["x"], _trend["y"], color="crimson", linewidth=2)
    _grid.set_axis_labels("Days Since First Review", "PLB@14")
    _grid.set_titles("{col_name}")
    _grid.figure
    return


@app.cell
def _(final_test_with_sims_df, sns):
    # Recency (in days before the target's first review) of the items that feed each
    # median: the causal recent-k window vs. the uncensored similarity neighbors.
    _long_df = final_test_with_sims_df.melt(
        id_vars=["word_instance_id", "learner_code"],
        value_vars=["recent_k_mean_days_ago", "neighbor_mean_days_ago"],
        var_name="source",
        value_name="days_ago",
    ).dropna(subset=["days_ago"])

    _ax = sns.histplot(
        data=_long_df,
        x="days_ago",
        hue="source",
        kde=True,
        stat="density",
        common_norm=False,
        element="step",
    )

    _both_days_filled = final_test_with_sims_df[["recent_k_mean_days_ago", "neighbor_mean_days_ago"]].dropna()
    print(f"n = {len(_both_days_filled)} test items with both windows filled")
    print(_both_days_filled.corr(method="spearman"))

    _ax.figure
    return


@app.cell
def _(final_test_with_sims_df, pd):
    # Numeric stand-in for the target's calendar date, since .corr() skips datetimes.
    # Timedelta division rather than .astype("int64") — datetime64 resolution isn't
    # guaranteed to be nanoseconds (pandas 2.x may parse to microseconds instead).
    _df = final_test_with_sims_df.copy()
    _df["first_reviewed_at_epoch_days"] = (
        _df["first_reviewed_at_dt"] - pd.Timestamp("1970-01-01", tz="UTC")
    ) / pd.Timedelta(days=1)

    _cols = ["neighbor_mean_plb_log", "first_reviewed_at_epoch_days"]
    _both_filled = _df[_cols].dropna()
    print(f"n = {len(_both_filled)} test items with neighbor_mean_plb_log filled")
    print(_both_filled.corr(method="spearman"))

    # Within-learner breakdown, same as the other per-learner corr cell.
    _df.groupby("learner_code")[_cols].corr(method="spearman")
    return


@app.cell
def _(final_test_with_sims_df, np, pd):
    from scipy.stats import spearmanr

    # Does neighbor_mean_plb_log track the true target, or just the calendar-time
    # trend both happen to share? Detrend each series against time (per learner, via
    # a linear fit) before correlating the leftovers — a confound shared by both
    # variables gets subtracted out first, so whatever correlation survives can't be
    # explained by "both drift with time."
    _df = final_test_with_sims_df.copy()
    _df["first_reviewed_at_epoch_days"] = (
        _df["first_reviewed_at_dt"] - pd.Timestamp("1970-01-01", tz="UTC")
    ) / pd.Timedelta(days=1)
    _df = _df.dropna(subset=["neighbor_mean_plb_log", "log_plb_14", "first_reviewed_at_epoch_days"])

    def _partial_corr(g):
        t = g["first_reviewed_at_epoch_days"].to_numpy()
        f = g["neighbor_mean_plb_log"].to_numpy()
        y = g["log_plb_14"].to_numpy()
        rf = f - np.poly1d(np.polyfit(t, f, 1))(t)  # neighbor feature, time trend removed
        ry = y - np.poly1d(np.polyfit(t, y, 1))(t)  # true target, time trend removed
        return spearmanr(rf, ry).statistic

    def _raw_corr(g):
        return spearmanr(g["neighbor_mean_plb_log"], g["log_plb_14"]).statistic

    _by_learner = pd.DataFrame({
        "raw": _df.groupby("learner_code").apply(_raw_corr, include_groups=False),
        "partial_on_time": _df.groupby("learner_code").apply(_partial_corr, include_groups=False),
    })
    _by_learner
    return (spearmanr,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Does the neighbourhood just count elapsed time?

    `n_neighbors` is the count of *selected* neighbours, but selection admits roughly a
    fixed fraction of candidates, and the candidate pool is every word the learner
    started earlier. So the count may be a stand-in for "how far into the journey is
    this item" — which both the similar and the random ablation share, and which would
    explain their tie.

    Two possible drivers, separated below: elapsed days since the learner's first ever
    review, and the size of the prior vocabulary itself.
    """)
    return


@app.cell
def _(pd, test_df):
    import json

    from plb.utils import get_dataset_as_path as _path

    # Neighbour counts as the model sees them: uncensored only, union across channels.
    _counts = {}
    with open(_path("processed/similarities.jsonl")) as _handle:
        for _line in _handle:
            _row = json.loads(_line)
            _counts[_row["word_instance_id"]] = sum(
                1 for _n in _row["neighbors"] if not _n["is_censored"]
            )

    offsets_df = test_df.copy()
    offsets_df["n_neighbors"] = offsets_df["word_instance_id"].map(_counts)

    # Offset measured from each learner's own first review, over their whole history
    # rather than the test split, so the axis is "days into the journey".
    _all_words = pd.concat(
        [
            pd.read_json(_path(f"processed/model_b/{_split}_word_instances.jsonl"), lines=True)
            for _split in ("train", "eval", "test")
        ]
    )
    _all_words["first_reviewed_at_dt"] = pd.to_datetime(
        _all_words["first_reviewed_at"], utc=True
    )
    _start = _all_words.groupby("user_id")["first_reviewed_at_dt"].min()

    offsets_df["days_since_first_review"] = (
        offsets_df["first_reviewed_at_dt"] - offsets_df["user_id"].map(_start)
    ).dt.total_seconds() / 86400

    # Prior vocabulary: words the learner started strictly earlier, counted across all
    # splits so it reflects real history rather than the split's slice of it.
    _starts = _all_words[["user_id", "first_reviewed_at_dt"]].sort_values(
        "first_reviewed_at_dt"
    )
    offsets_df["prior_vocabulary"] = [
        int(
            (
                _starts.loc[_starts["user_id"] == _user, "first_reviewed_at_dt"] < _when
            ).sum()
        )
        for _user, _when in zip(
            offsets_df["user_id"], offsets_df["first_reviewed_at_dt"], strict=True
        )
    ]
    offsets_df = offsets_df[offsets_df["learner_code"].isin(["L01", "L02", "L03"])]
    offsets_df[["learner_code", "n_neighbors", "days_since_first_review", "prior_vocabulary"]].describe().round(1)
    return (offsets_df,)


@app.cell
def _(offsets_df, sns):
    _long = offsets_df.melt(
        id_vars=["learner_code", "n_neighbors"],
        value_vars=["days_since_first_review", "prior_vocabulary"],
        var_name="driver",
        value_name="x",
    )

    _grid = sns.relplot(
        data=_long,
        x="x",
        y="n_neighbors",
        col="learner_code",
        row="driver",
        col_order=["L01", "L02", "L03"],
        alpha=0.35,
        height=3.6,
        facet_kws={"sharex": False, "sharey": True},
    )
    # A median trend per facet: the cloud is dense, the drift is the point.
    for (_driver, _code), _ax in _grid.axes_dict.items():
        _sub = _long[(_long["driver"] == _driver) & (_long["learner_code"] == _code)]
        _bins = _sub.assign(_b=_sub["x"].rank(pct=True).mul(20).astype(int))
        _trend = _bins.groupby("_b").agg(x=("x", "median"), y=("n_neighbors", "median"))
        _ax.plot(_trend["x"], _trend["y"], color="crimson", linewidth=2)
    _grid.set_titles("{col_name} — {row_name}")
    _grid.figure
    return


@app.cell
def _(offsets_df, pd, spearmanr):
    # Which of the two drivers the count actually tracks, and whether either reaches
    # the target: if prior vocabulary dominates, the neighbour block is a timeline index.
    _rows = []
    for _code, _g in offsets_df.groupby("learner_code"):
        _rows.append(
            {
                "learner_code": _code,
                "n": len(_g),
                "rho(n_nbr, days)": spearmanr(_g["n_neighbors"], _g["days_since_first_review"]).statistic,
                "rho(n_nbr, prior_vocab)": spearmanr(_g["n_neighbors"], _g["prior_vocabulary"]).statistic,
                "rho(days, log_plb)": spearmanr(_g["days_since_first_review"], _g["log_plb_14"]).statistic,
                "rho(prior_vocab, log_plb)": spearmanr(_g["prior_vocabulary"], _g["log_plb_14"]).statistic,
                "rho(n_nbr, log_plb)": spearmanr(_g["n_neighbors"], _g["log_plb_14"]).statistic,
            }
        )
    pd.DataFrame(_rows).set_index("learner_code").round(3)
    return


if __name__ == "__main__":
    app.run()
