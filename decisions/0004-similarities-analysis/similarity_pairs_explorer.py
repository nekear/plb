import marimo

__generated_with = "0.23.14"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Similarity-pair analysis

    This notebook expands `similarities.jsonl` into one row per target–neighbor
    pair and substitutes the word-instance IDs with each side's Japanese form,
    learner-specific meaning, and reading.

    `L01`–`L06` are arbitrary stable learner codes; they encode neither
    learner characteristics nor outcome ordering.

    The displayed state (`s`, `d`, and `reviews`) is the neighbor's state
    immediately before the target word's first review.

    A neighbor is kept when at least one channel's score is at or above its
    configured percentile of that channel's null distribution
    (`similarities.percentiles` in `params.yaml`). `selected_by` lists those
    channels, exactly as `select_similarity_neighbors` recorded them, and the
    `*_pct` columns show each score's percentile.
    """)
    return


@app.cell
def _():
    from pathlib import Path
    from tempfile import TemporaryDirectory

    import duckdb

    from plb.learners import get_learner_code, validate_learner_ids
    from plb.similarities.config import load_similarities_config

    return (
        Path,
        TemporaryDirectory,
        duckdb,
        get_learner_code,
        load_similarities_config,
        validate_learner_ids,
    )


@app.cell
def _(Path, TemporaryDirectory, duckdb, load_similarities_config):
    config = load_similarities_config()

    required_paths = {
        "similarities": config.output_path,
        "word instances": config.word_instances_path,
        "lexemes": config.lexemes_path,
        "decks": config.decks_path,
    }
    missing_paths = [
        f"{name}: {path}" for name, path in required_paths.items() if not path.exists()
    ]
    if missing_paths:
        raise FileNotFoundError(
            "Required similarity-analysis inputs are missing:\n"
            + "\n".join(missing_paths)
        )

    connection = duckdb.connect()
    # Prevent the nested JSON reader from retaining several gigabytes while
    # expanding the neighbor lists.
    connection.execute("SET memory_limit = '512MB'")
    connection.execute("SET threads = 2")

    def sql_path(path: Path) -> str:
        return str(path.resolve()).replace("'", "''")

    similarities_path = sql_path(config.output_path)
    word_instances_path = sql_path(config.word_instances_path)
    lexemes_path = sql_path(config.lexemes_path)
    decks_path = sql_path(config.decks_path)

    similarity_json_columns = """
        {
            'word_instance_id': 'VARCHAR',
            'neighbors': 'STRUCT(
                word_instance_id VARCHAR,
                s DOUBLE,
                d DOUBLE,
                reviews BIGINT,
                plb_14 DOUBLE,
                is_censored BOOLEAN,
                sem_sim DOUBLE,
                phon_sim DOUBLE,
                mor_sim DOUBLE,
                ortho_stroke_sim DOUBLE,
                ortho_stroke_coverage DOUBLE,
                ortho_kana_confusion DOUBLE,
                ortho_kana_coverage DOUBLE,
                sem_pct DOUBLE,
                phon_pct DOUBLE,
                mor_pct DOUBLE,
                ortho_stroke_pct DOUBLE,
                ortho_kana_pct DOUBLE,
                selected_by VARCHAR[]
            )[]'
        }
    """

    source_summary = connection.execute(
        f"""
        SELECT
            count(*) AS target_rows,
            count(*) FILTER (WHERE len(neighbors) = 0) AS empty_target_rows,
            sum(len(neighbors)) AS pair_rows,
            max(len(neighbors)) AS maximum_neighbors
        FROM read_json(
            '{similarities_path}',
            format = 'newline_delimited',
            columns = {similarity_json_columns}
        )
        """
    ).fetchone()

    # Querying the nested JSON directly on every UI interaction is expensive.
    # Flatten it once into a compact, session-scoped Parquet file.
    pair_cache_directory = TemporaryDirectory(prefix="plb_similarity_analysis_")
    pair_cache_path = Path(pair_cache_directory.name) / "similarity_pairs.parquet"
    pair_cache_sql_path = sql_path(pair_cache_path)

    connection.execute(
        f"""
        COPY (
            SELECT
                s.word_instance_id AS target_word_instance_id,
                neighbor.word_instance_id AS neighbor_word_instance_id,
                neighbor.s,
                neighbor.d,
                neighbor.reviews,
                neighbor.plb_14,
                neighbor.is_censored,
                neighbor.sem_sim,
                neighbor.phon_sim,
                neighbor.mor_sim,
                neighbor.ortho_stroke_sim,
                neighbor.ortho_stroke_coverage,
                neighbor.ortho_kana_confusion,
                neighbor.ortho_kana_coverage,
                neighbor.sem_pct,
                neighbor.phon_pct,
                neighbor.mor_pct,
                neighbor.ortho_stroke_pct,
                neighbor.ortho_kana_pct,
                neighbor.selected_by
            FROM read_json(
                '{similarities_path}',
                format = 'newline_delimited',
                columns = {similarity_json_columns}
            ) AS s,
            UNNEST(s.neighbors) AS expanded(neighbor)
        )
        TO '{pair_cache_sql_path}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    connection.execute(
        f"""
        CREATE TEMP TABLE word_metadata AS
        SELECT
            CAST(word_instance.word_instance_id AS VARCHAR) AS word_instance_id,
            CAST(deck.user_id AS VARCHAR) AS user_id,
            lexeme.form,
            word_instance.meaning,
            lexeme.reading,
            word_instance.first_reviewed_at
        FROM read_json_auto(
            '{word_instances_path}',
            format = 'newline_delimited'
        ) AS word_instance
        INNER JOIN read_json_auto(
            '{lexemes_path}',
            format = 'newline_delimited'
        ) AS lexeme
            USING (lexeme_id)
        INNER JOIN read_json_auto(
            '{decks_path}',
            format = 'newline_delimited'
        ) AS deck
            USING (deck_id)
        WHERE lexeme.language = 'ja'
        """
    )

    connection.execute(
        f"""
        CREATE TEMP VIEW enriched_pairs AS
        SELECT
            target.user_id,
            pair.target_word_instance_id,
            target.form AS target_form,
            target.meaning AS target_meaning,
            target.reading AS target_reading,
            target.first_reviewed_at AS target_first_reviewed_at,
            pair.neighbor_word_instance_id,
            neighbor.form AS neighbor_form,
            neighbor.meaning AS neighbor_meaning,
            neighbor.reading AS neighbor_reading,
            neighbor.first_reviewed_at AS neighbor_first_reviewed_at,
            pair.s,
            pair.d,
            pair.reviews,
            pair.plb_14,
            pair.is_censored,
            pair.sem_sim,
            pair.phon_sim,
            pair.mor_sim,
            pair.ortho_stroke_sim,
            pair.ortho_stroke_coverage,
            pair.ortho_kana_confusion,
            pair.ortho_kana_coverage,
            pair.sem_pct,
            pair.phon_pct,
            pair.mor_pct,
            pair.ortho_stroke_pct,
            pair.ortho_kana_pct,
            pair.selected_by,
            target.user_id = neighbor.user_id AS same_user,
            neighbor.first_reviewed_at < target.first_reviewed_at
                AS neighbor_precedes_target
        FROM read_parquet('{pair_cache_sql_path}') AS pair
        LEFT JOIN word_metadata AS target
            ON target.word_instance_id = pair.target_word_instance_id
        LEFT JOIN word_metadata AS neighbor
            ON neighbor.word_instance_id = pair.neighbor_word_instance_id
        """
    )

    input_summary = {
        "target_rows": int(source_summary[0]),
        "empty_target_rows": int(source_summary[1]),
        "pair_rows": int(source_summary[2]),
        "maximum_neighbors": int(source_summary[3]),
    }

    return config, connection, input_summary, pair_cache_directory


@app.cell
def _(connection, input_summary, mo, pair_cache_directory):
    integrity = connection.execute(
        """
        SELECT
            count(*) FILTER (
                WHERE target_form IS NULL OR neighbor_form IS NULL
            ) AS missing_metadata,
            count(*) FILTER (WHERE same_user IS NOT TRUE) AS cross_user_pairs,
            count(*) FILTER (
                WHERE neighbor_precedes_target IS NOT TRUE
            ) AS chronology_violations
        FROM enriched_pairs
        """
    ).fetchone()

    # Referencing this object keeps the temporary Parquet cache alive for the
    # duration of the notebook session.
    _cache_location = pair_cache_directory.name

    mo.vstack(
        [
            mo.hstack(
                [
                    mo.stat(
                        f"{input_summary['target_rows']:,}",
                        label="Japanese targets",
                    ),
                    mo.stat(
                        f"{input_summary['pair_rows']:,}",
                        label="Neighbor pairs",
                    ),
                    mo.stat(
                        f"{input_summary['empty_target_rows']:,}",
                        label="Targets without prior neighbors",
                    ),
                    mo.stat(
                        f"{input_summary['maximum_neighbors']:,}",
                        label="Maximum neighbors",
                    ),
                ],
                widths="equal",
            ),
            mo.hstack(
                [
                    mo.stat(f"{int(integrity[0]):,}", label="Missing metadata"),
                    mo.stat(f"{int(integrity[1]):,}", label="Cross-learner pairs"),
                    mo.stat(
                        f"{int(integrity[2]):,}",
                        label="Chronology violations",
                    ),
                ],
                widths="equal",
            ),
        ],
        gap=1,
    )
    return


@app.cell
def _(config, connection, mo):
    percentiles = config.percentiles
    metric_summary = connection.execute(
        f"""
        SELECT
            'semantic' AS axis,
            count(*) AS defined_pairs,
            round(avg(sem_sim), 4) AS mean,
            round(median(sem_sim), 4) AS median,
            round(quantile_cont(sem_sim, 0.9), 4) AS p90,
            round(quantile_cont(sem_sim, 0.99), 4) AS p99,
            {percentiles.semantic} AS percentile_threshold,
            count(*) FILTER (
                WHERE list_contains(selected_by, 'semantic')
            ) AS selected_pairs
        FROM enriched_pairs

        UNION ALL

        SELECT
            'phonological',
            count(phon_sim),
            round(avg(phon_sim), 4),
            round(median(phon_sim), 4),
            round(quantile_cont(phon_sim, 0.9), 4),
            round(quantile_cont(phon_sim, 0.99), 4),
            {percentiles.phonological},
            count(*) FILTER (
                WHERE list_contains(selected_by, 'phonological')
            )
        FROM enriched_pairs

        UNION ALL

        SELECT
            'morphological',
            count(*),
            round(avg(mor_sim), 4),
            round(median(mor_sim), 4),
            round(quantile_cont(mor_sim, 0.9), 4),
            round(quantile_cont(mor_sim, 0.99), 4),
            {percentiles.morphological},
            count(*) FILTER (
                WHERE list_contains(selected_by, 'morphological')
            )
        FROM enriched_pairs

        UNION ALL

        SELECT
            'stroke',
            count(*) FILTER (WHERE ortho_stroke_coverage > 0),
            round(avg(ortho_stroke_sim)
                FILTER (WHERE ortho_stroke_coverage > 0), 4),
            round(median(ortho_stroke_sim)
                FILTER (WHERE ortho_stroke_coverage > 0), 4),
            round(quantile_cont(ortho_stroke_sim, 0.9)
                FILTER (WHERE ortho_stroke_coverage > 0), 4),
            round(quantile_cont(ortho_stroke_sim, 0.99)
                FILTER (WHERE ortho_stroke_coverage > 0), 4),
            {percentiles.ortho_stroke},
            count(*) FILTER (
                WHERE list_contains(selected_by, 'ortho_stroke')
            )
        FROM enriched_pairs

        UNION ALL

        SELECT
            'kana',
            count(*) FILTER (WHERE ortho_kana_coverage > 0),
            round(avg(ortho_kana_confusion)
                FILTER (WHERE ortho_kana_coverage > 0), 4),
            round(median(ortho_kana_confusion)
                FILTER (WHERE ortho_kana_coverage > 0), 4),
            round(quantile_cont(ortho_kana_confusion, 0.9)
                FILTER (WHERE ortho_kana_coverage > 0), 4),
            round(quantile_cont(ortho_kana_confusion, 0.99)
                FILTER (WHERE ortho_kana_coverage > 0), 4),
            {percentiles.ortho_kana},
            count(*) FILTER (
                WHERE list_contains(selected_by, 'ortho_kana')
            )
        FROM enriched_pairs
        """
    ).fetchdf()

    metric_table = mo.ui.table(
        metric_summary,
        selection=None,
        pagination=False,
        show_column_summaries=False,
        show_data_types=False,
        show_download=False,
        label="Score distributions among retained pairs",
    )

    mo.vstack(
        [
            mo.md(
                """
                ## Retained-pair score distributions

                `selected_pairs` counts the pairs each channel selected, as
                recorded in the artifact; `percentile_threshold` is the value
                currently in `params.yaml` (re-run `select_similarity_neighbors`
                if you change it). These distributions describe the retained
                neighbors, not all eligible pairs.
                """
            ),
            metric_table,
        ]
    )
    return


@app.cell
def _(connection, get_learner_code, mo, validate_learner_ids):
    user_ids = [
        row[0]
        for row in connection.execute(
            "SELECT DISTINCT user_id FROM word_metadata ORDER BY user_id"
        ).fetchall()
    ]
    validate_learner_ids(user_ids)
    learner_options = {"All learners": ""}
    learner_options.update(
        {
            get_learner_code(user_id): user_id
            for user_id in sorted(user_ids, key=get_learner_code)
        }
    )

    user_filter = mo.ui.dropdown(
        options=learner_options,
        value="All learners",
        label="Learner",
    )
    axis_filter = mo.ui.dropdown(
        options={
            "Any retained pair": "any",
            "Selected by semantic": "semantic",
            "Selected by phonological": "phonological",
            "Selected by morphological": "morphological",
            "Selected by stroke": "ortho_stroke",
            "Selected by kana": "ortho_kana",
        },
        value="Any retained pair",
        label="Selecting channel",
    )
    sort_metric = mo.ui.dropdown(
        options={
            "Semantic": "sem_sim",
            "Phonological": "phon_sim",
            "Morphological": "mor_sim",
            "Stroke": "ortho_stroke_sim",
            "Kana": "ortho_kana_confusion",
        },
        value="Semantic",
        label="Sort and filter score",
    )
    minimum_score = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.01,
        value=0.0,
        show_value=True,
        include_input=True,
        label="Minimum selected score",
    )
    text_filter = mo.ui.text(
        placeholder="form, meaning, or reading",
        debounce=300,
        full_width=True,
        label="Search either side",
    )
    row_limit = mo.ui.number(
        start=20,
        stop=2_000,
        step=20,
        value=200,
        label="Maximum displayed rows",
    )

    mo.vstack(
        [
            mo.md("## Inspect enriched target–neighbor pairs"),
            mo.hstack(
                [user_filter, axis_filter, sort_metric],
                widths="equal",
            ),
            mo.hstack(
                [minimum_score, row_limit],
                widths=[2, 1],
            ),
            text_filter,
        ],
        gap=1,
    )
    return (
        axis_filter,
        minimum_score,
        row_limit,
        sort_metric,
        text_filter,
        user_filter,
    )


@app.cell
def _(
    axis_filter,
    connection,
    get_learner_code,
    minimum_score,
    row_limit,
    sort_metric,
    text_filter,
    user_filter,
):
    metric_columns = {
        "sem_sim",
        "phon_sim",
        "mor_sim",
        "ortho_stroke_sim",
        "ortho_kana_confusion",
    }
    selected_metric = sort_metric.value
    if selected_metric not in metric_columns:
        raise ValueError(f"Unsupported similarity metric: {selected_metric}")

    channels = {"semantic", "phonological", "morphological", "ortho_stroke", "ortho_kana"}
    selected_axis = axis_filter.value
    if selected_axis != "any" and selected_axis not in channels:
        raise ValueError(f"Unsupported selecting channel: {selected_axis}")

    conditions = [f"coalesce({selected_metric}, 0.0) >= ?"]
    parameters = [float(minimum_score.value)]
    if selected_axis != "any":
        conditions.append("list_contains(selected_by, ?)")
        parameters.append(selected_axis)

    if user_filter.value:
        conditions.append("user_id = ?")
        parameters.append(user_filter.value)

    search_text = text_filter.value.strip()
    if search_text:
        conditions.append(
            """
            concat_ws(
                ' ',
                target_form,
                target_meaning,
                target_reading,
                neighbor_form,
                neighbor_meaning,
                neighbor_reading
            ) ILIKE ?
            """
        )
        parameters.append(f"%{search_text}%")

    display_limit = int(row_limit.value)
    parameters.append(display_limit)
    where_clause = " AND ".join(f"({condition})" for condition in conditions)

    filtered_pairs = connection.execute(
        f"""
        SELECT
            count(*) OVER () AS matching_pair_count,
            user_id,
            target_form,
            target_meaning,
            target_reading,
            target_first_reviewed_at,
            neighbor_form,
            neighbor_meaning,
            neighbor_reading,
            neighbor_first_reviewed_at,
            array_to_string(selected_by, ', ') AS selected_by,
            reviews,
            s,
            d,
            plb_14,
            is_censored,
            sem_sim,
            phon_sim,
            mor_sim,
            ortho_stroke_sim,
            ortho_stroke_coverage,
            ortho_kana_confusion,
            ortho_kana_coverage,
            sem_pct,
            phon_pct,
            mor_pct,
            ortho_stroke_pct,
            ortho_kana_pct,
            same_user,
            neighbor_precedes_target,
            target_word_instance_id,
            neighbor_word_instance_id
        FROM enriched_pairs
        WHERE {where_clause}
        ORDER BY
            {selected_metric} DESC NULLS LAST,
            reviews DESC,
            target_first_reviewed_at,
            target_word_instance_id,
            neighbor_word_instance_id
        LIMIT ?
        """,
        parameters,
    ).fetchdf()

    filtered_pairs.insert(
        1,
        "learner_code",
        filtered_pairs["user_id"].map(get_learner_code),
    )

    if filtered_pairs.empty:
        matching_pair_count = 0
    else:
        matching_pair_count = int(filtered_pairs["matching_pair_count"].iloc[0])
        filtered_pairs = filtered_pairs.drop(columns="matching_pair_count")

    score_columns = [
        "s",
        "d",
        "sem_sim",
        "phon_sim",
        "mor_sim",
        "ortho_stroke_sim",
        "ortho_stroke_coverage",
        "ortho_kana_confusion",
        "ortho_kana_coverage",
        "sem_pct",
        "phon_pct",
        "mor_pct",
        "ortho_stroke_pct",
        "ortho_kana_pct",
    ]
    filtered_pairs[score_columns] = filtered_pairs[score_columns].round(3)

    return display_limit, filtered_pairs, matching_pair_count


@app.cell
def _(display_limit, filtered_pairs, matching_pair_count, mo):
    pair_table = mo.ui.table(
        filtered_pairs,
        selection=None,
        pagination=True,
        page_size=20,
        show_column_summaries=False,
        show_data_types=False,
        show_download=True,
        show_search=False,
        freeze_columns_left=["target_form", "neighbor_form"],
        wrapped_columns=["target_meaning", "neighbor_meaning"],
        column_widths={
            "learner_code": 90,
            "target_form": 120,
            "neighbor_form": 120,
            "target_meaning": 320,
            "neighbor_meaning": 320,
            "target_reading": 140,
            "neighbor_reading": 140,
            "selected_by": 180,
        },
        hidden_columns=[
            "user_id",
            "same_user",
            "neighbor_precedes_target",
            "target_word_instance_id",
            "neighbor_word_instance_id",
        ],
        max_height=720,
        label="Enriched similarity pairs",
    )

    mo.vstack(
        [
            mo.md(
                f"""
                **{matching_pair_count:,} pairs match the filters.**
                Showing at most {display_limit:,}; the download button exports
                the currently displayed subset.
                """
            ),
            pair_table,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
