import numpy as np
import pandas as pd
import pytest

from plb.learners import get_user_id
from plb.model_b.ablations import gbdt
from plb.model_b.ablations.gbdt import XGBoostModel

L01 = get_user_id("L01")
L02 = get_user_id("L02")
L03 = get_user_id("L03")

N_ROWS = 40

LEXEMES = pd.DataFrame(
    {
        "lexeme_id": [f"lex-{i}" for i in range(N_ROWS)],
        "form": [f"語{i}" for i in range(N_ROWS)],
    }
)
TUBELEX = pd.DataFrame(
    {
        "form": [f"語{i}" for i in range(0, N_ROWS, 2)],
        "count": range(10, 10 + N_ROWS, 2),
    }
)
KVL = pd.DataFrame(
    {
        "word_instance_id": [f"wi-{i}" for i in range(N_ROWS)],
        "kvl_score": np.linspace(1.0, 5.0, N_ROWS),
    }
)
INPUTS = {"lexemes": LEXEMES, "tubelex": TUBELEX, "kvl": KVL}


def _nbr(pct: float, log_plb: float | None, censored: bool = False) -> dict:
    return {
        "word_instance_id": "n",
        "s": 2.0,
        "d": 5.0,
        "reviews": 3,
        "plb_14": None if censored else 3,
        "log_plb_14": log_plb,
        "is_censored": censored,
        "sem_pct": pct,
        "phon_pct": None,
        "mor_pct": None,
        "ortho_stroke_pct": None,
        "ortho_kana_pct": None,
        "selected_by": ["semantic"],
    }


def _frame(
    rows: int = N_ROWS, *, with_neighbors: bool = False
) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "user_id": [L01 if i % 2 else L02 for i in range(rows)],
            "word_instance_id": [f"wi-{i}" for i in range(rows)],
            "lexeme_id": [f"lex-{i}" for i in range(rows)],
            # Staggered starts so days_elapsed varies; L01/L02 alternate, as above.
            "first_reviewed_at": pd.date_range(
                "2024-01-01", periods=rows, freq="D", tz="UTC"
            ),
        }
    )
    if with_neighbors:
        frame["neighbors"] = [
            [_nbr(0.9, 1.0), _nbr(0.8, None, censored=True)] for _ in range(rows)
        ]
    # Target correlated with kvl_score so the model has something to learn.
    y = np.linspace(1.0, 5.0, rows) * 0.5 + rng.normal(0, 0.1, rows)
    return frame, y


def _params(**overrides) -> dict:
    return {
        "include_nbrs": False,
        "seed": 42,
        "early_stopping_rounds": 5,
        "xgb": {"n_estimators": 50, "max_depth": 3},
        **overrides,
    }


@pytest.fixture(autouse=True)
def _synthetic_inputs(monkeypatch):
    monkeypatch.setattr(gbdt, "load_intralexical_inputs", lambda: INPUTS)


def test_include_censored_without_include_nbrs_raises() -> None:
    with pytest.raises(ValueError, match="include_censored requires include_nbrs"):
        XGBoostModel(_params(include_censored=True))


def test_unexpected_params_raise() -> None:
    with pytest.raises(ValueError, match="Unexpected XGBoostModel params"):
        XGBoostModel(_params(typo=1))
    with pytest.raises(ValueError, match="include_nbrs is required"):
        XGBoostModel({"seed": 42, "early_stopping_rounds": 5})


def test_spec_pinned_keys_inside_xgb_raise() -> None:
    with pytest.raises(ValueError, match="must not set.*objective"):
        XGBoostModel(_params(xgb={"objective": "reg:squarederror"}))
    with pytest.raises(ValueError, match="must not set.*random_state"):
        XGBoostModel(_params(xgb={"random_state": 1}))


def test_requires_neighbors_follows_include_nbrs() -> None:
    assert XGBoostModel(_params()).requires_neighbors is False
    assert XGBoostModel(_params(include_nbrs=True)).requires_neighbors is True


def test_count_only_keeps_just_the_neighborhood_size() -> None:
    X, y = _frame(with_neighbors=True)
    model = XGBoostModel(_params(include_nbrs=True, nbr_features="count")).fit(X, y)

    assert model.feature_columns == [
        "kvl_score",
        "log_freq",
        "is_oov",
        "learner_median_plb",
        "n_neighbors",
    ]
    # The split still has to carry neighbors; only the aggregation is dropped.
    assert (
        XGBoostModel(
            _params(include_nbrs=True, nbr_features="count")
        ).requires_neighbors
        is True
    )
    assert np.isfinite(model.predict(X)).all()


def test_days_variant_measures_from_each_learner_own_training_start() -> None:
    X, y = _frame()
    model = XGBoostModel(_params(include_days=True)).fit(X, y)

    assert model.feature_columns[-1] == "days_elapsed"
    matrix = model._build_matrix(X, INPUTS)
    # Consecutive rows are one calendar day apart, so the feature is in days.
    spans = matrix["days_elapsed"].to_numpy()
    assert spans.max() == float(len(X) - 2)
    # Each learner's earliest training row sits at day zero on its own timeline.
    for user_id in (L01, L02):
        rows = matrix.loc[(X["user_id"] == user_id).to_numpy(), "days_elapsed"]
        assert rows.min() == 0.0
    assert np.isfinite(model.predict(X)).all()


def test_days_variant_falls_back_to_the_cohort_start_for_an_unseen_learner() -> None:
    X, y = _frame()
    model = XGBoostModel(_params(include_days=True)).fit(X, y)

    unseen = X.head(1).assign(user_id=L03)
    value = model._build_matrix(unseen, INPUTS)["days_elapsed"].iloc[0]

    expected = XGBoostModel._epoch_days(unseen["first_reviewed_at"]).iloc[0]
    assert value == expected - model.global_start


def test_count_only_rejects_censored_and_missing_neighbors() -> None:
    with pytest.raises(ValueError, match="incompatible with include_censored"):
        XGBoostModel(
            _params(include_nbrs=True, include_censored=True, nbr_features="count")
        )
    with pytest.raises(ValueError, match="nbr_features='count' requires"):
        XGBoostModel(_params(nbr_features="count"))
    with pytest.raises(ValueError, match="nbr_features must be"):
        XGBoostModel(_params(include_nbrs=True, nbr_features="all"))


def test_fit_then_predict_returns_one_finite_value_per_row() -> None:
    X, y = _frame()
    model = XGBoostModel(_params()).fit(X, y)

    predictions = model.predict(X)

    assert predictions.shape == (len(X),)
    assert np.isfinite(predictions).all()


def test_neighbor_variant_consumes_the_neighbors_column() -> None:
    X, y = _frame(with_neighbors=True)
    model = XGBoostModel(_params(include_nbrs=True, include_censored=True)).fit(X, y)

    assert "cen_share" in model.feature_columns
    assert np.isfinite(model.predict(X)).all()


def test_an_unseen_learner_gets_the_global_median_as_learner_median_plb() -> None:
    X, y = _frame()
    model = XGBoostModel(_params()).fit(X, y)

    unseen = X.head(1).assign(user_id=L03)
    matrix = model._build_matrix(unseen, INPUTS)

    assert matrix["learner_median_plb"].iloc[0] == model.global_median


def test_save_load_round_trip_yields_identical_predictions(tmp_path) -> None:
    X, y = _frame(with_neighbors=True)
    model = XGBoostModel(_params(include_nbrs=True)).fit(X, y)
    path = tmp_path / "model.json"
    model.save(path)

    restored = XGBoostModel.load(path)

    assert np.array_equal(restored.predict(X), model.predict(X))
    assert restored.feature_columns == model.feature_columns
    assert restored.learner_medians == model.learner_medians


def test_feature_column_drift_is_rejected() -> None:
    X, y = _frame()
    model = XGBoostModel(_params()).fit(X, y)

    model.feature_columns = list(reversed(model.feature_columns))

    with pytest.raises(ValueError, match="Feature columns drifted"):
        model._build_matrix(X, INPUTS)


def test_early_stopping_stops_before_the_estimator_cap() -> None:
    X, y = _frame()
    eval_X, eval_y = _frame()
    cap = 500
    model = XGBoostModel(
        _params(early_stopping_rounds=3, xgb={"n_estimators": cap, "max_depth": 3})
    ).fit(X, y, eval_X=eval_X, eval_y=eval_y)

    assert model.best_iteration is not None
    assert model.regressor.get_booster().num_boosted_rounds() < cap


def test_fit_without_an_eval_set_trains_without_early_stopping() -> None:
    X, y = _frame()
    model = XGBoostModel(_params()).fit(X, y)

    assert model.best_iteration is None


def test_providing_only_half_an_eval_set_raises() -> None:
    X, y = _frame()
    with pytest.raises(ValueError, match="provided together"):
        XGBoostModel(_params()).fit(X, y, eval_X=X)
