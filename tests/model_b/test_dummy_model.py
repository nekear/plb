import numpy as np
import pandas as pd
import pytest

from plb.learners import get_user_id
from plb.model_b.ablations.dummy import DummyModel

L01 = get_user_id("L01")
L02 = get_user_id("L02")
L03 = get_user_id("L03")


def _train(
    entries: list[tuple[str, str, str, str, float]],
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build (X, y) from (user_id, wid, first_reviewed_at, plb_14_achieved_at, log_plb)."""
    frame = pd.DataFrame(
        {
            "user_id": [e[0] for e in entries],
            "word_instance_id": [e[1] for e in entries],
            "first_reviewed_at": pd.to_datetime([e[2] for e in entries], utc=True),
            "plb_14_achieved_at": pd.to_datetime([e[3] for e in entries], utc=True),
        }
    )
    return frame, np.array([e[4] for e in entries], dtype=np.float64)


def _targets(entries: list[tuple[str, str]]) -> pd.DataFrame:
    """Build a predict-time frame from (user_id, first_reviewed_at) rows."""
    return pd.DataFrame(
        {
            "user_id": [e[0] for e in entries],
            "first_reviewed_at": pd.to_datetime([e[1] for e in entries], utc=True),
        }
    )


# Every item starts a month before its label is achieved — the gap the leak lived in.
X_TRAIN, Y_TRAIN = _train(
    [
        (L01, "a", "2024-12-01", "2025-01-01", 1.0),
        (L01, "b", "2024-12-02", "2025-01-02", 2.0),
        (L01, "c", "2024-12-03", "2025-01-03", 3.0),
        (L02, "d", "2024-12-01", "2025-01-01", 10.0),
        (L02, "e", "2024-12-02", "2025-01-02", 20.0),
    ]
)
GLOBAL_MEDIAN = 3.0  # median of [1, 2, 3, 10, 20]


def test_global_type_predicts_the_global_median_for_every_row() -> None:
    model = DummyModel({"type": "global"}).fit(X_TRAIN, Y_TRAIN)

    predictions = model.predict(_targets([(L01, "2025-02-01"), (L02, "2025-02-01")]))

    assert predictions.tolist() == [GLOBAL_MEDIAN, GLOBAL_MEDIAN]


def test_within_type_predicts_each_learners_own_median() -> None:
    model = DummyModel({"type": "within"}).fit(X_TRAIN, Y_TRAIN)

    predictions = model.predict(_targets([(L01, "2025-02-01"), (L02, "2025-02-01")]))

    assert predictions.tolist() == [2.0, 15.0]


def test_an_unseen_learner_falls_back_to_the_global_median() -> None:
    within = DummyModel({"type": "within"}).fit(X_TRAIN, Y_TRAIN)
    recent = DummyModel({"type": "recent", "k": 2}).fit(X_TRAIN, Y_TRAIN)

    targets = _targets([(L03, "2025-02-01")])

    assert within.predict(targets).tolist() == [GLOBAL_MEDIAN]
    assert recent.predict(targets).tolist() == [GLOBAL_MEDIAN]


def test_recent_uses_the_k_most_recently_achieved_items() -> None:
    model = DummyModel({"type": "recent", "k": 2}).fit(X_TRAIN, Y_TRAIN)

    # All three L01 labels are achieved before the target; only the two most
    # recently achieved count.
    predictions = model.predict(_targets([(L01, "2025-02-01")]))

    assert predictions.tolist() == [2.5]  # median of [2.0, 3.0], not [1, 2, 3]


def test_recent_excludes_items_achieved_at_the_targets_exact_timestamp() -> None:
    model = DummyModel({"type": "recent", "k": 10}).fit(X_TRAIN, Y_TRAIN)

    # Strict precedence: the label achieved at 2025-01-03 (y=3.0) is NOT before it.
    predictions = model.predict(_targets([(L01, "2025-01-03")]))

    assert predictions.tolist() == [1.5]  # median of [1.0, 2.0]


def test_recent_ignores_items_started_but_not_yet_achieved_before_the_target() -> None:
    # The leak regression: every L01 item STARTED (2024-12-XX) before this target,
    # but none had its label achieved yet — the window must be empty, falling back
    # to the learner median rather than reading labels from the future.
    model = DummyModel({"type": "recent", "k": 10}).fit(X_TRAIN, Y_TRAIN)

    predictions = model.predict(_targets([(L01, "2024-12-15")]))

    assert predictions.tolist() == [2.0]  # L01's median, not median of [1, 2, 3]


def test_recent_falls_back_to_the_learner_median_when_nothing_precedes() -> None:
    model = DummyModel({"type": "recent", "k": 3}).fit(X_TRAIN, Y_TRAIN)

    # Target predates every L01 achievement, so the window is empty.
    predictions = model.predict(_targets([(L01, "2024-01-01")]))

    assert predictions.tolist() == [2.0]  # L01's own median


def test_save_load_round_trip_yields_identical_predictions(tmp_path) -> None:
    targets = _targets(
        [
            (L01, "2025-01-02"),
            (L01, "2025-02-01"),
            (L02, "2024-01-01"),
            (L03, "2025-02-01"),
        ]
    )
    for params in ({"type": "global"}, {"type": "within"}, {"type": "recent", "k": 2}):
        model = DummyModel(params).fit(X_TRAIN, Y_TRAIN)
        path = tmp_path / f"{params['type']}.json"
        model.save(path)

        restored = DummyModel.load(path)

        assert restored.predict(targets).tolist() == model.predict(targets).tolist()
        assert restored.train_rows == len(X_TRAIN)


def test_invalid_params_raise_with_informative_messages() -> None:
    with pytest.raises(ValueError, match="type must be one of"):
        DummyModel({"type": "median"})
    with pytest.raises(ValueError, match="requires an integer k >= 1"):
        DummyModel({"type": "recent"})
    with pytest.raises(ValueError, match="requires an integer k >= 1"):
        DummyModel({"type": "recent", "k": 0})
    with pytest.raises(ValueError, match="k is only valid for type 'recent'"):
        DummyModel({"type": "global", "k": 5})
    with pytest.raises(ValueError, match="Unexpected DummyModel params"):
        DummyModel({"type": "global", "typo": True})


def test_fit_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="cannot be None"):
        DummyModel({"type": "global"}).fit(None, Y_TRAIN)
    with pytest.raises(ValueError, match="5 rows but y has 2 values"):
        DummyModel({"type": "global"}).fit(X_TRAIN, Y_TRAIN[:2])
    with pytest.raises(
        ValueError, match="missing required columns.*plb_14_achieved_at"
    ):
        DummyModel({"type": "recent", "k": 2}).fit(
            X_TRAIN.drop(columns=["plb_14_achieved_at"]), Y_TRAIN
        )


def test_predicting_before_fitting_raises() -> None:
    with pytest.raises(ValueError, match="must be fitted"):
        DummyModel({"type": "global"}).predict(_targets([(L01, "2025-02-01")]))


def test_the_optional_eval_pair_is_accepted_and_ignored() -> None:
    plain = DummyModel({"type": "within"}).fit(X_TRAIN, Y_TRAIN)
    with_eval = DummyModel({"type": "within"}).fit(
        X_TRAIN, Y_TRAIN, eval_X=X_TRAIN.head(2), eval_y=Y_TRAIN[:2]
    )

    targets = _targets([(L01, "2025-02-01"), (L02, "2025-02-01")])

    assert with_eval.predict(targets).tolist() == plain.predict(targets).tolist()
