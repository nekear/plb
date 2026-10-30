import numpy as np
import pandas as pd
import pytest

from plb.learners import get_user_id
from plb.model_b.ablations import st
from plb.model_b.ablations.registry import model_path
from plb.model_b.ablations.st import SetTransformerModel

L01 = get_user_id("L01")
L02 = get_user_id("L02")

N_ROWS = 30
NEIGHBOR_IDS = ["wi-100", "wi-101"]

LEXEMES = pd.DataFrame(
    {
        "lexeme_id": [f"lex-{i}" for i in range(N_ROWS)] + ["lex-100", "lex-101"],
        "form": [f"語{i}" for i in range(N_ROWS)] + ["猫", "犬"],
    }
)
TUBELEX = pd.DataFrame({"form": ["猫"], "count": [99]})
KVL = pd.DataFrame(
    {
        "word_instance_id": [f"wi-{i}" for i in range(N_ROWS)] + NEIGHBOR_IDS,
        "kvl_score": list(np.linspace(1.0, 5.0, N_ROWS)) + [2.5, 3.5],
    }
)
INPUTS = {"lexemes": LEXEMES, "tubelex": TUBELEX, "kvl": KVL}
WORD_INDEX = pd.DataFrame(
    {"word_instance_id": NEIGHBOR_IDS, "lexeme_id": ["lex-100", "lex-101"]}
)


def _nbr(
    wid: str = "wi-100",
    *,
    censored: bool = False,
    s: float | None = 2.0,
    phon_pct: float | None = 0.7,
) -> dict:
    return {
        "word_instance_id": wid,
        "s": s,
        "d": 5.0 if s is not None else None,
        "reviews": 3,
        "plb_14": None if censored else 4,
        "log_plb_14": None if censored else 1.386,
        "is_censored": censored,
        "sem_pct": 0.9,
        "phon_pct": phon_pct,
        "mor_pct": 0.5,
        "ortho_stroke_pct": None,
        "ortho_kana_pct": None,
        "ortho_stroke_coverage": 0.8,
        "ortho_kana_coverage": 0.0,
        "selected_by": ["semantic"],
    }


def _frame(*, with_neighbors: bool = False) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "user_id": [L01 if i % 2 else L02 for i in range(N_ROWS)],
            "word_instance_id": [f"wi-{i}" for i in range(N_ROWS)],
            "lexeme_id": [f"lex-{i}" for i in range(N_ROWS)],
        }
    )
    if with_neighbors:
        # Mixed set sizes, including empty, plus one censored neighbor per full row.
        frame["neighbors"] = [
            [] if i % 5 == 0 else [_nbr("wi-100"), _nbr("wi-101", censored=True)]
            for i in range(N_ROWS)
        ]
    y = np.linspace(1.0, 5.0, N_ROWS) * 0.5 + rng.normal(0, 0.1, N_ROWS)
    return frame, y


def _params(**overrides) -> dict:
    return {
        "include_nbrs": False,
        "seed": 42,
        "epochs": 5,
        "batch_size": 8,
        "learning_rate": 1.0e-2,
        "weight_decay": 1.0e-4,
        "patience": 3,
        "dim": 8,
        "num_heads": 2,
        **overrides,
    }


@pytest.fixture(autouse=True)
def _synthetic_inputs(monkeypatch):
    monkeypatch.setattr(st, "load_intralexical_inputs", lambda: INPUTS)
    monkeypatch.setattr(st, "load_word_instance_index", lambda: WORD_INDEX)


def _feature(name: str, model: SetTransformerModel) -> int:
    return model.neighbor_feature_names.index(name)


def _built_neighbors(model: SetTransformerModel, neighbors: list[dict]) -> np.ndarray:
    model.global_median = 0.0
    frame = pd.DataFrame(
        {
            "user_id": [L01],
            "word_instance_id": ["wi-0"],
            "lexeme_id": ["lex-0"],
            "neighbors": [neighbors],
        }
    )
    examples = model._build_examples(frame, None, INPUTS, WORD_INDEX)
    return examples[0].neighbor_features


def test_invalid_params_raise_with_informative_messages() -> None:
    with pytest.raises(ValueError, match="Unexpected SetTransformerModel params"):
        SetTransformerModel(_params(typo=1))
    with pytest.raises(ValueError, match="include_nbrs is required"):
        SetTransformerModel({"seed": 42})
    with pytest.raises(ValueError, match="include_censored requires include_nbrs"):
        SetTransformerModel(_params(include_censored=True))
    with pytest.raises(ValueError, match="num_heads must divide dim"):
        SetTransformerModel(_params(dim=8, num_heads=3))


def test_requires_neighbors_follows_include_nbrs() -> None:
    assert SetTransformerModel(_params()).requires_neighbors is False
    assert SetTransformerModel(_params(include_nbrs=True)).requires_neighbors is True


def test_missing_stability_is_zero_filled_and_flagged() -> None:
    model = SetTransformerModel(_params(include_nbrs=True))

    features = _built_neighbors(model, [_nbr(s=None)])

    assert features[0, _feature("log1p_s", model)] == 0.0
    assert features[0, _feature("s_missing", model)] == 1.0
    assert features[0, _feature("d", model)] == 0.0


def test_missing_phonology_is_zero_filled_and_flagged() -> None:
    model = SetTransformerModel(_params(include_nbrs=True))

    features = _built_neighbors(model, [_nbr(phon_pct=None)])

    assert features[0, _feature("phon_pct", model)] == 0.0
    assert features[0, _feature("phon_missing", model)] == 1.0


def test_censored_neighbors_are_dropped_from_the_completed_variant() -> None:
    model = SetTransformerModel(_params(include_nbrs=True))

    features = _built_neighbors(model, [_nbr(), _nbr("wi-101", censored=True)])

    assert features.shape[0] == 1
    assert "is_censored" not in model.neighbor_feature_names


def test_censored_neighbors_stay_visible_and_flagged_in_the_censored_variant() -> None:
    model = SetTransformerModel(_params(include_nbrs=True, include_censored=True))

    features = _built_neighbors(model, [_nbr(), _nbr("wi-101", censored=True)])

    assert features.shape[0] == 2
    assert features[1, _feature("is_censored", model)] == 1.0
    assert features[1, _feature("log_plb_14", model)] == 0.0  # zero-filled burden


def test_neighbor_word_representation_is_joined_via_the_index() -> None:
    model = SetTransformerModel(_params(include_nbrs=True))

    features = _built_neighbors(model, [_nbr("wi-101")])  # 犬: kvl 3.5, OOV

    assert features[0, _feature("kvl_score", model)] == np.float32(3.5)
    assert features[0, _feature("is_oov", model)] == 1.0


def test_fit_reduces_the_training_loss_on_synthetic_data() -> None:
    X, y = _frame()
    model = SetTransformerModel(_params()).fit(X, y)

    assert model.history[-1]["train_l1"] < model.history[0]["train_l1"]


def test_rows_without_neighbors_train_and_predict() -> None:
    X, y = _frame(with_neighbors=True)
    eval_X, eval_y = _frame(with_neighbors=True)
    model = SetTransformerModel(_params(include_nbrs=True, include_censored=True)).fit(
        X, y, eval_X=eval_X, eval_y=eval_y
    )

    predictions = model.predict(X)

    assert predictions.shape == (len(X),)
    assert predictions.dtype == np.float64
    assert np.isfinite(predictions).all()
    assert model.best_epoch is not None


def test_save_then_load_returns_identical_predictions(tmp_path) -> None:
    X, y = _frame(with_neighbors=True)
    model = SetTransformerModel(_params(include_nbrs=True)).fit(X, y)
    path = tmp_path / "model.pt"
    model.save(path)

    restored = SetTransformerModel.load(path)

    assert np.array_equal(restored.predict(X), model.predict(X))
    assert restored.learner_medians == model.learner_medians
    assert restored.best_epoch == model.best_epoch


def test_the_seed_alone_determines_the_fitted_model() -> None:
    X, y = _frame()

    first = SetTransformerModel(_params()).fit(X, y).predict(X)
    same = SetTransformerModel(_params()).fit(X, y).predict(X)

    assert np.array_equal(first, same)


def test_st_artifacts_are_torch_checkpoints_and_older_paths_are_unchanged() -> None:
    assert model_path("st_no_nbrs").name == "model.pt"
    assert model_path("gbdt_no_nbrs").name == "model.json"
    assert model_path("global_median").name == "model.json"
