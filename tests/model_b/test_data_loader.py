import json

import pandas as pd
import pytest

from plb.learners import get_user_id
from plb.model_b.core import data
from plb.model_b.preparation.create_splits import TEST

L01 = get_user_id("L01")


def _split_row(word_instance_id: str) -> dict:
    return {
        "user_id": L01,
        "deck_id": "deck-1",
        "lexeme_id": f"lex-{word_instance_id}",
        "word_instance_id": word_instance_id,
        "language": "ja",
        "plb_14": 3,
        "log_plb_14": 1.0986,
        "first_reviewed_at": "2025-01-01T10:00:00.000Z",
        "last_reviewed_at": "2025-02-01T10:00:00.000Z",
        "plb_14_achieved_at": "2025-01-20T10:00:00.000Z",
        "is_shared_test": False,
    }


def _write_jsonl(path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.fixture
def split_file(tmp_path, monkeypatch):
    path = tmp_path / "test_word_instances.jsonl"
    _write_jsonl(path, [_split_row("wi-1"), _split_row("wi-2")])
    monkeypatch.setitem(data.OUTPUT_PATHS, TEST, path)
    return path


def test_timestamps_come_back_tz_aware_utc(split_file) -> None:
    frame = data.load_split(TEST)

    assert str(frame["first_reviewed_at"].dtype).endswith("UTC]")
    assert frame["first_reviewed_at"].iloc[0] == pd.Timestamp(
        "2025-01-01 10:00", tz="UTC"
    )


def test_neighbors_arrive_as_a_list_column_without_changing_row_count(
    split_file, tmp_path, monkeypatch
) -> None:
    similarities = tmp_path / "similarities.jsonl"
    _write_jsonl(
        similarities,
        [
            {"word_instance_id": "wi-1", "neighbors": [{"word_instance_id": "wi-9"}]},
            {"word_instance_id": "wi-2", "neighbors": []},
        ],
    )
    monkeypatch.setitem(data.NEIGHBOR_PATHS, "similar", similarities)

    frame = data.load_split(TEST, with_neighbors=True)

    assert len(frame) == 2
    assert frame["neighbors"].iloc[0] == [{"word_instance_id": "wi-9"}]
    assert frame["neighbors"].iloc[1] == []


def test_a_split_row_without_similarities_coverage_names_the_ids(
    split_file, tmp_path, monkeypatch
) -> None:
    similarities = tmp_path / "similarities.jsonl"
    _write_jsonl(similarities, [{"word_instance_id": "wi-1", "neighbors": []}])
    monkeypatch.setitem(data.NEIGHBOR_PATHS, "similar", similarities)

    with pytest.raises(ValueError, match=r"1 split rows have no similar neighbors.*wi-2"):
        data.load_split(TEST, with_neighbors=True)


def test_the_random_source_reads_the_other_file(
    split_file, tmp_path, monkeypatch
) -> None:
    """The drift control lives in its own file; only the source flag selects it."""
    for name, neighbor in (("similarities", "wi-similar"), ("randomities", "wi-random")):
        path = tmp_path / f"{name}.jsonl"
        _write_jsonl(
            path,
            [
                {"word_instance_id": "wi-1", "neighbors": [{"word_instance_id": neighbor}]},
                {"word_instance_id": "wi-2", "neighbors": []},
            ],
        )
        monkeypatch.setitem(
            data.NEIGHBOR_PATHS, "similar" if name == "similarities" else "random", path
        )

    frame = data.load_split(TEST, with_neighbors=True, neighbors_source="random")

    assert frame["neighbors"].iloc[0] == [{"word_instance_id": "wi-random"}]


def test_an_unknown_neighbors_source_is_rejected(split_file) -> None:
    with pytest.raises(ValueError, match="neighbors_source must be one of"):
        data.load_split(TEST, with_neighbors=True, neighbors_source="shuffled")


def test_an_unknown_split_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="split must be one of"):
        data.load_split("validation")
