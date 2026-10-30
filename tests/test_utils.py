import json

import pytest

from plb.utils import PROJECT_ROOT, get_dataset_as_path, write_json


def test_get_dataset_as_path_prefixes_data_directory() -> None:
    assert get_dataset_as_path("raw/tubelex") == PROJECT_ROOT / "data" / "raw" / "tubelex"


def test_get_dataset_as_path_rejects_absolute_routes(tmp_path) -> None:
    with pytest.raises(ValueError):
        get_dataset_as_path(tmp_path)


def test_write_json_uses_project_format(tmp_path) -> None:
    path = tmp_path / "payload.json"

    write_json(path, {"greeting": "こんにちは", "values": [1, 2]})

    assert json.loads(path.read_text()) == {"greeting": "こんにちは", "values": [1, 2]}
    assert path.read_text() == '{\n  "greeting": "こんにちは",\n  "values": [\n    1,\n    2\n  ]\n}'


def test_write_json_rejects_non_json_numbers(tmp_path) -> None:
    with pytest.raises(ValueError):
        write_json(tmp_path / "payload.json", {"value": float("nan")})
