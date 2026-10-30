import pytest
from pydantic import ValidationError

from plb.similarities.types import PhonologicalSimilarityEntry


@pytest.fixture(autouse=True)
def clear_reading_similarity_cache():
    phonological = pytest.importorskip("plb.similarities.phonological")
    phonological._compute_reading_similarity.cache_clear()
    yield
    phonological._compute_reading_similarity.cache_clear()


def valid_entry() -> dict[str, str]:
    return {
        "id": "word-instance",
        "form": "猫",
        "meaning": "cat",
        "reading": "ねこ",
    }


def test_phonological_entry_accepts_required_fields() -> None:
    entry = PhonologicalSimilarityEntry(**valid_entry())

    assert entry.form == "猫"
    assert entry.meaning == "cat"
    assert entry.reading == "ねこ"


@pytest.mark.parametrize("field", ["form", "reading"])
def test_phonological_entry_rejects_missing_required_fields(field: str) -> None:
    data = valid_entry()
    data.pop(field)

    with pytest.raises(ValidationError):
        PhonologicalSimilarityEntry(**data)


@pytest.mark.parametrize("field", ["form", "reading"])
def test_phonological_entry_rejects_empty_required_fields(field: str) -> None:
    data = valid_entry()
    data[field] = ""

    with pytest.raises(ValidationError):
        PhonologicalSimilarityEntry(**data)


def test_compute_phonological_similarity_validates_mapping_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phonological = pytest.importorskip("plb.similarities.phonological")

    def fake_kana_to_ipa(reading: str) -> str:
        return f"ipa:{reading}"

    monkeypatch.setattr(phonological, "kana_to_ipa", fake_kana_to_ipa)
    monkeypatch.setattr(
        phonological,
        "feature_distance",
        type(
            "FakeDistance",
            (),
            {
                "feature_edit_distance_div_maxlen": lambda self, left, right: 0.25
            },
        )(),
    )

    result = phonological.compute_phonological_similarity(valid_entry(), valid_entry())

    assert result == 0.75


@pytest.mark.parametrize(
    ("distance", "expected_similarity"),
    [(-0.1, 1.0), (1.2, 0.0)],
)
def test_compute_phonological_similarity_clamps_score_to_valid_range(
    monkeypatch: pytest.MonkeyPatch,
    distance: float,
    expected_similarity: float,
) -> None:
    phonological = pytest.importorskip("plb.similarities.phonological")
    monkeypatch.setattr(phonological, "kana_to_ipa", lambda reading: "ipa")
    monkeypatch.setattr(
        phonological,
        "feature_distance",
        type(
            "FakeDistance",
            (),
            {
                "feature_edit_distance_div_maxlen": lambda self, left, right: distance
            },
        )(),
    )

    result = phonological.compute_phonological_similarity(valid_entry(), valid_entry())

    assert result == expected_similarity


@pytest.mark.parametrize("invalid_position", [0, 1])
def test_compute_phonological_similarity_rejects_entries_without_reading(
    invalid_position: int,
) -> None:
    phonological = pytest.importorskip("plb.similarities.phonological")
    entries = [valid_entry(), valid_entry()]
    entries[invalid_position].pop("reading")

    with pytest.raises(ValidationError):
        phonological.compute_phonological_similarity(*entries)


def test_compute_phonological_similarity_raises_when_ipa_conversion_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phonological = pytest.importorskip("plb.similarities.phonological")
    monkeypatch.setattr(phonological, "kana_to_ipa", lambda reading: None)

    with pytest.raises(
        phonological.PhonologicalSimilarityComputationError,
        match="Failed to convert readings to valid IPA",
    ):
        phonological.compute_phonological_similarity(valid_entry(), valid_entry())


def test_real_phonological_similarity_is_one_for_identical_readings() -> None:
    phonological = pytest.importorskip("plb.similarities.phonological")
    entry = valid_entry()

    result = phonological.compute_phonological_similarity(entry, entry)

    assert result == pytest.approx(1.0)


def test_real_phonological_similarity_is_in_range_for_different_readings() -> None:
    phonological = pytest.importorskip("plb.similarities.phonological")
    first = valid_entry()
    second = {**valid_entry(), "form": "犬", "meaning": "dog", "reading": "いぬ"}

    result = phonological.compute_phonological_similarity(first, second)

    assert 0.0 <= result < 1.0


def test_phonological_similarity_caches_symmetric_reading_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phonological = pytest.importorskip("plb.similarities.phonological")
    calls = 0

    class FakeDistance:
        def feature_edit_distance_div_maxlen(self, left: str, right: str) -> float:
            nonlocal calls
            calls += 1
            return 0.25

    monkeypatch.setattr(phonological, "kana_to_ipa", lambda reading: f"ipa:{reading}")
    monkeypatch.setattr(phonological, "feature_distance", FakeDistance())
    first = valid_entry()
    second = {**valid_entry(), "id": "dog", "reading": "いぬ"}

    forward = phonological.compute_phonological_similarity(first, second)
    reverse = phonological.compute_phonological_similarity(second, first)

    assert forward == reverse == 0.75
    assert calls == 1
