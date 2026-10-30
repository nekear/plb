import pytest

from plb.esm.kana_confusability import KanaConfusability
from plb.similarities import orthographic
from plb.similarities.types import SimilarityEntry


def entry(form: str) -> SimilarityEntry:
    return SimilarityEntry(id=form, form=form, meaning="test")


def test_kana_confusability_exposes_matrix_coverage() -> None:
    confusability = KanaConfusability(orthographic._ESM_PATH)

    assert confusability.supports("シ")
    assert not confusability.supports("が")
    assert confusability.compare("シ", "シ")["score"] == 1.0

    with pytest.raises(KeyError, match="No empirical kana comparison"):
        confusability.compare("が", "が")


def test_weighted_edit_similarity_uses_soft_substitution_cost() -> None:
    def character_similarity(first: str, second: str) -> float | None:
        if first == second:
            return 1.0
        if (first, second) == ("b", "c"):
            return 0.75
        return None

    result = orthographic._weighted_edit_similarity(
        "ab",
        "ac",
        character_similarity,
    )

    assert result.similarity == pytest.approx(0.875)
    assert result.coverage == 1.0


def test_weighted_edit_coverage_excludes_insertions() -> None:
    result = orthographic._weighted_edit_similarity(
        "猫",
        "子猫",
        lambda first, second: 1.0 if first == second == "猫" else None,
    )

    assert result.similarity == pytest.approx(0.5)
    assert result.coverage == 1.0


def test_weighted_edit_coverage_is_symmetric_when_paths_tie() -> None:
    def character_similarity(first: str, second: str) -> float | None:
        return 1.0 if first == second and first in {"K", "k"} else None

    forward = orthographic._weighted_edit_similarity(
        "KkK",
        "kxKk",
        character_similarity,
    )
    reverse = orthographic._weighted_edit_similarity(
        "kxKk",
        "KkK",
        character_similarity,
    )

    assert forward == reverse


def test_compute_orthographic_similarity_returns_stroke_and_kana_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def stroke_similarity(first: str, second: str) -> float | None:
        return 1.0 if first == second == "食" else None

    def kana_similarity(first: str, second: str) -> float | None:
        if first == second == "べ":
            return 1.0
        if (first, second) == ("る", "た"):
            return 0.25
        return None

    monkeypatch.setattr(orthographic, "compute_niteru_similarity", stroke_similarity)
    monkeypatch.setattr(orthographic, "_kana_character_similarity", kana_similarity)

    result = orthographic.compute_orthographic_similarity(
        entry("食べる"),
        entry("食べた"),
    )

    assert result.ortho_stroke_sim == pytest.approx(1 / 3)
    assert result.ortho_stroke_coverage == pytest.approx(1 / 3)
    assert result.ortho_kana_confusion == pytest.approx(5 / 12)
    assert result.ortho_kana_coverage == pytest.approx(2 / 3)
    assert set(result.model_dump()) == {
        "ortho_stroke_sim",
        "ortho_stroke_coverage",
        "ortho_kana_confusion",
        "ortho_kana_coverage",
    }


@pytest.mark.parametrize(
    ("first", "second", "feature", "expected"),
    [
        ("待", "持", "ortho_stroke_sim", 2 / 3),
        ("シ", "ツ", "ortho_kana_confusion", 0.997730530339226),
    ],
)
def test_real_character_backends_are_used_by_the_lexeme_comparison(
    first: str,
    second: str,
    feature: str,
    expected: float,
) -> None:
    result = orthographic.compute_orthographic_similarity(
        entry(first),
        entry(second),
    )

    assert getattr(result, feature) == pytest.approx(expected)


def test_unsupported_channel_returns_zero_similarity_and_coverage() -> None:
    result = orthographic.compute_orthographic_similarity(
        entry("が"),
        entry("ざ"),
    )

    assert result.ortho_stroke_sim == 0.0
    assert result.ortho_stroke_coverage == 0.0
    assert result.ortho_kana_confusion == 0.0
    assert result.ortho_kana_coverage == 0.0
