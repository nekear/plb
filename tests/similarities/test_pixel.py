import pytest

from plb.similarities.pixel import compute_pixel_similarity


def test_pixel_similarity_is_one_for_identical_characters() -> None:
    assert compute_pixel_similarity("シ", "シ") == 1.0


def test_pixel_similarity_is_symmetric_for_affine_pair() -> None:
    forward = compute_pixel_similarity("シ", "ツ")
    reverse = compute_pixel_similarity("ツ", "シ")

    assert forward == pytest.approx(0.530408, abs=1e-6)
    assert reverse == forward


def test_pixel_similarity_requires_single_characters() -> None:
    with pytest.raises(ValueError, match="single characters"):
        compute_pixel_similarity("シツ", "ツ")
