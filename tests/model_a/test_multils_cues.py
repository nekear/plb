import pytest

from plb.model_a.multils.cues import make_l1_clue


@pytest.mark.parametrize(
    ("gloss", "expected"),
    [
        ("debate", "d_____"),
        ("even though", "e___ ______"),
        ("討論", "討_"),
        ("l'état", "l'____"),
        ("2 items", "2 _____"),
    ],
)
def test_l1_clue_masks_unicode_letters_and_numbers_after_first(
    gloss: str, expected: str
) -> None:
    assert make_l1_clue(gloss) == expected


def test_l1_clue_rejects_nonlexical_gloss() -> None:
    with pytest.raises(ValueError, match="no Unicode letter or number"):
        make_l1_clue("---")
