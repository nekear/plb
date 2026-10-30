import pytest
from pydantic import ValidationError

from plb.cohort import Cohort, load_cohort
from plb.learners import LEARNER_CODE_BY_USER_ID, get_user_id


def test_user_id_lookup_round_trips() -> None:
    for user_id, code in LEARNER_CODE_BY_USER_ID.items():
        assert get_user_id(code) == user_id


def test_unknown_learner_code_is_rejected() -> None:
    with pytest.raises(ValidationError, match="Unknown learner code"):
        Cohort(learners=["L01", "L99"])


def test_train_only_must_be_part_of_the_cohort() -> None:
    # Otherwise the setting reads as if a learner were included when it is not.
    with pytest.raises(ValidationError, match="not in cohort.learners"):
        Cohort(learners=["L01", "L02"], train_only=["L03"])


def test_a_cohort_of_only_train_only_learners_is_rejected() -> None:
    with pytest.raises(ValidationError, match="no test set could be built"):
        Cohort(learners=["L06"], train_only=["L06"])


def test_duplicate_learners_are_rejected() -> None:
    with pytest.raises(ValidationError, match="Duplicate learner code"):
        Cohort(learners=["L01", "L01"])


def test_membership_partitions_into_test_eligible_and_train_only() -> None:
    cohort = Cohort(learners=["L01", "L02", "L06"], train_only=["L06"])

    assert cohort.user_ids() == {get_user_id(c) for c in ("L01", "L02", "L06")}
    assert cohort.test_eligible_user_ids() == {get_user_id(c) for c in ("L01", "L02")}
    assert cohort.train_only_user_ids() == {get_user_id("L06")}
    assert cohort.is_train_only(get_user_id("L06"))
    assert not cohort.is_train_only(get_user_id("L01"))


def test_load_cohort_reads_params(tmp_path) -> None:
    params = tmp_path / "params.yaml"
    params.write_text("cohort:\n  learners: [L01, L02, L06]\n  train_only: [L06]\n")

    cohort = load_cohort(params)

    assert cohort.learners == ["L01", "L02", "L06"]
    assert cohort.train_only == ["L06"]
