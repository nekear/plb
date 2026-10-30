"""Which learners the study analyses, and which of them can carry a test set.

This is study-participant policy, the same category of fact as `learners.py`, and it
lives at the top level for the same reason: several stages need it. `similarities`
scopes its scoring to the cohort and `model_b` builds its splits from it, so the
definition cannot sit inside either without one importing the other.

Two learners are excluded from the analysis entirely, and one (L06) is kept but marked
train-only: 480 labelled rows are enough to contribute training signal, not enough to
support a test metric.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from plb.learners import LEARNER_CODE_ORDER, get_user_id
from plb.utils import PROJECT_ROOT


class Cohort(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    learners: list[str] = Field(min_length=1)
    # Learners kept for training but never evaluated, because their labelled row
    # count cannot support a per-learner test metric.
    train_only: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_membership(self) -> Cohort:
        unknown = sorted(set(self.learners) - set(LEARNER_CODE_ORDER))
        if unknown:
            raise ValueError(
                f"Unknown learner code(s) {unknown} in cohort.learners; "
                f"known codes are {list(LEARNER_CODE_ORDER)}."
            )

        duplicates = sorted({code for code in self.learners if self.learners.count(code) > 1})
        if duplicates:
            raise ValueError(f"Duplicate learner code(s) in cohort.learners: {duplicates}")

        outside = sorted(set(self.train_only) - set(self.learners))
        if outside:
            raise ValueError(
                f"cohort.train_only lists learner(s) {outside} that are not in "
                "cohort.learners; a learner cannot be train-only without being in "
                "the study."
            )

        if not set(self.learners) - set(self.train_only):
            raise ValueError(
                "Every cohort learner is train-only, so no test set could be built."
            )
        return self

    def user_ids(self) -> set[str]:
        return {get_user_id(code) for code in self.learners}

    def test_eligible_user_ids(self) -> set[str]:
        return {
            get_user_id(code)
            for code in self.learners
            if code not in set(self.train_only)
        }

    def train_only_user_ids(self) -> set[str]:
        return {get_user_id(code) for code in self.train_only}

    def is_train_only(self, user_id: str) -> bool:
        return user_id in self.train_only_user_ids()


def load_cohort(params_path: Path = PROJECT_ROOT / "params.yaml") -> Cohort:
    raw = yaml.safe_load(params_path.read_text()) or {}
    return Cohort(**raw.get("cohort", {}))
