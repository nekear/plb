"""Stable, paper-facing codes for learners (participants).

Since the study reads the public release (`data/release`), a learner's ``user_id``
*is* its code: the release never carries the original submission IDs. The mapping
below is therefore the identity, kept so callers can stay agnostic of that fact.
"""

from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Final
from uuid import UUID


LEARNER_CODE_ORDER: Final[tuple[str, ...]] = ("L01", "L02", "L03", "L04", "L05", "L06")

LEARNER_CODE_BY_USER_ID: Final[Mapping[str, str]] = MappingProxyType(
    {code: code for code in LEARNER_CODE_ORDER}
)

# Derived, never written by hand: a second literal would be free to drift.
USER_ID_BY_LEARNER_CODE: Final[Mapping[str, str]] = MappingProxyType(
    {code: user_id for user_id, code in LEARNER_CODE_BY_USER_ID.items()}
)

# Freezing the palette as hex values so library upgrades and filtered plots do
# not silently change a learner's colour between analyses.
# TODO: it may be better to infer colors from the colorblind palette of seaborn directly to make them accessible to readers with disabilities
LEARNER_COLOR_BY_CODE: Final[Mapping[str, str]] = MappingProxyType(
    {
        "L01": "#0173b2",
        "L02": "#de8f05",
        "L03": "#029e73",
        "L04": "#d55e00",
        "L05": "#cc78bc",
        "L06": "#ca9161",
    }
)


def get_learner_code(user_id: str | UUID) -> str:
    """Returns the stable learner code for a study ``user_id``.

    Unknown identifiers fail rather than producing a missing plot label. New
    learners must be assigned the next unused code explicitly; existing codes
    must never be derived again, renumbered, or reused.
    """

    normalized_user_id = str(user_id)
    try:
        return LEARNER_CODE_BY_USER_ID[normalized_user_id]
    except KeyError as error:
        raise ValueError(
            f"No learner code is assigned to user_id {normalized_user_id!r}."
        ) from error


def get_user_id(learner_code: str) -> str:
    """Returns the study ``user_id`` behind a paper-facing learner code.

    The inverse of `get_learner_code`, so configuration can be written in codes
    that mean something to a reader rather than in opaque UUIDs.
    """

    try:
        return USER_ID_BY_LEARNER_CODE[learner_code]
    except KeyError as error:
        raise ValueError(
            f"Unknown learner code {learner_code!r}; "
            f"known codes are {list(LEARNER_CODE_ORDER)}."
        ) from error


def validate_learner_ids(user_ids: Iterable[str | UUID]) -> None:
    """Raises if any study ``user_id`` lacks a paper-facing learner code."""

    unknown_user_ids = sorted(
        {
            str(user_id)
            for user_id in user_ids
            if str(user_id) not in LEARNER_CODE_BY_USER_ID
        }
    )
    if unknown_user_ids:
        raise ValueError(
            "Unmapped learner user_id(s): " + ", ".join(unknown_user_ids)
        )
