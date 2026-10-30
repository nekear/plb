from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .helpers import _valid_string

# ---- Input configuration ---- #

# An extractor receives the note's fields already zipped to {field_name: value}
# and returns {"form","meaning","reading","pitch"} (any may be None/absent).
Extractor = Callable[[dict[str, Any]], Optional[dict[str, Optional[str]]]]


class Submission(BaseModel):
    # Learner code in the public release (data/release), e.g. "L01".
    learner_id: str
    # Keyed by model_id. Only these models are read.
    # Extractors are used to get `from`, `meaning`, `reading` and `pitch` data.
    model_extractors: dict[str, Extractor]


# ---- Output configuration (Parquet types) ---- #
ProficiencyLevel = Literal["A1", "A2", "B1", "B2", "C1", "C2", "native", "Unsure"]
ReadingSource = Literal["original", "derived"]


# Base confirmation of a row model's fields
class StrictRowModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class TargetLanguage(StrictRowModel):
    lang: str
    proficiency: ProficiencyLevel

    @field_validator("lang")
    @classmethod
    def validate_language_code(cls, value: str) -> str:
        validated = _valid_string(value, "language code cannot be empty")
        if validated is None:
            raise ValueError("language code cannot be empty")
        return validated


class UserRow(StrictRowModel):
    """Final row schema for users.parquet."""

    user_id: str
    native_languages: list[str] = Field(default_factory=list)
    target_languages: list[TargetLanguage] = Field(default_factory=list)
    domain_interests: list[str] = Field(default_factory=list)

    @field_validator("native_languages", "domain_interests")
    @classmethod
    def validate_non_empty_strings(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("list values cannot be empty strings")
        return cleaned


class DeckRow(StrictRowModel):
    """Final row schema for decks.parquet."""

    deck_id: UUID
    user_id: str
    fsrs_weights: list[float] = Field(default_factory=list)
    desired_retention: float = Field(..., gt=0, le=1)
    target_language: str

    @field_validator("target_language")
    @classmethod
    def validate_optional_language_code(cls, value: str) -> str:
        return _valid_string(value, "target_language cannot be an empty string")


class LexemeRow(StrictRowModel):
    """Final row schema for lexemes.parquet."""

    lexeme_id: UUID
    language: str
    form: str
    reading: str | None = None
    reading_source: ReadingSource | None = None
    pitch: str | None = None

    @field_validator("form", "reading", "pitch")
    @classmethod
    def validate_required_strings(cls, value: str) -> str:
        return _valid_string(value, "Either form, reading, or pitch is empty or none")

    @model_validator(mode="after")
    def validate_reading_provenance(self) -> "LexemeRow":
        if (self.reading is None) != (self.reading_source is None):
            raise ValueError("reading and reading_source must either both be set or both be null")
        return self


class WordInstanceRow(StrictRowModel):
    """Final row schema for words.parquet."""

    word_instance_id: UUID
    lexeme_id: UUID
    deck_id: UUID
    meaning: str
    created_at: datetime
    first_reviewed_at: datetime | None = None

    @field_validator("meaning")
    @classmethod
    def validate_meaning(cls, value: str) -> str:
        return _valid_string(value, "Meaning cannot be empty or none")


class ReviewRow(StrictRowModel):
    """Final row schema for reviews.parquet."""

    review_id: UUID
    word_instance_id: UUID
    timestamp_ms: int = Field(..., ge=0)
    rating: int = Field(..., ge=1, le=4)
    interval_before: int
    interval_after: int
    factor_after: int
    time_taken_ms: int = Field(..., ge=0)
    review_type: int = Field(..., ge=0)


__all__ = [
    "DeckRow",
    "ProficiencyLevel",
    "ReadingSource",
    "ReviewRow",
    "TargetLanguage",
    "UserRow",
    "LexemeRow",
    "WordInstanceRow",
]
