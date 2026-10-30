from pydantic import BaseModel, ConfigDict, Field, model_validator


class SimilarityEntry(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    id: str
    form: str = Field(min_length=1)
    meaning: str = Field(min_length=1)
    reading: str | None = None


class PhonologicalSimilarityEntry(SimilarityEntry):
    reading: str = Field(min_length=1)


class OrthographicSimilarityResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    ortho_stroke_sim: float = Field(ge=0.0, le=1.0)
    ortho_stroke_coverage: float = Field(ge=0.0, le=1.0)
    ortho_kana_confusion: float = Field(ge=0.0, le=1.0)
    ortho_kana_coverage: float = Field(ge=0.0, le=1.0)


class NeighborSimilarityResult(BaseModel):
    """One neighbour of a target word, as known at the target's first review.

    Every field describing the neighbour's learning state is as-of that moment, never
    end-of-history. In particular `plb_14` — personalized learning burden, the review
    count at which the word reached the stability target — is **right-censored**:

    - `is_censored` false -> the neighbour had reached the target by then, after
      `plb_14` reviews;
    - `is_censored` true  -> it had not, and all that is known is that its burden
      exceeds `reviews` (which may be 0 when the learner had not started the word
      yet).

    So a censored row is a lower bound, not missing data. Aggregating only the
    uncensored ones biases the neighbourhood's burden downward, because censoring
    correlates with difficulty.
    """

    model_config = ConfigDict(frozen=True)

    word_instance_id: str = Field(min_length=1)
    s: float | None = Field(default=None, allow_inf_nan=False)
    d: float | None = Field(default=None, allow_inf_nan=False)
    reviews: int = Field(ge=0)
    plb_14: int | None = Field(default=None, ge=1)
    log_plb_14: float | None = Field(
        default=None,
        ge=0.0,
        allow_inf_nan=False,
    )
    # Redundant with a null burden, but stated explicitly so downstream filters read
    # as intent ("drop censored neighbours") rather than as a null check.
    is_censored: bool
    sem_sim: float = Field(ge=0.0, le=1.0)
    phon_sim: float | None = Field(default=None, ge=0.0, le=1.0)
    mor_sim: float = Field(ge=0.0, le=1.0)
    ortho_stroke_sim: float = Field(ge=0.0, le=1.0)
    ortho_stroke_coverage: float = Field(ge=0.0, le=1.0)
    ortho_kana_confusion: float = Field(ge=0.0, le=1.0)
    ortho_kana_coverage: float = Field(ge=0.0, le=1.0)

    # Each raw score as a percentile of the null distribution of random pairs, which
    # is what makes the channels comparable. Null where the channel does not apply to
    # this pair (no reading, or no character coverage).
    sem_pct: float = Field(ge=0.0, le=1.0)
    phon_pct: float | None = Field(default=None, ge=0.0, le=1.0)
    mor_pct: float = Field(ge=0.0, le=1.0)
    ortho_stroke_pct: float | None = Field(default=None, ge=0.0, le=1.0)
    ortho_kana_pct: float | None = Field(default=None, ge=0.0, le=1.0)

    # Channels that selected this neighbour into the target's list. A pair chosen by
    # several channels is stored once; per-channel features filter on this.
    selected_by: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_censoring_is_consistent(self) -> "NeighborSimilarityResult":
        if self.is_censored != (self.plb_14 is None):
            raise ValueError(
                "is_censored must agree with plb_14: a censored "
                "neighbour has no observed burden, an uncensored one does."
            )
        if self.is_censored and self.log_plb_14 is not None:
            raise ValueError(
                "A censored neighbour cannot carry log_plb_14."
            )
        return self


class WordSimilaritiesResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    word_instance_id: str = Field(min_length=1)
    neighbors: list[NeighborSimilarityResult]
