import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Collection, Iterable, Optional, TypedDict
from uuid import UUID

from tqdm import tqdm

from plb.learners import validate_learner_ids
from plb.utils import write_jsonl

from .helpers import _clean_text, _norm_key, _uuid_from_parts
from .readings import resolve_reading
from .schema import (
    DeckRow,
    LexemeRow,
    ReadingSource,
    ReviewRow,
    Submission,
    TargetLanguage,
    UserRow,
    WordInstanceRow,
)

# Further processing identified a language code mismatch in the collected dataset. Languages listed in this mapping will be converted to their correct language codes.
LANG_COMPATIBILITY_MAPPING = {"jp": "ja"}

# ---- Data structures ---- #


@dataclass
class Policy:
    should_keep_lexeme: Callable[[str, Optional[str]], bool]
    # The study analyses Japanese only, so decks in any other target language are
    # dropped here and never reach the lexeme, word-instance or review tables. A
    # collection rather than a single code, so widening the study to another L2 is a
    # configuration change instead of an edit to the filter. The default is a tuple
    # because a dataclass rejects a mutable default; callers may pass a list.
    keep_languages: Collection[str] = ("ja",)


@dataclass
class ReviewDict(TypedDict):
    card_id: int
    timestamp_ms: int
    interval_before: int
    interval_after: int
    factor_after: int
    time_taken_ms: int
    review_type: int
    rating: int


@dataclass
class WordCandidate:
    user_id: str
    deck_id: UUID
    target_language: str
    form: str
    meaning: str
    reading: Optional[str]
    reading_source: ReadingSource | None
    pitch: Optional[str]
    created_at_ms: int
    note_id: int
    card_id: int
    select_score: float  # efficiency of selected card (for dedup ranking)
    reviews: list[ReviewDict]  # raw review dicts for the selected card
    first_reviewed_at_ms: Optional[int]


@dataclass
class SubmissionResult:
    user: UserRow
    decks: list[DeckRow]
    candidates: list[WordCandidate] = field(default_factory=list)


def write_users(rows: list[UserRow], path: Path) -> None:
    write_jsonl(sorted(rows, key=lambda r: r.user_id), path)


def write_decks(rows: list[DeckRow], path: Path) -> None:
    write_jsonl(sorted(rows, key=lambda r: (r.user_id, r.deck_id)), path)


def write_lexemes(rows: list[LexemeRow], path: Path) -> None:
    write_jsonl(sorted(rows, key=lambda r: (r.language, r.form, r.lexeme_id)), path)


def write_word_instances(rows: list[WordInstanceRow], path: Path) -> None:
    write_jsonl(
        sorted(rows, key=lambda r: (r.deck_id, r.created_at, r.word_instance_id)),
        path,
    )


def write_reviews(rows: list[ReviewRow], path: Path) -> None:
    write_jsonl(
        sorted(rows, key=lambda r: (r.word_instance_id, r.timestamp_ms, r.review_id)),
        path,
    )


# ---- Multi-card resolution ---- #


def _efficiency(c: dict) -> float:
    """Efficiency is a stability per review ratio which is later used to pick the easier card."""
    stab = c.get("fsrs_stability")
    stab = float(stab) if isinstance(stab, (int, float)) else 0.0
    reps = c.get("reps_total") or 0
    return stab / reps if reps > 0 else 0.0


def _lapse_ratio(c: dict) -> float:
    reps = c.get("reps_total") or 0
    return (c.get("lapses_total") or 0) / reps if reps > 0 else 1.0


def select_receptive_card(cards: list[dict]) -> Optional[dict]:
    """
    Picks the receptive/passive-recall card for a note.

    Essentially, it ranks by stability/reps (desc), then prefers fewer failures (lapse ratio,
    asc), then card_id (asc) for determinism. Cards with no reviews carry no receptive signal and sort last.
    """
    if not cards:
        return None
    return sorted(
        cards,
        key=lambda c: (-_efficiency(c), _lapse_ratio(c), c.get("card_id", 0)),
    )[0]


# ---- Submission processing ---- #
def _ms_to_dt(ms: int) -> datetime:
    """Converts a timestamp in milliseconds to a timezone-aware datetime."""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def load_release(release_dir: Path) -> dict[str, dict]:
    """Regroups the release tables into one raw-submission-shaped dict per learner.

    The release keeps each learner's rows in their original order, which the dedup
    tie-break and first-seen lexeme logic below depend on.
    """
    tables = {
        "user_profile": "learners",
        "decks": "decks",
        "models": "note_types",
        "notes": "notes",
        "cards": "cards",
        "reviews": "reviews",
    }
    learners: dict[str, dict] = defaultdict(lambda: {key: [] for key in tables})
    for key, table in tables.items():
        with (release_dir / f"{table}.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                learners[row["learner_id"]][key].append(row)
    for raw in learners.values():
        (profile,) = raw["user_profile"]
        raw["user_profile"] = profile
        # The release writes profile fields only with consent, and null otherwise.
        raw["consent"] = {"publish_userinfo": profile["native_languages"] is not None}
    return dict(learners)


def process_submission(sub: Submission, raw: dict, policy: Policy) -> SubmissionResult:
    user_id = sub.learner_id

    consent = raw.get("consent") or {}
    publish_userinfo = consent.get("publish_userinfo", True)
    drop_profile = not publish_userinfo

    # ---- user row ----
    profile = raw.get("user_profile") or {}
    if drop_profile:
        user = UserRow(user_id=user_id)
    else:
        tls = []
        for tl in profile.get("target_languages") or []:
            lang = (tl.get("lang") or "").strip()
            prof = tl.get("proficiency")
            tls.append(TargetLanguage(lang=lang, proficiency=prof))
        user = UserRow(
            user_id=user_id,
            native_languages=[
                s for s in (profile.get("native_languages") or []) if s and s.strip()
            ],
            target_languages=tls,
            domain_interests=[
                s for s in (profile.get("domain_interests") or []) if s and s.strip()
            ],
        )

    # ---- decks (FSRS only) + lookups ----
    deck_rows: list[DeckRow] = []
    deck_ids_to_uuids: dict[int, UUID] = {}
    deck_lang: dict[int, str] = {}
    for deck_raw in raw.get("decks") or []:
        if (deck_raw.get("scheduler") or "").upper() != "FSRS":
            continue  # all collected decks are FSRS; guard anyway
        deck_id = deck_raw["deck_id"]
        # some decks (containers / "Default") carry target_language "" -> None
        lang: str = deck_raw.get("target_language")
        if not lang:
            print(
                f"Skipping {deck_id} of {sub.learner_id} because target_language is empty..."
            )
            continue

        if lang in LANG_COMPATIBILITY_MAPPING:
            lang = LANG_COMPATIBILITY_MAPPING[lang]

        # Runs after the compatibility remap, so a "jp"-tagged deck is kept.
        if lang not in policy.keep_languages:
            continue

        deck_uuid = _uuid_from_parts(user_id, "deck", deck_id)
        deck_ids_to_uuids[deck_id] = deck_uuid
        deck_lang[deck_id] = lang
        deck_rows.append(
            DeckRow(
                deck_id=deck_uuid,
                user_id=user_id,
                fsrs_weights=list(deck_raw.get("fsrs_weights") or []),
                desired_retention=deck_raw.get("desired_retention"),
                target_language=lang,
            )
        )
    # ---- index notes / cards / reviews ----
    # Keying on model_id (string-normalized): schema 1.4 has no model_key, and within
    # a single submission model_id is already unambiguous.
    models_by_model_id = {str(m["model_id"]): m for m in raw.get("models") or []}
    cards_by_note_id: dict[int, list[dict]] = defaultdict(list)
    for c in raw.get("cards") or []:
        cards_by_note_id[c["note_id"]].append(c)
    reviews_by_card_id: dict[int, list[ReviewDict]] = defaultdict(list)
    for r in raw.get("reviews") or []:
        reviews_by_card_id[r["card_id"]].append(r)

    result = SubmissionResult(user=user, decks=deck_rows)

    # ---- notes -> candidates ----
    for note in raw.get("notes") or []:
        model_id = str(note.get("model_id"))
        extractor = sub.model_extractors.get(model_id)
        if extractor is None:
            continue  # if model is not configured, skipping it

        model = models_by_model_id.get(model_id)
        field_names = (model or {}).get("field_names") or []
        values = note.get("fields") or []
        named = {
            name: values[i] if i < len(values) else None
            for i, name in enumerate(field_names)
        }

        content = extractor(named)
        if content is None:
            continue
        form = _clean_text(content.get("form"))
        meaning = _clean_text(content.get("meaning"))

        if not form or not meaning:
            continue

        # picking the receptive card (if several cards are present per note)
        note_cards = [
            c
            for c in cards_by_note_id.get(note["note_id"], [])
            if c["deck_id"] in deck_ids_to_uuids
        ]
        card = select_receptive_card(note_cards)
        if card is None:
            continue  # no FSRS card -> no review signal -> drop

        deck_id: int = card["deck_id"]
        # WARN: deck_lang may not have an entry for deck_id, but I'm assuming it does
        # if typing is needed, replace with Optional[str] = deck_lang.get(deck_id)
        lang = deck_lang[deck_id]

        if not policy.should_keep_lexeme(form, lang):
            continue

        reading, reading_source = resolve_reading(
            form=form,
            reading=_clean_text(content.get("reading")),
            language=lang,
        )

        raw_reviews = list(reviews_by_card_id.get(card["card_id"], []))
        first_ms = (
            min((r["timestamp_ms"] for r in raw_reviews), default=None)
            if raw_reviews
            else None
        )

        result.candidates.append(
            WordCandidate(
                user_id=user_id,
                deck_id=deck_ids_to_uuids[deck_id],
                target_language=lang,
                form=form,
                meaning=meaning,
                reading=reading,
                reading_source=reading_source,
                pitch=content.get("pitch"),  # kept raw (may be SVG); not cleaned
                created_at_ms=note.get("created_at_ms"),
                note_id=note["note_id"],
                card_id=card["card_id"],
                select_score=_efficiency(card),
                reviews=raw_reviews,
                first_reviewed_at_ms=first_ms,
            )
        )

    return result


def dedup_candidates(cands: list[WordCandidate]) -> list[WordCandidate]:
    """
    There're cases when the same lexical item can be present in different decks as different notes.
    This function collapses duplicate lexical items within a user to the most-receptive
    (highest "efficiency") representative, and carries its earliest first_reviewed_at across the group.
    Review logs are NOT spliced.
    """
    groups: dict[tuple, list[WordCandidate]] = defaultdict(list)
    for c in cands:
        key = (c.target_language, _norm_key(c.form), _norm_key(c.reading))
        groups[key].append(c)

    out: list[WordCandidate] = []
    for members in groups.values():
        rep = max(members, key=lambda c: c.select_score)
        if any(c.reading_source == "original" for c in members):
            rep.reading_source = "original"
        first_candidates = [
            c.first_reviewed_at_ms
            for c in members
            if c.first_reviewed_at_ms is not None
        ]
        rep.first_reviewed_at_ms = min(first_candidates) if first_candidates else None
        out.append(rep)
    return out


def candidates_to_rows(
    # (language, NFKC-normalized form, NFKC-normalized reading) -> lexeme row.
    # Reading is part of the identity, because (as an example) same-form heteronyms (e.g. Japanese
    # 上手 じょうず vs うわて) are different lexemes.
    form_to_lexemes: dict[tuple[str, str, str], LexemeRow],
    candidates: list[WordCandidate],  # candidates of a single user
) -> tuple[
    dict[tuple[str, str, str], LexemeRow], list[WordInstanceRow], list[ReviewRow]
]:
    word_instance_rows: list[WordInstanceRow] = []
    review_rows: list[ReviewRow] = []
    for c in candidates:
        key = (c.target_language, _norm_key(c.form), _norm_key(c.reading))
        if key not in form_to_lexemes:
            lexeme_id = _uuid_from_parts("lexeme", *key)
            form_to_lexemes[key] = LexemeRow(
                lexeme_id=lexeme_id,
                form=c.form,
                reading=c.reading,
                reading_source=c.reading_source,
                pitch=c.pitch,
                language=c.target_language,
            )
        else:
            lexeme_id = form_to_lexemes[key].lexeme_id
            form_to_lexemes[key].pitch = form_to_lexemes[key].pitch or c.pitch
            if c.reading_source == "original":
                form_to_lexemes[key].reading_source = "original"

        word_instance_id = _uuid_from_parts(
            c.user_id, "word_instance", c.note_id, c.card_id
        )
        word_instance_rows.append(
            WordInstanceRow(
                lexeme_id=lexeme_id,
                word_instance_id=word_instance_id,
                deck_id=c.deck_id,
                meaning=c.meaning,
                created_at=_ms_to_dt(c.created_at_ms),
                # TODO: there may be a more elegant solution
                first_reviewed_at=_ms_to_dt(c.first_reviewed_at_ms)
                if c.first_reviewed_at_ms
                else None,
            )
        )

        # TODO: add fsrs recomputation
        for r in c.reviews:
            rating = r.get("rating")
            if rating not in (1, 2, 3, 4):
                continue  # drop manual/rescheduled non-grades
            # Anki's revlog is keyed by (card_id, timestamp_ms)
            review_id = _uuid_from_parts(
                word_instance_id, "review", r.get("timestamp_ms")
            )
            review_rows.append(
                ReviewRow(
                    review_id=review_id,
                    word_instance_id=word_instance_id,
                    timestamp_ms=r.get("timestamp_ms"),
                    rating=rating,
                    interval_before=r.get("interval_before"),
                    interval_after=r.get("interval_after"),
                    factor_after=r.get("factor_after"),
                    time_taken_ms=r.get("time_taken_ms"),
                    review_type=r.get("review_type"),
                )
            )
    return form_to_lexemes, word_instance_rows, review_rows


def build_dataset(
    submissions: Iterable[Submission],
    release_dir: str | Path,
    output_dir: str | Path,
    policy: Policy,
) -> dict[str, int]:
    releases = load_release(Path(release_dir))

    # Sort so output row order (and therefore DVC's output hash) doesn't
    # depend on the caller's iteration order.
    submissions = sorted(submissions, key=lambda s: s.learner_id)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    users: list[UserRow] = []
    decks: list[DeckRow] = []
    lexemes: list[LexemeRow] = []
    word_instances: list[WordInstanceRow] = []
    reviews: list[ReviewRow] = []

    # (language, normalized form, normalized reading) -> lexeme row, used to
    # deduplicate lexeme rows
    form_to_lexemes: dict[tuple[str, str, str], LexemeRow] = {}

    failures: list[str] = []

    progress = tqdm(
        submissions,
        desc="Processing submissions",
        unit="submission",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )
    for submission in progress:
        progress.set_postfix_str(submission.learner_id, refresh=False)
        try:
            raw = releases[submission.learner_id]
            result = process_submission(submission, raw, policy)

            if not result.decks:
                # Every deck was filtered out by policy.keep_languages — writing the
                # user row would leave a learner who owns nothing.
                continue

            deduped = dedup_candidates(result.candidates)
            updated_form_to_lexemes, word_rows, review_rows = candidates_to_rows(
                form_to_lexemes, deduped
            )

            users.append(result.user)
            decks.extend(result.decks)
            form_to_lexemes = updated_form_to_lexemes
            word_instances.extend(word_rows)
            reviews.extend(review_rows)
        except Exception as exc:
            failures.append(submission.learner_id)
            tqdm.write(f"  ✗ Failed to process {submission.learner_id}: {exc}")

    lexemes = list(form_to_lexemes.values())

    # The surviving cohort is the study's cohort: a new submission or a broken
    # language filter must fail here rather than silently redefine it.
    validate_learner_ids(user.user_id for user in users)

    write_users(users, out / "users.jsonl")
    write_decks(decks, out / "decks.jsonl")
    write_lexemes(lexemes, out / "lexemes.jsonl")
    write_word_instances(word_instances, out / "word_instances.jsonl")
    write_reviews(reviews, out / "reviews.jsonl")

    return {
        "users": len(users),
        "decks": len(decks),
        "lexemes": len(lexemes),
        "word_instances": len(word_instances),
        "reviews": len(reviews),
        "failed_submissions": len(failures),
    }
