import numpy as np
import pytest

from plb.similarities import cache


class _Word:
    def __init__(self, word_instance_id: str) -> None:
        self.word_instance_id = word_instance_id


def _words(*ids: str) -> list[_Word]:
    return [_Word(word_instance_id) for word_instance_id in ids]


def _manifest(words_by_user: dict[str, list[_Word]], **counts: int) -> cache.CacheManifest:
    return cache.CacheManifest(
        learners={
            user_id: cache.LearnerCacheEntry(
                pair_count=counts.get(user_id, 0),
                word_count=len(words),
                words_fingerprint=cache.fingerprint_words(
                    [word.word_instance_id for word in words]
                ),
            )
            for user_id, words in words_by_user.items()
        }
    )


def test_offsets_are_an_exclusive_prefix_sum() -> None:
    offsets = cache.pair_offsets([0, 1, 1, 3, 4])

    assert offsets.tolist() == [0, 0, 1, 2, 5, 9]
    # Target i occupies offsets[i]:offsets[i + 1]
    assert offsets[3 + 1] - offsets[3] == 3


def test_offsets_of_an_empty_list() -> None:
    assert cache.pair_offsets([]).tolist() == [0]


def test_fingerprint_depends_on_order() -> None:
    # Order is exactly what the positional layout rests on, so a reordering must
    # produce a different fingerprint even though the set of words is unchanged.
    assert cache.fingerprint_words(["a", "b"]) != cache.fingerprint_words(["b", "a"])
    assert cache.fingerprint_words(["a", "b"]) == cache.fingerprint_words(["a", "b"])


def test_manifest_round_trips(tmp_path) -> None:
    words_by_user = {"u1": _words("a", "b")}
    manifest = _manifest(words_by_user, u1=1)

    cache.write_manifest(tmp_path, manifest)

    assert cache.load_manifest(tmp_path) == manifest


def test_missing_manifest_points_at_the_scoring_stage(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="run the scoring stage"):
        cache.load_manifest(tmp_path)


def test_reordered_words_are_rejected() -> None:
    manifest = _manifest({"u1": _words("a", "b", "c")}, u1=3)

    with pytest.raises(ValueError, match="different word list"):
        manifest.verify_against({"u1": _words("c", "b", "a")})


def test_a_learner_absent_from_the_cache_is_rejected() -> None:
    manifest = _manifest({"u1": _words("a")}, u1=0)

    with pytest.raises(ValueError, match="no entry for learner"):
        manifest.verify_against({"u1": _words("a"), "u2": _words("b")})


def test_matching_words_verify() -> None:
    words_by_user = {"u1": _words("a", "b")}
    manifest = _manifest(words_by_user, u1=1)

    manifest.verify_against(words_by_user)  # does not raise


def test_scores_round_trip_including_nan(tmp_path) -> None:
    written = cache.open_scores(tmp_path, "u1", 3, mode="w+")
    written[0] = np.arange(len(cache.SCORE_COLUMNS), dtype=cache.SCORE_DTYPE)
    written[1] = np.nan
    written[2] = 0.5
    written.flush()
    del written

    read = cache.open_scores(tmp_path, "u1", 3, mode="r")

    assert read[0].tolist() == list(range(len(cache.SCORE_COLUMNS)))
    assert np.isnan(read[1]).all()
    assert read[2].tolist() == [0.5] * len(cache.SCORE_COLUMNS)


def test_reading_a_missing_block_points_at_the_scoring_stage(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="run the scoring stage"):
        cache.open_scores(tmp_path, "absent", 1, mode="r")


def test_progress_survives_a_torn_final_line(tmp_path) -> None:
    cache.record_completed_chunk(tmp_path, "u1", 0)
    cache.record_completed_chunk(tmp_path, "u1", 8)
    # A hard kill mid-append can leave a partial number; that chunk is simply redone.
    with cache.progress_path(tmp_path, "u1").open("a", encoding="utf-8") as handle:
        handle.write("1")

    assert cache.load_completed_chunks(tmp_path, "u1") == {0, 8, 1}

    cache.clear_progress(tmp_path, "u1")
    assert cache.load_completed_chunks(tmp_path, "u1") == set()
