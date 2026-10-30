"""The random control must swap the word but keep the slot."""

import numpy as np

from plb.similarities.select_neighbors import RANDOM_WINDOW, draw_replacement

DAY = np.timedelta64(1, "D")
T0 = np.datetime64("2025-06-15T00:00:00")


def _pool(offsets_in_days: list[int]) -> tuple[np.ndarray, list[str]]:
    times = np.array([T0 + offset * DAY for offset in offsets_in_days])
    return times, [f"w{offset}" for offset in offsets_in_days]


def _index(ids: list[str]) -> dict[str, int]:
    return {word_instance_id: position for position, word_instance_id in enumerate(ids)}


def test_draw_stays_inside_the_window_and_never_returns_the_original() -> None:
    offsets = [-90, -20, -5, 0, 5, 20, 90]
    times, ids = _pool(offsets)
    generator = np.random.default_rng(0)

    drawn = {
        draw_replacement(
            (times, ids), T0, T0 + 365 * DAY, "w0", _index(ids), len(ids), generator
        )
        for _ in range(200)
    }

    chosen = {ids[index] for index in drawn}
    assert chosen == {"w-20", "w-5", "w5", "w20"}  # within +/-30d, minus the original


def test_the_cutoff_bounds_the_window() -> None:
    times, ids = _pool([-5, 5, 20])
    generator = np.random.default_rng(0)

    # Target first reviewed 1 day after T0: nothing completed later may be drawn,
    # which is what keeps the replacement uncensored as well as leak-free.
    drawn = {
        draw_replacement(
            (times, ids), T0, T0 + DAY, "w20", _index(ids), len(ids), generator
        )
        for _ in range(50)
    }
    assert {ids[index] for index in drawn} == {"w-5"}


def test_words_outside_the_candidate_prefix_are_skipped() -> None:
    times, ids = _pool([-5, 5])
    generator = np.random.default_rng(0)

    # candidate_end=1 admits only position 0; position 1 is not a candidate of this
    # target, so its cached score row would describe a different pair.
    drawn = draw_replacement(
        (times, ids), T0, T0 + 365 * DAY, "w5", _index(ids), 1, generator
    )
    assert drawn == 0


def test_an_empty_window_reports_no_draw() -> None:
    times, ids = _pool([0])
    generator = np.random.default_rng(0)

    assert (
        draw_replacement(
            (times, ids), T0, T0 + 365 * DAY, "w0", _index(ids), len(ids), generator
        )
        is None
    )


def test_window_is_thirty_days() -> None:
    assert RANDOM_WINDOW == np.timedelta64(30, "D")
