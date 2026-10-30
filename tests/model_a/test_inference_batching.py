import pandas as pd

from plb.model_a.common.inference import PromptDataset, length_bucketed_batches


def test_batches_cover_every_row_exactly_once() -> None:
    lengths = [10, 400, 12, 33, 7, 1200, 64, 64]
    batches = length_bucketed_batches(lengths, batch_size=4, max_batch_tokens=512)
    covered = sorted(index for batch in batches for index in batch)
    assert covered == list(range(len(lengths)))


def test_batches_respect_the_row_and_token_budgets() -> None:
    lengths = [100] * 10
    batches = length_bucketed_batches(lengths, batch_size=4, max_batch_tokens=4096)
    assert all(len(batch) <= 4 for batch in batches)

    batches = length_bucketed_batches(lengths, batch_size=32, max_batch_tokens=250)
    # 250 // 100 == 2 rows per batch, the padded cost being rows * longest row.
    assert all(len(batch) * 100 <= 250 for batch in batches)


def test_rows_longer_than_the_budget_get_their_own_batch() -> None:
    lengths = [8, 5000, 8]
    batches = length_bucketed_batches(lengths, batch_size=8, max_batch_tokens=512)
    assert [1] in batches
    assert sorted(index for batch in batches for index in batch) == [0, 1, 2]


def test_bucketing_keeps_similar_lengths_together() -> None:
    lengths = [1000, 10, 1000, 10]
    batches = length_bucketed_batches(lengths, batch_size=2, max_batch_tokens=4096)
    assert sorted(sorted(batch) for batch in batches) == [[0, 2], [1, 3]]


def test_prompt_dataset_indexes_by_position_not_frame_index() -> None:
    prompts = pd.DataFrame({"text": ["a", "b", "c"]}, index=[7, 8, 9])
    dataset = PromptDataset(prompts)
    assert len(dataset) == 3
    assert [dataset[index]["text"] for index in range(3)] == ["a", "b", "c"]
