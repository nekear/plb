from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from plb.model_a.common.modeling import (
    get_mask_logits,
    get_score_token_ids,
    predict_continuous_score,
)
from plb.model_a.kvl.data import make_collate_fn
from plb.utils import autocast_context


def length_bucketed_batches(
    lengths: Sequence[int],
    batch_size: int,
    max_batch_tokens: int,
) -> list[list[int]]:
    """Group row indices into batches bounded by *padded* token count.

    The MLM head emits a `rows x padded_length x vocab_size` logit tensor, and
    mmBERT's vocabulary is 256k entries wide — roughly 0.5 MB per token position.
    A fixed batch size therefore blows up whenever one long prompt drags the whole
    batch's padding with it (a single 1.7k-token gloss in a 32-row batch asks for
    ~14 GB). Sorting by length keeps padding waste near zero, and the token budget
    caps peak memory regardless of how long the longest prompt is. Rows longer than
    the budget still get their own single-row batch rather than being dropped.
    """
    order = sorted(range(len(lengths)), key=lambda index: lengths[index])
    batches: list[list[int]] = []
    current: list[int] = []
    current_max = 0
    for index in order:
        candidate_max = max(current_max, lengths[index])
        exceeds_rows = len(current) + 1 > batch_size
        exceeds_tokens = (len(current) + 1) * candidate_max > max_batch_tokens
        if current and (exceeds_rows or exceeds_tokens):
            batches.append(current)
            current = []
            candidate_max = lengths[index]
        current.append(index)
        current_max = candidate_max
    if current:
        batches.append(current)
    return batches


class PromptDataset(Dataset):
    def __init__(self, prompts: pd.DataFrame):
        self.prompts = prompts.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, index: int) -> dict:
        return {"text": self.prompts.iloc[index]["text"]}


def predict_prompts(
    prompts: pd.DataFrame,
    *,
    model,
    tokenizer,
    device: torch.device,
    batch_size: int,
    desc: str = "Predicting",
    max_batch_tokens: int | None = None,
) -> list[float]:
    """Score every prompt, returning one 1-5 prediction per row in input order.

    `max_batch_tokens` switches on length-bucketed batching (see
    `length_bucketed_batches`); leave it unset for uniformly short prompt sets.
    """
    score_token_ids, score_token_values = get_score_token_ids(tokenizer, device)
    placeholder_token_id = score_token_ids[0].item()
    collate_batch = make_collate_fn(tokenizer, placeholder_token_id)
    dataset = PromptDataset(prompts)
    if max_batch_tokens is None:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch
        )
        prediction_order = None
    else:
        lengths = [
            len(input_ids)
            for input_ids in tokenizer(prompts["text"].tolist())["input_ids"]
        ]
        batches = length_bucketed_batches(lengths, batch_size, max_batch_tokens)
        loader = DataLoader(dataset, batch_sampler=batches, collate_fn=collate_batch)
        prediction_order = [index for batch in batches for index in batch]
    predictions: list[float] = []
    model.eval()
    with torch.no_grad(), autocast_context(device):
        for batch in tqdm(loader, desc=desc):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            mask_logits = get_mask_logits(outputs, input_ids, tokenizer.mask_token_id)
            scores = predict_continuous_score(mask_logits, score_token_ids, score_token_values)
            predictions.extend(scores.float().cpu().tolist())
    if len(predictions) != len(prompts):
        raise RuntimeError(f"Produced {len(predictions)} predictions for {len(prompts)} prompts")
    if prediction_order is None:
        return predictions
    # Length bucketing scored the rows out of order; put them back so the caller
    # can zip predictions against its own frame.
    restored = [0.0] * len(predictions)
    for position, index in enumerate(prediction_order):
        restored[index] = predictions[position]
    return restored
