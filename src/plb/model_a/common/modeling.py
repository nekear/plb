import torch
from torch.nn import functional as F
from transformers.modeling_outputs import MaskedLMOutput


def get_score_token_ids(
    tokenizer, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve the token id for each digit 1-5.

    mmBERT's tokenizer has no vocab entry for a bare digit with a merged
    leading space (unlike common words, e.g. " Difficulty" is one token but
    " 1" is two: a standalone space token followed by "1"). Tokenizing a digit
    in isolation therefore yields either 1 or 2 tokens depending on the digit;
    the digit itself is always the *last* one, so take encoded[-1] rather than
    assuming a fixed length.
    """
    ids = []
    for digit in "12345":
        encoded = tokenizer(digit, add_special_tokens=False)["input_ids"]
        if len(encoded) not in (1, 2):
            raise ValueError(
                f"Digit {digit!r} tokenizes to {len(encoded)} tokens ({encoded}); "
                "expected 1 or 2 — the soft-target loss assumes a single digit "
                "token per score value."
            )
        ids.append(encoded[-1])

    score_token_ids = torch.tensor(ids, device=device)
    score_token_values = torch.tensor(
        [int(d) for d in "12345"], dtype=torch.float32, device=device
    )
    return score_token_ids, score_token_values


def get_mask_logits(
    outputs: MaskedLMOutput, input_ids: torch.Tensor, mask_token_id: int
) -> torch.Tensor:
    mask_indices = torch.where(input_ids == mask_token_id)
    return outputs.logits[mask_indices]


def soft_target_loss(
    mask_logits: torch.Tensor, y: torch.Tensor, score_token_ids: torch.Tensor
) -> torch.Tensor:
    """
    mask_logits: [batch_size, vocab_size]
    y: [batch_size], continuous remapped scores in [1, 5]
    """
    log_probs = F.log_softmax(mask_logits, dim=-1)
    y = y.clamp(1.0, 5.0)

    lower = torch.floor(y).long()
    upper = torch.ceil(y).long()

    lower_idx = lower - 1
    upper_idx = upper - 1

    lower_token_ids = score_token_ids[lower_idx]
    upper_token_ids = score_token_ids[upper_idx]

    upper_weight = y - lower.float()
    lower_weight = 1.0 - upper_weight

    lower_log_probs = log_probs.gather(1, lower_token_ids[:, None]).squeeze(1)
    upper_log_probs = log_probs.gather(1, upper_token_ids[:, None]).squeeze(1)

    loss = -(lower_weight * lower_log_probs + upper_weight * upper_log_probs)
    return loss.mean()


def predict_continuous_score(
    mask_logits: torch.Tensor,
    score_token_ids: torch.Tensor,
    score_token_values: torch.Tensor,
) -> torch.Tensor:
    """
    mask_logits: [batch_size, vocab_size]
    returns: [batch_size], score in [1, 5]
    """
    probs = F.softmax(mask_logits, dim=-1)

    digit_probs = probs[:, score_token_ids]
    digit_probs = digit_probs / digit_probs.sum(dim=-1, keepdim=True)

    return (digit_probs * score_token_values).sum(dim=-1)
