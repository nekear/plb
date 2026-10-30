from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

# torch/transformers come with the optional `train` extra, so they're imported inside
# the helpers that need them: the CPU-only stages import this module for paths alone.
if TYPE_CHECKING:
    import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_device() -> torch.device:
    import torch

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return device


def load_masked_model_and_tokenizer(model_id: str, revision: str, device: torch.device):
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForMaskedLM.from_pretrained(model_id, revision=revision)
    model = model.to(device)
    return model, tokenizer


def load_model_and_tokenizer(model_id: str, revision: str, device: torch.device):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModel.from_pretrained(model_id, revision=revision)
    model = model.to(device)
    return model, tokenizer


def autocast_context(device: torch.device):
    """bf16 mixed precision for the forward pass on CUDA GPUs that support it
    (Ampere+, including Ada Lovelace) — a no-op everywhere else.

    Unlike fp16, bf16 keeps fp32's 8-bit exponent, so it doesn't have fp16's
    underflow/overflow problem on small gradient values; no GradScaler needed.
    Wrap only the forward pass + loss computation, not the backward() call —
    autograd replays the dtypes recorded during the forward pass on its own.
    """
    import torch

    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def get_dataset_as_path(route: str | Path) -> Path:
    """Return an absolute path for a route relative to the project's data directory."""
    route_path = Path(route)
    if route_path.is_absolute():
        raise ValueError(f"Dataset route must be relative to data/: {route}")
    return PROJECT_ROOT / "data" / route_path


def write_json(path: Path, payload: object) -> None:
    """Write JSON using the project's standard human-readable formatting."""
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


def write_jsonl(rows: Sequence[object], path: Path) -> None:
    """Write Pydantic models as newline-delimited JSON."""
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row.model_dump_json())
            handle.write("\n")
