import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm.auto import tqdm

from plb.utils import get_dataset_as_path, get_device, load_model_and_tokenizer

from .config import load_direct_embeddings_config


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def embed_word_instances(
        word_instances_df: pd.DataFrame,
        model,
        tokenizer,
        device: torch.device,
        batch_size: int,
        max_length: int,
        limit: int | None = None,
) -> pd.DataFrame:
    if limit is not None and limit < 0:
        raise ValueError("limit cannot be negative")

    row_count = (
        len(word_instances_df)
        if limit is None
        else min(limit, len(word_instances_df))
    )
    ids: list[str] = []
    embeddings: list[list[float]] = []

    model.eval()

    with torch.inference_mode():
        for start in tqdm(range(0, row_count, batch_size)):
            batch = word_instances_df.iloc[start: min(start + batch_size, row_count)]

            batch_ids = batch["word_instance_id"].to_list()
            batch_texts = ("query: " + batch["meaning"]).to_list()

            batch_dict = tokenizer(
                batch_texts,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            outputs = model(**batch_dict)

            batch_embeddings = average_pool(
                outputs.last_hidden_state, batch_dict["attention_mask"]
            )

            batch_embeddings = (
                F.normalize(batch_embeddings, p=2, dim=1).cpu().numpy().tolist()
            )

            ids.extend(batch_ids)
            embeddings.extend(batch_embeddings)

            if len(ids) != len(embeddings):
                raise ValueError(
                    f"Iteration {start // batch_size} ids and embeddings must have the same length"
                )

    if len(ids) != len(embeddings):
        raise ValueError(
            "Final ids and embeddings must have the same length, while "
            f"ids={len(ids)} and embeddings={len(embeddings)}"
        )

    return pd.DataFrame({"word_instance_id": ids, "embedding": embeddings})


def main():
    config = load_direct_embeddings_config()
    device = get_device()
    model, tokenizer = load_model_and_tokenizer(
        config.model_id,
        config.model_revision,
        device,
    )

    word_instances_df = pd.read_json(config.word_instances_path, lines=True)
    embeddings_df = embed_word_instances(
        word_instances_df,
        model,
        tokenizer,
        device=device,
        batch_size=config.batch_size,
        max_length=config.max_length,
    )

    output_path = config.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    embeddings_df.to_json(output_path, lines=True, orient="records")


if __name__ == "__main__":
    main()
