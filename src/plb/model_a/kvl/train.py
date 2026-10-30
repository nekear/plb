import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plb.model_a.common.config import load_model_a_config
from plb.model_a.common.modeling import (
    get_mask_logits,
    get_score_token_ids,
    soft_target_loss,
)
from plb.model_a.kvl.data import (
    KVLDataset,
    fit_score_range,
    load_kvl_split,
    make_collate_fn,
    remap_scores,
)
from plb.utils import (
    autocast_context,
    get_device,
    load_masked_model_and_tokenizer,
    write_json,
)

config = load_model_a_config()
torch.manual_seed(config.seed)

device = get_device()

model, tokenizer = load_masked_model_and_tokenizer(
    config.model_id, config.model_revision, device
)
score_token_ids, score_token_values = get_score_token_ids(tokenizer, device)
placeholder_token_id = score_token_ids[0].item()  # token id for digit "1"
collate_batch = make_collate_fn(tokenizer, placeholder_token_id)

print(f"\nDevice: {device}")
print(f"Languages: {config.languages}")

train_df = load_kvl_split(config.data_dir, "train", config.languages)

# Fit the 1-5 remap on train only; the exact same score_min/score_max is
# persisted in training_metadata.json and reused (never refit) at eval time.
score_min, score_max = fit_score_range(train_df)
train_df["remapped_score"] = remap_scores(train_df, score_min, score_max)

train_df.describe()

def fine_tune(train_df, epochs, batch_size, lr):
    model.train()
    dataset = KVLDataset(train_df)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    optimizer = AdamW(model.parameters(), lr=lr)
    history = {"loss": []}

    for epoch in range(epochs):
        total_loss = 0.0
        progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for batch in progress:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            scores = batch["score"].to(device)

            with autocast_context(device):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                mask_logits = get_mask_logits(
                    outputs, input_ids, tokenizer.mask_token_id
                )
                loss = soft_target_loss(mask_logits, scores, score_token_ids)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(loader)
        history["loss"].append(avg_loss)
        print(f"Epoch {epoch + 1}: loss = {avg_loss:.4f}")

    return history


history = fine_tune(train_df, config.epochs, config.batch_size, config.lr)

run_dir = config.run_dir_models
run_dir.mkdir(parents=True, exist_ok=True)

model.save_pretrained(run_dir)
tokenizer.save_pretrained(run_dir)

metadata = {
    "model_id": config.model_id,
    "model_revision": config.model_revision,
    "languages": list(config.languages),
    "epochs": config.epochs,
    "batch_size": config.batch_size,
    "lr": config.lr,
    "seed": config.seed,
    "score_min": score_min,
    "score_max": score_max,
    "train_loss_history": history["loss"],
}
write_json(run_dir / "training_metadata.json", metadata)

print(f"Saved model + tokenizer + metadata to {run_dir}")

output_path = config.run_dir_evaluations / "train_loss.png"
output_path.parent.mkdir(parents=True, exist_ok=True)

plt.plot(range(1, len(history["loss"]) + 1), history["loss"], marker="o")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss Curve")
plt.grid(True)
plt.tight_layout()
plt.savefig(output_path, dpi=150)
plt.close()

print(f"Saved loss curve to {output_path}")
