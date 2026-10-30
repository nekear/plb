# %% Cell 1
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plb.model_a.common.config import load_model_a_config
from plb.model_a.kvl.data import (
    KVLDataset,
    invert_remap,
    load_kvl_split,
    make_collate_fn,
    remap_scores,
)
from plb.model_a.common.modeling import (
    get_mask_logits,
    get_score_token_ids,
    predict_continuous_score,
)
from plb.utils import write_json, autocast_context

SPLIT = os.environ.get("MODEL_A_EVAL_SPLIT", "dev")

config = load_model_a_config()

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

run_dir = config.run_dir_models
metadata = json.loads((run_dir / "training_metadata.json").read_text())
score_min, score_max = metadata["score_min"], metadata["score_max"]

tokenizer = AutoTokenizer.from_pretrained(run_dir)
model = AutoModelForMaskedLM.from_pretrained(run_dir).to(device)
model.eval()

score_token_ids, score_token_values = get_score_token_ids(tokenizer, device)
placeholder_token_id = score_token_ids[0].item()  # token id for digit "1"
collate_batch = make_collate_fn(tokenizer, placeholder_token_id)

print(f"\nDevice: {device}")
print(f"Evaluating run {config.run_name!r} on split={SPLIT!r}")

eval_df = load_kvl_split(config.data_dir, SPLIT, config.languages)
eval_df["remapped_score"] = remap_scores(eval_df, score_min, score_max)

dataset = KVLDataset(eval_df)
loader = DataLoader(
    dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_batch
)

predictions: list[float] = []
with torch.no_grad(), autocast_context(device):
    for batch in tqdm(loader, desc=f"Evaluating on {SPLIT}"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        mask_logits = get_mask_logits(outputs, input_ids, tokenizer.mask_token_id)
        scores = predict_continuous_score(mask_logits, score_token_ids, score_token_values)
        predictions.extend(scores.cpu().tolist())

eval_df["predicted_score"] = predictions
# Inverting back to the raw GLMM_score scale the BEA leaderboard reports in
eval_df["predicted_GLMM_score"] = invert_remap(eval_df["predicted_score"], score_min, score_max)

def compute_metrics(df: pd.DataFrame) -> dict:
    gold_glmm = df["GLMM_score"].to_numpy()
    pred_glmm = df["predicted_GLMM_score"].to_numpy()
    return {
        "n": len(df),
        "spearman": float(spearmanr(gold_glmm, pred_glmm).statistic),
        "pearson": float(pearsonr(gold_glmm, pred_glmm)[0]),
        "mae_glmm": float(np.mean(np.abs(gold_glmm - pred_glmm))),
        "rmse_glmm": float(np.sqrt(np.mean((gold_glmm - pred_glmm) ** 2))),
    }


metrics = {"overall": compute_metrics(eval_df)}
for lang, group in eval_df.groupby("L1"):
    metrics[lang] = compute_metrics(group)

print(json.dumps(metrics, indent=2))

eval_dir = config.run_dir_evaluations / SPLIT
eval_dir.mkdir(parents=True, exist_ok=True)

write_json(eval_dir / "metrics.json", metrics)
eval_df.to_csv(eval_dir / "predictions.csv", index=False)

plt.scatter(eval_df["GLMM_score"], eval_df["predicted_GLMM_score"], alpha=0.3, s=10)
score_range = [
    min(eval_df["GLMM_score"].min(), eval_df["predicted_GLMM_score"].min()),
    max(eval_df["GLMM_score"].max(), eval_df["predicted_GLMM_score"].max()),
]
plt.plot(score_range, score_range, color="black", linestyle="--", linewidth=1)
plt.xlabel("Gold GLMM_score")
plt.ylabel("Predicted GLMM_score")
plt.title(f"Model A calibration — {SPLIT} split")
plt.tight_layout()
plt.savefig(eval_dir / "calibration.png", dpi=150)
plt.close()

print(f"Saved metrics + calibration plot to {eval_dir}")
