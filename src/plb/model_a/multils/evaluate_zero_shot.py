import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from plb.model_a.common.config import load_model_a_config
from plb.model_a.multils.evaluation import evaluate_multils


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    config = load_model_a_config()
    device = select_device()
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id, revision=config.model_revision
    )
    model = AutoModelForMaskedLM.from_pretrained(
        config.model_id, revision=config.model_revision
    ).to(device)
    metrics = evaluate_multils(
        model=model,
        tokenizer=tokenizer,
        device=device,
        data_dir=config.multils_data_dir,
        cues_path=config.multils.cues_path,
        condition=config.multils.condition,
        batch_size=config.batch_size,
        bootstrap_samples=config.multils.bootstrap_samples,
        bootstrap_seed=config.multils.bootstrap_seed,
        output_dir=config.multils_base_dir,
        model_kind="untouched_base",
        model_identifier=f"{config.model_id}@{config.model_revision}",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
