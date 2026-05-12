"""
Hyperparameter ablation script.
Runs a small grid over aggregation / probe configs and reports
5-fold mean test AUROC.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from aggregation import aggregation_and_feature_extraction
from evaluate import run_evaluation
from model import MAX_LENGTH, get_model_and_tokenizer
from probe import HallucinationProbe
from splitting import split_data


def extract_features(
    df: pd.DataFrame,
    model,
    tokenizer,
    device: torch.device,
    use_geometric: bool,
    n_layers: int,
    batch_size: int = 4,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Extract features for a given aggregation configuration."""
    import aggregation as agg_mod
    old_n_layers = agg_mod.N_LAYERS_TO_CONCAT
    agg_mod.N_LAYERS_TO_CONCAT = n_layers

    all_texts = [f"{row['prompt']}{row['response']}" for _, row in df.iterrows()]
    all_labels = np.array([int(float(h)) for h in df["label"]])

    prompt_texts = [row['prompt'] for _, row in df.iterrows()]
    prompt_lens: list[int] = []
    for p_start in range(0, len(prompt_texts), batch_size):
        p_batch = prompt_texts[p_start : p_start + batch_size]
        p_enc = tokenizer(p_batch, padding=False, truncation=True, max_length=MAX_LENGTH)
        for ids in p_enc["input_ids"]:
            prompt_lens.append(len(ids))

    all_features: list = []
    for start in tqdm(range(0, len(all_texts), batch_size),
                      desc="Extracting", unit="batch", leave=False):
        batch_texts = all_texts[start : start + batch_size]
        encoding = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        hidden = torch.stack(outputs.hidden_states, dim=1).float()
        mask = attention_mask.cpu()

        for i in range(hidden.size(0)):
            feat = aggregation_and_feature_extraction(
                hidden[i],
                mask[i],
                use_geometric=use_geometric,
                response_start=prompt_lens[start + i],
            )
            all_features.append(feat.cpu())

    agg_mod.N_LAYERS_TO_CONCAT = old_n_layers
    X = np.vstack([f.numpy() for f in all_features])
    return X, all_labels, prompt_lens


def eval_config(X, y, probe_kwargs, n_splits=5):
    """Run 5-fold CV and return mean test AUROC."""
    splits = split_data(y, n_splits=n_splits)
    fold_results = run_evaluation(splits, X, y, lambda: HallucinationProbe(**probe_kwargs))
    test_aurocs = [r["test_auroc"] for r in fold_results]
    return float(np.mean(test_aurocs)), float(np.std(test_aurocs)), fold_results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv("./data/dataset.csv")
    model, tokenizer = get_model_and_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)

    configs: list[dict] = [
        # (n_layers, use_geometric, probe_kwargs)
        {"n_layers": 4, "geo": True,  "probe": {"hidden_dims": (512, 256, 128), "dropout": 0.4, "weight_decay": 1e-3, "max_epochs": 400, "use_pca": False}},
        {"n_layers": 4, "geo": True,  "probe": {"hidden_dims": (256, 128),      "dropout": 0.5, "weight_decay": 5e-3, "max_epochs": 300, "use_pca": True,  "pca_components": 512}},
        {"n_layers": 4, "geo": False, "probe": {"hidden_dims": (256, 128),      "dropout": 0.5, "weight_decay": 5e-3, "max_epochs": 300, "use_pca": True,  "pca_components": 512}},
        {"n_layers": 2, "geo": True,  "probe": {"hidden_dims": (256, 128),      "dropout": 0.5, "weight_decay": 5e-3, "max_epochs": 300, "use_pca": True,  "pca_components": 512}},
        {"n_layers": 2, "geo": False, "probe": {"hidden_dims": (256, 128),      "dropout": 0.5, "weight_decay": 5e-3, "max_epochs": 300, "use_pca": True,  "pca_components": 512}},
        {"n_layers": 4, "geo": True,  "probe": {"hidden_dims": (128, 64),       "dropout": 0.6, "weight_decay": 1e-2, "max_epochs": 200, "use_pca": True,  "pca_components": 256}},
        {"n_layers": 4, "geo": False, "probe": {"hidden_dims": (128, 64),       "dropout": 0.6, "weight_decay": 1e-2, "max_epochs": 200, "use_pca": True,  "pca_components": 256}},
        {"n_layers": 1, "geo": False, "probe": {"hidden_dims": (128, 64),       "dropout": 0.5, "weight_decay": 5e-3, "max_epochs": 300, "use_pca": False}},
        {"n_layers": 4, "geo": False, "probe": {"hidden_dims": (),              "dropout": 0.0, "weight_decay": 1e-1, "max_epochs": 500, "use_pca": False}},
    ]

    results: list[dict] = []
    for cfg in configs:
        print(f"\n{'='*60}")
        print(f"Config: layers={cfg['n_layers']}, geo={cfg['geo']}, probe={cfg['probe']}")
        t0 = time.time()
        X, y, _ = extract_features(df, model, tokenizer, device, cfg["geo"], cfg["n_layers"])
        mean_auroc, std_auroc, fold_res = eval_config(X, y, cfg["probe"])
        elapsed = time.time() - t0
        print(f"Feature dim: {X.shape[1]} | Mean test AUROC: {mean_auroc:.4f} +- {std_auroc:.4f} | Time: {elapsed:.1f}s")
        results.append({
            "config": cfg,
            "feature_dim": int(X.shape[1]),
            "mean_test_auroc": mean_auroc,
            "std_test_auroc": std_auroc,
            "time_s": elapsed,
            "folds": fold_res,
        })

    # Save results
    with open("ablation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print sorted summary
    print("\n" + "="*60)
    print("SUMMARY (sorted by mean test AUROC)")
    print("="*60)
    for r in sorted(results, key=lambda x: x["mean_test_auroc"], reverse=True):
        cfg = r["config"]
        print(f"AUROC {r['mean_test_auroc']:.4f} +- {r['std_test_auroc']:.4f}  |  "
              f"layers={cfg['n_layers']} geo={cfg['geo']} dim={r['feature_dim']}  |  "
              f"probe={cfg['probe']}")


if __name__ == "__main__":
    main()
