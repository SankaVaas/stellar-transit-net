"""
mc_dropout.py
-------------
Utilities for Monte Carlo dropout inference and uncertainty decomposition.
The actual MC sampling logic lives in bayesian_net.py; this module provides
higher-level analysis tools built on top of those raw samples.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader


def uncertainty_summary(mc_results: dict) -> dict:
    """
    Summarize MC dropout output into human-readable uncertainty categories.

    Thresholds (empirical, adjust based on your calibration set):
        - High epistemic (>0.05): model hasn't seen this type of star → collect more data
        - High aleatoric (>0.3): signal is intrinsically ambiguous → flag for human review
        - High predictive entropy (>0.5): uncertain in aggregate → do not auto-classify

    Args:
        mc_results: output dict from bayesian_net.mc_predict()

    Returns:
        summary dict with per-sample uncertainty flags
    """
    epi   = mc_results["epistemic_uncertainty"]
    ale   = mc_results["aleatoric_uncertainty"]
    entr  = mc_results["predictive_entropy"]
    probs = mc_results["mean_probs"]

    return {
        "pred_class":          mc_results["pred_class"],
        "confidence":          probs.max(axis=1),
        "epistemic":           epi,
        "aleatoric":           ale,
        "predictive_entropy":  entr,
        "flag_epistemic":      epi  > 0.05,
        "flag_aleatoric":      ale  > 0.30,
        "flag_low_confidence": probs.max(axis=1) < 0.70,
        "flag_any":            (epi > 0.05) | (ale > 0.30) | (probs.max(axis=1) < 0.70),
    }


def run_mc_over_loader(
    model,
    loader: DataLoader,
    n_samples: int = 50,
    device: str = "cpu",
) -> dict:
    """
    Run MC dropout inference over an entire DataLoader.
    Concatenates results across all batches.
    """
    from src.models.bayesian_net import mc_predict

    all_keys = ["mean_probs", "std_probs", "pred_class",
                "epistemic_uncertainty", "aleatoric_uncertainty", "predictive_entropy"]
    collected = {k: [] for k in all_keys}
    all_labels = []

    for x, y in loader:
        result = mc_predict(model, x, n_samples=n_samples, device=device)
        for k in all_keys:
            collected[k].append(result[k])
        all_labels.append(y.numpy())

    return {
        k: np.concatenate(collected[k]) for k in all_keys
    } | {"labels": np.concatenate(all_labels)}