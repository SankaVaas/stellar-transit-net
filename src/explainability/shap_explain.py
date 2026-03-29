"""
shap_explain.py
---------------
SHAP (SHapley Additive exPlanations) for light curve attribution.

SHAP answers: "Which timesteps in this light curve most influenced
the model's decision to call it a planet?"

For transit detection this is scientifically meaningful:
    - A good model should assign high SHAP values to the transit dip region
    - If the model is keying off a non-transit region, that's a red flag
    - SHAP values on false positives reveal what the model "confused" for a transit

We use DeepSHAP (gradient-based, fast for neural nets) via the shap library.
A background dataset of ~100 randomly sampled training examples is used
as the reference distribution.

Reference: Lundberg & Lee, "A Unified Approach to Interpreting Model
Predictions" (NeurIPS 2017)
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from pathlib import Path


def build_shap_explainer(
    model: nn.Module,
    background_loader: torch.utils.data.DataLoader,
    n_background: int = 100,
    device: str = "cpu",
) -> shap.DeepExplainer:
    """
    Build a DeepSHAP explainer using a background dataset.

    The background represents the "expected" input — SHAP values measure
    deviation from this expected output when a feature changes.

    Args:
        model: trained PyTorch model
        background_loader: DataLoader for background samples
        n_background: number of background samples to use
        device: torch device

    Returns:
        shap.DeepExplainer instance
    """
    model.eval()
    background_samples = []

    for x, _ in background_loader:
        background_samples.append(x)
        if sum(b.shape[0] for b in background_samples) >= n_background:
            break

    background = torch.cat(background_samples, dim=0)[:n_background].to(device)
    explainer = shap.DeepExplainer(model, background)
    return explainer


def compute_shap_values(
    explainer: shap.DeepExplainer,
    x: torch.Tensor,
    target_class: int = 1,
) -> np.ndarray:
    """
    Compute SHAP values for a batch of light curves.

    Args:
        explainer: fitted DeepSHAP explainer
        x: input tensor (batch, 1, seq_len)
        target_class: class index to explain (1 = planet)

    Returns:
        shap_values: (batch, seq_len) attribution values
                     positive = pushes toward planet prediction
                     negative = pushes toward false positive prediction
    """
    shap_vals = explainer.shap_values(x)
    if isinstance(shap_vals, list):
        # List of arrays, one per class
        vals = shap_vals[target_class]
    else:
        vals = shap_vals

    # Squeeze channel dim: (batch, 1, seq_len) → (batch, seq_len)
    if vals.ndim == 3:
        vals = vals.squeeze(1)
    return vals


def plot_shap_attribution(
    flux: np.ndarray,
    shap_values: np.ndarray,
    label: int,
    pred_prob: float,
    title: str = "SHAP attribution",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot a single light curve with SHAP attribution overlay.

    Top panel: flux with SHAP-coloured background
        - Red regions: pushed model toward "planet"
        - Blue regions: pushed model toward "false positive"
    Bottom panel: SHAP value bar plot

    Args:
        flux: (seq_len,) normalized flux
        shap_values: (seq_len,) SHAP attribution values
        label: true label (0 or 1)
        pred_prob: model's predicted probability of planet
        save_path: optional path to save figure
    """
    seq_len = len(flux)
    x = np.arange(seq_len)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})

    # Top panel: flux with shap background shading
    ax1.plot(x, flux, color="steelblue", linewidth=0.8, label="Flux")
    # Overlay SHAP as coloured fill
    pos_shap = np.where(shap_values > 0, shap_values, 0)
    neg_shap = np.where(shap_values < 0, shap_values, 0)
    ax1_twin = ax1.twinx()
    ax1_twin.fill_between(x, pos_shap, alpha=0.3, color="red",  label="↑ planet")
    ax1_twin.fill_between(x, neg_shap, alpha=0.3, color="blue", label="↑ false positive")
    ax1_twin.set_ylabel("SHAP value", fontsize=10)
    ax1_twin.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    true_str = "planet" if label == 1 else "false positive"
    ax1.set_title(f"{title} | True: {true_str} | P(planet) = {pred_prob:.3f}")
    ax1.set_ylabel("Normalized flux")
    ax1.legend(loc="upper left", fontsize=9)
    ax1_twin.legend(loc="upper right", fontsize=9)

    # Bottom panel: SHAP bar chart
    ax2.bar(x, shap_values, color=["red" if v > 0 else "blue" for v in shap_values],
            alpha=0.6, width=1.0)
    ax2.axhline(0, color="gray", linewidth=0.5)
    ax2.set_xlabel("Timestep (phase-folded)")
    ax2.set_ylabel("SHAP")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def shap_summary_plot(
    shap_values: np.ndarray,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Mean absolute SHAP value across all samples — shows which timestep
    regions are globally most important for the model's decisions.

    Args:
        shap_values: (N, seq_len) SHAP values across test set
        save_path: optional save path
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    x = np.arange(len(mean_abs_shap))

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(x, mean_abs_shap, alpha=0.6, color="steelblue")
    ax.plot(x, mean_abs_shap, color="steelblue", linewidth=0.8)
    ax.set_xlabel("Timestep (phase-folded)")
    ax.set_ylabel("Mean |SHAP value|")
    ax.set_title("Global feature importance — mean |SHAP| across test set")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig