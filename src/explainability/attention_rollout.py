"""
attention_rollout.py
--------------------
Attention rollout for the Transformer model — maps attention weights
back to input patch positions to produce a saliency map over the flux.

Problem with raw attention weights:
    Each transformer layer attends to outputs of the previous layer,
    not directly to the input. Looking at only the last layer's attention
    misses how information flowed through earlier layers.

Attention rollout (Abnar & Zuidema, 2020) recursively multiplies attention
matrices across layers, accounting for residual connections:
    A_rollout = A_1 · A_2 · ... · A_L
    where each A_i = 0.5 * attention_i + 0.5 * I  (residual identity added)

The [CLS] row of A_rollout tells us: how much did each input patch
contribute to the final classification token?

For transit detection: high rollout weight on the transit dip patch
means the model correctly focused on the transit to make its decision.
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from src.models.transformer import TransitTransformer


def attention_rollout(
    model: TransitTransformer,
    x: torch.Tensor,
    device: str = "cpu",
    discard_ratio: float = 0.9,
) -> np.ndarray:
    """
    Compute attention rollout scores for a batch of inputs.

    Args:
        model: trained TransitTransformer
        x: input tensor (batch, 1, seq_len)
        device: torch device
        discard_ratio: fraction of lowest attention weights to zero out
                       before rollout (reduces noise from weak connections)

    Returns:
        rollout: (batch, num_patches) attention importance per patch
                 (excludes CLS token from output)
    """
    model.eval()
    x = x.to(device)

    attention_weights = model.get_attention_weights(x)
    # Each element: (batch, nhead, seq_len+1, seq_len+1)

    # Average over heads
    attn_maps = []
    for attn in attention_weights:
        avg_attn = attn.mean(dim=1)   # (batch, seq+1, seq+1)
        attn_maps.append(avg_attn.cpu().numpy())

    batch_size = attn_maps[0].shape[0]
    seq_plus_1 = attn_maps[0].shape[1]

    rollout = np.eye(seq_plus_1)[np.newaxis].repeat(batch_size, axis=0)

    for attn in attn_maps:
        # Add identity for residual connection: A_hat = 0.5*A + 0.5*I
        attn_hat = 0.5 * attn + 0.5 * np.eye(seq_plus_1)[np.newaxis]

        # Discard lowest attention weights (set to 0 and renormalize)
        if discard_ratio > 0:
            flat = attn_hat.reshape(batch_size, seq_plus_1, seq_plus_1)
            threshold = np.quantile(flat, discard_ratio, axis=-1, keepdims=True)
            attn_hat = np.where(attn_hat >= threshold, attn_hat, 0)
            # Renormalize rows
            row_sums = attn_hat.sum(axis=-1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1.0, row_sums)
            attn_hat = attn_hat / row_sums

        # Matrix multiply: rollout = rollout @ attn_hat
        rollout = np.matmul(rollout, attn_hat)

    # Extract CLS token row (index 0) — tells us how much each patch influenced CLS
    cls_rollout = rollout[:, 0, 1:]   # (batch, num_patches) — skip CLS itself
    return cls_rollout


def rollout_to_flux_importance(
    rollout: np.ndarray,
    seq_len: int,
    patch_size: int,
) -> np.ndarray:
    """
    Upsample patch-level rollout scores to full flux sequence length.

    Args:
        rollout: (batch, num_patches) rollout importance per patch
        seq_len: original sequence length
        patch_size: number of timesteps per patch

    Returns:
        importance: (batch, seq_len) importance per timestep
    """
    batch_size = rollout.shape[0]
    importance = np.repeat(rollout, patch_size, axis=1)[:, :seq_len]
    return importance


def plot_attention_rollout(
    flux: np.ndarray,
    importance: np.ndarray,
    label: int,
    pred_prob: float,
    patch_size: int = 50,
    title: str = "Attention rollout",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Visualize attention rollout importance overlaid on the light curve.

    Args:
        flux: (seq_len,) normalized flux
        importance: (seq_len,) rollout importance values
        label: true label
        pred_prob: predicted probability of planet
        patch_size: for drawing patch boundaries
        save_path: optional save path
    """
    seq_len = len(flux)
    x = np.arange(seq_len)

    # Normalize importance to [0, 1] for visualization
    imp_norm = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})

    # Top: flux with importance-coloured background
    ax1.plot(x, flux, color="steelblue", linewidth=0.8, zorder=3)
    # Shade background by importance
    for i in range(0, seq_len - patch_size, patch_size):
        patch_imp = imp_norm[i:i + patch_size].mean()
        ax1.axvspan(i, i + patch_size, alpha=patch_imp * 0.5,
                    color="orange", zorder=1)

    true_str = "planet" if label == 1 else "false positive"
    ax1.set_title(f"{title} | True: {true_str} | P(planet) = {pred_prob:.3f}")
    ax1.set_ylabel("Normalized flux")

    # Draw patch boundaries
    for i in range(0, seq_len, patch_size):
        ax1.axvline(i, color="gray", linewidth=0.3, alpha=0.5, zorder=2)

    # Bottom: importance curve
    ax2.fill_between(x, imp_norm, alpha=0.6, color="orange")
    ax2.plot(x, imp_norm, color="darkorange", linewidth=0.8)
    ax2.set_xlabel("Timestep (phase-folded)")
    ax2.set_ylabel("Attention importance")
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig