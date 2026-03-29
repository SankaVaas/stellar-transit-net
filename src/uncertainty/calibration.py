"""
calibration.py
--------------
Probability calibration evaluation and visualization.

A well-calibrated model means: when it says "80% confident this is a planet",
it should be correct ~80% of the time across many such predictions.

Most neural networks are overconfident out of the box. Temperature scaling
(in ensemble.py) corrects this. This module measures how much correction
was achieved using Expected Calibration Error (ECE).

For NASA: an uncalibrated model giving "99% confidence" on a false positive
is worse than a calibrated model saying "60% — I'm not sure."
Scientists need to know how much to trust the probability numbers.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    Algorithm:
        1. Group predictions into n_bins equal-width confidence bins
        2. For each bin: compute mean confidence and mean accuracy
        3. ECE = weighted average of |confidence - accuracy| across bins

    Args:
        probs: (N, C) softmax probabilities
        labels: (N,) true labels
        n_bins: number of calibration bins

    Returns:
        ECE value in [0, 1], lower is better (0 = perfect calibration)
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_conf = confidences[mask].mean()
        bin_acc  = correct[mask].mean()
        bin_weight = mask.sum() / len(labels)
        ece += bin_weight * abs(bin_conf - bin_acc)

    return float(ece)


def reliability_diagram(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    title: str = "Reliability diagram",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot reliability diagram (calibration curve).

    Perfect calibration = diagonal line from (0,0) to (1,1).
    Points above diagonal = underconfident (model hedges more than needed).
    Points below diagonal = overconfident (model too sure of itself).

    Args:
        probs: (N, C) softmax probs
        labels: (N,) true labels
        save_path: if provided, save figure to this path
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_accs, bin_confs, bin_sizes = [], [], []

    for i in range(n_bins):
        mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_accs.append(correct[mask].mean())
        bin_confs.append(confidences[mask].mean())
        bin_sizes.append(mask.sum())

    bin_accs  = np.array(bin_accs)
    bin_confs = np.array(bin_confs)
    ece = expected_calibration_error(probs, labels, n_bins)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Reliability diagram
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration", alpha=0.6)
    ax1.bar(bin_confs, bin_accs, width=1.0/n_bins, alpha=0.7,
            color="steelblue", edgecolor="white", label="Model")
    ax1.bar(bin_confs, bin_confs, width=1.0/n_bins, alpha=0.2,
            color="red", label="Gap (overconfidence)")
    ax1.set_xlabel("Mean confidence")
    ax1.set_ylabel("Fraction correct")
    ax1.set_title(f"{title}\nECE = {ece:.4f}")
    ax1.legend()
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)

    # Confidence histogram
    ax2.hist(confidences, bins=n_bins, color="steelblue", edgecolor="white", alpha=0.8)
    ax2.set_xlabel("Confidence")
    ax2.set_ylabel("Count")
    ax2.set_title("Confidence distribution")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def calibration_report(
    probs_before: np.ndarray,
    probs_after: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> dict:
    """Compare ECE before and after temperature scaling."""
    ece_before = expected_calibration_error(probs_before, labels, n_bins)
    ece_after  = expected_calibration_error(probs_after,  labels, n_bins)

    return {
        "ece_before_calibration": ece_before,
        "ece_after_calibration":  ece_after,
        "improvement": ece_before - ece_after,
        "improved": ece_after < ece_before,
    }