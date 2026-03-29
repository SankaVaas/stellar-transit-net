"""
query_strategies.py
-------------------
Query strategies for active learning — deciding WHICH unlabelled light curves
the model should ask a human (oracle) to label next.

Context: We start with only 10% of the dataset labelled. In each AL iteration,
we select a batch of the most "informative" unlabelled examples, label them,
retrain, and measure how accuracy improves per label used.

This simulates a real NASA scenario: astronomer time is the bottleneck.
A smart query strategy achieves the same accuracy with far fewer labels
than random sampling — which directly translates to fewer telescope hours.

Three strategies implemented:
    1. Max entropy: label the examples the model is most uncertain about
    2. BALD (Bayesian Active Learning by Disagreement): maximize information
       gained about model parameters (requires MC Dropout)
    3. Least confident: label examples closest to the decision boundary
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


def max_entropy_query(
    probs: np.ndarray,
    n_query: int,
    already_labeled: np.ndarray | None = None,
) -> np.ndarray:
    """
    Select samples with highest predictive entropy.
    H = -sum(p * log(p))

    Args:
        probs: (N, C) softmax probabilities for all unlabelled samples
        n_query: number of samples to select
        already_labeled: indices already labelled (excluded from selection)

    Returns:
        indices: indices of selected samples (into the unlabelled pool)
    """
    eps = 1e-8
    entropy = -(probs * np.log(probs + eps)).sum(axis=1)   # (N,)

    if already_labeled is not None:
        entropy[already_labeled] = -np.inf

    return np.argsort(entropy)[::-1][:n_query]


def bald_query(
    mean_probs: np.ndarray,
    sample_probs: np.ndarray,
    n_query: int,
) -> np.ndarray:
    """
    BALD: Bayesian Active Learning by Disagreement.
    Selects samples that maximize mutual information between predictions
    and model parameters.

    BALD score = H[mean_probs] - E[H[sample_probs]]
               = predictive entropy - mean of individual sample entropies
               = how much disagreement exists AMONG the MC samples

    A sample with high BALD score means: individual MC passes disagree with
    each other a lot, so labelling this example will most reduce model uncertainty.

    Args:
        mean_probs: (N, C) mean softmax probability across MC samples
        sample_probs: (T, N, C) per-sample softmax probabilities from T MC passes

    Returns:
        indices of top n_query samples by BALD score
    """
    eps = 1e-8
    # Predictive entropy: H[mean]
    H_mean = -(mean_probs * np.log(mean_probs + eps)).sum(axis=1)   # (N,)

    # Expected entropy: E[H[samples]]
    T = sample_probs.shape[0]
    H_samples = -(sample_probs * np.log(sample_probs + eps)).sum(axis=-1)  # (T, N)
    E_H = H_samples.mean(axis=0)   # (N,)

    bald_scores = H_mean - E_H
    return np.argsort(bald_scores)[::-1][:n_query]


def least_confident_query(probs: np.ndarray, n_query: int) -> np.ndarray:
    """
    Select samples where the model's maximum class probability is lowest.
    These are the examples closest to the decision boundary.

    Args:
        probs: (N, C) softmax probabilities
        n_query: number to select

    Returns:
        indices of n_query least confident samples
    """
    max_probs = probs.max(axis=1)
    return np.argsort(max_probs)[:n_query]