"""
conformal.py
------------
Conformal prediction for statistically guaranteed prediction sets.

Standard softmax gives probabilities but NO coverage guarantee:
    "90% confidence" might be right 70% of the time in practice.

Conformal prediction gives a DISTRIBUTION-FREE guarantee:
    Given a calibration set and significance level alpha=0.05,
    the prediction SET (possibly {planet}, or {planet, false_positive})
    contains the true label with probability ≥ 1-alpha = 95%.

This guarantee holds regardless of model architecture, data distribution,
or whether the model is calibrated. The only assumption: calibration
examples are exchangeable with test examples.

For NASA: this is the difference between "the model says 90% confident"
(which could mean anything) and "with 95% statistical guarantee, the true
class is in this set" (which is a rigorous scientific statement).

Reference: Angelopoulos & Bates, "A Gentle Introduction to Conformal
Prediction and Distribution-Free Uncertainty Quantification" (2021)
"""

import numpy as np
from torch.utils.data import DataLoader


class ConformalPredictor:
    """
    Split conformal predictor using softmax scores as the conformity measure.

    Steps:
        1. On calibration set: compute nonconformity scores
           score_i = 1 - softmax_prob[true_class_i]
           (higher score = less conforming = model was less confident on true class)

        2. Compute q_hat = (1-alpha) quantile of calibration scores

        3. At test time: prediction set = all classes where
           1 - softmax_prob[class] ≤ q_hat
           i.e., classes the model is at least as confident about as the
           threshold learned from calibration

    Properties:
        - If true class is "planet", its score = 1 - p(planet)
        - If this score ≤ q_hat (model was confident enough on similar examples),
          planet is included in the prediction set
        - Marginal coverage: P(true class ∈ set) ≥ 1-alpha
    """

    def __init__(self, alpha: float = 0.05):
        """
        Args:
            alpha: miscoverage rate. alpha=0.05 → 95% coverage guarantee.
        """
        self.alpha = alpha
        self.q_hat = None

    def calibrate(self, probs: np.ndarray, labels: np.ndarray):
        """
        Compute q_hat from a calibration set.

        Args:
            probs: (N, C) softmax probabilities on calibration set
            labels: (N,) true class labels
        """
        n = len(labels)
        true_class_probs = probs[np.arange(n), labels]
        scores = 1.0 - true_class_probs   # nonconformity scores

        # Finite-sample corrected quantile: ceil((n+1)(1-alpha)) / n
        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        level = min(level, 1.0)
        self.q_hat = float(np.quantile(scores, level))
        print(f"Conformal q_hat = {self.q_hat:.4f} (alpha={self.alpha}, n={n})")

    def predict_set(self, probs: np.ndarray) -> list[list[int]]:
        """
        Produce prediction sets for a batch of test examples.

        Args:
            probs: (N, C) softmax probabilities

        Returns:
            list of N prediction sets (each is a list of class indices)

        Example:
            [[1], [0, 1], [0]]
            → sample 0: certain it's class 1 (planet)
            → sample 1: uncertain, could be either class
            → sample 2: certain it's class 0 (false positive)
        """
        if self.q_hat is None:
            raise RuntimeError("Call calibrate() before predict_set()")

        prediction_sets = []
        for prob_row in probs:
            included = [c for c, p in enumerate(prob_row) if (1.0 - p) <= self.q_hat]
            if len(included) == 0:
                # Fallback: always include the most likely class
                included = [int(np.argmax(prob_row))]
            prediction_sets.append(included)

        return prediction_sets

    def coverage(self, probs: np.ndarray, labels: np.ndarray) -> float:
        """
        Empirical coverage on a test set.
        Should be ≥ 1-alpha if conformal guarantee holds.
        """
        sets = self.predict_set(probs)
        covered = sum(labels[i] in sets[i] for i in range(len(labels)))
        return covered / len(labels)

    def average_set_size(self, probs: np.ndarray) -> float:
        """
        Average size of prediction sets. Smaller = more informative.
        Set size of 1 means the model is certain.
        Set size of C means the model abstains entirely.
        """
        sets = self.predict_set(probs)
        return np.mean([len(s) for s in sets])

    def efficiency_report(self, probs: np.ndarray, labels: np.ndarray) -> dict:
        """Full evaluation: coverage, average set size, singleton rate."""
        sets = self.predict_set(probs)
        coverage = sum(labels[i] in sets[i] for i in range(len(labels))) / len(labels)
        avg_size = np.mean([len(s) for s in sets])
        singleton_rate = np.mean([len(s) == 1 for s in sets])
        empty_rate = np.mean([len(s) == 0 for s in sets])

        return {
            "coverage": coverage,
            "target_coverage": 1.0 - self.alpha,
            "coverage_satisfied": coverage >= (1.0 - self.alpha),
            "avg_set_size": avg_size,
            "singleton_rate": singleton_rate,
            "empty_rate": empty_rate,
            "q_hat": self.q_hat,
        }