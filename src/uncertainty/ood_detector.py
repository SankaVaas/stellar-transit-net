"""
ood_detector.py
---------------
Out-of-Distribution (OOD) detection for light curves.

A model trained on Kepler data will behave unpredictably on:
    - TESS light curves (different noise profile, 2-min cadence)
    - Stars with extreme flaring (M-dwarfs)
    - Very long or short period transits outside training distribution
    - Light curves with data gaps larger than anything seen in training

The model doesn't know it's confused — it will still output a confident
probability. OOD detection is a separate mechanism that asks: "Is this
input sufficiently similar to what the model was trained on?"

Three methods implemented:
    1. Isolation Forest — fast, works on feature vectors
    2. Mahalanobis distance — uses training set feature statistics
    3. VAE reconstruction error — uses the trained VAE from vae.py
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path


class IsolationForestOOD:
    """
    Isolation Forest trained on feature vectors from the encoder.

    Anomaly score in sklearn: negative values = more anomalous
    We flip sign so higher score = more OOD.
    """

    def __init__(self, contamination: float = 0.05, n_estimators: int = 100):
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()

    def fit(self, features: np.ndarray):
        """Fit on in-distribution (training set) features."""
        features_scaled = self.scaler.fit_transform(features)
        self.model.fit(features_scaled)

    def score(self, features: np.ndarray) -> np.ndarray:
        """
        Returns OOD score per sample. Higher = more OOD.
        Score > 0 means anomalous (decision_function < 0 in sklearn).
        """
        features_scaled = self.scaler.transform(features)
        raw_scores = self.model.decision_function(features_scaled)
        return -raw_scores   # flip so higher = more anomalous

    def predict(self, features: np.ndarray, threshold: float = 0.0) -> np.ndarray:
        """Returns boolean array: True = OOD."""
        return self.score(features) > threshold

    def save(self, path: str):
        joblib.dump({"model": self.model, "scaler": self.scaler}, path)

    def load(self, path: str):
        obj = joblib.load(path)
        self.model = obj["model"]
        self.scaler = obj["scaler"]


class MahalanobisOOD:
    """
    Mahalanobis distance-based OOD detection.

    Computes distance from a test feature vector to the multivariate Gaussian
    fitted on training features. Large distance = OOD.

    More principled than Isolation Forest when the in-distribution is
    approximately Gaussian in feature space (often true after a few FC layers).
    """

    def __init__(self):
        self.mean = None
        self.cov_inv = None

    def fit(self, features: np.ndarray):
        """Fit on in-distribution feature vectors."""
        self.mean = features.mean(axis=0)
        cov = np.cov(features.T)
        # Regularize covariance matrix (add small diagonal to ensure invertibility)
        cov += np.eye(cov.shape[0]) * 1e-5
        self.cov_inv = np.linalg.inv(cov)

    def score(self, features: np.ndarray) -> np.ndarray:
        """
        Compute Mahalanobis distance for each sample.
        d = sqrt((x - mu)^T * Sigma^{-1} * (x - mu))
        """
        if self.mean is None:
            raise RuntimeError("Call fit() first")
        diff = features - self.mean   # (N, D)
        # Efficient batch computation: sum(diff @ cov_inv * diff, axis=1)
        left = diff @ self.cov_inv    # (N, D)
        distances = np.sqrt((left * diff).sum(axis=1))   # (N,)
        return distances


def extract_features(
    model,
    loader: DataLoader,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract penultimate-layer feature vectors from any model with a
    get_features() method (CNN-TCN, Transformer, BayesianNet all implement this).

    Returns:
        features: (N, feature_dim)
        labels: (N,)
    """
    model.eval()
    all_features, all_labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            feats = model.get_features(x)
            all_features.append(feats.cpu().numpy())
            all_labels.append(y.numpy())

    return np.concatenate(all_features), np.concatenate(all_labels)


def ood_report(
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    threshold_percentile: float = 95.0,
) -> dict:
    """
    Produce an OOD detection report comparing train and test score distributions.

    Args:
        train_scores: OOD scores on training set (in-distribution reference)
        test_scores: OOD scores on test / deployment set
        threshold_percentile: top X% of training scores = OOD threshold

    Returns:
        dict with threshold, flagged fraction, and score statistics
    """
    threshold = np.percentile(train_scores, threshold_percentile)
    flagged = (test_scores > threshold).mean()

    return {
        "threshold": threshold,
        "threshold_percentile": threshold_percentile,
        "train_score_mean": train_scores.mean(),
        "train_score_std":  train_scores.std(),
        "test_score_mean":  test_scores.mean(),
        "test_score_std":   test_scores.std(),
        "fraction_flagged_as_ood": flagged,
        "n_flagged": int((test_scores > threshold).sum()),
        "n_total":   len(test_scores),
    }