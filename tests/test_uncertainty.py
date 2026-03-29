"""
test_uncertainty.py
-------------------
Unit tests for uncertainty quantification, conformal prediction,
calibration, and OOD detection modules.
Run with: pytest tests/test_uncertainty.py -v
"""

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset, DataLoader

from src.uncertainty.conformal import ConformalPredictor
from src.uncertainty.calibration import expected_calibration_error, calibration_report
from src.uncertainty.ood_detector import IsolationForestOOD, MahalanobisOOD, ood_report
from src.uncertainty.mc_dropout import uncertainty_summary


N = 500
N_CLASSES = 2


@pytest.fixture
def calibrated_probs():
    """Synthetic well-calibrated probabilities."""
    np.random.seed(42)
    confs = np.random.uniform(0.5, 1.0, N)
    probs = np.stack([1 - confs, confs], axis=1)
    # Generate labels consistent with probabilities
    labels = (np.random.rand(N) < confs).astype(int)
    return probs, labels


@pytest.fixture
def overconfident_probs():
    """Synthetic overconfident probabilities (model too sure)."""
    np.random.seed(42)
    confs = np.random.uniform(0.9, 1.0, N)   # always >90% confident
    probs = np.stack([1 - confs, confs], axis=1)
    # Labels are actually only 70% correct
    correct = np.random.rand(N) < 0.70
    labels = np.where(correct, np.ones(N, int), np.zeros(N, int))
    return probs, labels


# ── Conformal Prediction tests ────────────────────────────────────────────────

class TestConformalPredictor:
    def test_coverage_guarantee(self, calibrated_probs):
        """
        The empirical coverage must be >= 1-alpha on the test set.
        This is the fundamental guarantee of conformal prediction.
        """
        probs, labels = calibrated_probs
        cal_probs, cal_labels = probs[:N//2], labels[:N//2]
        test_probs, test_labels = probs[N//2:], labels[N//2:]

        cp = ConformalPredictor(alpha=0.10)
        cp.calibrate(cal_probs, cal_labels)
        coverage = cp.coverage(test_probs, test_labels)

        assert coverage >= 0.90 - 0.03, (
            f"Coverage {coverage:.3f} violated 90% guarantee. "
            "This test has a small tolerance for finite-sample variation."
        )

    def test_set_sizes_are_integers(self, calibrated_probs):
        probs, labels = calibrated_probs
        cp = ConformalPredictor(alpha=0.10)
        cp.calibrate(probs[:N//2], labels[:N//2])
        sets = cp.predict_set(probs[N//2:])
        for s in sets:
            assert isinstance(s, list)
            assert all(isinstance(c, int) for c in s)

    def test_set_classes_are_valid(self, calibrated_probs):
        probs, labels = calibrated_probs
        cp = ConformalPredictor(alpha=0.10)
        cp.calibrate(probs[:N//2], labels[:N//2])
        sets = cp.predict_set(probs[N//2:])
        for s in sets:
            for c in s:
                assert 0 <= c < N_CLASSES

    def test_uncalibrated_raises(self, calibrated_probs):
        probs, _ = calibrated_probs
        cp = ConformalPredictor()
        with pytest.raises(RuntimeError):
            cp.predict_set(probs)

    def test_q_hat_is_set_after_calibration(self, calibrated_probs):
        probs, labels = calibrated_probs
        cp = ConformalPredictor(alpha=0.05)
        assert cp.q_hat is None
        cp.calibrate(probs, labels)
        assert cp.q_hat is not None
        assert 0.0 <= cp.q_hat <= 1.0

    def test_efficiency_report_keys(self, calibrated_probs):
        probs, labels = calibrated_probs
        cp = ConformalPredictor(alpha=0.10)
        cp.calibrate(probs[:N//2], labels[:N//2])
        report = cp.efficiency_report(probs[N//2:], labels[N//2:])
        for key in ["coverage", "avg_set_size", "singleton_rate", "q_hat", "coverage_satisfied"]:
            assert key in report

    def test_stricter_alpha_gives_larger_sets(self, calibrated_probs):
        """Lower alpha (higher coverage requirement) → larger prediction sets."""
        probs, labels = calibrated_probs
        cp_strict = ConformalPredictor(alpha=0.01)   # 99% coverage
        cp_loose  = ConformalPredictor(alpha=0.20)   # 80% coverage

        cp_strict.calibrate(probs[:N//2], labels[:N//2])
        cp_loose.calibrate(probs[:N//2], labels[:N//2])

        size_strict = cp_strict.average_set_size(probs[N//2:])
        size_loose  = cp_loose.average_set_size(probs[N//2:])

        assert size_strict >= size_loose, (
            "Stricter coverage requirement should produce equal or larger sets"
        )


# ── Calibration tests ─────────────────────────────────────────────────────────

class TestCalibration:
    def test_ece_perfect_calibration(self):
        """Perfectly calibrated model should have ECE near 0."""
        # Create probs where confidence = accuracy
        np.random.seed(42)
        confs = np.linspace(0.5, 1.0, N)
        probs = np.stack([1 - confs, confs], axis=1)
        labels = (np.random.rand(N) < confs).astype(int)
        ece = expected_calibration_error(probs, labels, n_bins=10)
        assert ece < 0.15, f"Well-calibrated model should have low ECE, got {ece:.4f}"

    def test_ece_overconfident_is_higher(self, calibrated_probs, overconfident_probs):
        cal_probs, cal_labels = calibrated_probs
        oc_probs, oc_labels = overconfident_probs
        ece_cal = expected_calibration_error(cal_probs, cal_labels)
        ece_oc  = expected_calibration_error(oc_probs, oc_labels)
        assert ece_oc > ece_cal, "Overconfident model should have higher ECE"

    def test_ece_in_range(self, calibrated_probs):
        probs, labels = calibrated_probs
        ece = expected_calibration_error(probs, labels)
        assert 0.0 <= ece <= 1.0

    def test_calibration_report_keys(self, calibrated_probs, overconfident_probs):
        cal_probs, cal_labels = calibrated_probs
        oc_probs, _ = overconfident_probs
        report = calibration_report(oc_probs, cal_probs, cal_labels)
        assert "ece_before_calibration" in report
        assert "ece_after_calibration" in report
        assert "improvement" in report


# ── OOD Detection tests ───────────────────────────────────────────────────────

class TestIsolationForestOOD:
    @pytest.fixture
    def in_dist_features(self):
        np.random.seed(42)
        return np.random.normal(0, 1, (300, 64))

    @pytest.fixture
    def ood_features(self):
        np.random.seed(42)
        return np.random.normal(10, 1, (50, 64))   # far from training distribution

    def test_ood_scores_higher_for_ood(self, in_dist_features, ood_features):
        detector = IsolationForestOOD(contamination=0.05)
        detector.fit(in_dist_features)
        in_scores  = detector.score(in_dist_features)
        ood_scores = detector.score(ood_features)
        assert ood_scores.mean() > in_scores.mean(), (
            "OOD samples should have higher anomaly scores than in-distribution"
        )

    def test_predict_returns_boolean(self, in_dist_features):
        detector = IsolationForestOOD()
        detector.fit(in_dist_features)
        preds = detector.predict(in_dist_features)
        assert preds.dtype == bool

    def test_save_load(self, tmp_path, in_dist_features):
        detector = IsolationForestOOD()
        detector.fit(in_dist_features)
        scores_before = detector.score(in_dist_features)

        save_path = str(tmp_path / "iforest.pkl")
        detector.save(save_path)

        detector2 = IsolationForestOOD()
        detector2.load(save_path)
        scores_after = detector2.score(in_dist_features)

        np.testing.assert_allclose(scores_before, scores_after, rtol=1e-5)


class TestMahalanobisOOD:
    @pytest.fixture
    def features(self):
        np.random.seed(42)
        train = np.random.normal(0, 1, (200, 32))
        ood   = np.random.normal(5, 1, (50, 32))
        return train, ood

    def test_ood_scores_higher(self, features):
        train, ood = features
        detector = MahalanobisOOD()
        detector.fit(train)
        in_scores  = detector.score(train)
        ood_scores = detector.score(ood)
        assert ood_scores.mean() > in_scores.mean()

    def test_raises_before_fit(self, features):
        train, _ = features
        detector = MahalanobisOOD()
        with pytest.raises(RuntimeError):
            detector.score(train)


class TestOODReport:
    def test_report_keys(self):
        train_scores = np.random.normal(0, 1, 200)
        test_scores  = np.random.normal(2, 1, 100)
        report = ood_report(train_scores, test_scores, threshold_percentile=95)
        for key in ["threshold", "fraction_flagged_as_ood", "n_flagged", "n_total"]:
            assert key in report

    def test_fraction_in_range(self):
        train_scores = np.random.normal(0, 1, 200)
        test_scores  = np.random.normal(0, 1, 100)
        report = ood_report(train_scores, test_scores)
        assert 0.0 <= report["fraction_flagged_as_ood"] <= 1.0


# ── MC Dropout Uncertainty Summary tests ─────────────────────────────────────

class TestUncertaintySummary:
    def test_all_keys_present(self):
        mc_results = {
            "mean_probs": np.array([[0.8, 0.2], [0.4, 0.6]]),
            "epistemic_uncertainty": np.array([0.02, 0.08]),
            "aleatoric_uncertainty": np.array([0.1, 0.35]),
            "predictive_entropy": np.array([0.2, 0.55]),
            "pred_class": np.array([0, 1]),
        }
        summary = uncertainty_summary(mc_results)
        for key in ["pred_class", "confidence", "flag_any", "flag_epistemic", "flag_aleatoric"]:
            assert key in summary

    def test_flags_high_uncertainty(self):
        mc_results = {
            "mean_probs": np.array([[0.5, 0.5]]),
            "epistemic_uncertainty": np.array([0.1]),   # > threshold
            "aleatoric_uncertainty": np.array([0.5]),   # > threshold
            "predictive_entropy": np.array([0.7]),
            "pred_class": np.array([1]),
        }
        summary = uncertainty_summary(mc_results)
        assert summary["flag_any"][0], "High uncertainty sample should be flagged"