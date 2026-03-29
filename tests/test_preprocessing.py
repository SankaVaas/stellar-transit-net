"""
test_preprocessing.py
---------------------
Unit tests for the preprocessing pipeline.
Run with: pytest tests/test_preprocessing.py -v
"""

import numpy as np
import pytest
import torch

from src.preprocessing.detrend import sigma_clip, interpolate_nans, savgol_detrend, detrend_flux
from src.preprocessing.normalize import normalize, pad_or_truncate, normalize_median_std
from src.preprocessing.augment import add_gaussian_noise, time_shift, flux_scale, inject_transit, augment
from src.preprocessing.phase_fold import phase_fold, center_transit


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def clean_flux():
    """Synthetic sinusoidal flux with a known transit dip."""
    n = 2000
    t = np.linspace(0, 100, n)
    flux = 1.0 + 0.05 * np.sin(2 * np.pi * t / 10)   # stellar variability
    # Inject transit at t=50, depth=0.01, duration=50 timesteps
    flux[975:1025] -= 0.01
    return flux.astype(np.float32)


@pytest.fixture
def noisy_flux(clean_flux):
    """Flux with outliers and NaNs."""
    flux = clean_flux.copy()
    flux[100] = 5.0    # outlier
    flux[200] = -3.0   # outlier
    flux[300:310] = np.nan
    return flux


# ── Detrending tests ──────────────────────────────────────────────────────────

class TestSigmaClip:
    def test_clips_outliers(self, noisy_flux):
        clipped = sigma_clip(noisy_flux, threshold=5.0)
        assert np.isnan(clipped[100]), "Large positive outlier should be NaN after clipping"
        assert np.isnan(clipped[200]), "Large negative outlier should be NaN after clipping"

    def test_preserves_normal_values(self, clean_flux):
        clipped = sigma_clip(clean_flux, threshold=5.0)
        finite_before = np.isfinite(clean_flux).sum()
        finite_after = np.isfinite(clipped).sum()
        assert finite_after >= finite_before * 0.99, "Should not clip >1% of normal values"

    def test_handles_all_nan(self):
        flux = np.full(100, np.nan)
        result = sigma_clip(flux)
        assert np.all(np.isnan(result))


class TestInterpolateNans:
    def test_removes_all_nans(self, noisy_flux):
        interpolated = interpolate_nans(noisy_flux)
        assert not np.any(np.isnan(interpolated)), "No NaNs should remain after interpolation"

    def test_length_preserved(self, noisy_flux):
        result = interpolate_nans(noisy_flux)
        assert len(result) == len(noisy_flux)

    def test_no_nans_unchanged(self, clean_flux):
        result = interpolate_nans(clean_flux)
        np.testing.assert_array_almost_equal(result, clean_flux)


class TestSavgolDetrend:
    def test_output_near_one(self, clean_flux):
        detrended = savgol_detrend(clean_flux, window_length=101, polyorder=2)
        # After dividing by trend, baseline should be near 1.0
        # (transit dip will pull median slightly below 1.0)
        assert 0.98 <= np.median(detrended) <= 1.02, "Baseline should be near 1.0 after detrending"

    def test_length_preserved(self, clean_flux):
        result = savgol_detrend(clean_flux)
        assert len(result) == len(clean_flux)

    def test_transit_preserved(self, clean_flux):
        detrended = savgol_detrend(clean_flux, window_length=101, polyorder=2)
        # Transit dip should still be the minimum region
        min_region = detrended[950:1050].min()
        rest_median = np.median(np.concatenate([detrended[:950], detrended[1050:]]))
        assert min_region < rest_median, "Transit dip should be preserved after detrending"


class TestDetrend:
    def test_full_pipeline(self, noisy_flux):
        result = detrend_flux(noisy_flux)
        assert not np.any(np.isnan(result))
        assert len(result) == len(noisy_flux)
        assert result.dtype == np.float32


# ── Normalization tests ───────────────────────────────────────────────────────

class TestNormalize:
    def test_median_std_zero_median(self, clean_flux):
        normed = normalize_median_std(clean_flux)
        assert abs(np.median(normed)) < 0.1, "Median should be near 0 after normalization"

    def test_pad_short(self):
        short = np.ones(500, dtype=np.float32)
        padded = pad_or_truncate(short, target_length=2000)
        assert len(padded) == 2000
        assert np.all(padded[500:] == 0.0), "Padded region should be zeros"

    def test_truncate_long(self):
        long = np.ones(3000, dtype=np.float32)
        truncated = pad_or_truncate(long, target_length=2000)
        assert len(truncated) == 2000

    def test_normalize_output_length(self, clean_flux):
        result = normalize(clean_flux, method="median_std", target_length=2000)
        assert len(result) == 2000
        assert result.dtype == np.float32

    def test_invalid_method(self, clean_flux):
        with pytest.raises(ValueError):
            normalize(clean_flux, method="nonexistent")


# ── Augmentation tests ────────────────────────────────────────────────────────

class TestAugment:
    def test_noise_changes_values(self, clean_flux):
        noisy = add_gaussian_noise(clean_flux, std=0.01)
        assert not np.array_equal(noisy, clean_flux), "Noise should change values"
        assert len(noisy) == len(clean_flux)

    def test_time_shift_same_length(self, clean_flux):
        shifted = time_shift(clean_flux, max_shift=50)
        assert len(shifted) == len(clean_flux)

    def test_flux_scale_range(self, clean_flux):
        scaled = flux_scale(clean_flux, scale_range=(0.95, 1.05))
        ratio = scaled / clean_flux
        assert ratio.min() >= 0.94 and ratio.max() <= 1.06

    def test_inject_transit_creates_dip(self):
        flat = np.ones(2000, dtype=np.float32)
        injected = inject_transit(flat, depth=0.01, duration=50, position=1000)
        # Verify the transit region is strictly below the baseline (1.0).
        # Check a wide window (900:1100) to safely contain the injected dip regardless
        # of exact boundary arithmetic. Compare mean of dip vs mean of untouched region.
        dip_mean = injected[900:1100].mean()
        baseline_mean = injected[:500].mean()
        assert dip_mean < baseline_mean, (
            f"Transit dip mean {dip_mean:.6f} should be below baseline mean {baseline_mean:.6f}"
        )
        # Verify the injected depth is approximately correct (within 50% tolerance)
        actual_depth = float(baseline_mean - injected[975:1025].mean())
        assert actual_depth > 0.005, f"Expected depth ~0.01, got {actual_depth:.6f}"

    def test_no_horizontal_flip(self, clean_flux):
        # Verify augment() never reverses the sequence (flipping is physically invalid
        # for transits because ingress/egress shapes are asymmetric).
        # Strategy: check that augmented output correlates positively with the original.
        # A flipped sequence would have strong NEGATIVE correlation with the original.
        np.random.seed(0)
        for _ in range(20):
            result, _ = augment(clean_flux, label=1)
            # Correlation between augmented and original must be positive
            # (time_shift, noise, scaling all preserve positive correlation;
            #  a flip would give strong negative correlation near -1)
            corr = float(np.corrcoef(clean_flux, result)[0, 1])
            assert corr > 0.0, (
                f"Augmented flux has negative correlation ({corr:.3f}) with original — "
                f"possible flip detected"
            )


# ── Phase folding tests ───────────────────────────────────────────────────────

class TestPhaseFold:
    def test_output_length(self):
        time = np.linspace(0, 100, 5000)
        flux = np.ones(5000) - 0.01 * ((time % 10) < 0.5).astype(float)
        folded = phase_fold(time, flux.astype(np.float32), period=10.0, n_bins=200)
        assert len(folded) == 200

    def test_center_transit_centers_minimum(self):
        # Create array with minimum at the edge
        arr = np.ones(200, dtype=np.float32)
        arr[10] = 0.9   # dip near start
        centered = center_transit(arr)
        min_idx = np.argmin(centered)
        assert 90 <= min_idx <= 110, f"Transit should be centered, got index {min_idx}"


# ── Dataset tests ─────────────────────────────────────────────────────────────

class TestTransitDataset:
    def test_output_shape(self, tmp_path, clean_flux):
        import pandas as pd
        from src.preprocessing.pipeline import TransitDataset

        # Save a processed curve
        p = tmp_path / "kic_1.npy"
        normed = (clean_flux - clean_flux.mean()) / (clean_flux.std() + 1e-8)
        np.save(p, normed[:2000])

        manifest = pd.DataFrame([{"processed_path": str(p), "label": 1}])
        ds = TransitDataset(manifest, augment_data=False)
        x, y = ds[0]

        assert x.shape == (1, 2000), f"Expected (1, 2000), got {x.shape}"
        assert isinstance(y, torch.Tensor)
        assert y.item() in [0, 1]

    def test_augmentation_changes_input(self, tmp_path, clean_flux):
        import pandas as pd
        from src.preprocessing.pipeline import TransitDataset

        p = tmp_path / "kic_2.npy"
        np.save(p, clean_flux[:2000])
        manifest = pd.DataFrame([{"processed_path": str(p), "label": 1}])

        ds_aug = TransitDataset(manifest, augment_data=True)
        x1, _ = ds_aug[0]
        x2, _ = ds_aug[0]
        assert not torch.equal(x1, x2), "Augmented samples should differ across calls"