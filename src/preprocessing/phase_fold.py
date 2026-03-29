"""
phase_fold.py
-------------
Phase-folds a detrended light curve onto a candidate orbital period,
aligning all transit dips on top of each other to produce a clean,
high-SNR stacked transit view.

This is a critical preprocessing step used by AstroNet and all subsequent
exoplanet ML pipelines. A phase-folded view dramatically reduces the
sequence length the model needs to reason over (from 50,000+ timesteps
down to ~200 around the transit) and makes the transit depth/shape
the dominant signal.
"""

import numpy as np
from scipy.signal import lombscargle


def bls_period_search(
    time: np.ndarray,
    flux: np.ndarray,
    min_period: float = 0.5,
    max_period: float = 500.0,
    n_periods: int = 10000,
) -> float:
    """
    Box Least Squares (BLS) period search.
    Finds the orbital period that produces the deepest, most consistent
    transit-like dip when the light curve is folded at that period.

    We implement a simplified BLS using scipy's lombscargle as a fast
    approximation. For production use, consider astropy.timeseries.BoxLeastSquares.

    Args:
        time: time array in days (BJD or BKJD)
        flux: detrended relative flux
        min_period: minimum search period in days
        max_period: maximum search period in days
        n_periods: number of trial periods

    Returns:
        best_period: period in days with strongest transit signal
    """
    try:
        from astropy.timeseries import BoxLeastSquares
        model = BoxLeastSquares(time, flux)
        periodogram = model.autopower(
            duration=[0.05, 0.1, 0.2],   # transit durations in days to test
            minimum_period=min_period,
            maximum_period=max_period,
            frequency_factor=1.0,
        )
        best_period = periodogram.period[np.argmax(periodogram.power)]
        return float(best_period)
    except Exception:
        # Fallback: Lomb-Scargle (less accurate for box-shaped transits)
        freqs = np.linspace(1.0 / max_period, 1.0 / min_period, n_periods)
        ang_freqs = 2 * np.pi * freqs
        power = lombscargle(time, flux - flux.mean(), ang_freqs, normalize=True)
        best_freq = freqs[np.argmax(power)]
        return float(1.0 / best_freq)


def phase_fold(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float | None = None,
    n_bins: int = 200,
) -> np.ndarray:
    """
    Phase-fold and bin a light curve onto a given period.

    Steps:
        1. Compute phase = ((time - t0) % period) / period  → [0, 1)
        2. Bin flux values into n_bins equally-spaced phase bins
        3. Return mean flux per bin as a fixed-length array

    Args:
        time: time array in days
        flux: detrended flux array
        period: orbital period in days
        t0: reference epoch (time of first transit); defaults to time[0]
        n_bins: output sequence length

    Returns:
        binned_flux: shape (n_bins,), phase-folded and binned relative flux
    """
    if t0 is None:
        t0 = time[0]

    phase = ((time - t0) % period) / period    # phase in [0, 1)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(phase, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    binned_flux = np.zeros(n_bins, dtype=np.float32)
    bin_counts = np.zeros(n_bins, dtype=np.int32)

    np.add.at(binned_flux, bin_indices, flux)
    np.add.at(bin_counts, bin_indices, 1)

    # Fill empty bins with local linear interpolation
    empty = bin_counts == 0
    if empty.any():
        x = np.arange(n_bins)
        binned_flux[empty] = np.interp(x[empty], x[~empty], binned_flux[~empty] / bin_counts[~empty])
        bin_counts[empty] = 1

    return binned_flux / bin_counts


def center_transit(folded_flux: np.ndarray) -> np.ndarray:
    """
    Shift the phase-folded array so the transit dip is centred.
    The model learns better when the transit is always at position n//2.

    Finds the minimum (deepest dip) and rolls the array accordingly.
    """
    min_idx = int(np.argmin(folded_flux))
    center = len(folded_flux) // 2
    shift = center - min_idx
    return np.roll(folded_flux, shift)