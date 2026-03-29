"""
detrend.py
----------
Removes long-term stellar variability (starspots, pulsations, instrument drift)
from raw flux time series, leaving only short-duration transit signals.

Method: Savitzky-Golay smoothing followed by division (not subtraction),
which preserves the relative depth of transit dips regardless of flux baseline.
This mirrors the approach used in the Kepler pipeline (PDC-MAP) and AstroNet.
"""

import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import sigmaclip


def sigma_clip(flux: np.ndarray, threshold: float = 5.0) -> np.ndarray:
    """
    Replace outlier flux values with NaN using iterative sigma clipping.
    Outliers arise from cosmic rays, sudden pixel sensitivity drops, and
    momentum dumps in the Kepler/TESS spacecraft.

    Args:
        flux: 1D flux array, may contain NaNs
        threshold: number of standard deviations beyond which a point is clipped

    Returns:
        flux array with outliers replaced by NaN
    """
    flux = flux.copy().astype(np.float64)
    finite_mask = np.isfinite(flux)
    if finite_mask.sum() < 10:
        return flux

    _, low, high = sigmaclip(flux[finite_mask], low=threshold, high=threshold)
    outlier_mask = (flux < low) | (flux > high)
    flux[outlier_mask] = np.nan
    return flux


def interpolate_nans(flux: np.ndarray) -> np.ndarray:
    """
    Linear interpolation over NaN gaps before smoothing.
    Savitzky-Golay requires no NaNs in input.
    """
    flux = flux.copy()
    nans = np.isnan(flux)
    if not nans.any():
        return flux
    x = np.arange(len(flux))
    flux[nans] = np.interp(x[nans], x[~nans], flux[~nans])
    return flux


def savgol_detrend(
    flux: np.ndarray,
    window_length: int = 101,
    polyorder: int = 2,
) -> np.ndarray:
    """
    Detrend flux by dividing by a Savitzky-Golay smooth baseline.

    Division (flux / trend) rather than subtraction (flux - trend) is used
    because it produces a fractional deviation (relative flux), making transit
    depths comparable across stars of different brightness.

    Args:
        flux: 1D flux array, NaN-free
        window_length: SG filter window in timesteps (must be odd, > polyorder)
        polyorder: polynomial order for SG filter

    Returns:
        detrended relative flux, centered near 1.0
    """
    if window_length % 2 == 0:
        window_length += 1
    window_length = min(window_length, len(flux) - 1)
    if window_length <= polyorder:
        window_length = polyorder + 2
        if window_length % 2 == 0:
            window_length += 1

    trend = savgol_filter(flux, window_length=window_length, polyorder=polyorder)
    # Avoid division by near-zero trend values
    trend = np.where(np.abs(trend) < 1e-8, 1e-8, trend)
    return flux / trend


def detrend_flux(
    flux: np.ndarray,
    window_length: int = 101,
    polyorder: int = 2,
    sigma_threshold: float = 5.0,
) -> np.ndarray:
    """
    Full detrending pipeline for a single light curve:
        1. Sigma clip outliers
        2. Interpolate NaN gaps
        3. Savitzky-Golay baseline removal

    Args:
        flux: raw PDC-SAP flux from Kepler/TESS
        window_length: SG window (timesteps)
        polyorder: SG polynomial order
        sigma_threshold: clipping threshold

    Returns:
        detrended relative flux array, same length as input
    """
    flux = sigma_clip(flux, threshold=sigma_threshold)
    flux = interpolate_nans(flux)
    flux = savgol_detrend(flux, window_length=window_length, polyorder=polyorder)
    return flux.astype(np.float32)