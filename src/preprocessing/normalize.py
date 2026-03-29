"""
normalize.py
------------
Normalizes flux arrays to a consistent scale before feeding to models.

Normalization strategy: (flux - median) / std
- Median is more robust than mean for flux data (outliers / transits skew the mean)
- Dividing by std makes transit depth comparable across stars with
  different intrinsic variability levels
- Output is centered near 0, with transit dips appearing as negative excursions
"""

import numpy as np


def normalize_median_std(flux: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Normalize flux to zero median and unit standard deviation.

    Args:
        flux: 1D flux array
        eps: small constant to prevent division by zero for constant signals

    Returns:
        normalized flux array
    """
    median = np.nanmedian(flux)
    std = np.nanstd(flux)
    return ((flux - median) / (std + eps)).astype(np.float32)


def normalize_minmax(flux: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Normalize flux to [0, 1] range.
    Less preferred than median-std for transit detection because it compresses
    the transit dip relative to the baseline range.
    Included for ablation comparison.
    """
    fmin = np.nanmin(flux)
    fmax = np.nanmax(flux)
    return ((flux - fmin) / (fmax - fmin + eps)).astype(np.float32)


def normalize_mad(flux: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Normalize using Median Absolute Deviation (MAD) — most robust to outliers.
    MAD = median(|flux - median(flux)|)
    Useful for noisy TESS light curves with frequent flares.
    """
    median = np.nanmedian(flux)
    mad = np.nanmedian(np.abs(flux - median))
    return ((flux - median) / (1.4826 * mad + eps)).astype(np.float32)


def pad_or_truncate(flux: np.ndarray, target_length: int) -> np.ndarray:
    """
    Resize a flux array to a fixed length by:
    - Truncating from the end if too long
    - Zero-padding at the end if too short

    Zero-padding is acceptable here because normalized flux is centered near 0,
    so pads blend with the baseline and do not mislead the model.

    Args:
        flux: 1D flux array of any length
        target_length: desired output length

    Returns:
        flux array of exactly target_length
    """
    n = len(flux)
    if n >= target_length:
        return flux[:target_length]
    pad = np.zeros(target_length - n, dtype=np.float32)
    return np.concatenate([flux, pad])


def normalize(flux: np.ndarray, method: str = "median_std", target_length: int = 2000) -> np.ndarray:
    """
    Full normalization: choose method, apply, then resize to target length.

    Args:
        flux: raw or detrended flux
        method: one of 'median_std', 'minmax', 'mad'
        target_length: output sequence length

    Returns:
        normalized and resized flux array of shape (target_length,)
    """
    method_map = {
        "median_std": normalize_median_std,
        "minmax": normalize_minmax,
        "mad": normalize_mad,
    }
    if method not in method_map:
        raise ValueError(f"Unknown normalization method '{method}'. Choose from {list(method_map)}")

    flux = method_map[method](flux)
    flux = pad_or_truncate(flux, target_length)
    return flux