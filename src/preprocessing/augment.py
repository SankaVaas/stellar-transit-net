"""
augment.py
----------
Data augmentation for light curve time series.

Augmentation is critical here because the Kepler positive class (~planet candidates)
is severely underrepresented (~1-5% of all observed stars). Rather than pure
oversampling (SMOTE), we augment existing positive examples with physically
plausible transformations that preserve the transit signal.

Key constraint: do NOT horizontally flip light curves. Transit ingress and
egress have asymmetric shapes that encode physical parameters (limb darkening,
impact parameter). Flipping would produce physically unrealistic curves and
confuse the model.
"""

import numpy as np


def add_gaussian_noise(flux: np.ndarray, std: float = 0.01) -> np.ndarray:
    """
    Add zero-mean Gaussian noise to simulate additional photon noise.
    Models instrumental noise floor variations across Kepler's 84-day rolls.

    Args:
        flux: normalized flux array
        std: noise standard deviation (relative to normalized flux scale)
    """
    noise = np.random.normal(0.0, std, size=flux.shape).astype(np.float32)
    return flux + noise


def time_shift(flux: np.ndarray, max_shift: int = 50) -> np.ndarray:
    """
    Circularly shift the flux array by a random number of timesteps.
    Simulates uncertainty in transit epoch (t0) and tests the model's
    translation invariance.

    Uses circular (roll) rather than zero-padded shift to avoid introducing
    artificial edges.
    """
    shift = np.random.randint(-max_shift, max_shift + 1)
    return np.roll(flux, shift).astype(np.float32)


def flux_scale(flux: np.ndarray, scale_range: tuple = (0.95, 1.05)) -> np.ndarray:
    """
    Multiply flux by a random scalar factor.
    Simulates star-to-star variation in photometric precision and dilution
    from nearby background stars contaminating the aperture.
    """
    scale = np.random.uniform(*scale_range)
    return (flux * scale).astype(np.float32)


def inject_transit(
    flux: np.ndarray,
    depth: float | None = None,
    duration: int | None = None,
    position: int | None = None,
) -> np.ndarray:
    """
    Inject a synthetic box-shaped transit into a negative (no-planet) example
    to create additional positive training samples.

    This is a simplified transit injection — a real implementation would use
    a Mandel-Agol limb-darkened transit model, but box injection is sufficient
    for augmentation purposes.

    Args:
        flux: normalized flux array (from a false-positive star)
        depth: transit depth as fraction of flux (default: random 0.001–0.02)
        duration: transit duration in timesteps (default: random 10–60)
        position: center position of transit (default: random)

    Returns:
        flux array with injected transit dip
    """
    flux = flux.copy()
    n = len(flux)

    if depth is None:
        depth = np.random.uniform(0.001, 0.02)
    if duration is None:
        duration = np.random.randint(10, 60)
    if position is None:
        position = np.random.randint(duration, n - duration)

    start = max(0, position - duration // 2)
    end = min(n, position + duration // 2)
    flux[start:end] -= depth
    return flux.astype(np.float32)


def augment(
    flux: np.ndarray,
    label: int,
    noise_std: float = 0.01,
    time_shift_max: int = 50,
    flux_scale_range: tuple = (0.95, 1.05),
    p_inject: float = 0.3,
) -> tuple[np.ndarray, int]:
    """
    Apply a random combination of augmentations to a single sample.

    Augmentation policy:
    - Gaussian noise: always applied (both classes)
    - Time shift: always applied (both classes)
    - Flux scale: always applied (both classes)
    - Transit injection: only applied to negatives, with probability p_inject
      (creates new positive examples from negative light curves)

    Args:
        flux: normalized flux array
        label: 0 or 1
        p_inject: probability of injecting a transit into a negative sample

    Returns:
        (augmented_flux, new_label)
    """
    flux = add_gaussian_noise(flux, std=noise_std)
    flux = time_shift(flux, max_shift=time_shift_max)
    flux = flux_scale(flux, scale_range=flux_scale_range)

    if label == 0 and np.random.random() < p_inject:
        flux = inject_transit(flux)
        label = 1   # this is now a synthetic positive

    return flux, label