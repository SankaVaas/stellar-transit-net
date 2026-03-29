"""
pipeline.py
-----------
End-to-end preprocessing pipeline: raw flux → model-ready tensor.

Also defines the PyTorch Dataset class used by all training scripts.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import yaml

from .detrend import detrend_flux
from .phase_fold import bls_period_search, phase_fold, center_transit
from .normalize import normalize
from .augment import augment

log = logging.getLogger(__name__)


def preprocess_single(
    flux_raw: np.ndarray,
    time: np.ndarray | None = None,
    target_length: int = 2000,
    window_length: int = 101,
    polyorder: int = 2,
    sigma_threshold: float = 5.0,
    do_phase_fold: bool = True,
    normalize_method: str = "median_std",
) -> np.ndarray:
    """
    Process one raw light curve into a fixed-length normalized array.

    Pipeline:
        raw flux
          → sigma clip + SG detrend
          → [optional] BLS period search + phase fold + center transit
          → normalize to fixed length

    Args:
        flux_raw: raw flux values (arbitrary length)
        time: time array in days (required if do_phase_fold=True)
        target_length: output sequence length
        do_phase_fold: if True, fold on best BLS period

    Returns:
        processed flux array of shape (target_length,)
    """
    flux = detrend_flux(
        flux_raw,
        window_length=window_length,
        polyorder=polyorder,
        sigma_threshold=sigma_threshold,
    )

    if do_phase_fold and time is not None and len(time) == len(flux):
        try:
            period = bls_period_search(time, flux)
            folded = phase_fold(time, flux, period=period, n_bins=target_length)
            flux = center_transit(folded)
        except Exception as e:
            log.debug(f"Phase folding failed, using raw detrended curve: {e}")
            flux = normalize(flux, method=normalize_method, target_length=target_length)
            return flux

    flux = normalize(flux, method=normalize_method, target_length=target_length)
    return flux


def build_processed_dataset(
    manifest_path: str | Path,
    output_dir: str | Path,
    config: dict,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Batch-process all light curves listed in a manifest CSV.
    Saves each processed curve as a .npy file.

    Args:
        manifest_path: path to manifest.csv from download scripts
        output_dir: directory to save processed .npy files
        config: data config dict (from data_config.yaml)
        overwrite: reprocess even if output already exists

    Returns:
        updated manifest DataFrame with 'processed_path' column
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_path)
    cfg = config["preprocessing"]
    processed_paths = []

    for _, row in manifest.iterrows():
        raw_path = Path(row["path"])
        processed_path = output_dir / raw_path.name

        if processed_path.exists() and not overwrite:
            processed_paths.append(str(processed_path))
            continue

        try:
            flux_raw = np.load(raw_path).astype(np.float32)
            processed = preprocess_single(
                flux_raw,
                target_length=cfg["target_length"],
                window_length=cfg["detrend_window_length"],
                polyorder=cfg["detrend_polyorder"],
                sigma_threshold=cfg["sigma_clip_threshold"],
                do_phase_fold=cfg["phase_fold"],
                normalize_method=cfg["normalize_method"],
            )
            np.save(processed_path, processed)
            processed_paths.append(str(processed_path))
        except Exception as e:
            log.warning(f"Failed to process {raw_path}: {e}")
            processed_paths.append(None)

    manifest["processed_path"] = processed_paths
    manifest = manifest[manifest["processed_path"].notna()].reset_index(drop=True)
    updated_manifest = output_dir / "manifest_processed.csv"
    manifest.to_csv(updated_manifest, index=False)
    log.info(f"Processed {len(manifest)} curves. Manifest saved to {updated_manifest}")
    return manifest


class TransitDataset(Dataset):
    """
    PyTorch Dataset for transit detection.

    Each item is a (flux_tensor, label) pair where:
        flux_tensor: shape (1, target_length)  — channel-first for Conv1d
        label: int, 0 = false positive, 1 = planet candidate/confirmed
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        augment_data: bool = False,
        augment_config: dict | None = None,
    ):
        """
        Args:
            manifest: DataFrame with columns ['processed_path', 'label']
            augment_data: if True, apply random augmentation each __getitem__
            augment_config: augmentation hyperparameters (from train_config.yaml)
        """
        self.manifest = manifest.reset_index(drop=True)
        self.augment_data = augment_data
        self.aug_cfg = augment_config or {}

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.manifest.iloc[idx]
        flux = np.load(row["processed_path"]).astype(np.float32)
        label = int(row["label"])

        if self.augment_data:
            flux, label = augment(
                flux,
                label,
                noise_std=self.aug_cfg.get("gaussian_noise_std", 0.01),
                time_shift_max=self.aug_cfg.get("time_shift_max", 50),
                flux_scale_range=self.aug_cfg.get("flux_scale_range", (0.95, 1.05)),
            )

        flux_tensor = torch.tensor(flux, dtype=torch.float32).unsqueeze(0)  # (1, L)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return flux_tensor, label_tensor


def build_dataloaders(
    train_manifest: pd.DataFrame,
    val_manifest: pd.DataFrame,
    test_manifest: pd.DataFrame,
    train_config: dict,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train, val, and test DataLoaders from split manifests.

    Args:
        train/val/test_manifest: DataFrames with processed_path and label
        train_config: training config dict (from train_config.yaml)

    Returns:
        (train_loader, val_loader, test_loader)
    """
    aug_cfg = train_config.get("augmentation", {})
    batch = train_config["training"]["batch_size"]
    workers = train_config["data"]["num_workers"]
    pin = train_config["data"]["pin_memory"]

    train_ds = TransitDataset(train_manifest, augment_data=aug_cfg.get("enabled", True), augment_config=aug_cfg)
    val_ds   = TransitDataset(val_manifest,   augment_data=False)
    test_ds  = TransitDataset(test_manifest,  augment_data=False)

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=batch, shuffle=False, num_workers=workers, pin_memory=pin)

    log.info(f"Dataset sizes — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")
    return train_loader, val_loader, test_loader