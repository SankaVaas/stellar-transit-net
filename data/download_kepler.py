"""
download_kepler.py
------------------
Downloads Kepler light curves and labels from NASA MAST + Exoplanet Archive.

Usage:
    python data/download_kepler.py --config configs/data_config.yaml --max_stars 5000

What this does:
    1. Fetches the KOI cumulative label table from NASA Exoplanet Archive
    2. Filters to CONFIRMED + CANDIDATE (positives) and FALSE POSITIVE (negatives)
    3. Downloads PDC-SAP flux light curves via lightkurve from MAST
    4. Saves each curve as a .npy file with metadata in a manifest CSV
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import lightkurve as lk
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def fetch_koi_table(label_url: str, cache_dir: Path) -> pd.DataFrame:
    """Download or load cached KOI cumulative table from NASA Exoplanet Archive."""
    cache_file = cache_dir / "koi_cumulative.csv"
    if cache_file.exists():
        log.info(f"Loading cached KOI table from {cache_file}")
        return pd.read_csv(cache_file, comment="#")

    log.info("Fetching KOI cumulative table from NASA Exoplanet Archive...")
    df = pd.read_csv(label_url, comment="#")
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_file, index=False)
    log.info(f"Saved {len(df)} KOI entries to {cache_file}")
    return df


def build_label_df(
    koi_df: pd.DataFrame,
    positive_dispositions: list,
    negative_dispositions: list,
) -> pd.DataFrame:
    """Filter KOI table to confirmed planets and false positives only."""
    pos_mask = koi_df["koi_disposition"].isin(positive_dispositions)
    neg_mask = koi_df["koi_disposition"].isin(negative_dispositions)
    filtered = koi_df[pos_mask | neg_mask].copy()
    filtered["label"] = filtered["koi_disposition"].apply(
        lambda d: 1 if d in positive_dispositions else 0
    )
    pos_count = filtered["label"].sum()
    neg_count = len(filtered) - pos_count
    log.info(f"Label distribution — positives: {pos_count}, negatives: {neg_count}")
    return filtered[["kepid", "kepoi_name", "koi_disposition", "label"]].reset_index(drop=True)


def download_light_curve(kepid: int) -> np.ndarray | None:
    """
    Download a single Kepler light curve via lightkurve 2.5.x.

    API notes for lightkurve >= 2.4:
        - `cadence` parameter removed; use `exptime=1800` for 30-min long cadence
        - `author="Kepler"` restricts to official Kepler pipeline products only
        - `quarter` omitted to return all available quarters (stitched below)

    Returns PDC-SAP flux as 1D float32 numpy array, or None on failure.
    """
    try:
        search = lk.search_lightcurve(
            f"KIC {kepid}",
            author="Kepler",
            exptime=1800,       # 1800s = 30-min long cadence
        )
        if len(search) == 0:
            return None

        lc_collection = search.download_all()
        if lc_collection is None or len(lc_collection) == 0:
            return None

        lc = lc_collection.stitch()
        lc = lc.remove_nans().remove_outliers(sigma=5.0)
        flux = lc.flux.value.astype(np.float32)
        return flux if len(flux) >= 200 else None
    except Exception as e:
        log.debug(f"Failed to download KIC {kepid}: {e}")
        return None


def run_download(config: dict, max_stars: int, output_dir: Path):
    cfg = config["kepler"]
    cache_dir = Path(cfg["cache_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    koi_df = fetch_koi_table(cfg["label_url"], cache_dir)
    label_df = build_label_df(
        koi_df, cfg["positive_dispositions"], cfg["negative_dispositions"]
    )

    # Balance classes — sample equal positives and negatives
    pos_df = label_df[label_df["label"] == 1]
    neg_df = label_df[label_df["label"] == 0]
    n_each = min(len(pos_df), len(neg_df), max_stars // 2)
    balanced = pd.concat([
        pos_df.sample(n_each, random_state=42),
        neg_df.sample(n_each, random_state=42),
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

    log.info(f"Downloading {len(balanced)} light curves ({n_each} per class)...")

    manifest_rows = []
    failed = 0

    for _, row in tqdm(balanced.iterrows(), total=len(balanced), desc="Downloading"):
        kepid = int(row["kepid"])
        save_path = output_dir / f"kic_{kepid}.npy"

        if save_path.exists():
            manifest_rows.append({
                "kepid": kepid,
                "kepoi_name": row["kepoi_name"],
                "label": row["label"],
                "disposition": row["koi_disposition"],
                "path": str(save_path),
                "length": len(np.load(save_path)),
            })
            continue

        flux = download_light_curve(kepid)
        if flux is None:
            failed += 1
            continue

        np.save(save_path, flux)
        manifest_rows.append({
            "kepid": kepid,
            "kepoi_name": row["kepoi_name"],
            "label": row["label"],
            "disposition": row["koi_disposition"],
            "path": str(save_path),
            "length": len(flux),
        })
        time.sleep(0.1)   # be polite to MAST servers

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = output_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    log.info(f"Download complete. Saved: {len(manifest_rows)}, Failed: {failed}")
    log.info(f"Manifest written to {manifest_path}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Kepler light curves from NASA MAST")
    parser.add_argument("--config", default="configs/data_config.yaml")
    parser.add_argument("--max_stars", type=int, default=5000,
                        help="Maximum number of stars to download (balanced per class)")
    parser.add_argument("--output_dir", default="data/raw/kepler/curves")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_download(cfg, args.max_stars, Path(args.output_dir))