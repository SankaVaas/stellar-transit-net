"""
download_tess.py
----------------
Downloads TESS light curves and TOI (TESS Object of Interest) labels
from NASA MAST + Exoplanet Archive. Used for cross-mission transfer learning.

Usage:
    python data/download_tess.py --config configs/data_config.yaml --max_stars 2000
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


def fetch_toi_table(label_url: str, cache_dir: Path) -> pd.DataFrame:
    """Download or load cached TOI table from NASA Exoplanet Archive."""
    cache_file = cache_dir / "toi_table.csv"
    if cache_file.exists():
        log.info(f"Loading cached TOI table from {cache_file}")
        return pd.read_csv(cache_file, comment="#")

    log.info("Fetching TOI table from NASA Exoplanet Archive...")
    df = pd.read_csv(label_url, comment="#")
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_file, index=False)
    log.info(f"Saved {len(df)} TOI entries to {cache_file}")
    return df


def build_toi_label_df(toi_df: pd.DataFrame) -> pd.DataFrame:
    """
    Map TESS TOI TFOPWG dispositions to binary labels.
    KP/CP = confirmed/candidate planet = 1
    FP/FA = false positive/false alarm = 0
    """
    disposition_map = {
        "KP": 1,   # Known Planet
        "CP": 1,   # Confirmed Planet
        "PC": 1,   # Planet Candidate
        "FP": 0,   # False Positive
        "FA": 0,   # False Alarm
    }
    col = "tfopwg_disp" if "tfopwg_disp" in toi_df.columns else "disposition"
    filtered = toi_df[toi_df[col].isin(disposition_map)].copy()
    filtered["label"] = filtered[col].map(disposition_map)

    # Extract TIC ID (TESS Input Catalog) for lightkurve queries
    if "tid" in filtered.columns:
        filtered["tic_id"] = filtered["tid"].astype(int)
    elif "tic_id" in filtered.columns:
        filtered["tic_id"] = filtered["tic_id"].astype(int)

    pos = filtered["label"].sum()
    neg = len(filtered) - pos
    log.info(f"TOI label distribution — positives: {pos}, negatives: {neg}")
    return filtered[["tic_id", "toi", "label", col]].reset_index(drop=True)


def download_tess_light_curve(
    tic_id: int, sectors: list, cadence: str
) -> np.ndarray | None:
    """Download TESS light curve for a given TIC ID."""
    try:
        search = lk.search_lightcurve(
            f"TIC {tic_id}",
            mission="TESS",
            cadence=cadence,
            sector=sectors,
        )
        if len(search) == 0:
            return None

        lc_collection = search.download_all()
        lc = lc_collection.stitch()
        lc = lc.remove_nans().remove_outliers(sigma=5.0)
        flux = lc.flux.value.astype(np.float32)
        return flux
    except Exception as e:
        log.debug(f"Failed to download TIC {tic_id}: {e}")
        return None


def run_download(config: dict, max_stars: int, output_dir: Path):
    cfg = config["tess"]
    cache_dir = Path(cfg["cache_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    toi_df = fetch_toi_table(cfg["label_url"], cache_dir)
    label_df = build_toi_label_df(toi_df)

    pos_df = label_df[label_df["label"] == 1]
    neg_df = label_df[label_df["label"] == 0]
    n_each = min(len(pos_df), len(neg_df), max_stars // 2)
    balanced = pd.concat([
        pos_df.sample(n_each, random_state=42),
        neg_df.sample(n_each, random_state=42),
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

    log.info(f"Downloading {len(balanced)} TESS light curves...")

    manifest_rows = []
    failed = 0

    for _, row in tqdm(balanced.iterrows(), total=len(balanced), desc="TESS download"):
        tic_id = int(row["tic_id"])
        save_path = output_dir / f"tic_{tic_id}.npy"

        if save_path.exists():
            manifest_rows.append({
                "tic_id": tic_id,
                "toi": row.get("toi", ""),
                "label": row["label"],
                "path": str(save_path),
                "length": len(np.load(save_path)),
            })
            continue

        flux = download_tess_light_curve(tic_id, cfg["sectors"], cfg["cadence"])
        if flux is None or len(flux) < 200:
            failed += 1
            continue

        np.save(save_path, flux)
        manifest_rows.append({
            "tic_id": tic_id,
            "toi": row.get("toi", ""),
            "label": row["label"],
            "path": str(save_path),
            "length": len(flux),
        })
        time.sleep(0.1)

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = output_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    log.info(f"TESS download complete. Saved: {len(manifest_rows)}, Failed: {failed}")
    log.info(f"Manifest written to {manifest_path}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download TESS light curves from NASA MAST")
    parser.add_argument("--config", default="configs/data_config.yaml")
    parser.add_argument("--max_stars", type=int, default=2000)
    parser.add_argument("--output_dir", default="data/raw/tess/curves")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_download(cfg, args.max_stars, Path(args.output_dir))