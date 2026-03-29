"""
oracle.py
---------
The "oracle" in active learning simulates the human expert (astronomer)
who provides labels for queried samples.

In a real deployment this would be a human annotation interface.
In our simulation, the oracle is the held-out ground truth labels —
we pretend not to know them until the model asks for them.

This design cleanly separates the labelling budget simulation from
the query strategy logic, making it easy to swap in a real human
annotation pipeline later.
"""

import numpy as np
import pandas as pd
from pathlib import Path


class SimulatedOracle:
    """
    Simulates an astronomer labelling light curves on demand.

    Tracks:
        - Total labels given (labelling budget consumed)
        - Which indices have been labelled
        - Label history (for learning curve analysis)

    Args:
        manifest: full dataset manifest with true labels
        initial_labeled_indices: indices already labelled at start
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        initial_labeled_indices: np.ndarray,
    ):
        self.manifest = manifest.reset_index(drop=True)
        self.labeled_indices = set(initial_labeled_indices.tolist())
        self.label_history = []   # list of (iteration, n_newly_labeled, total_labeled)
        self.query_log = []       # which indices were queried at each step

    @property
    def n_labeled(self) -> int:
        return len(self.labeled_indices)

    @property
    def unlabeled_indices(self) -> np.ndarray:
        all_idx = set(range(len(self.manifest)))
        return np.array(sorted(all_idx - self.labeled_indices))

    def query(self, indices: np.ndarray, iteration: int = 0) -> np.ndarray:
        """
        "Ask" the oracle to label a batch of indices.

        In simulation: immediately returns true labels from manifest.
        In real deployment: would trigger a human annotation task.

        Args:
            indices: pool indices to label (into unlabeled_indices)
            iteration: current AL iteration number (for logging)

        Returns:
            labels: true labels for the queried indices
        """
        # Map pool indices back to manifest indices
        pool = self.unlabeled_indices
        manifest_indices = pool[indices]

        # Reveal true labels
        labels = self.manifest.iloc[manifest_indices]["label"].values.astype(int)

        # Update labeled set
        for idx in manifest_indices:
            self.labeled_indices.add(int(idx))

        self.query_log.append({
            "iteration": iteration,
            "queried_indices": manifest_indices.tolist(),
            "labels_revealed": labels.tolist(),
        })
        self.label_history.append({
            "iteration": iteration,
            "n_newly_labeled": len(manifest_indices),
            "total_labeled": self.n_labeled,
            "positive_rate_in_query": labels.mean(),
        })

        return labels

    def get_labeled_manifest(self) -> pd.DataFrame:
        """Return manifest rows for all currently labelled samples."""
        return self.manifest.iloc[sorted(self.labeled_indices)].copy()

    def get_unlabeled_manifest(self) -> pd.DataFrame:
        """Return manifest rows for all currently unlabelled samples."""
        return self.manifest.iloc[self.unlabeled_indices].copy()

    def budget_summary(self) -> dict:
        """Summarise labelling budget usage."""
        total = len(self.manifest)
        labeled = self.n_labeled
        pos = self.manifest.iloc[sorted(self.labeled_indices)]["label"].sum()
        return {
            "total_samples": total,
            "labeled": labeled,
            "unlabeled": total - labeled,
            "label_fraction": labeled / total,
            "positives_labeled": int(pos),
            "negatives_labeled": labeled - int(pos),
        }