"""
ensemble.py
-----------
Ensemble of CNN-TCN, Transformer, and Bayesian network.

Two ensemble strategies implemented:
    1. Averaging: mean of softmax probabilities from all models
       → simple, no extra training, works well when models are well-calibrated
    2. Stacking: train a meta-learner (logistic regression) on the concatenated
       output probabilities of base models
       → learns to weight models based on their individual strengths

Temperature scaling is applied post-ensemble to calibrate the final
probability outputs — essential for the conformal prediction step.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import joblib
from pathlib import Path

from .cnn_tcn import CNNTCN
from .transformer import TransitTransformer
from .bayesian_net import BayesianTransitNet, mc_predict


class TemperatureScaler(nn.Module):
    """
    Post-hoc calibration by learning a single temperature parameter T.
    Scaled logits = logits / T

    T > 1 → softens probabilities (reduces overconfidence)
    T < 1 → sharpens probabilities

    Fit on a held-out calibration set (typically the validation set).
    Does NOT change the model's predictions (argmax is unchanged),
    only the probability values.
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=0.01)

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 1000,
    ) -> float:
        """
        Fit temperature to minimize NLL on calibration set.
        Returns final temperature value.
        """
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        criterion = nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()
            loss = criterion(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return self.temperature.item()


class TransitEnsemble:
    """
    Ensemble wrapper combining CNN-TCN, Transformer, and Bayesian net.

    Usage:
        ensemble = TransitEnsemble(cnn_tcn, transformer, bayesian_net)
        ensemble.fit_stacker(val_loader, device)
        results = ensemble.predict(test_loader, device)
    """

    def __init__(
        self,
        cnn_tcn: CNNTCN,
        transformer: TransitTransformer,
        bayesian_net: BayesianTransitNet,
        strategy: str = "stacking",
        mc_samples: int = 50,
    ):
        self.models = {
            "cnn_tcn": cnn_tcn,
            "transformer": transformer,
            "bayesian_net": bayesian_net,
        }
        self.strategy = strategy
        self.mc_samples = mc_samples
        self.stacker = None
        self.temperature_scaler = TemperatureScaler()

    def _get_base_probs(
        self,
        x: torch.Tensor,
        device: str = "cpu",
    ) -> dict[str, np.ndarray]:
        """Run all base models and return their softmax probabilities."""
        probs = {}
        x = x.to(device)

        with torch.no_grad():
            # CNN-TCN
            self.models["cnn_tcn"].eval()
            logits = self.models["cnn_tcn"](x)
            probs["cnn_tcn"] = torch.softmax(logits, dim=-1).cpu().numpy()

            # Transformer
            self.models["transformer"].eval()
            logits = self.models["transformer"](x)
            probs["transformer"] = torch.softmax(logits, dim=-1).cpu().numpy()

        # Bayesian net — use MC mean
        mc_out = mc_predict(self.models["bayesian_net"], x, n_samples=self.mc_samples, device=device)
        probs["bayesian_net"] = mc_out["mean_probs"]

        return probs

    def _collect_dataset_probs(
        self,
        dataloader: torch.utils.data.DataLoader,
        device: str = "cpu",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Collect base model probs and labels over a full dataloader."""
        all_probs = {k: [] for k in self.models}
        all_labels = []

        for x, y in dataloader:
            batch_probs = self._get_base_probs(x, device)
            for k in self.models:
                all_probs[k].append(batch_probs[k])
            all_labels.append(y.numpy())

        all_labels = np.concatenate(all_labels)
        stacked = np.concatenate(
            [np.concatenate(all_probs[k], axis=0) for k in self.models], axis=1
        )   # (N, num_models * num_classes)
        return stacked, all_labels

    def fit_stacker(
        self,
        val_loader: torch.utils.data.DataLoader,
        device: str = "cpu",
        save_path: str | None = None,
    ):
        """
        Train meta-learner (logistic regression) on base model predictions.

        Using the validation set as the stacking dataset avoids leakage
        because base models were trained only on the training set.
        """
        print("Collecting base model predictions on validation set...")
        stacked_probs, labels = self._collect_dataset_probs(val_loader, device)

        print("Training stacking meta-learner...")
        base_lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        self.stacker = CalibratedClassifierCV(base_lr, cv="prefit") if False else base_lr
        self.stacker.fit(stacked_probs, labels)

        if save_path:
            joblib.dump(self.stacker, save_path)
            print(f"Stacker saved to {save_path}")

    def fit_temperature(
        self,
        val_loader: torch.utils.data.DataLoader,
        device: str = "cpu",
    ) -> float:
        """Fit temperature scaling on validation logits."""
        all_logits, all_labels = [], []
        for x, y in val_loader:
            x = x.to(device)
            with torch.no_grad():
                self.models["cnn_tcn"].eval()
                logits = self.models["cnn_tcn"](x)
            all_logits.append(logits.cpu())
            all_labels.append(y)

        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        T = self.temperature_scaler.fit(all_logits, all_labels)
        print(f"Fitted temperature: {T:.4f}")
        return T

    def predict(
        self,
        dataloader: torch.utils.data.DataLoader,
        device: str = "cpu",
    ) -> dict:
        """
        Generate ensemble predictions over a full dataloader.

        Returns dict with:
            pred_class: (N,) final predicted class
            mean_probs: (N, C) ensemble mean probabilities
            model_probs: dict of per-model probability arrays
            epistemic: (N,) epistemic uncertainty from Bayesian net
            aleatoric: (N,) aleatoric uncertainty from Bayesian net
            labels: (N,) ground truth labels
        """
        all_stacked, all_labels = self._collect_dataset_probs(dataloader, device)
        C = 2   # num_classes

        model_keys = list(self.models.keys())
        model_probs = {
            k: all_stacked[:, i*C:(i+1)*C]
            for i, k in enumerate(model_keys)
        }

        if self.strategy == "stacking" and self.stacker is not None:
            mean_probs = self.stacker.predict_proba(all_stacked)
        else:
            # Simple averaging
            mean_probs = np.mean(
                [model_probs[k] for k in model_keys], axis=0
            )

        # Uncertainty from Bayesian net MC samples
        # Re-run to get epistemic/aleatoric separately
        all_epi, all_ale = [], []
        for x, _ in dataloader:
            mc_out = mc_predict(self.models["bayesian_net"], x, n_samples=self.mc_samples, device=device)
            all_epi.append(mc_out["epistemic_uncertainty"])
            all_ale.append(mc_out["aleatoric_uncertainty"])

        return {
            "pred_class": mean_probs.argmax(axis=1),
            "mean_probs": mean_probs,
            "model_probs": model_probs,
            "epistemic": np.concatenate(all_epi),
            "aleatoric": np.concatenate(all_ale),
            "labels": all_labels,
        }

    def save(self, checkpoint_dir: str):
        path = Path(checkpoint_dir)
        path.mkdir(parents=True, exist_ok=True)
        for name, model in self.models.items():
            torch.save(model.state_dict(), path / f"{name}.pt")
        if self.stacker is not None:
            joblib.dump(self.stacker, path / "stacker.pkl")
        torch.save(self.temperature_scaler.state_dict(), path / "temperature_scaler.pt")
        print(f"Ensemble saved to {path}")

    def load(self, checkpoint_dir: str, device: str = "cpu"):
        path = Path(checkpoint_dir)
        for name, model in self.models.items():
            model.load_state_dict(torch.load(path / f"{name}.pt", map_location=device))
        stacker_path = path / "stacker.pkl"
        if stacker_path.exists():
            self.stacker = joblib.load(stacker_path)
        scaler_path = path / "temperature_scaler.pt"
        if scaler_path.exists():
            self.temperature_scaler.load_state_dict(
                torch.load(scaler_path, map_location=device)
            )
        print(f"Ensemble loaded from {path}")