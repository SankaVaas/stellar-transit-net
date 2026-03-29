"""
bayesian_net.py
---------------
Bayesian Neural Network using Monte Carlo (MC) Dropout for uncertainty estimation.

Key idea (Gal & Ghahramani, 2016):
    Dropout at inference time (not just training) turns a standard neural network
    into an approximate Bayesian neural network. Running T stochastic forward
    passes gives a distribution over predictions, not just a point estimate.

This gives us two types of uncertainty:
    - Epistemic uncertainty: model uncertainty, reducible with more data
      → high variance across MC samples = model hasn't seen this type of star
    - Aleatoric uncertainty: data uncertainty, irreducible
      → high mean entropy = the signal itself is ambiguous (borderline case)

For NASA: knowing WHICH uncertainty dominates tells scientists whether to
collect more observations (epistemic) or accept the ambiguity (aleatoric).
"""

import torch
import torch.nn as nn
import numpy as np


class MCDropout(nn.Dropout):
    """
    Dropout that stays active during inference (eval mode).
    Standard nn.Dropout turns off in eval(); this version does not.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.dropout(x, p=self.p, training=True, inplace=False)


class BayesianTransitNet(nn.Module):
    """
    Fully-connected Bayesian network for transit classification.

    Simpler architecture than CNN-TCN by design — the uncertainty value
    comes from the MC sampling, not architectural complexity.

    Input: flattened flux vector of length input_dim
    Output: logits of shape (batch, num_classes)

    Args:
        input_dim: length of flattened flux sequence
        hidden_dims: list of hidden layer sizes
        dropout_rate: kept ON during inference for MC sampling
        num_classes: 2
    """

    def __init__(
        self,
        input_dim: int = 2000,
        hidden_dims: list = None,
        dropout_rate: float = 0.3,
        num_classes: int = 2,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [512, 256, 128]

        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                MCDropout(p=dropout_rate),    # stays active at inference
            ]
            in_dim = h

        self.backbone = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1, seq_len) or (batch, seq_len)
        Returns:
            logits: (batch, num_classes)
        """
        if x.dim() == 3:
            x = x.squeeze(1)   # (batch, seq_len)
        x = self.backbone(x)
        return self.classifier(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)
        return self.backbone(x)


def mc_predict(
    model: BayesianTransitNet,
    x: torch.Tensor,
    n_samples: int = 50,
    device: str = "cpu",
) -> dict:
    """
    Run T stochastic forward passes and compute prediction statistics.

    The model must be in eval() mode — MCDropout stays active regardless.

    Args:
        model: BayesianTransitNet instance
        x: input tensor (batch, 1, seq_len)
        n_samples: number of MC samples (T)
        device: torch device string

    Returns dict with:
        mean_probs: (batch, num_classes) — mean softmax probability across samples
        std_probs:  (batch, num_classes) — std across samples (spread = uncertainty)
        pred_class: (batch,) — argmax of mean_probs
        epistemic_uncertainty: (batch,) — mean variance across classes (model uncertainty)
        aleatoric_uncertainty: (batch,) — mean entropy of individual sample predictions
        predictive_entropy: (batch,) — entropy of mean prediction (total uncertainty)
    """
    model.eval()
    x = x.to(device)

    # Collect T predictions: shape (T, batch, num_classes)
    sample_probs = []
    with torch.no_grad():
        for _ in range(n_samples):
            logits = model(x)
            probs = torch.softmax(logits, dim=-1)
            sample_probs.append(probs.cpu().numpy())

    sample_probs = np.stack(sample_probs, axis=0)   # (T, batch, C)

    mean_probs = sample_probs.mean(axis=0)           # (batch, C)
    std_probs  = sample_probs.std(axis=0)            # (batch, C)

    pred_class = mean_probs.argmax(axis=-1)          # (batch,)

    # Epistemic uncertainty: variance of predictions across MC samples
    # High variance = model is uncertain (hasn't seen this input type)
    epistemic = sample_probs.var(axis=0).mean(axis=-1)   # (batch,)

    # Aleatoric uncertainty: expected entropy of individual sample predictions
    # H[p] = -sum(p * log(p)); average over T samples
    eps = 1e-8
    sample_entropy = -(sample_probs * np.log(sample_probs + eps)).sum(axis=-1)  # (T, batch)
    aleatoric = sample_entropy.mean(axis=0)              # (batch,)

    # Predictive entropy: entropy of the mean prediction (total uncertainty)
    predictive_entropy = -(mean_probs * np.log(mean_probs + eps)).sum(axis=-1)  # (batch,)

    return {
        "mean_probs": mean_probs,
        "std_probs": std_probs,
        "pred_class": pred_class,
        "epistemic_uncertainty": epistemic,
        "aleatoric_uncertainty": aleatoric,
        "predictive_entropy": predictive_entropy,
    }