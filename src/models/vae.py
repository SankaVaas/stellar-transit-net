"""
vae.py
------
Variational Autoencoder (VAE) for unsupervised anomaly detection in light curves.

How it works:
    1. Train VAE only on NORMAL (false positive / no-transit) light curves
    2. The VAE learns a compressed latent representation of normal stellar variability
    3. At inference, reconstruct any light curve through the VAE
    4. High reconstruction error → the curve is unlike anything in training → ANOMALY
    5. Anomalies include: real planet transits, novel transit morphologies,
       instrument artifacts, flaring stars, unusual binaries

Why this matters for NASA:
    This is an UNSUPERVISED discovery tool. It doesn't just classify known types —
    it flags the genuinely unexpected. Several interesting astrophysical objects
    (heartbeat stars, tidally distorted binaries, disintegrating planets) were
    found by astronomers following up on "anomalous" Kepler light curves.
    Your VAE automates that anomaly-triage step.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, latent_dim: int):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.ReLU(inplace=True)]
            in_dim = h
        self.net = nn.Sequential(*layers)
        self.fc_mu     = nn.Linear(in_dim, latent_dim)
        self.fc_logvar = nn.Linear(in_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dims: list, output_dim: int):
        super().__init__()
        layers = []
        in_dim = latent_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.ReLU(inplace=True)]
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class TransitVAE(nn.Module):
    """
    Beta-VAE for light curve anomaly detection.

    beta > 1 encourages more disentangled latent representations,
    but at the cost of reconstruction quality. beta=1 is the standard VAE.

    Args:
        input_dim: length of flux sequence
        latent_dim: size of latent space
        encoder_dims: hidden layer sizes for encoder
        decoder_dims: hidden layer sizes for decoder
        beta: weight on KL divergence term
    """

    def __init__(
        self,
        input_dim: int = 2000,
        latent_dim: int = 32,
        encoder_dims: list = None,
        decoder_dims: list = None,
        beta: float = 1.0,
    ):
        super().__init__()
        encoder_dims = encoder_dims or [512, 256, 128]
        decoder_dims = decoder_dims or [128, 256, 512]
        self.beta = beta
        self.input_dim = input_dim

        self.encoder = Encoder(input_dim, encoder_dims, latent_dim)
        self.decoder = Decoder(latent_dim, decoder_dims, input_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + eps * sigma
        Allows gradients to flow through the sampling operation.
        During inference, we can set eps=0 (use mean only) for deterministic reconstruction.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu   # deterministic at inference

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, 1, seq_len) or (batch, seq_len)
        Returns:
            (reconstruction, mu, logvar)
        """
        if x.dim() == 3:
            x = x.squeeze(1)
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def loss(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> dict:
        """
        Beta-VAE loss = reconstruction loss + beta * KL divergence

        Reconstruction loss: MSE between input and reconstructed flux
        KL divergence: measures how much latent distribution deviates from N(0,1)
        """
        if x.dim() == 3:
            x = x.squeeze(1)
        recon_loss = F.mse_loss(recon, x, reduction="mean")
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total = recon_loss + self.beta * kl_loss
        return {"total": total, "recon": recon_loss, "kl": kl_loss}


def compute_anomaly_scores(
    model: TransitVAE,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cpu",
    n_samples: int = 10,
) -> np.ndarray:
    """
    Compute reconstruction error for all samples in a dataloader.

    We run multiple stochastic forward passes (training=True to sample z)
    and average the reconstruction error, giving a more stable anomaly score.

    Returns:
        anomaly_scores: (N,) array, higher = more anomalous
    """
    model.eval()
    scores = []

    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            if x.dim() == 3:
                x_flat = x.squeeze(1)
            else:
                x_flat = x

            batch_scores = torch.zeros(x.shape[0], device=device)
            for _ in range(n_samples):
                model.train()   # enable stochastic sampling
                recon, _, _ = model(x)
                error = F.mse_loss(recon, x_flat, reduction="none").mean(dim=-1)
                batch_scores += error

            batch_scores /= n_samples
            scores.append(batch_scores.cpu().numpy())

    return np.concatenate(scores)


def anomaly_threshold(scores: np.ndarray, percentile: float = 95.0) -> float:
    """
    Compute anomaly threshold from the distribution of reconstruction errors
    on the normal (false positive) training set.

    Curves with scores above this threshold are flagged as anomalous.

    Args:
        scores: reconstruction errors from training set (normal examples only)
        percentile: e.g. 95 means top 5% of normal curves are threshold

    Returns:
        threshold value
    """
    return float(np.percentile(scores, percentile))