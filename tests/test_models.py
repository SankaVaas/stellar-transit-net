"""
test_models.py
--------------
Unit tests for all model architectures.
Run with: pytest tests/test_models.py -v
"""

import numpy as np
import pytest
import torch

from src.models.cnn_tcn import CNNTCN, TCNBlock, CausalConv1d
from src.models.transformer import TransitTransformer, PatchEmbedding
from src.models.bayesian_net import BayesianTransitNet, mc_predict
from src.models.vae import TransitVAE, compute_anomaly_scores, anomaly_threshold
from src.models.ensemble import TemperatureScaler


BATCH = 4
SEQ_LEN = 2000
N_CLASSES = 2


@pytest.fixture
def dummy_batch():
    return torch.randn(BATCH, 1, SEQ_LEN)


# ── CNN-TCN tests ─────────────────────────────────────────────────────────────

class TestCNNTCN:
    def test_output_shape(self, dummy_batch):
        model = CNNTCN()
        out = model(dummy_batch)
        assert out.shape == (BATCH, N_CLASSES), f"Expected ({BATCH}, {N_CLASSES}), got {out.shape}"

    def test_forward_no_nan(self, dummy_batch):
        model = CNNTCN()
        out = model(dummy_batch)
        assert not torch.isnan(out).any(), "Output contains NaN"

    def test_get_features_shape(self, dummy_batch):
        model = CNNTCN(tcn_channels=[128, 128, 128])
        feats = model.get_features(dummy_batch)
        assert feats.shape == (BATCH, 128), f"Expected ({BATCH}, 128), got {feats.shape}"

    def test_causal_conv_no_future_leakage(self):
        """
        Causal conv should not allow information from future timesteps.
        Test: if we change the last timestep of the input, only the last
        output position should change (for a single causal conv).
        """
        conv = CausalConv1d(1, 1, kernel_size=3, dilation=1)
        x1 = torch.zeros(1, 1, 100)
        x2 = x1.clone()
        x2[0, 0, -1] = 1.0   # change only last timestep

        with torch.no_grad():
            out1 = conv(x1)
            out2 = conv(x2)

        diff = (out1 - out2).abs()
        # Only the last few positions should differ
        assert diff[0, 0, :-3].max().item() < 1e-5, "Early positions changed — causality violated"

    def test_tcn_block_output_shape(self, dummy_batch):
        block = TCNBlock(1, 64, kernel_size=3, dilation=1)
        out = block(dummy_batch)
        assert out.shape == (BATCH, 64, SEQ_LEN)

    def test_gradient_flow(self, dummy_batch):
        model = CNNTCN()
        out = model(dummy_batch)
        loss = out.sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"


# ── Transformer tests ─────────────────────────────────────────────────────────

class TestTransformer:
    def test_output_shape(self, dummy_batch):
        model = TransitTransformer(patch_size=50, d_model=128, nhead=4, num_layers=2)
        out = model(dummy_batch)
        assert out.shape == (BATCH, N_CLASSES)

    def test_forward_no_nan(self, dummy_batch):
        model = TransitTransformer()
        out = model(dummy_batch)
        assert not torch.isnan(out).any()

    def test_patch_embedding_shape(self):
        embed = PatchEmbedding(patch_size=50, d_model=128)
        x = torch.randn(BATCH, 1, SEQ_LEN)
        out = embed(x)
        expected_patches = SEQ_LEN // 50
        assert out.shape == (BATCH, expected_patches, 128)

    def test_get_features_shape(self, dummy_batch):
        model = TransitTransformer(d_model=128)
        feats = model.get_features(dummy_batch)
        assert feats.shape == (BATCH, 128)

    def test_attention_weights_returned(self, dummy_batch):
        model = TransitTransformer(num_layers=2)
        attn_weights = model.get_attention_weights(dummy_batch)
        assert len(attn_weights) == 2, "Should return one attention map per layer"


# ── Bayesian Net tests ────────────────────────────────────────────────────────

class TestBayesianNet:
    def test_output_shape(self, dummy_batch):
        model = BayesianTransitNet(input_dim=SEQ_LEN)
        out = model(dummy_batch)
        assert out.shape == (BATCH, N_CLASSES)

    def test_mc_dropout_stochastic(self, dummy_batch):
        """MC dropout should produce different outputs on each forward pass."""
        model = BayesianTransitNet(input_dim=SEQ_LEN, dropout_rate=0.5)
        model.eval()
        out1 = model(dummy_batch)
        out2 = model(dummy_batch)
        assert not torch.equal(out1, out2), "MC dropout should produce stochastic outputs at eval time"

    def test_mc_predict_returns_uncertainty(self, dummy_batch):
        model = BayesianTransitNet(input_dim=SEQ_LEN)
        result = mc_predict(model, dummy_batch, n_samples=10)

        assert "epistemic_uncertainty" in result
        assert "aleatoric_uncertainty" in result
        assert "predictive_entropy" in result
        assert result["epistemic_uncertainty"].shape == (BATCH,)
        assert result["mean_probs"].shape == (BATCH, N_CLASSES)

    def test_uncertainty_non_negative(self, dummy_batch):
        model = BayesianTransitNet(input_dim=SEQ_LEN)
        result = mc_predict(model, dummy_batch, n_samples=10)
        assert (result["epistemic_uncertainty"] >= 0).all()
        assert (result["aleatoric_uncertainty"] >= 0).all()

    def test_mean_probs_sum_to_one(self, dummy_batch):
        model = BayesianTransitNet(input_dim=SEQ_LEN)
        result = mc_predict(model, dummy_batch, n_samples=10)
        row_sums = result["mean_probs"].sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(BATCH), atol=1e-5)


# ── VAE tests ─────────────────────────────────────────────────────────────────

class TestVAE:
    def test_forward_output_shapes(self, dummy_batch):
        model = TransitVAE(input_dim=SEQ_LEN)
        recon, mu, logvar = model(dummy_batch)
        assert recon.shape == (BATCH, SEQ_LEN)
        assert mu.shape[1] == 32   # latent_dim
        assert logvar.shape[1] == 32

    def test_loss_dict_keys(self, dummy_batch):
        model = TransitVAE(input_dim=SEQ_LEN)
        recon, mu, logvar = model(dummy_batch)
        loss_dict = model.loss(dummy_batch, recon, mu, logvar)
        assert "total" in loss_dict
        assert "recon" in loss_dict
        assert "kl" in loss_dict

    def test_loss_positive(self, dummy_batch):
        model = TransitVAE(input_dim=SEQ_LEN)
        recon, mu, logvar = model(dummy_batch)
        loss_dict = model.loss(dummy_batch, recon, mu, logvar)
        assert loss_dict["total"].item() > 0

    def test_deterministic_at_eval(self, dummy_batch):
        """At eval time, reparameterize should use mean (no sampling)."""
        model = TransitVAE(input_dim=SEQ_LEN)
        model.eval()
        with torch.no_grad():
            recon1, mu1, _ = model(dummy_batch)
            recon2, mu2, _ = model(dummy_batch)
        torch.testing.assert_close(mu1, mu2)
        torch.testing.assert_close(recon1, recon2)

    def test_anomaly_threshold(self):
        normal_scores = np.random.normal(0, 1, 1000)
        threshold = anomaly_threshold(normal_scores, percentile=95)
        flagged_fraction = (normal_scores > threshold).mean()
        assert abs(flagged_fraction - 0.05) < 0.02, "~5% of normal examples should exceed threshold"


# ── Temperature Scaler tests ──────────────────────────────────────────────────

class TestTemperatureScaler:
    def test_output_shape(self, dummy_batch):
        model = CNNTCN()
        logits = model(dummy_batch)
        scaler = TemperatureScaler()
        scaled = scaler(logits)
        assert scaled.shape == logits.shape

    def test_temperature_softens_distribution(self):
        logits = torch.tensor([[2.0, -2.0], [1.0, -1.0]])
        scaler = TemperatureScaler()
        scaler.temperature.data = torch.tensor([2.0])
        scaled = scaler(logits)
        # Scaled logits should be closer to zero (softer distribution)
        assert scaled.abs().max() < logits.abs().max()