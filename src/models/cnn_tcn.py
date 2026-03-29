"""
cnn_tcn.py
----------
CNN + Temporal Convolutional Network (TCN) for transit classification.

Architecture:
    Input (1, L)
      → 3× Conv1d blocks (feature extraction)
      → 3× TCN residual blocks with dilated causal convolutions
      → Global average pooling
      → Fully connected classifier

Why TCN over LSTM:
    - Dilated convolutions grow the receptive field exponentially:
      a 10-layer TCN with dilation 2^i sees 2^10 = 1024 timesteps back
    - Parallelizable (unlike LSTM which is sequential)
    - Trains 3-5× faster on CPU
    - Matches or beats LSTM on most time-series benchmarks (Bai et al. 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """
    1D convolution with causal (left) padding.
    Ensures the output at time t only depends on inputs at times ≤ t.
    Required for autoregressive / time-series use cases.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return x[:, :, :-self.padding] if self.padding > 0 else x


class TCNBlock(nn.Module):
    """
    Single TCN residual block.
    Two causal dilated conv layers + residual connection + dropout.
    Residual connection allows gradients to flow unchanged through the block,
    enabling much deeper networks without vanishing gradient.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

        # 1×1 conv to match channel dimensions for residual if needed
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.residual_conv is None else self.residual_conv(x)
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.dropout(out)
        out = self.relu(self.norm2(self.conv2(out)))
        out = self.dropout(out)
        return self.relu(out + residual)


class CNNTCN(nn.Module):
    """
    Full CNN-TCN classifier for exoplanet transit detection.

    Args:
        in_channels: number of input channels (1 for univariate flux)
        cnn_channels: list of channel sizes for 3 CNN layers
        cnn_kernel_size: kernel size for CNN layers
        tcn_channels: list of channel sizes for TCN blocks
        tcn_kernel_size: kernel size for TCN (odd recommended)
        tcn_dropout: dropout rate inside TCN blocks
        num_classes: 2 (planet vs false positive)
        fc_hidden: hidden size of final FC layer
    """

    def __init__(
        self,
        in_channels: int = 1,
        cnn_channels: list = None,
        cnn_kernel_size: int = 5,
        tcn_channels: list = None,
        tcn_kernel_size: int = 3,
        tcn_dropout: float = 0.2,
        num_classes: int = 2,
        fc_hidden: int = 256,
    ):
        super().__init__()
        cnn_channels = cnn_channels or [32, 64, 128]
        tcn_channels = tcn_channels or [128, 128, 128]

        # ── CNN feature extractor ─────────────────────────────────────────────
        cnn_layers = []
        ch_in = in_channels
        for ch_out in cnn_channels:
            cnn_layers += [
                nn.Conv1d(ch_in, ch_out, kernel_size=cnn_kernel_size, padding=cnn_kernel_size // 2),
                nn.BatchNorm1d(ch_out),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2),
            ]
            ch_in = ch_out
        self.cnn = nn.Sequential(*cnn_layers)

        # ── TCN blocks with exponential dilation ──────────────────────────────
        tcn_layers = []
        ch_in = cnn_channels[-1]
        for i, ch_out in enumerate(tcn_channels):
            dilation = 2 ** i
            tcn_layers.append(TCNBlock(ch_in, ch_out, tcn_kernel_size, dilation, tcn_dropout))
            ch_in = ch_out
        self.tcn = nn.Sequential(*tcn_layers)

        # ── Classifier head ───────────────────────────────────────────────────
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(tcn_channels[-1], fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(fc_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (batch, 1, seq_len)
        Returns:
            logits: shape (batch, num_classes)
        """
        x = self.cnn(x)
        x = self.tcn(x)
        x = self.global_pool(x)
        return self.classifier(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return feature vector before classifier (for ensemble stacking)."""
        x = self.cnn(x)
        x = self.tcn(x)
        x = self.global_pool(x)
        return x.squeeze(-1)   # (batch, tcn_channels[-1])