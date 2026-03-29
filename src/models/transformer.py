"""
transformer.py
--------------
Patch-based Transformer encoder for transit classification.

Inspired by Vision Transformer (ViT) but adapted for 1D time series:
    - Flux sequence is split into non-overlapping patches (like image patches in ViT)
    - Each patch is linearly projected into d_model dimensions
    - Positional embeddings are added (learned, not sinusoidal)
    - Standard Transformer encoder blocks with multi-head self-attention
    - [CLS] token aggregates global sequence information for classification

Why Transformer over 1D-CNN for this task:
    - Self-attention captures long-range dependencies across the full orbit period
    - A CNN with kernel 5 sees 5 timesteps; attention sees ALL timesteps simultaneously
    - This is important for detecting secondary eclipses (at phase 0.5 from primary)
      which are strong evidence AGAINST a planet (eclipsing binary impostor)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """
    Split flux sequence into fixed-size patches and project each to d_model.

    For a sequence of length L and patch_size P:
        num_patches = L // P
        output shape: (batch, num_patches, d_model)
    """

    def __init__(self, patch_size: int, d_model: int, in_channels: int = 1):
        super().__init__()
        self.patch_size = patch_size
        # Treat each patch as a flat vector; project with a linear layer
        self.projection = nn.Linear(patch_size * in_channels, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, seq_len)
        Returns:
            patches: (batch, num_patches, d_model)
        """
        B, C, L = x.shape
        P = self.patch_size
        num_patches = L // P

        # Truncate to exact multiple of patch_size
        x = x[:, :, :num_patches * P]
        # Reshape to (batch, num_patches, patch_size * channels)
        x = x.reshape(B, C, num_patches, P).permute(0, 2, 3, 1)
        x = x.reshape(B, num_patches, P * C)
        return self.projection(x)


class TransitTransformer(nn.Module):
    """
    Transformer encoder for exoplanet transit classification.

    Architecture:
        Input → Patch embedding → [CLS] token prepend → positional embedding
          → N × TransformerEncoderLayer → [CLS] token → classifier head

    Args:
        input_dim: number of input channels (1 for flux)
        patch_size: number of timesteps per patch
        d_model: transformer embedding dimension
        nhead: number of attention heads (d_model must be divisible by nhead)
        num_layers: number of stacked transformer encoder layers
        dim_feedforward: inner dimension of FFN in each transformer block
        dropout: dropout applied in attention and FFN
        num_classes: output classes (2)
        max_seq_len: maximum sequence length (for positional embedding size)
    """

    def __init__(
        self,
        input_dim: int = 1,
        patch_size: int = 50,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        num_classes: int = 2,
        max_seq_len: int = 2000,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        num_patches = max_seq_len // patch_size

        self.patch_embed = PatchEmbedding(patch_size, d_model, in_channels=input_dim)

        # Learnable [CLS] token — aggregates global information
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Learned positional embeddings (num_patches + 1 for CLS token)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,         # (batch, seq, features)
            norm_first=True,          # Pre-LN: more stable training
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1, seq_len)
        Returns:
            logits: (batch, num_classes)
        """
        B = x.shape[0]

        # Patch embedding → (batch, num_patches, d_model)
        x = self.patch_embed(x)

        # Prepend [CLS] token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        # Add positional embedding (truncate if sequence differs from max_seq_len)
        pos = self.pos_embed[:, :x.size(1), :]
        x = x + pos

        # Transformer encoder
        x = self.encoder(x)
        x = self.norm(x)

        # Use [CLS] token output for classification
        cls_output = x[:, 0, :]
        return self.classifier(cls_output)

    def get_attention_weights(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Extract attention weight matrices from all layers.
        Used by explainability/attention_rollout.py for saliency analysis.

        Returns:
            list of attention weight tensors, one per encoder layer,
            each of shape (batch, nhead, seq_len+1, seq_len+1)
        """
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, :x.size(1), :]

        attention_weights = []
        for layer in self.encoder.layers:
            # Access internal MHA to get attention weights
            x_norm = layer.norm1(x) if hasattr(layer, 'norm1') else x
            _, attn_w = layer.self_attn(x_norm, x_norm, x_norm, need_weights=True, average_attn_weights=False)
            attention_weights.append(attn_w.detach())
            # Continue forward pass
            x = layer(x)

        return attention_weights

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return [CLS] feature vector (for ensemble stacking)."""
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.encoder(x)
        x = self.norm(x)
        return x[:, 0, :]