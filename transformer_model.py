"""
Transformer-based Encoder Module for Lunar Ice Detection
Hybrid CNN-Transformer architecture for improved global context
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import math
import logging

logger = logging.getLogger(__name__)


class PatchEmbedding(nn.Module):
    """Convert image patches into token embeddings."""

    def __init__(
        self,
        patch_size: int = 8,
        in_channels: int = 7,
        embed_dim: int = 128
    ):
        """
        Initialize patch embedding.

        Args:
            patch_size: Size of patches to extract
            in_channels: Number of input channels
            embed_dim: Dimension of embedded tokens
        """
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # Use convolution for patch embedding (more efficient than linear)
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        Convert input to patch tokens.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Tuple of (tokens, num_patches_h, num_patches_w)
        """
        B, C, H, W = x.shape

        # Create patch tokens
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)

        num_patches_h = x.shape[2]
        num_patches_w = x.shape[3]

        # Flatten spatial dimensions
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)

        return x, num_patches_h, num_patches_w


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply self-attention.

        Args:
            x: Input tokens (B, N, D)

        Returns:
            Attended tokens (B, N, D)
        """
        B, N, D = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)

        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.1
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(
            embed_dim, num_heads, attn_dropout
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transformer block."""
        # Self-attention with residual
        x = x + self.attn(self.norm1(x))

        # MLP with residual
        x = x + self.mlp(self.norm2(x))

        return x


class PositionalEncoding(nn.Module):
    """Learnable positional embeddings."""

    def __init__(self, num_patches: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.drop = nn.Dropout(dropout)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding."""
        B, N, D = x.shape
        x = x + self.pos_embed[:, :N, :]
        return self.drop(x)


class CNNStem(nn.Module):
    """CNN stem for initial feature extraction."""

    def __init__(self, in_channels: int = 7, out_channels: int = 64):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=4, padding=3)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv2d(out_channels, out_channels * 2, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels * 2)
        self.act2 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract initial features."""
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        return x


class HybridTransformerVAE(nn.Module):
    """
    Hybrid CNN-Transformer VAE for lunar terrain reconstruction.

    Combines CNN local feature extraction with Transformer global context.
    """

    def __init__(
        self,
        input_channels: int = 7,
        cnn_channels: int = 64,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 6,
        mlp_ratio: float = 4.0,
        patch_size: int = 8,
        latent_dim: int = 128,
        dropout: float = 0.1
    ):
        """
        Initialize Hybrid Transformer VAE.

        Args:
            input_channels: Number of input channels
            cnn_channels: CNN stem output channels
            embed_dim: Transformer embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            mlp_ratio: MLP expansion ratio
            patch_size: Patch size for embedding
            latent_dim: Dimension of latent space
            dropout: Dropout probability
        """
        super().__init__()

        self.input_channels = input_channels
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim

        # CNN stem for local features
        self.cnn_stem = CNNStem(input_channels, cnn_channels)

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            patch_size=patch_size,
            in_channels=cnn_channels,
            embed_dim=embed_dim
        )

        # Get number of patches for positional encoding
        # Assuming 64x64 input, CNN reduces to 16x16, then patch embedding
        dummy_input = torch.zeros(1, input_channels, 64, 64)
        with torch.no_grad():
            cnn_out = self.cnn_stem(dummy_input)
            _, num_patches_h, num_patches_w = self.patch_embed(cnn_out)
        num_patches = num_patches_h * num_patches_w

        # Positional encoding
        self.pos_embed = PositionalEncoding(num_patches, embed_dim, dropout)

        # Transformer encoder
        self.transformer = nn.ModuleList([
            TransformerBlock(
                embed_dim, num_heads, mlp_ratio, dropout
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Latent space
        self.fc_mu = nn.Linear(embed_dim, latent_dim)
        self.fc_logvar = nn.Linear(embed_dim, latent_dim)

        # Decoder
        self.decoder_embed = nn.Linear(latent_dim, embed_dim)

        # Decoder transformer (with cross-attention style)
        self.decoder_transformer = nn.ModuleList([
            TransformerBlock(
                embed_dim, num_heads, mlp_ratio, dropout
            )
            for _ in range(num_layers // 2)
        ])

        self.decoder_norm = nn.LayerNorm(embed_dim)

        # Upsampling to image
        self.decoder_proj = nn.ConvTranspose2d(
            embed_dim,
            cnn_channels,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Final reconstruction
        self.reconstruction_head = nn.Sequential(
            nn.Conv2d(cnn_channels, cnn_channels // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(cnn_channels // 2, input_channels, kernel_size=1)
        )

        self.patch_size = patch_size

        logger.info(
            f"HybridTransformerVAE: {input_channels} channels, "
            f"{num_layers} transformer layers, {num_heads} heads"
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent space."""
        # CNN features
        x = self.cnn_stem(x)

        # Patch embedding
        x, h, w = self.patch_embed(x)

        # Add positional encoding
        x = self.pos_embed(x)

        # Transformer encoding
        for block in self.transformer:
            x = block(x)

        x = self.norm(x)

        # Global average pooling for latent
        x_pooled = x.mean(dim=1)

        mu = self.fc_mu(x_pooled)
        logvar = self.fc_logvar(x_pooled)

        return mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to reconstruction."""
        # Expand to sequence
        B = z.shape[0]
        dummy_input = torch.zeros(B, 1, self.embed_dim, device=z.device)
        x = dummy_input.expand(B, 64, -1)  # Reconstruct sequence

        # Decode
        x = self.decoder_embed(x)

        for block in self.decoder_transformer:
            x = block(x)

        x = self.decoder_norm(x)

        # Reshape to 2D
        # Assume 8x8 = 64 patches
        h = w = int(math.sqrt(x.shape[1]))
        x = x.transpose(1, 2).reshape(B, self.embed_dim, h, w)

        # Upsample
        x = self.decoder_proj(x)

        # Reconstruction
        x = self.reconstruction_head(x)

        return x

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)

        return reconstruction, mu, logvar

    def get_anomaly_score(
        self,
        x: torch.Tensor,
        use_mc_dropout: bool = False,
        mc_samples: int = 10
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute anomaly score."""
        self.eval()

        if use_mc_dropout:
            self.train()
            anomaly_scores = []

            for _ in range(mc_samples):
                with torch.no_grad():
                    reconstruction, _, _ = self.forward(x)
                    error = (x - reconstruction) ** 2
                    score = torch.mean(error, dim=1)
                    anomaly_scores.append(score)

            self.eval()

            anomaly_scores = torch.stack(anomaly_scores, dim=0)
            anomaly_mean = torch.mean(anomaly_scores, dim=0)
            anomaly_std = torch.std(anomaly_scores, dim=0)

            return anomaly_mean, anomaly_std
        else:
            with torch.no_grad():
                reconstruction, _, _ = self.forward(x)
                error = (x - reconstruction) ** 2
                anomaly_score = torch.mean(error, dim=1)

            return anomaly_score, None


class LightweightTransformerVAE(nn.Module):
    """
    Lightweight Transformer VAE for faster inference.
    """

    def __init__(
        self,
        input_channels: int = 7,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        latent_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()

        # Simple patch embedding
        self.patch_embed = nn.Conv2d(
            input_channels, embed_dim, kernel_size=8, stride=8
        )

        # Positional encoding
        num_patches = (64 // 8) ** 2  # 64x64 input
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Latent
        self.fc_mu = nn.Linear(embed_dim, latent_dim)
        self.fc_logvar = nn.Linear(embed_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_patches * embed_dim)
        )

        # Output
        self.output_conv = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 32, kernel_size=8, stride=8),
            nn.ReLU(),
            nn.Conv2d(32, input_channels, kernel_size=1)
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Patch embed
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, embed, 8, 8)
        x = x.flatten(2).transpose(1, 2)  # (B, 64, embed)

        # Positional encoding
        x = x + self.pos_embed

        # Transform
        x = self.transformer(x)

        # Pool and latent
        x = x.mean(dim=1)
        return self.fc_mu(x), self.fc_logvar(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        x = self.decoder(z)  # (B, 64*embed)
        x = x.reshape(B, 8, 8, -1).permute(0, 3, 1, 2)  # (B, embed, 8, 8)
        x = self.output_conv(x)
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def create_hybrid_vae(
    input_channels: int = 7,
    model_size: str = "large",
    device: str = "cuda"
) -> Tuple[nn.Module, nn.Module]:
    """
    Create a Hybrid Transformer VAE.

    Args:
        input_channels: Number of input channels
        model_size: "small", "medium", or "large"
        device: Device for model

    Returns:
        Tuple of (model, loss_fn)
    """
    configs = {
        "small": {
            "cnn_channels": 32,
            "embed_dim": 64,
            "num_heads": 4,
            "num_layers": 3,
            "latent_dim": 64
        },
        "medium": {
            "cnn_channels": 64,
            "embed_dim": 128,
            "num_heads": 8,
            "num_layers": 6,
            "latent_dim": 128
        },
        "large": {
            "cnn_channels": 96,
            "embed_dim": 192,
            "num_heads": 12,
            "num_layers": 8,
            "latent_dim": 256
        }
    }

    config = configs[model_size]

    if model_size == "small":
        model = LightweightTransformerVAE(
            input_channels=input_channels,
            **config
        )
    else:
        model = HybridTransformerVAE(
            input_channels=input_channels,
            **config
        )

    model = model.to(device)
    loss_fn = VAELoss(reconstruction_weight=1.0, kl_weight=0.1)

    return model, loss_fn


# Import VAELoss from model
from model import VAELoss
