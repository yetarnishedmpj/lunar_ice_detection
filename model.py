"""
Model Architecture Module for Lunar Ice Detection Pipeline
Convolutional Variational Autoencoder for anomaly detection in lunar terrain
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import numpy.typing as npt
from typing import Tuple, Optional, List, Dict
import logging

logger = logging.getLogger(__name__)


class ConvVAE(nn.Module):
    """
    Convolutional Variational Autoencoder for lunar terrain reconstruction.

    The VAE learns to reconstruct normal lunar terrain from multi-modal
    remote sensing data. Anomalous regions (potential ice deposits) will
    have higher reconstruction errors.

    Architecture:
    - Encoder: Series of conv layers that compress spatial patches into latent space
    - Reparameterization: Sample from learned mean and variance
    - Decoder: Series of transposed conv layers to reconstruct the input
    """

    def __init__(
        self,
        input_channels: int = 7,
        hidden_channels: List[int] = [32, 64, 128, 256],
        latent_dim: int = 128,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        dropout: float = 0.2
    ):
        """
        Initialize the Convolutional VAE.

        Args:
            input_channels: Number of input channels (LOLA, Diviner, LEND, Mini-RF bands)
            hidden_channels: List of hidden channel dimensions for each conv layer
            latent_dim: Dimension of the latent space
            kernel_size: Convolution kernel size
            stride: Convolution stride
            padding: Convolution padding
            dropout: Dropout probability
        """
        super(ConvVAE, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.latent_dim = latent_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Build encoder
        encoder_layers = []
        in_channels = input_channels

        for i, out_channels in enumerate(hidden_channels):
            encoder_layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding
                )
            )
            encoder_layers.append(nn.BatchNorm2d(out_channels))
            encoder_layers.append(nn.LeakyReLU(0.2))
            encoder_layers.append(nn.Dropout2d(dropout))
            in_channels = out_channels

        self.encoder = nn.Sequential(*encoder_layers)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        # Calculate flattened size after encoder
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, 64, 64)
            dummy_output = self.pool(self.encoder(dummy_input))
            self.flat_features = dummy_output.view(dummy_output.size(0), -1).shape[1]

        # Latent space layers (mean and log variance)
        self.fc_mu = nn.Linear(self.flat_features, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_features, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, self.flat_features)

        decoder_layers = []
        hidden_channels_rev = hidden_channels[::-1]

        for i in range(len(hidden_channels_rev)):
            in_ch = hidden_channels_rev[i]
            out_ch = hidden_channels_rev[i + 1] if i < len(hidden_channels_rev) - 1 else input_channels

            # Internal decoder layers use output_padding = stride - 1 to undo
            # the encoder downsampling; the final layer uses 0 to land exactly
            # on the original spatial size.
            output_padding = stride - 1 if i < len(hidden_channels_rev) - 1 else 0

            decoder_layers.append(
                nn.ConvTranspose2d(
                    in_ch,
                    out_ch,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding
                )
            )

            if i < len(hidden_channels_rev) - 1:
                decoder_layers.append(nn.BatchNorm2d(out_ch))
                decoder_layers.append(nn.LeakyReLU(0.2))
                decoder_layers.append(nn.Dropout2d(dropout))

        self.decoder = nn.Sequential(*decoder_layers)

        logger.info(
            f"ConvVAE initialized: {input_channels} input channels, "
            f"hidden {hidden_channels}, latent {latent_dim}"
        )

        self.final_encoder_channels = hidden_channels[-1]

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent space parameters.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Tuple of (mu, logvar) for the latent distribution
        """
        h = self.encoder(x)
        h = self.pool(h)
        h = h.view(h.size(0), -1)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar

    def _decoder_spatial_shape(self) -> Tuple[int, int]:
        """Infer the (H, W) the decoder reshapes latent vectors to.

        Falls back to 8x8 if the encoder spatial size cannot be inferred
        from flat_features (e.g. for non-square / non-power-of-two inputs).
        """
        if self.flat_features % self.final_encoder_channels != 0:
            return 8, 8
        side = int(np.sqrt(self.flat_features / self.final_encoder_channels))
        if side * side * self.final_encoder_channels != self.flat_features:
            return 8, 8
        return side, side

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for sampling from N(mu, var).

        Args:
            mu: Mean of the latent distribution
            logvar: Log variance of the latent distribution

        Returns:
            Sampled latent vector
        """
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, target_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Decode latent vector to reconstructed output.

        Args:
            z: Latent vector
            target_size: Optional spatial (H, W) to interpolate reconstruction to

        Returns:
            Reconstructed tensor
        """
        h = self.fc_decode(z)
        side_h, side_w = self._decoder_spatial_shape()
        h = h.view(h.size(0), self.hidden_channels[-1], side_h, side_w)
        reconstruction = self.decoder(h)
        if target_size is not None and reconstruction.shape[-2:] != target_size:
            reconstruction = F.interpolate(reconstruction, size=target_size, mode='bilinear', align_corners=False)
        return reconstruction

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VAE.

        Args:
            x: Input tensor

        Returns:
            Tuple of (reconstruction, mu, logvar)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z, target_size=x.shape[-2:])

        return reconstruction, mu, logvar

    def get_anomaly_score(
        self,
        x: torch.Tensor,
        use_mc_dropout: bool = False,
        mc_samples: int = 10
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute reconstruction error as anomaly score.

        Args:
            x: Input tensor
            use_mc_dropout: If True, use Monte Carlo dropout for uncertainty estimation
            mc_samples: Number of MC dropout samples for uncertainty estimation

        Returns:
            Tuple of (anomaly_score, uncertainty_map) where uncertainty_map is None if use_mc_dropout=False
        """
        self.eval()

        if use_mc_dropout:
            # Enable dropout at inference time
            self.train()
            anomaly_scores = []

            for _ in range(mc_samples):
                with torch.no_grad():
                    reconstruction, _, _ = self.forward(x)
                    error = (x - reconstruction) ** 2
                    score = torch.mean(error, dim=1)
                    anomaly_scores.append(score)

            self.eval()

            # Stack and compute statistics
            anomaly_scores = torch.stack(anomaly_scores, dim=0)  # (mc_samples, batch, H, W)
            anomaly_mean = torch.mean(anomaly_scores, dim=0)
            anomaly_std = torch.std(anomaly_scores, dim=0)

            return anomaly_mean, anomaly_std
        else:
            with torch.no_grad():
                reconstruction, _, _ = self.forward(x)
                error = (x - reconstruction) ** 2
                anomaly_score = torch.mean(error, dim=1)

            return anomaly_score, None


class VAELoss(nn.Module):
    """
    Combined VAE loss function.

    Loss = Reconstruction Loss + KL Divergence
    """

    def __init__(self, reconstruction_weight: float = 1.0, kl_weight: float = 0.1):
        """
        Initialize VAE loss.

        Args:
            reconstruction_weight: Weight for reconstruction loss
            kl_weight: Weight for KL divergence loss
        """
        super(VAELoss, self).__init__()
        self.reconstruction_weight = reconstruction_weight
        self.kl_weight = kl_weight

    def forward(
        self,
        reconstruction: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the VAE loss.

        Args:
            reconstruction: Reconstructed input
            x: Original input
            mu: Latent mean
            logvar: Latent log variance

        Returns:
            Tuple of (total_loss, reconstruction_loss, kl_loss)
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstruction, x, reduction='sum')

        # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Combined loss
        total_loss = self.reconstruction_weight * recon_loss + self.kl_weight * kl_loss

        return total_loss, recon_loss, kl_loss


class LunarPatchDataset(Dataset):
    """
    Dataset for lunar terrain patches.

    Extracts spatial patches from the full feature tensor for training
    or inference.
    """

    def __init__(
        self,
        feature_tensor: npt.NDArray[np.float32],
        patch_size: int = 64,
        stride: int = 32,
        valid_mask: Optional[npt.NDArray[np.bool_]] = None,
        augment: bool = False
    ):
        """
        Initialize the dataset.

        Args:
            feature_tensor: Input feature array (height, width, channels)
            patch_size: Size of square patches to extract
            stride: Stride for patch extraction (non-overlapping if equal to patch_size)
            valid_mask: Optional mask of valid pixels
            augment: Whether to apply data augmentation
        """
        self.feature_tensor = feature_tensor
        self.patch_size = patch_size
        self.stride = stride
        self.augment = augment

        # Calculate valid patch positions
        self.height, self.width, self.channels = feature_tensor.shape

        # Generate patch coordinates
        self.patch_coords = self._generate_patch_coords(valid_mask)

        logger.info(
            f"LunarPatchDataset: {len(self.patch_coords)} patches from "
            f"{self.height}x{self.width} input, patch_size={patch_size}"
        )

    def _generate_patch_coords(
        self,
        valid_mask: Optional[npt.NDArray[np.bool_]]
    ) -> List[Tuple[int, int]]:
        """Generate list of (row, col) coordinates for patches."""
        coords = []

        for row in range(0, self.height - self.patch_size + 1, self.stride):
            for col in range(0, self.width - self.patch_size + 1, self.stride):
                if valid_mask is not None:
                    # Check if patch has sufficient valid pixels
                    patch_mask = valid_mask[
                        row:row + self.patch_size,
                        col:col + self.patch_size
                    ]
                    valid_ratio = patch_mask.sum() / (self.patch_size * self.patch_size)

                    if valid_ratio < 0.5:  # Require at least 50% valid
                        continue

                coords.append((row, col))

        return coords

    def __len__(self) -> int:
        return len(self.patch_coords)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Get a single patch.

        Returns:
            Tuple of (patch_tensor, (row, col))
        """
        row, col = self.patch_coords[idx]

        patch = self.feature_tensor[
            row:row + self.patch_size,
            col:col + self.patch_size,
            :
        ]

        # Convert to torch tensor (channels first for Conv2d)
        patch = torch.from_numpy(patch).permute(2, 0, 1).float()

        # Simple augmentation (rotation and flip)
        if self.augment:
            k = np.random.randint(0, 4)
            patch = torch.rot90(patch, k, dims=[1, 2])

            if np.random.random() > 0.5:
                patch = torch.flip(patch, dims=[2])

        return patch, (row, col)


def create_vae_model(
    input_channels: int = 7,
    patch_size: int = 64,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Tuple[ConvVAE, VAELoss]:
    """
    Create and initialize the VAE model.

    Args:
        input_channels: Number of input channels
        patch_size: Input patch size (determines architecture)
        device: Device to place model on

    Returns:
        Tuple of (model, loss_fn)
    """
    # Adjust hidden channels based on patch size
    if patch_size <= 32:
        hidden_channels = [32, 64, 128]
    else:
        hidden_channels = [32, 64, 128, 256]

    model = ConvVAE(
        input_channels=input_channels,
        hidden_channels=hidden_channels,
        latent_dim=128,
        dropout=0.2
    )

    loss_fn = VAELoss(
        reconstruction_weight=1.0,
        kl_weight=0.1
    )

    model = model.to(device)
    logger.info(f"VAE model created on device: {device}")

    return model, loss_fn


def train_epoch(
    model: ConvVAE,
    loss_fn: VAELoss,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    clip_grad: float = 1.0
) -> Tuple[float, float, float]:
    """
    Train for one epoch.

    Args:
        model: VAE model
        loss_fn: Loss function
        dataloader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        clip_grad: Gradient clipping value

    Returns:
        Tuple of (total_loss, recon_loss, kl_loss)
    """
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0

    for batch_idx, (patches, _) in enumerate(dataloader):
        patches = patches.to(device)

        optimizer.zero_grad()

        reconstruction, mu, logvar = model(patches)
        loss, recon_loss, kl_loss = loss_fn(reconstruction, patches, mu, logvar)

        loss.backward()

        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()

    num_batches = len(dataloader)
    return (
        total_loss / num_batches,
        total_recon / num_batches,
        total_kl / num_batches
    )


def evaluate(
    model: ConvVAE,
    dataloader: DataLoader,
    loss_fn: VAELoss,
    device: str
) -> Tuple[float, float, float]:
    """
    Evaluate the model.

    Args:
        model: VAE model
        dataloader: Evaluation data loader
        loss_fn: Loss function
        device: Device

    Returns:
        Tuple of (total_loss, recon_loss, kl_loss)
    """
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0

    with torch.no_grad():
        for patches, _ in dataloader:
            patches = patches.to(device)

            reconstruction, mu, logvar = model(patches)
            loss, recon_loss, kl_loss = loss_fn(reconstruction, patches, mu, logvar)

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()

    num_batches = len(dataloader)
    return (
        total_loss / num_batches,
        total_recon / num_batches,
        total_kl / num_batches
    )
