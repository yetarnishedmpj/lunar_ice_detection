"""
Data Augmentation Module for Lunar Ice Detection
Advanced augmentation strategies for improving model robustness
"""

import torch
import numpy as np
import numpy.typing as npt
from typing import Tuple, Optional, List, Callable
import random
import logging

logger = logging.getLogger(__name__)


class LunarAugmentation:
    """
    Domain-specific augmentation for lunar terrain data.

    Applies physically plausible transformations that preserve
    ice detection labels.
    """

    def __init__(
        self,
        rotation: bool = True,
        flip: bool = True,
        scale: bool = False,
        noise: bool = True,
        elastic: bool = False,
        mixup: bool = False,
        cutout: bool = False
    ):
        """
        Initialize augmentation pipeline.

        Args:
            rotation: Enable random rotation (0, 90, 180, 270)
            flip: Enable horizontal/vertical flips
            scale: Enable random scaling
            noise: Enable Gaussian noise
            elastic: Enable elastic deformation
            mixup: Enable mixup between samples
            cutout: Enable random cutout
        """
        self.rotation = rotation
        self.flip = flip
        self.scale = scale
        self.noise = noise
        self.elastic = elastic
        self.mixup = mixup
        self.cutout = cutout

    def __call__(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply augmentation to input tensor.

        Args:
            x: Input tensor (C, H, W)

        Returns:
            Augmented tensor
        """
        if self.rotation:
            x = self.random_rotation(x)

        if self.flip:
            x = self.random_flip(x)

        if self.scale:
            x = self.random_scale(x)

        if self.noise:
            x = self.add_noise(x)

        if self.elastic:
            x = self.elastic_deformation(x)

        if self.cutout:
            x = self.cutout(x)

        return x

    def random_rotation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random 90-degree rotation."""
        k = random.randint(0, 3)
        if k > 0:
            x = torch.rot90(x, k, dims=[1, 2])
        return x

    def random_flip(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random flip."""
        if random.random() > 0.5:
            x = torch.flip(x, dims=[2])  # Horizontal

        if random.random() > 0.5:
            x = torch.flip(x, dims=[1])  # Vertical

        return x

    def random_scale(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random scale variation (for elevation/roughness)."""
        # Only scale certain channels (elevation, roughness)
        # Temperature and radar should remain similar
        channels_to_scale = [0, 1, 2]  # elevation, slope, roughness

        scale_factor = random.uniform(0.95, 1.05)

        x_scaled = x.clone()
        for c in channels_to_scale:
            x_scaled[c] = x[c] * scale_factor

        return x_scaled

    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise."""
        # Different noise levels for different channels
        noise_levels = {
            0: 0.01,   # elevation
            1: 0.02,   # slope
            2: 0.02,   # roughness
            3: 0.005,  # temp_max
            4: 0.005,  # temp_min
            5: 0.01,   # neutron
            6: 0.01    # radar
        }

        for c, level in noise_levels.items():
            if c < x.shape[0]:
                noise = torch.randn_like(x[c]) * level
                x[c] = x[c] + noise

        return x

    def elastic_deformation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply elastic deformation."""
        # Simplified elastic deformation
        C, H, W = x.shape

        # Create displacement field
        alpha = H * 0.1  # Deformation magnitude
        sigma = H * 0.05  # Smoothing

        # This is a simplified version - in practice would use scipy.ndimage
        # For now, just skip if scipy not available
        return x

    def cutout(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random cutout."""
        C, H, W = x.shape

        # Random cutout region
        if random.random() > 0.5:
            cut_size = random.randint(H // 8, H // 4)
            y = random.randint(0, H - cut_size)
            x_ = random.randint(0, W - cut_size)

            # Zero out region
            x[:, y:y+cut_size, x_:x_+cut_size] = 0

        return x

    def mixup(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        alpha: float = 0.2
    ) -> torch.Tensor:
        """
        Apply mixup between two samples.

        Args:
            x1: First sample
            x2: Second sample
            alpha: Mixup parameter

        Returns:
            Mixed sample
        """
        lam = np.random.beta(alpha, alpha)
        return lam * x1 + (1 - lam) * x2


class PhysicsPreservingAugmentation:
    """
    Augmentation that preserves physical constraints.

    Ensures augmented data still represents physically plausible
    lunar terrain.
    """

    def __init__(self):
        self.aug = LunarAugmentation(
            rotation=True,
            flip=True,
            scale=True,
            noise=True,
            cutout=True
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply physics-preserving augmentation."""
        x = self.aug(x)

        # Apply physical constraints
        x = self.apply_physics_constraints(x)

        return x

    def apply_physics_constraints(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure physical constraints are maintained."""
        C, H, W = x.shape

        # Temperature: must be in valid range (50-400K)
        if x.shape[0] > 4:
            x[3] = torch.clamp(x[3], 50, 400)  # temp_max
            x[4] = torch.clamp(x[4], 20, 400)  # temp_min

        # Neutron suppression: must be in valid range (0-1.5)
        if x.shape[0] > 5:
            x[5] = torch.clamp(x[5], 0, 1.5)

        # Radar CPR: must be in valid range (0-2)
        if x.shape[0] > 6:
            x[6] = torch.clamp(x[6], 0, 2)

        # Temperature relation: min <= max
        if x.shape[0] > 4:
            x[4] = torch.minimum(x[4], x[3])

        return x


class CutMixAugmentation:
    """
    CutMix augmentation for improved generalization.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def __call__(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """
        Apply CutMix.

        Args:
            x1: First sample (C, H, W)
            x2: Second sample (C, H, W)

        Returns:
            Tuple of (mixed sample, lambda)
        """
        C, H, W = x1.shape

        # Sample lambda
        lam = np.random.beta(self.alpha, self.alpha)

        # Get bounding box
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # Random center
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        # Bounding box
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        # Apply cutmix
        x_mixed = x1.clone()
        x_mixed[:, bby1:bby2, bbx1:bbx2] = x2[:, bby1:bby2, bbx1:bbx2]

        # Adjust lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

        return x_mixed, lam


class SpectralAugmentation:
    """
    Augmentation in spectral domain.

    Randomly mixes or drops spectral bands.
    """

    def __init__(self, band_dropout: float = 0.1, band_mix: float = 0.1):
        """
        Initialize spectral augmentation.

        Args:
            band_dropout: Probability of dropping a band
            band_mix: Probability of mixing bands
        """
        self.band_dropout = band_dropout
        self.band_mix = band_mix

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spectral augmentation."""
        C = x.shape[0]

        # Random band dropout
        if random.random() < self.band_dropout:
            band_to_drop = random.randint(0, C - 1)
            x[band_to_drop] = 0

        # Random band mixing
        if random.random() < self.band_mix:
            band1 = random.randint(0, C - 1)
            band2 = random.randint(0, C - 1)
            if band1 != band2:
                # Swap or mix
                mix_factor = random.uniform(0.1, 0.3)
                x[band1] = x[band1] * (1 - mix_factor) + x[band2] * mix_factor

        return x


class Compose:
    """Compose multiple augmentations."""

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            x = t(x)
        return x


def get_standard_augmentation() -> Compose:
    """
    Get standard augmentation pipeline.

    Returns:
        Composed augmentation transforms
    """
    return Compose([
        LunarAugmentation(rotation=True, flip=True, noise=True, cutout=True),
        PhysicsPreservingAugmentation(),
        SpectralAugmentation()
    ])


def get_light_augmentation() -> Compose:
    """Get light augmentation for inference-time augmentation."""
    return Compose([
        LunarAugmentation(rotation=True, flip=True)
    ])


def augment_batch(
    batch: torch.Tensor,
    num_augmented: int = 8,
    augmentation: Optional[Compose] = None
) -> List[torch.Tensor]:
    """
    Create multiple augmented versions of a batch.

    Args:
        batch: Input batch (B, C, H, W)
        num_augmented: Number of augmented versions
        augmentation: Augmentation pipeline

    Returns:
        List of augmented batches
    """
    if augmentation is None:
        augmentation = get_light_augmentation()

    augmented = []

    for i in range(batch.shape[0]):
        x = batch[i]
        versions = [x]

        for _ in range(num_augmented):
            versions.append(augmentation(x.clone()))

        augmented.extend(versions)

    return augmented


def tta_predict(
    model: torch.nn.Module,
    batch: torch.Tensor,
    device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Test-time augmentation for improved predictions.

    Args:
        model: VAE model
        batch: Input batch (B, C, H, W)
        device: Device

    Returns:
        Tuple of (mean_prediction, std_prediction)
    """
    model.eval()

    augmentation = get_light_augmentation()

    predictions = []

    with torch.no_grad():
        # Original
        pred = model.get_anomaly_score(batch.to(device))[0]
        predictions.append(pred)

        # Augmented versions
        for _ in range(4):
            aug_batch = torch.stack([augmentation(x) for x in batch])
            pred = model.get_anomaly_score(aug_batch.to(device))[0]
            predictions.append(pred)

    predictions = torch.stack(predictions, dim=0)
    mean_pred = predictions.mean(dim=0)
    std_pred = predictions.std(dim=0)

    return mean_pred.cpu(), std_pred.cpu()
