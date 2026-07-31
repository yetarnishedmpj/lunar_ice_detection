"""
Ensemble Model Module for Lunar Ice Detection
Combines multiple models for improved predictions
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Dict
import logging
from pathlib import Path
import copy

try:
    from .model import ConvVAE, create_vae_model, VAELoss
except ImportError:
    from model import ConvVAE, create_vae_model, VAELoss

logger = logging.getLogger(__name__)


class EnsembleModel(nn.Module):
    """
    Ensemble of VAE models for improved anomaly detection.

    Combines predictions from multiple models to reduce variance
    and improve robustness.
    """

    def __init__(
        self,
        models: Optional[List[nn.Module]] = None,
        num_models: int = 5,
        input_channels: int = 7,
        patch_size: int = 64,
        device: str = "cuda"
    ):
        """
        Initialize ensemble.

        Args:
            models: List of pre-trained models (if None, creates new ones)
            num_models: Number of models to create if models is None
            input_channels: Number of input channels
            patch_size: Input patch size
            device: Device to use
        """
        super().__init__()

        self.input_channels = input_channels
        self.patch_size = patch_size
        self.device = device

        if models is not None:
            self.models = nn.ModuleList(models)
        else:
            # Create ensemble of VAEs with different initializations
            self.models = nn.ModuleList([
                create_vae_model(
                    input_channels=input_channels,
                    patch_size=patch_size,
                    device=device
                )[0]
                for _ in range(num_models)
            ])

        self.num_models = len(self.models)

        logger.info(f"Ensemble initialized with {self.num_models} models")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through all models.

        Returns mean and variance of reconstructions.
        """
        reconstructions = []
        mus = []
        logvars = []

        for model in self.models:
            model.eval()
            with torch.no_grad():
                recon, mu, logvar = model(x)
                reconstructions.append(recon)
                mus.append(mu)
                logvars.append(logvar)

        # Stack and compute statistics
        reconstructions = torch.stack(reconstructions, dim=0)
        mus = torch.stack(mus, dim=0)
        logvars = torch.stack(logvars, dim=0)

        # Mean across models
        recon_mean = torch.mean(reconstructions, dim=0)
        mu_mean = torch.mean(mus, dim=0)
        logvar_mean = torch.mean(logvars, dim=0)

        return recon_mean, mu_mean, logvar_mean

    def get_anomaly_scores(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get anomaly scores from all models.

        Returns:
            Dictionary with mean, std, and individual model scores
        """
        all_scores = []

        for model in self.models:
            model.eval()
            with torch.no_grad():
                recon, _, _ = model(x)
                error = (x - recon) ** 2
                score = torch.mean(error, dim=1)
                all_scores.append(score)

        scores = torch.stack(all_scores, dim=0)  # (num_models, batch, H, W)

        return {
            'mean': torch.mean(scores, dim=0),
            'std': torch.std(scores, dim=0),
            'min': torch.min(scores, dim=0)[0],
            'max': torch.max(scores, dim=0)[0],
            'individual': all_scores
        }

    def get_uncertainty(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get prediction with uncertainty.

        Returns:
            Tuple of (mean_anomaly, uncertainty_std)
        """
        scores = self.get_anomaly_scores(x)
        return scores['mean'], scores['std']


class BootstrapEnsemble:
    """
    Bootstrap ensemble for improved training.

    Trains multiple models on different bootstrap samples.
    """

    def __init__(
        self,
        num_models: int = 5,
        input_channels: int = 7,
        patch_size: int = 64,
        device: str = "cuda"
    ):
        self.num_models = num_models
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.device = device

        self.models: List[Tuple[nn.Module, VAELoss]] = []

    def train_models(
        self,
        train_loader,
        val_loader,
        num_epochs: int = 50,
        save_dir: Optional[Path] = None
    ) -> List[Dict[str, List[float]]]:
        """
        Train ensemble with different initializations.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Epochs per model
            save_dir: Directory to save models

        Returns:
            List of training histories
        """
        histories = []

        for i in range(self.num_models):
            logger.info(f"Training model {i+1}/{self.num_models}")

            # Create model with different seed
            torch.manual_seed(42 + i)
            np.random.seed(42 + i)

            model, loss_fn = create_vae_model(
                input_channels=self.input_channels,
                patch_size=self.patch_size,
                device=self.device
            )

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

            # Training loop
            history = {
                'train_loss': [], 'train_recon': [], 'train_kl': [],
                'val_loss': [], 'val_recon': [], 'val_kl': []
            }

            for epoch in range(num_epochs):
                # Train
                model.train()
                train_loss, train_recon, train_kl = 0, 0, 0
                for patches, _ in train_loader:
                    patches = patches.to(self.device)
                    optimizer.zero_grad()
                    recon, mu, logvar = model(patches)
                    loss, recon_loss, kl_loss = loss_fn(recon, patches, mu, logvar)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    train_recon += recon_loss.item()
                    train_kl += kl_loss.item()

                train_loss /= len(train_loader)
                train_recon /= len(train_loader)
                train_kl /= len(train_loader)

                # Validate
                model.eval()
                val_loss, val_recon, val_kl = 0, 0, 0
                with torch.no_grad():
                    for patches, _ in val_loader:
                        patches = patches.to(self.device)
                        recon, mu, logvar = model(patches)
                        loss, recon_loss, kl_loss = loss_fn(recon, patches, mu, logvar)
                        val_loss += loss.item()
                        val_recon += recon_loss.item()
                        val_kl += kl_loss.item()

                val_loss /= len(val_loader)
                val_recon /= len(val_loader)
                val_kl /= len(val_loader)

                history['train_loss'].append(train_loss)
                history['train_recon'].append(train_recon)
                history['train_kl'].append(train_kl)
                history['val_loss'].append(val_loss)
                history['val_recon'].append(val_recon)
                history['val_kl'].append(val_kl)

            # Save model
            if save_dir:
                save_path = save_dir / f"ensemble_model_{i}.pth"
                torch.save(model.state_dict(), save_path)
                logger.info(f"Saved model to {save_path}")

            self.models.append((model, loss_fn))
            histories.append(history)

        return histories

    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Make ensemble prediction.

        Args:
            x: Input tensor

        Returns:
            Dictionary with mean prediction and uncertainty
        """
        all_scores = []

        for model, _ in self.models:
            model.eval()
            with torch.no_grad():
                recon, _, _ = model(x)
                error = (x - recon) ** 2
                score = torch.mean(error, dim=1)
                all_scores.append(score)

        scores = torch.stack(all_scores, dim=0)

        return {
            'mean': torch.mean(scores, dim=0),
            'std': torch.std(scores, dim=0),
            'individual': all_scores
        }


class ModelSnapshot:
    """
    Model snapshot ensemble (Snapshot Ensembling).

    Saves model checkpoints during training and combines them.
    """

    def __init__(
        self,
        input_channels: int = 7,
        patch_size: int = 64,
        device: str = "cuda"
    ):
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.device = device
        self.snapshots: List[nn.Module] = []

    def create_model(self):
        """Create a new model instance."""
        model, loss_fn = create_vae_model(
            input_channels=self.input_channels,
            patch_size=self.patch_size,
            device=self.device
        )
        return model, loss_fn

    def save_snapshot(self, model: nn.Module):
        """Save a model snapshot."""
        snapshot = copy.deepcopy(model.state_dict())
        self.snapshots.append(snapshot)
        logger.info(f"Saved snapshot {len(self.snapshots)}")

    def predict_with_snapshots(self, x: torch.Tensor, model: nn.Module) -> Dict[str, torch.Tensor]:
        """
        Make predictions using all snapshots.

        Args:
            x: Input tensor
            model: Model architecture (for forward pass)

        Returns:
            Dictionary with mean and std predictions
        """
        all_scores = []

        original_state = model.state_dict()

        for snapshot in self.snapshots:
            model.load_state_dict(snapshot)
            model.eval()
            with torch.no_grad():
                recon, _, _ = model(x)
                error = (x - recon) ** 2
                score = torch.mean(error, dim=1)
                all_scores.append(score)

        # Restore original state
        model.load_state_dict(original_state)

        scores = torch.stack(all_scores, dim=0)

        return {
            'mean': torch.mean(scores, dim=0),
            'std': torch.std(scores, dim=0),
            'n_snapshots': len(self.snapshots)
        }


def load_ensemble(
    model_paths: List[Path],
    input_channels: int = 7,
    patch_size: int = 64,
    device: str = "cuda"
) -> EnsembleModel:
    """
    Load ensemble from saved model checkpoints.

    Args:
        model_paths: List of paths to model checkpoints
        input_channels: Number of input channels
        patch_size: Input patch size
        device: Device to load models

    Returns:
        Loaded ensemble model
    """
    models = []

    for path in model_paths:
        model, _ = create_vae_model(
            input_channels=input_channels,
            patch_size=patch_size,
            device=device
        )
        model.load_state_dict(torch.load(path, map_location=device))
        models.append(model)

    ensemble = EnsembleModel(models=models, device=device)

    logger.info(f"Loaded ensemble with {len(models)} models")

    return ensemble


def compute_ensemble_metrics(
    predictions: Dict[str, torch.Tensor],
    threshold: Optional[float] = None
) -> Dict[str, float]:
    """
    Compute metrics for ensemble predictions.

    Args:
        predictions: Output from ensemble predict
        threshold: Optional threshold for binary predictions

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    mean = predictions['mean']
    std = predictions['std']

    # Mean statistics
    metrics['mean_score_mean'] = float(mean.mean())
    metrics['mean_score_std'] = float(mean.std())
    metrics['mean_score_max'] = float(mean.max())

    # Uncertainty statistics
    metrics['uncertainty_mean'] = float(std.mean())
    metrics['uncertainty_std'] = float(std.std())
    metrics['uncertainty_max'] = float(std.max())

    # Coefficient of variation (uncertainty / mean)
    with torch.no_grad():
        cv = std / (mean + 1e-8)
        metrics['coefficient_of_variation_mean'] = float(cv.mean())

    # Confidence interval width (95%)
    metrics['ci_95_width'] = float(1.96 * std.mean())

    return metrics
