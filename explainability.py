"""
Feature Importance and Explainability Module for Lunar Ice Detection
Provides attribution methods for understanding model predictions
"""

import torch
import torch.nn as nn
import numpy as np
import numpy.typing as npt
from typing import Tuple, Optional, Dict, List, Callable
import logging

logger = logging.getLogger(__name__)


class GradientAttribution:
    """
    Gradient-based attribution for model explanations.

    Computes input gradients to understand feature importance.
    """

    def __init__(self, model: nn.Module):
        """
        Initialize with a model.

        Args:
            model: Trained VAE model
        """
        self.model = model
        self.model.eval()

    def compute_gradients(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute gradients of output with respect to input.

        Args:
            x: Input tensor (requires_grad=True)
            target: Target for gradient computation (uses reconstruction error if None)

        Returns:
            Gradient tensor of same shape as input
        """
        x = x.requires_grad_(True)

        self.model.eval()
        reconstruction, _, _ = self.model(x)

        if target is None:
            # Use reconstruction error as target
            target = ((x - reconstruction) ** 2).mean()

        # Backward
        target.backward()

        gradients = x.grad

        return gradients

    def integrated_gradients(
        self,
        x: torch.Tensor,
        baseline: Optional[torch.Tensor] = None,
        steps: int = 50
    ) -> torch.Tensor:
        """
        Compute Integrated Gradients.

        Args:
            x: Input tensor
            baseline: Baseline input (zeros if None)
            steps: Number of interpolation steps

        Returns:
            Integrated gradients
        """
        if baseline is None:
            baseline = torch.zeros_like(x)

        # Interpolate between baseline and input
        alphas = torch.linspace(0, 1, steps).to(x.device)
        gradients_list = []

        for alpha in alphas:
            interpolated = alpha * x + (1 - alpha) * baseline
            interpolated = interpolated.requires_grad_(True)

            reconstruction, _, _ = self.model(interpolated)
            target = ((interpolated - reconstruction) ** 2).mean()

            target.backward()
            gradients_list.append(interpolated.grad.detach())
            interpolated.grad.zero_()

        # Average gradients
        integrated_grad = torch.stack(gradients_list).mean(dim=0)

        # Scale by input difference
        integrated_grad = integrated_grad * (x - baseline)

        return integrated_grad

    def saliency_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute simple saliency map.

        Args:
            x: Input tensor

        Returns:
            Saliency map (absolute gradients)
        """
        gradients = self.compute_gradients(x)
        saliency = torch.abs(gradients)

        # Average across channels
        saliency = saliency.mean(dim=1, keepdim=True)

        return saliency


class PerturbationAttribution:
    """
    Perturbation-based attribution.

    Measures impact of masking input regions on output.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()

    def occlusion_sensitivity(
        self,
        x: torch.Tensor,
        patch_size: int = 8,
        stride: int = 4
    ) -> npt.NDArray[np.float32]:
        """
        Compute occlusion sensitivity map.

        Args:
            x: Input tensor (C, H, W)
            patch_size: Size of occlusion patch
            stride: Stride for sliding window

        Returns:
            Sensitivity map (H, W)
        """
        C, H, W = x.shape

        # Get baseline prediction
        with torch.no_grad():
            _, _, _ = self.model(x.unsqueeze(0))
            baseline_error = ((x.unsqueeze(0) - self.model(x.unsqueeze(0))[0]) ** 2).mean().item()

        sensitivity = np.zeros((H, W), dtype=np.float32)

        # Slide occlusion window
        for row in range(0, H - patch_size + 1, stride):
            for col in range(0, W - patch_size + 1, stride):
                # Create perturbed input
                perturbed = x.clone()
                perturbed[:, row:row+patch_size, col:col+patch_size] = 0

                # Get perturbed prediction
                with torch.no_grad():
                    recon, _, _ = self.model(perturbed.unsqueeze(0))
                    perturbed_error = ((perturbed.unsqueeze(0) - recon) ** 2).mean().item()

                # Sensitivity is difference in error
                sensitivity[row:row+patch_size, col:col+patch_size] = (
                    perturbed_error - baseline_error
                )

        return sensitivity

    def feature_ablation(
        self,
        x: torch.Tensor,
        feature_idx: int,
        ablation_value: float = 0.0
    ) -> float:
        """
        Ablate a single feature channel and measure impact.

        Args:
            x: Input tensor (C, H, W)
            feature_idx: Index of feature to ablate
            ablation_value: Value to replace feature with

        Returns:
            Change in reconstruction error
        """
        with torch.no_grad():
            # Original prediction
            recon_orig, _, _ = self.model(x.unsqueeze(0))
            orig_error = ((x.unsqueeze(0) - recon_orig) ** 2).mean().item()

            # Ablated prediction
            x_ablated = x.clone()
            x_ablated[feature_idx] = ablation_value

            recon_ablated, _, _ = self.model(x_ablated.unsqueeze(0))
            ablated_error = ((x_ablated.unsqueeze(0) - recon_ablated) ** 2).mean().item()

        return ablated_error - orig_error


class ChannelAttribution:
    """
    Analyze importance of input channels/bands.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()

    def compute_channel_importance(
        self,
        x: torch.Tensor,
        num_samples: int = 100
    ) -> Dict[str, npt.NDArray[np.float32]]:
        """
        Compute importance of each input channel.

        Args:
            x: Input tensor (C, H, W) or (B, C, H, W)
            num_samples: Number of random samples

        Returns:
            Dictionary with importance scores
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)

        B, C, H, W = x.shape

        importance = np.zeros(C, dtype=np.float32)

        with torch.no_grad():
            # Original error
            recon, _, _ = self.model(x)
            baseline_error = ((x - recon) ** 2).mean().item()

            # Test each channel
            for c in range(C):
                errors = []
                for _ in range(num_samples):
                    x_test = x.clone()
                    # Random permutation of channel
                    perm_idx = torch.randperm(B)
                    x_test[:, c] = x[perm_idx, c]

                    recon_test, _, _ = self.model(x_test)
                    error = ((x_test - recon_test) ** 2).mean().item()
                    errors.append(error)

                # Importance = increase in error when channel is shuffled
                importance[c] = np.mean(errors) - baseline_error

        channel_names = ['elevation', 'slope', 'roughness', 'temp_max',
                        'temp_min', 'neutron_suppression', 'radar_cpr'][:C]

        return {
            'importance': importance,
            'baseline_error': baseline_error,
            'channel_names': channel_names
        }

    def compute_pairwise_interaction(
        self,
        x: torch.Tensor,
        channel_pairs: Optional[List[Tuple[int, int]]] = None
    ) -> npt.NDArray[np.float32]:
        """
        Compute pairwise channel interactions.

        Args:
            x: Input tensor
            channel_pairs: List of (i, j) pairs to test

        Returns:
            Interaction matrix
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)

        B, C, H, W = x.shape

        if channel_pairs is None:
            channel_pairs = [(i, j) for i in range(C) for j in range(i+1, C)]

        interactions = np.zeros((C, C), dtype=np.float32)

        with torch.no_grad():
            # Baseline
            recon, _, _ = self.model(x)
            baseline = ((x - recon) ** 2).mean().item()

            for i, j in channel_pairs:
                # Ablate both channels
                x_test = x.clone()
                x_test[:, i] = 0
                x_test[:, j] = 0

                recon_test, _, _ = self.model(x_test)
                error = ((x_test - recon_test) ** 2).mean().item()

                interactions[i, j] = error - baseline
                interactions[j, i] = interactions[i, j]

        return interactions


class LimeAttribution:
    """
    LIME (Local Interpretable Model-agnostic Explanations) simplified implementation.
    """

    def __init__(self, model: nn.Module, num_samples: int = 100):
        self.model = model
        self.model.eval()
        self.num_samples = num_samples

    def explain(
        self,
        x: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[npt.NDArray[np.float32], float]:
        """
        Generate LIME-style explanation.

        Args:
            x: Input tensor (C, H, W)
            target_class: Not used for VAE (uses reconstruction error)

        Returns:
            Tuple of (explanation_map, local_model_weight)
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)

        B, C, H, W = x.shape

        # Flatten spatial dimensions for perturbation
        x_flat = x.view(B, C, -1)  # (B, C, H*W)
        num_features = x_flat.shape[2]

        # Get baseline prediction
        with torch.no_grad():
            recon, _, _ = self.model(x)
            baseline = ((x - recon) ** 2).mean()

        # Sample perturbations
        perturbations = []
        predictions = []

        for _ in range(self.num_samples):
            # Binary mask for perturbation
            mask = (torch.rand(B, num_features) > 0.5).float()

            # Apply perturbation
            x_pert = x_flat * mask.view(B, 1, -1).expand(-1, C, -1)
            x_pert = x_pert.view(B, C, H, W)

            with torch.no_grad():
                recon_p, _, _ = self.model(x_pert)
                pred = ((x_pert - recon_p) ** 2).mean()

            perturbations.append(mask[0].cpu().numpy())
            predictions.append(pred.item())

        # Fit simple linear model
        perturbations = np.array(perturbations)
        predictions = np.array(predictions)

        # Simple weighted linear regression
        weights = np.exp(-np.abs(predictions - baseline.item()))
        weights = weights / weights.sum()

        # Feature importance
        importance = np.average(perturbations.T * weights, axis=1)
        importance = importance.reshape(H, W)

        # Normalize
        importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)

        return importance.astype(np.float32), float(baseline.item())


def create_explanations(
    model: nn.Module,
    x: torch.Tensor,
    methods: List[str] = ["gradients", "saliency", "channel"]
) -> Dict[str, npt.NDArray[np.float32]]:
    """
    Create multiple types of explanations for a single input.

    Args:
        model: Trained VAE model
        x: Input tensor (C, H, W)
        methods: List of explanation methods to use

    Returns:
        Dictionary of explanation maps
    """
    explanations = {}

    if "gradients" in methods or "saliency" in methods:
        grad_attr = GradientAttribution(model)

        if x.dim() == 3:
            x = x.unsqueeze(0)

        if "gradients" in methods:
            explanations["gradients"] = grad_attr.compute_gradients(x).cpu().numpy()[0]

        if "saliency" in methods:
            explanations["saliency"] = grad_attr.saliency_map(x).cpu().numpy()[0, 0]

    if "channel" in methods:
        if x.dim() == 3:
            x = x.unsqueeze(0)

        channel_attr = ChannelAttribution(model)
        channel_results = channel_attr.compute_channel_importance(x[0])

        # Store as per-channel importance
        explanations["channel_importance"] = channel_results['importance']
        explanations["channel_names"] = channel_results['channel_names']

    return explanations


def visualize_explanations(
    explanations: Dict[str, npt.NDArray[np.float32]],
    band_names: Optional[List[str]] = None
) -> 'matplotlib.figure.Figure':
    """
    Visualize explanation maps.

    Args:
        explanations: Dictionary of explanation maps
        band_names: Names of input bands

    Returns:
        Matplotlib figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available")
        return None

    n_plots = len(explanations)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    axes = axes.flatten()

    for i, (name, data) in enumerate(explanations.items()):
        if data.ndim == 1 and name == "channel_importance":
            # Bar chart for channel importance
            if band_names is None:
                band_names = [f"Channel {i}" for i in range(len(data))]
            axes[i].barh(band_names, data)
            axes[i].set_xlabel("Importance")
            axes[i].set_title(name)
        else:
            # Image for spatial maps
            im = axes[i].imshow(data, cmap='hot', origin='lower')
            axes[i].set_title(name)
            plt.colorbar(im, ax=axes[i])

    # Hide unused axes
    for i in range(len(explanations), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    return fig


def compute_detection_confidence(
    ice_probability: npt.NDArray[np.float32],
    uncertainty: Optional[npt.NDArray[np.float32]] = None,
    threshold_high: float = 0.7,
    threshold_low: float = 0.3
) -> Dict[str, npt.NDArray[np.float32]]:
    """
    Compute detection confidence categories.

    Args:
        ice_probability: Ice probability map
        uncertainty: Uncertainty map (optional)
        threshold_high: High confidence threshold
        threshold_low: Low confidence threshold

    Returns:
        Dictionary with confidence categories
    """
    confidence = np.zeros_like(ice_probability)

    # High confidence detection
    high_conf = ice_probability >= threshold_high

    # Low confidence / no detection
    low_conf = ice_probability < threshold_low

    # Medium confidence
    medium_conf = ~high_conf & ~low_conf

    confidence[high_conf] = 2  # High
    confidence[medium_conf] = 1  # Medium
    confidence[low_conf] = 0  # Low/None

    result = {
        'confidence': confidence,
        'high_confidence_pixels': int(high_conf.sum()),
        'medium_confidence_pixels': int(medium_conf.sum()),
        'low_confidence_pixels': int(low_conf.sum())
    }

    if uncertainty is not None:
        # Compute confidence-adjusted probability
        # High uncertainty reduces confidence
        uncertainty_factor = 1 - np.clip(uncertainty * 2, 0, 1)
        adjusted_prob = ice_probability * uncertainty_factor

        result['adjusted_probability'] = adjusted_prob
        result['uncertainty_weight'] = uncertainty_factor

    return result
