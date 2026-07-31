"""
Training Visualization Module for Lunar Ice Detection
Provides plotting functions for training history and model metrics
"""

import numpy as np
import numpy.typing as npt
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def plot_training_history(
    history: Dict[str, List[float]],
    title: str = "Training History"
) -> 'matplotlib.figure.Figure':
    """
    Plot training and validation loss curves.

    Args:
        history: Dictionary with 'train_loss', 'val_loss', etc.
        title: Plot title

    Returns:
        Matplotlib figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available")
        return None

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    epochs = range(1, len(history['train_loss']) + 1)

    # Total loss
    axes[0].plot(epochs, history['train_loss'], label='Train', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Total Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Reconstruction loss
    axes[1].plot(epochs, history['train_recon'], label='Train', linewidth=2)
    axes[1].plot(epochs, history['val_recon'], label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Reconstruction Loss')
    axes[1].set_title('Reconstruction Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # KL divergence
    axes[2].plot(epochs, history['train_kl'], label='Train', linewidth=2)
    axes[2].plot(epochs, history['val_kl'], label='Validation', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('KL Divergence')
    axes[2].set_title('KL Divergence')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    return fig


def plot_loss_components(
    history: Dict[str, List[float]]
) -> 'matplotlib.figure.Figure':
    """
    Plot detailed loss component analysis.

    Args:
        history: Training history dictionary

    Returns:
        Matplotlib figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    epochs = range(1, len(history['train_loss']) + 1)

    # Training losses
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Total', linewidth=2)
    axes[0, 0].plot(epochs, history['train_recon'], 'g--', label='Reconstruction', linewidth=1.5)
    axes[0, 0].plot(epochs, history['train_kl'], 'r:', label='KL', linewidth=1.5)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss Components')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')

    # Validation losses
    axes[0, 1].plot(epochs, history['val_loss'], 'b-', label='Total', linewidth=2)
    axes[0, 1].plot(epochs, history['val_recon'], 'g--', label='Reconstruction', linewidth=1.5)
    axes[0, 1].plot(epochs, history['val_kl'], 'r:', label='KL', linewidth=1.5)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Validation Loss Components')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')

    # Loss ratio (KL / Reconstruction)
    train_ratio = np.array(history['train_kl']) / (np.array(history['train_recon']) + 1e-8)
    val_ratio = np.array(history['val_kl']) / (np.array(history['val_recon']) + 1e-8)

    axes[1, 0].plot(epochs, train_ratio, 'b-', label='Train', linewidth=2)
    axes[1, 0].plot(epochs, val_ratio, 'orange', label='Validation', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('KL / Reconstruction Ratio')
    axes[1, 0].set_title('Loss Balance')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Loss improvement rate
    train_improvement = -np.diff(history['train_loss']) / (np.array(history['train_loss'][:-1]) + 1e-8) * 100
    val_improvement = -np.diff(history['val_loss']) / (np.array(history['val_loss'][:-1]) + 1e-8) * 100

    axes[1, 1].bar(np.arange(len(train_improvement)) - 0.2, train_improvement, width=0.4, label='Train', alpha=0.7)
    axes[1, 1].bar(np.arange(len(val_improvement)) + 0.2, val_improvement, width=0.4, label='Validation', alpha=0.7)
    axes[1, 1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Improvement (%)')
    axes[1, 1].set_title('Epoch-to-Epoch Improvement')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_anomaly_distribution(
    anomaly_scores: npt.NDArray[np.float32],
    threshold: Optional[float] = None,
    title: str = "Anomaly Score Distribution"
) -> 'matplotlib.figure.Figure':
    """
    Plot distribution of anomaly scores.

    Args:
        anomaly_scores: Flattened anomaly scores
        threshold: Optional threshold for anomaly detection
        title: Plot title

    Returns:
        Matplotlib figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram
    axes[0].hist(anomaly_scores.flatten(), bins=100, edgecolor='black', alpha=0.7)
    if threshold is not None:
        axes[0].axvline(x=threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.4f}')
    axes[0].set_xlabel('Anomaly Score')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Anomaly Score Histogram')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # CDF
    sorted_scores = np.sort(anomaly_scores.flatten())
    cdf = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
    axes[1].plot(sorted_scores, cdf, linewidth=2)
    if threshold is not None:
        idx = np.searchsorted(sorted_scores, threshold)
        axes[1].axvline(x=threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.4f}')
        if idx < len(sorted_scores):
            axes[1].axhline(y=cdf[idx], color='r', linestyle=':', alpha=0.5)
    axes[1].set_xlabel('Anomaly Score')
    axes[1].set_ylabel('Cumulative Probability')
    axes[1].set_title('Cumulative Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    return fig


def plot_indicator_correlations(
    indicators: Dict[str, npt.NDArray[np.float32]],
    title: str = "Indicator Correlations"
) -> 'matplotlib.figure.Figure':
    """
    Plot correlation matrix between indicators.

    Args:
        indicators: Dictionary of indicator arrays
        title: Plot title

    Returns:
        Matplotlib figure
    """
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        return None

    # Flatten and stack arrays
    valid_keys = [k for k in indicators.keys() if indicators[k] is not None]
    data = {}
    for key in valid_keys:
        flat = indicators[key].flatten()
        # Remove NaN/inf
        valid = flat[np.isfinite(flat)]
        if len(valid) > 0:
            data[key] = valid

    if not data:
        return None

    # Create DataFrame for correlation
    min_len = min(len(v) for v in data.values())
    df_data = {k: v[:min_len] for k, v in data.items()}
    df = pd.DataFrame(df_data)

    # Compute correlation
    corr = df.corr()

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right')
    ax.set_yticklabels(corr.columns)

    # Add correlation values
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            text = ax.text(j, i, f'{corr.iloc[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=9)

    plt.colorbar(im, ax=ax, label='Correlation')
    ax.set_title(title, fontsize=14)

    plt.tight_layout()
    return fig


def plot_ice_probability_vs_indicators(
    ice_probability: npt.NDArray[np.float32],
    indicators: Dict[str, npt.NDArray[np.float32]],
    title: str = "Ice Probability vs Indicators"
) -> 'matplotlib.figure.Figure':
    """
    Plot ice probability vs each indicator.

    Args:
        ice_probability: Ice probability map
        indicators: Dictionary of indicator maps
        title: Plot title

    Returns:
        Matplotlib figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    # Sample for scatter plot (too many points otherwise)
    n_samples = min(10000, ice_probability.size)
    flat_prob = ice_probability.flatten()
    idx = np.random.choice(len(flat_prob), n_samples, replace=False)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    indicator_names = [
        ('temperature_indicator', 'Temperature'),
        ('neutron_indicator', 'Neutron Suppression'),
        ('radar_indicator', 'Radar CPR'),
        ('physics_combined', 'Physics Combined'),
        ('reconstruction_error', 'Reconstruction Error'),
        ('uncertainty', 'Uncertainty')
    ]

    for i, (key, name) in enumerate(indicator_names):
        if key in indicators and indicators[key] is not None:
            flat_ind = indicators[key].flatten()[idx]
            flat_p = flat_prob[idx]

            # Remove invalid points
            valid = np.isfinite(flat_ind) & np.isfinite(flat_p)
            axes[i].scatter(flat_ind[valid], flat_p[valid], alpha=0.1, s=1)
            axes[i].set_xlabel(name)
            axes[i].set_ylabel('Ice Probability')
            axes[i].set_title(f'vs {name}')
            axes[i].grid(True, alpha=0.3)

            # Add correlation
            corr = np.corrcoef(flat_ind[valid], flat_p[valid])[0, 1]
            axes[i].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[i].transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    return fig


def plot_spatial_heatmaps(
    ice_probability: npt.NDArray[np.float32],
    indicators: Dict[str, npt.NDArray[np.float32]]
) -> 'matplotlib.figure.Figure':
    """
    Create heatmap grid of all spatial outputs.

    Args:
        ice_probability: Ice probability map
        indicators: Dictionary of indicator maps

    Returns:
        Matplotlib figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    # Collect all maps
    maps = {'Ice Probability': ice_probability}
    maps.update({k: v for k, v in indicators.items() if v is not None})

    n_maps = len(maps)
    n_cols = 3
    n_rows = (n_maps + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for i, (name, data) in enumerate(maps.items()):
        row = i // n_cols
        col = i % n_cols

        # Determine colormap and range
        if 'probability' in name.lower() or 'indicator' in name.lower():
            cmap = 'YlOrRd'
            vmin, vmax = 0, 1
        elif 'error' in name.lower() or 'uncertainty' in name.lower():
            cmap = 'magma'
            vmin, vmax = None, None
        else:
            cmap = 'viridis'
            vmin, vmax = None, None

        im = axes[row, col].imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
        axes[row, col].set_title(name, fontsize=11)
        axes[row, col].axis('off')
        plt.colorbar(im, ax=axes[row, col], shrink=0.6)

    # Hide unused axes
    for i in range(n_maps, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')

    plt.tight_layout()
    return fig


def compute_training_metrics(
    history: Dict[str, List[float]]
) -> Dict[str, float]:
    """
    Compute summary metrics from training history.

    Args:
        history: Training history dictionary

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Best losses
    metrics['best_train_loss'] = float(min(history['train_loss']))
    metrics['best_val_loss'] = float(min(history['val_loss']))
    metrics['best_epoch'] = int(np.argmin(history['val_loss'])) + 1

    # Final losses
    metrics['final_train_loss'] = float(history['train_loss'][-1])
    metrics['final_val_loss'] = float(history['val_loss'][-1])

    # Loss improvement
    metrics['train_improvement'] = (
        (history['train_loss'][0] - history['train_loss'][-1]) /
        history['train_loss'][0] * 100
    )
    metrics['val_improvement'] = (
        (history['val_loss'][0] - history['val_loss'][-1]) /
        history['val_loss'][0] * 100
    )

    # Convergence metrics
    # Check if validation loss is increasing (overfitting)
    last_10_val = history['val_loss'][-10:]
    metrics['val_trend'] = 'increasing' if last_10_val[-1] > last_10_val[0] else 'decreasing'

    # KL divergence balance
    metrics['final_kl_ratio'] = history['train_kl'][-1] / (history['train_recon'][-1] + 1e-8)

    return metrics
