"""
Feature Normalization Module for Lunar Ice Detection Pipeline
Robust scaling to handle extreme outliers in lunar terrain data
"""

import numpy as np
import numpy.typing as npt
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import logging

try:
    from .data_ingestion import DataStatistics
except ImportError:
    from data_ingestion import DataStatistics

logger = logging.getLogger(__name__)


@dataclass
class NormalizationParams:
    """Parameters for robust scaling normalization."""
    center: float  # Median (robust center)
    scale: float  # IQR / 1.35 (robust scale, similar to std dev)
    clip_range: Optional[Tuple[float, float]] = None


class RobustNormalizer:
    """
    Robust scaling normalizer using median and IQR.

    This normalization method is designed to handle extreme outliers
    common in lunar remote sensing data (crater depths, temperatures).
    """

    def __init__(
        self,
        statistics: Dict[str, DataStatistics],
        clip_percentiles: Tuple[float, float] = (0.5, 99.5)
    ):
        """
        Initialize the robust normalizer.

        Args:
            statistics: Dictionary of DataStatistics per band
            clip_percentiles: Percentiles to clip extreme values at
        """
        self.statistics = statistics
        self.clip_percentiles = clip_percentiles
        self._params: Dict[str, NormalizationParams] = {}

    def _compute_params(self, band_name: str) -> NormalizationParams:
        """Compute normalization parameters for a band."""
        stats = self.statistics[band_name]

        # Robust scale: IQR / 1.35 approximates standard deviation for normal distributions
        # This makes the scale robust to outliers
        scale = stats.iqr / 1.35

        # Avoid division by zero for constant bands
        if scale < 1e-10:
            logger.warning(f"Band '{band_name}' has near-zero IQR ({stats.iqr}), setting scale to 1.0")
            scale = 1.0

        # Compute clip range based on percentiles
        # These will be computed on-the-fly during transform
        clip_range = self.clip_percentiles

        return NormalizationParams(
            center=stats.median,
            scale=scale,
            clip_range=clip_range
        )

    def fit(self, data: npt.NDArray[np.float32], band_names: list[str]) -> None:
        """
        Fit the normalizer (computes parameters from data if not pre-computed).

        Args:
            data: Input data array (height, width, channels)
            band_names: List of band names corresponding to channels
        """
        for i, band_name in enumerate(band_names):
            if band_name not in self._params:
                if band_name in self.statistics:
                    self._params[band_name] = self._compute_params(band_name)
                else:
                    # Compute statistics from data if not provided
                    channel_data = data[:, :, i]
                    valid_mask = ~np.isnan(channel_data)
                    valid_data = channel_data[valid_mask]

                    median = np.median(valid_data)
                    q1 = np.percentile(valid_data, 25)
                    q3 = np.percentile(valid_data, 75)
                    iqr = q3 - q1
                    scale = iqr / 1.35

                    if scale < 1e-10:
                        scale = 1.0

                    self._params[band_name] = NormalizationParams(
                        center=median,
                        scale=scale,
                        clip_range=self.clip_percentiles
                    )

                    logger.info(
                        f"Fitted normalizer for '{band_name}': center={median:.2f}, scale={scale:.2f}"
                    )

    def transform(
        self,
        data: npt.NDArray[np.float32],
        band_names: list[str],
        clip: bool = True
    ) -> npt.NDArray[np.float32]:
        """
        Apply robust scaling normalization.

        Transform: z = (x - center) / scale

        Args:
            data: Input data array (height, width, channels)
            band_names: List of band names corresponding to channels
            clip: Whether to clip values to standard normal range

        Returns:
            Normalized array of same shape
        """
        self.fit(data, band_names)

        normalized = np.zeros_like(data, dtype=np.float32)

        for i, band_name in enumerate(band_names):
            params = self._params[band_name]
            channel_data = data[:, :, i].copy()

            # Apply robust scaling
            if params.scale > 0:
                z = (channel_data - params.center) / params.scale
            else:
                z = channel_data - params.center

            # Optionally clip to prevent extreme outliers
            if clip:
                z = np.clip(z, -3.0, 3.0)

            # Preserve NaN values
            nan_mask = np.isnan(channel_data)
            z[nan_mask] = np.nan

            normalized[:, :, i] = z

        return normalized

    def fit_transform(
        self,
        data: npt.NDArray[np.float32],
        band_names: list[str],
        clip: bool = True
    ) -> npt.NDArray[np.float32]:
        """
        Fit and transform in one step.

        Args:
            data: Input data array
            band_names: Band names
            clip: Whether to clip

        Returns:
            Normalized data
        """
        self.fit(data, band_names)
        return self.transform(data, band_names, clip=clip)

    def inverse_transform(
        self,
        data: npt.NDArray[np.float32],
        band_names: list[str]
    ) -> npt.NDArray[np.float32]:
        """
        Inverse transform normalized data back to original scale.

        Args:
            data: Normalized data
            band_names: Band names

        Returns:
            Data in original scale
        """
        original = np.zeros_like(data, dtype=np.float32)

        for i, band_name in enumerate(band_names):
            params = self._params[band_name]
            channel_data = data[:, :, i]

            # Inverse: x = z * scale + center
            original[:, :, i] = channel_data * params.scale + params.center

            # Preserve NaN
            nan_mask = np.isnan(channel_data)
            original[:, :, i][nan_mask] = np.nan

        return original


class ChannelWiseNormalizer:
    """
    Simplified channel-wise normalizer for quick prototyping.
    Applies per-channel standardization with option for robust scaling.
    """

    def __init__(self, method: str = "robust"):
        """
        Initialize normalizer.

        Args:
            method: Normalization method - "robust" or "standard" (mean/std)
        """
        if method not in ("robust", "standard"):
            raise ValueError("method must be 'robust' or 'standard'")
        self.method = method
        self.means: Optional[npt.NDArray[np.float32]] = None
        self.stds: Optional[npt.NDArray[np.float32]] = None
        self.medians: Optional[npt.NDArray[np.float32]] = None
        self.iqrs: Optional[npt.NDArray[np.float32]] = None

    def fit(self, data: npt.NDArray[np.float32]) -> None:
        """
        Compute normalization parameters from training data.

        Args:
            data: Input array of shape (height, width, channels)
        """
        num_channels = data.shape[2]
        self.means = np.zeros(num_channels, dtype=np.float32)
        self.stds = np.zeros(num_channels, dtype=np.float32)
        self.medians = np.zeros(num_channels, dtype=np.float32)
        self.iqrs = np.zeros(num_channels, dtype=np.float32)

        for i in range(num_channels):
            channel = data[:, :, i]
            valid = channel[~np.isnan(channel)]

            self.means[i] = np.mean(valid)
            self.stds[i] = np.std(valid)
            self.medians[i] = np.median(valid)
            q1 = np.percentile(valid, 25)
            q3 = np.percentile(valid, 75)
            self.iqrs[i] = q3 - q1

        logger.info(
            f"Fitted ChannelWiseNormalizer: {num_channels} channels, "
            f"method={self.method}"
        )

    def transform(self, data: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        Normalize the data.

        Args:
            data: Input array

        Returns:
            Normalized array
        """
        normalized = np.zeros_like(data, dtype=np.float32)

        for i in range(data.shape[2]):
            channel = data[:, :, i]

            if self.method == "standard":
                if self.stds[i] > 1e-10:
                    z = (channel - self.means[i]) / self.stds[i]
                else:
                    z = channel - self.means[i]
            else:  # robust
                scale = self.iqrs[i] / 1.35
                if scale > 1e-10:
                    z = (channel - self.medians[i]) / scale
                else:
                    z = channel - self.medians[i]

            # Clip extreme values
            z = np.clip(z, -3.0, 3.0)

            # Preserve NaN
            z[np.isnan(channel)] = np.nan

            normalized[:, :, i] = z

        return normalized

    def fit_transform(self, data: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Fit and transform in one step."""
        self.fit(data)
        return self.transform(data)
