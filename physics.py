"""
Physics Constraints Module for Lunar Ice Detection
Implements domain knowledge for ice probability estimation
"""

import numpy as np
import numpy.typing as npt
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class PhysicsConstraints:
    """
    Physics-based constraints for lunar ice detection.

    Implements the physical relationships between temperature, neutron suppression,
    and radar brightness to infer ice presence in PSRs.
    """

    # Temperature threshold (K) - above this, ice is unstable
    TEMP_ICE_THRESHOLD = 110.0

    # Temperature range where ice is potentially stable (K)
    TEMP_STABLE_RANGE = (50.0, 110.0)

    # Minimum neutron suppression factor to indicate ice (unitless ratio)
    NEUTRON_SUPPRESSION_THRESHOLD = 0.85

    # Minimum CPR for ice detection (radar brightness)
    CPR_ICE_THRESHOLD = 0.1

    def __init__(
        self,
        temp_max_band: str = "temp_max",
        temp_min_band: str = "temp_min",
        neutron_band: str = "neutron_suppression",
        radar_band: str = "radar_cpr"
    ):
        """
        Initialize physics constraints.

        Args:
            temp_max_band: Name of max temperature band
            temp_min_band: Name of min temperature band
            neutron_band: Name of neutron suppression band
            radar_band: Name of CPR radar band
        """
        self.temp_max_band = temp_max_band
        self.temp_min_band = temp_min_band
        self.neutron_band = neutron_band
        self.radar_band = radar_band

    def compute_ice_probability(
        self,
        feature_tensor: npt.NDArray[np.float32],
        band_names: list[str],
        reconstruction_error: Optional[npt.NDArray[np.float32]] = None
    ) -> Tuple[npt.NDArray[np.float32], Dict[str, npt.NDArray[np.float32]]]:
        """
        Compute ice probability using physics constraints.

        Combines:
        1. Temperature: Ice requires T < 110K
        2. Neutron suppression: Hydrogen absorbs neutrons, reducing count
        3. Radar CPR: Ice creates distinctive radar signature
        4. Reconstruction error: Anomalous regions from VAE

        Args:
            feature_tensor: Input feature tensor (H, W, channels)
            band_names: List of band names
            reconstruction_error: Optional anomaly scores from VAE

        Returns:
            Tuple of (ice_probability_map, indicator_maps)
        """
        height, width, _ = feature_tensor.shape

        # Get band indices
        try:
            temp_max_idx = band_names.index(self.temp_max_band)
            temp_min_idx = band_names.index(self.temp_min_band)
            neutron_idx = band_names.index(self.neutron_band)
            radar_idx = band_names.index(self.radar_band)
        except ValueError as e:
            raise ValueError(f"Required band not found: {e}")

        # Extract physical values
        temp_max = feature_tensor[:, :, temp_max_idx]
        temp_min = feature_tensor[:, :, temp_min_idx]
        neutron_suppression = feature_tensor[:, :, neutron_idx]
        radar_cpr = feature_tensor[:, :, radar_idx]

        # Initialize indicator maps (0-1 range)
        temp_indicator = self._temperature_indicator(temp_max, temp_min)
        neutron_indicator = self._neutron_indicator(neutron_suppression)
        radar_indicator = self._radar_indicator(radar_cpr)

        # Combine physics indicators
        # Ice likely where ALL conditions are favorable
        physics_prob = (
            temp_indicator * 0.4 +
            neutron_indicator * 0.35 +
            radar_indicator * 0.25
        )

        # If reconstruction error available, incorporate it
        if reconstruction_error is not None:
            # Normalize reconstruction error to 0-1
            error_normalized = self._normalize_anomaly(reconstruction_error)
            # Combine: physics constrains, reconstruction error refines
            ice_probability = 0.6 * physics_prob + 0.4 * error_normalized
        else:
            ice_probability = physics_prob

        # Apply hard physics constraint: T > 110K = 0 probability. NaN
        # pixels are also zeroed (they represent no-data, not ice).
        bad_mask = (temp_max > self.TEMP_ICE_THRESHOLD) | ~np.isfinite(temp_max)
        ice_probability[bad_mask] = 0.0

        # Create diagnostic indicator maps
        indicators = {
            "temperature_indicator": temp_indicator,
            "neutron_indicator": neutron_indicator,
            "radar_indicator": radar_indicator,
            "physics_combined": physics_prob
        }

        if reconstruction_error is not None:
            indicators["reconstruction_error"] = reconstruction_error
            indicators["reconstruction_normalized"] = error_normalized

        return ice_probability, indicators

    def _temperature_indicator(
        self,
        temp_max: npt.NDArray[np.float32],
        temp_min: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """
        Compute temperature-based ice indicator.

        Uses both max and min temperature:
        - Coldest regions (< 110K) more likely to contain ice
        - Regions with high max temp but low min temp (high thermal variation)
          are less stable for ice

        Args:
            temp_max: Maximum temperature (K)
            temp_min: Minimum temperature (K)

        Returns:
            Indicator map (0-1)
        """
        # Cold temperature indicator (inverse of normalized temperature)
        # Map [50, 150] K to [1, 0]
        temp_norm = np.clip((temp_max - 50) / 100, 0, 1)
        cold_indicator = 1.0 - temp_norm

        # Stability indicator: low thermal variation
        temp_range = temp_max - temp_min
        variation_norm = np.clip(temp_range / 50, 0, 1)  # 0-50K range
        stability_indicator = 1.0 - variation_norm

        # Combined: cold AND stable
        indicator = cold_indicator * 0.7 + stability_indicator * 0.3

        # Hard cutoff above threshold
        indicator[temp_max > self.TEMP_ICE_THRESHOLD] = 0.0

        return indicator.astype(np.float32)

    def _neutron_indicator(
        self,
        neutron_suppression: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """
        Compute neutron suppression indicator.

        Lower neutron counts indicate hydrogen (water ice) presence.

        Args:
            neutron_suppression: Neutron suppression factor (0-1, where lower = more suppression)

        Returns:
            Indicator map (0-1)
        """
        # Suppression < threshold indicates ice
        # Map [0, 1] to [1, 0]
        suppression_norm = np.clip(neutron_suppression, 0, 1)

        # Strong suppression = high ice probability
        indicator = 1.0 - suppression_norm

        return indicator.astype(np.float32)

    def _radar_indicator(
        self,
        radar_cpr: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """
        Compute radar CPR indicator.

        Ice deposits have distinctive CPR signatures.

        Args:
            radar_cpr: Circular polarization ratio

        Returns:
            Indicator map (0-1)
        """
        # CPR > threshold indicates ice
        cpr_norm = np.clip(radar_cpr / 0.5, 0, 1)  # Normalize to 0-0.5 range

        return cpr_norm.astype(np.float32)

    def _normalize_anomaly(
        self,
        anomaly_scores: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """
        Normalize anomaly scores to probability-like values.

        Uses percentile-based normalization.

        Args:
            anomaly_scores: Raw anomaly scores

        Returns:
            Normalized scores (0-1)
        """
        # Flatten for percentile computation
        flat = anomaly_scores.flatten()

        # Use 95th percentile as normalization factor
        p95 = np.percentile(flat, 95)
        p99 = np.percentile(flat, 99)

        if p99 > p95:
            normalized = (anomaly_scores - p95) / (p99 - p95 + 1e-8)
        else:
            normalized = anomaly_scores / (p95 + 1e-8)

        return np.clip(normalized, 0, 1).astype(np.float32)


def compute_ice_probability_simple(
    temp_min: npt.NDArray[np.float32],
    neutron_suppression: npt.NDArray[np.float32],
    radar_cpr: Optional[npt.NDArray[np.float32]] = None,
    reconstruction_error: Optional[npt.NDArray[np.float32]] = None
) -> npt.NDArray[np.float32]:
    """
    Simplified ice probability computation.

    Convenience function for quick calculations with basic inputs.

    Args:
        temp_min: Minimum temperature (K)
        neutron_suppression: Neutron suppression factor
        radar_cpr: Optional CPR values
        reconstruction_error: Optional anomaly scores

    Returns:
        Ice probability map
    """
    # Temperature indicator
    temp_indicator = np.where(
        temp_min > PhysicsConstraints.TEMP_ICE_THRESHOLD,
        0.0,
        1.0 - (temp_min - 50) / 60  # Map 50-110K to 1-0
    )
    temp_indicator = np.clip(temp_indicator, 0, 1)

    # Neutron indicator
    neutron_indicator = np.clip(1.0 - neutron_suppression, 0, 1)

    # Combine
    ice_prob = temp_indicator * 0.5 + neutron_indicator * 0.5

    # Incorporate radar if available
    if radar_cpr is not None:
        radar_indicator = np.clip(radar_cpr / 0.3, 0, 1)
        ice_prob = ice_prob * 0.7 + radar_indicator * 0.3

    # Incorporate reconstruction error if available
    if reconstruction_error is not None:
        error_norm = np.percentile(reconstruction_error, 95)
        if error_norm > 0:
            error_prob = reconstruction_error / error_norm
        else:
            error_prob = reconstruction_error
        error_prob = np.clip(error_prob, 0, 1)
        ice_prob = ice_prob * 0.6 + error_prob * 0.4

    # Apply hard constraint
    ice_prob[temp_min > PhysicsConstraints.TEMP_ICE_THRESHOLD] = 0.0

    return ice_prob.astype(np.float32)
