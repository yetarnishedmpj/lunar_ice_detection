"""
Ice Depth Estimation Module for Lunar Ice Detection
Estimates subsurface ice thickness using radar penetration depth and thermal models
"""

import numpy as np
import numpy.typing as npt
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


# Physical constants
class RadarConstants:
    """Physical constants for radar ice depth estimation."""

    # Radar frequency for Mini-RF (S-band, ~2.5 GHz)
    FREQUENCY = 2.5e9  # Hz

    # Speed of light
    C = 299792458  # m/s

    # Wavelength in vacuum
    WAVELENGTH = C / FREQUENCY  # ~12 cm

    # Dielectric constants
    DIELECTRIC_VACUUM = 1.0
    DIELECTRIC_ICE = 3.1  # Pure water ice at lunar temperatures
    DIELECTRIC_REGOLITH = 2.5  # Dry lunar regolith

    # Attenuation in ice (dB/m) - frequency dependent
    # S-band ice attenuation is very low (< 0.01 dB/m)
    ATTENUATION_ICE = 0.01  # dB/m

    # Penetration depth where signal drops to 1/e (~37%)
    PENETRATION_DEPTH_FACTOR = 8.686  # Convert dB to nepers


class ThermalConstants:
    """Thermal constants for ice stability modeling."""

    # Ice sublimation rate temperature threshold
    ICE_UNSTABLE_TEMP = 110  # K

    # Thermal diffusivity of ice (m²/s)
    THERMAL_DIFFUSIVITY_ICE = 1.2e-6  # m²/s

    # Thermal conductivity (W/(m·K))
    THERMAL_CONDUCTIVITY_ICE = 2.2  # W/(m·K)
    THERMAL_CONDUCTIVITY_REGOLITH = 0.01  # W/(m·K)


def estimate_ice_depth_from_radar(
    radar_cpr: npt.NDArray[np.float32],
    temp_min: npt.NDArray[np.float32],
    ice_probability: npt.NDArray[np.float32],
    radar_frequency: float = RadarConstants.FREQUENCY
) -> Tuple[npt.NDArray[np.float32], Dict[str, npt.NDArray[np.float32]]]:
    """
    Estimate ice depth from radar data.

    Uses the radar CPR (Circular Polarization Ratio) and temperature
    to estimate the depth of subsurface ice.

    The method combines:
    1. Radar penetration depth based on dielectric properties
    2. Thermal stability constraints
    3. Ice probability as a confidence mask

    Args:
        radar_cpr: Radar CPR values (unitless ratio)
        temp_min: Minimum temperature (Kelvin)
        ice_probability: Ice probability from main model (0-1)
        radar_frequency: Radar frequency in Hz (default: Mini-RF S-band)

    Returns:
        Tuple of (ice_depth_meters, diagnostic_maps)
    """
    height, width = radar_cpr.shape

    # Calculate wavelength in ice
    wavelength_vacuum = RadarConstants.C / radar_frequency
    wavelength_ice = wavelength_vacuum / np.sqrt(RadarConstants.DIELECTRIC_ICE)

    # Estimate penetration depth from CPR
    # Higher CPR indicates surface scattering (shallow ice)
    # Lower CPR indicates volume scattering (deeper ice)
    penetration_depth = _cpr_to_penetration_depth(radar_cpr, wavelength_ice)

    # Apply temperature constraint
    # Ice is only stable below ~110K
    thermal_stability = _thermal_stability_factor(temp_min)

    # Combine with ice probability confidence
    # Depth estimate is more reliable where ice probability is high
    confidence_mask = ice_probability

    # Final depth estimate
    ice_depth = penetration_depth * thermal_stability * confidence_mask

    # Physical limits: ice depth between 0 and 10 meters
    ice_depth = np.clip(ice_depth, 0, 10).astype(np.float32)

    # Create diagnostic maps
    diagnostics = {
        'penetration_depth': penetration_depth,
        'thermal_stability': thermal_stability,
        'confidence_mask': confidence_mask,
        'wavelength_ice': np.full_like(radar_cpr, wavelength_ice, dtype=np.float32),
        'estimated_depth_meters': ice_depth
    }

    return ice_depth, diagnostics


def _cpr_to_penetration_depth(
    cpr: npt.NDArray[np.float32],
    wavelength_ice: float
) -> npt.NDArray[np.float32]:
    """
    Convert CPR to radar penetration depth.

    Uses empirical relationship between CPR and scattering mechanism:
    - High CPR (>0.5): Surface scattering, indicates exposed ice or very shallow
    - Medium CPR (0.2-0.5): Mixed scattering, moderate depth
    - Low CPR (<0.2): Volume scattering, deeper ice

    Args:
        cpr: Circular polarization ratio
        wavelength_ice: Radar wavelength in ice (meters)

    Returns:
        Penetration depth in meters
    """
    # Empirical depth model based on CPR
    # CPR > 0.6 → very shallow (< 10 cm)
    # CPR ~ 0.3 → moderate depth (~50 cm)
    # CPR < 0.1 → deep (several meters)

    # Convert CPR to depth estimate
    # Using a power law relationship
    depth = np.where(
        cpr > 0.6,
        0.1,  # Surface exposure
        np.where(
            cpr > 0.3,
            0.1 + (0.6 - cpr) * 1.0,  # 10-40 cm
            0.4 + (0.3 - cpr) * 5.0    # 40 cm to several meters
        )
    )

    # Scale by wavelength (longer wavelength = deeper penetration)
    depth = depth * (wavelength_ice / 0.12)  # Normalized to 12cm wavelength

    return depth.astype(np.float32)


def _thermal_stability_factor(temp: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """
    Calculate thermal stability factor for ice.

    Ice is stable below ~110K. Above this temperature, sublimation
    becomes significant over geological timescales.

    Args:
        temp: Temperature in Kelvin

    Returns:
        Stability factor (0-1, where 1 = fully stable)
    """
    # Ice stable below 110K
    # Use sigmoid for smooth transition
    k = 0.5  # Steepness
    threshold = 110.0

    stability = 1.0 / (1.0 + np.exp(k * (temp - threshold)))

    return stability.astype(np.float32)


def estimate_ice_volume(
    ice_depth: npt.NDArray[np.float32],
    ice_probability: npt.NDArray[np.float32],
    pixel_size: float = 20.0  # Default 20m pixels for lunar data
) -> Dict[str, float]:
    """
    Estimate total ice volume from depth map.

    Args:
        ice_depth: Ice depth in meters
        ice_probability: Ice probability (used as confidence weight)
        pixel_size: Pixel size in meters

    Returns:
        Dictionary with volume estimates
    """
    # Only count pixels with >50% probability
    valid_mask = ice_probability > 0.5

    # Volume per pixel = depth * pixel_area * probability
    pixel_area = pixel_size ** 2  # m²
    volume_per_pixel = ice_depth * pixel_area * ice_probability

    # Total volume
    total_volume = np.sum(volume_per_pixel[valid_mask])

    # Volume in cubic kilometers
    volume_km3 = total_volume / 1e9

    # Mass estimate (ice density = 917 kg/m³)
    density_ice = 917  # kg/m³
    total_mass_kg = total_volume * density_ice
    total_mass_mtons = total_mass_kg / 1e9  # Million metric tons

    return {
        'total_volume_m3': float(total_volume),
        'total_volume_km3': float(volume_km3),
        'total_mass_kg': float(total_mass_kg),
        'total_mass_million_tons': float(total_mass_mtons),
        'valid_pixels': int(valid_mask.sum()),
        'mean_depth_m': float(np.mean(ice_depth[valid_mask])) if valid_mask.sum() > 0 else 0.0,
        'max_depth_m': float(np.max(ice_depth[valid_mask])) if valid_mask.sum() > 0 else 0.0
    }


def compute_thermal_timescale(
    temp_min: npt.NDArray[np.float32],
    depth: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """
    Compute the thermal timescale for ice sublimation.

    This estimates how long ice would survive at a given temperature
    and depth.

    Args:
        temp_min: Minimum temperature (K)
        depth: Ice depth (meters)

    Returns:
        Timescale in years
    """
    # Simplified sublimation model
    # Higher temperature = shorter lifetime
    # Deeper ice = longer lifetime (protected from space)

    # Sublimation rate constant (simplified)
    k_sublimation = np.where(
        temp_min < 50,
        1e-6,  # Very cold, extremely stable
        np.where(
            temp_min < 80,
            1e-4,  # Cold, very stable
            np.where(
                temp_min < 100,
                1e-2,  # Moderately cold
                1e-1    # Near threshold
            )
        )
    )

    # Lifetime increases with depth (thermal insulation)
    # Depth provides exponential protection from solar heating & vacuum sublimation
    depth_factor = np.exp(depth * 5.0)

    # Timescale in years
    timescale_years = depth_factor / k_sublimation / (365 * 24 * 3600)

    return timescale_years.astype(np.float32)


def create_depth_visualization(
    ice_depth: npt.NDArray[np.float32],
    ice_probability: npt.NDArray[np.float32],
    temp_min: Optional[npt.NDArray[np.float32]] = None
) -> Dict[str, npt.NDArray[np.float32]]:
    """
    Create visualization-ready depth maps with different thresholds.

    Args:
        ice_depth: Estimated ice depth (meters)
        ice_probability: Ice probability (0-1)
        temp_min: Optional temperature for additional constraints

    Returns:
        Dictionary of visualization layers
    """
    # Binary mask at different confidence levels
    vis_layers = {
        'depth_shallow': np.where(
            (ice_depth < 0.2) & (ice_probability > 0.5),
            ice_depth,
            0
        ).astype(np.float32),

        'depth_medium': np.where(
            (ice_depth >= 0.2) & (ice_depth < 1.0) & (ice_probability > 0.5),
            ice_depth,
            0
        ).astype(np.float32),

        'depth_deep': np.where(
            (ice_depth >= 1.0) & (ice_probability > 0.5),
            ice_depth,
            0
        ).astype(np.float32),

        'depth_log': np.log10(ice_depth + 0.01).astype(np.float32),

        'probability_weighted_depth': (ice_depth * ice_probability).astype(np.float32)
    }

    if temp_min is not None:
        # Add thermally stable regions
        thermal_stability = _thermal_stability_factor(temp_min)
        vis_layers['thermally_stable_depth'] = (
            ice_depth * thermal_stability * (ice_probability > 0.5)
        ).astype(np.float32)

    return vis_layers


def validate_depth_estimates(
    ice_depth: npt.NDArray[np.float32],
    radar_cpr: npt.NDArray[np.float32],
    temp_min: npt.NDArray[np.float32]
) -> Dict[str, float]:
    """
    Validate depth estimates against physical constraints.

    Returns statistics on estimate quality.

    Args:
        ice_depth: Estimated depth
        radar_cpr: Radar CPR values
        temp_min: Temperature values

    Returns:
        Validation metrics
    """
    # Check for physical consistency

    # 1. Depth should be > 0 where ice is probable
    valid_prob = ice_depth[(ice_depth > 0) & np.isfinite(ice_depth)]

    # 2. Check temperature consistency
    # Deep ice should be in very cold regions
    deep_ice_mask = ice_depth > 1.0
    if deep_ice_mask.sum() > 0:
        deep_temp_mean = np.mean(temp_min[deep_ice_mask])
    else:
        deep_temp_mean = np.nan

    # 3. Check CPR consistency
    # Deep ice should have lower CPR
    if deep_ice_mask.sum() > 0:
        deep_cpr_mean = np.mean(radar_cpr[deep_ice_mask])
    else:
        deep_cpr_mean = np.nan

    return {
        'valid_depth_pixels': len(valid_prob),
        'mean_depth_m': float(np.mean(valid_prob)) if len(valid_prob) > 0 else 0.0,
        'median_depth_m': float(np.median(valid_prob)) if len(valid_prob) > 0 else 0.0,
        'max_depth_m': float(np.max(valid_prob)) if len(valid_prob) > 0 else 0.0,
        'deep_ice_temp_K': float(deep_temp_mean) if np.isfinite(deep_temp_mean) else None,
        'deep_ice_cpr': float(deep_cpr_mean) if np.isfinite(deep_cpr_mean) else None,
        'shallow_ice_fraction': float(np.sum(ice_depth < 0.2) / np.sum(ice_depth > 0)) if np.sum(ice_depth > 0) > 0 else 0.0
    }
