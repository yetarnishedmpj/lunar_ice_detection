"""
Output Module for Lunar Ice Detection Pipeline
Projects anomaly scores back to georeferenced GeoTIFF format
"""

import numpy as np
import numpy.typing as npt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import rasterio
from rasterio.io import DatasetReader

logger = logging.getLogger(__name__)


class GeoTIFFExporter:
    """
    Export ice probability maps to georeferenced GeoTIFF format.

    Maintains spatial reference information from input datasets
    and writes single-band output with proper georeferencing.
    """

    # Default CRS for lunar south pole stereographic
    DEFAULT_CRS = "EPSG:30120"

    # NoData value for output
    NODATA_VALUE = -9999.0

    def __init__(
        self,
        reference_dataset: DatasetReader,
        output_crs: str = DEFAULT_CRS
    ):
        """
        Initialize the exporter with a reference dataset.

        Args:
            reference_dataset: Open rasterio dataset to copy georeferencing from
            output_crs: Output coordinate reference system
        """
        self.reference_dataset = reference_dataset
        self.output_crs = output_crs
        self._transform = reference_dataset.transform
        try:
            self._crs = CRS.from_string(output_crs) if isinstance(output_crs, str) else output_crs
        except Exception:
            if hasattr(reference_dataset, 'crs') and reference_dataset.crs:
                self._crs = reference_dataset.crs
            else:
                self._crs = CRS.from_epsg(4326)

    @classmethod
    def from_paths(
        cls,
        reference_paths: List[Path],
        output_crs: str = "EPSG:30120"
    ) -> 'GeoTIFFExporter':
        """
        Create exporter from file paths.

        Args:
            reference_paths: List of reference GeoTIFF paths
            output_crs: Output CRS

        Returns:
            GeoTIFFExporter instance
        """
        # Use first available dataset as reference
        for path in reference_paths:
            if path.exists():
                dataset = rasterio.open(path)
                return cls(dataset, output_crs)

        raise FileNotFoundError("No valid reference datasets found")

    def export_ice_probability(
        self,
        ice_probability: npt.NDArray[np.float32],
        output_path: Path,
        valid_mask: Optional[npt.NDArray[np.bool_]] = None,
        compression: str = "lzw"
    ) -> None:
        """
        Export ice probability map to GeoTIFF.

        Args:
            ice_probability: Ice probability array (H, W) with values 0-1
            output_path: Output file path
            valid_mask: Optional mask of valid pixels
            compression: GeoTIFF compression type
        """
        height, width = ice_probability.shape

        # Apply valid mask if provided
        output_data = ice_probability.copy()
        if valid_mask is not None:
            output_data[~valid_mask] = self.NODATA_VALUE
        else:
            # Set NaN to NoData
            output_data[np.isnan(output_data)] = self.NODATA_VALUE

        # Ensure proper dtype
        output_data = output_data.astype(np.float32)

        # Write GeoTIFF
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=np.float32,
            crs=self._crs,
            transform=self._transform,
            nodata=self.NODATA_VALUE,
            compress=compression,
            tiled=True,
            blockysize=256,
            blockxsize=256
        ) as dst:
            dst.write(output_data, 1)

            # Add description
            dst.descriptions = ("Ice Probability",)

            # Add metadata
            dst.update_tags(
                IceProbability="1.0",
                Source="Lunar Ice Detection VAE",
                TemperatureThreshold="110K",
                Method="VAE Reconstruction + Physics Constraints"
            )

        logger.info(f"Exported ice probability map to {output_path}")

    def export_anomaly_scores(
        self,
        anomaly_scores: npt.NDArray[np.float32],
        output_path: Path,
        valid_mask: Optional[npt.NDArray[np.bool_]] = None,
        compression: str = "lzw"
    ) -> None:
        """
        Export raw anomaly scores to GeoTIFF.

        Args:
            anomaly_scores: Anomaly score array (H, W)
            output_path: Output file path
            valid_mask: Optional valid pixel mask
            compression: Compression type
        """
        self.export_ice_probability(
            anomaly_scores,
            output_path,
            valid_mask,
            compression
        )

    def export_multi_band(
        self,
        bands: Dict[str, npt.NDArray[np.float32]],
        output_path: Path,
        compression: str = "lzw"
    ) -> None:
        """
        Export multiple bands to a single multi-band GeoTIFF.

        Args:
            bands: Dictionary mapping band names to arrays
            output_path: Output file path
            compression: Compression type
        """
        # Ensure all bands have same shape
        shapes = [arr.shape for arr in bands.values()]
        if not all(s == shapes[0] for s in shapes):
            raise ValueError("All bands must have the same shape")

        height, width = shapes[0]
        num_bands = len(bands)

        # Stack bands
        data = np.stack(list(bands.values()), axis=0).astype(np.float32)

        # Get band names in order
        band_names = list(bands.keys())

        # Handle NaN
        data = np.nan_to_num(data, nan=self.NODATA_VALUE)

        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=num_bands,
            dtype=np.float32,
            crs=self._crs,
            transform=self._transform,
            nodata=self.NODATA_VALUE,
            compress=compression,
            tiled=True,
            blockysize=256,
            blockxsize=256
        ) as dst:
            for i, band_name in enumerate(band_names):
                dst.write(data[i], i + 1)
                # rasterio exposes per-band descriptions via set_band_description
                dst.set_band_description(i + 1, band_name)

        logger.info(f"Exported multi-band GeoTIFF to {output_path}")


class IceProbabilityMapper:
    """
    Maps VAE outputs and physics constraints to final ice probability.

    Handles windowed inference and stitching for large datasets.
    """

    def __init__(
        self,
        reference_dataset: DatasetReader,
        patch_size: int = 64,
        overlap: int = 16
    ):
        """
        Initialize the mapper.

        Args:
            reference_dataset: Reference dataset for georeferencing
            patch_size: Size of patches used for inference
            overlap: Overlap between patches for stitching
        """
        self.reference_dataset = reference_dataset
        self.patch_size = patch_size
        self.overlap = overlap
        self._height = reference_dataset.height
        self._width = reference_dataset.width
        self._transform = reference_dataset.transform
        self._crs = reference_dataset.crs

    def create_output_arrays(
        self
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """
        Create output arrays for accumulating results.

        Returns:
            Tuple of (accumulator array, weight array)
        """
        accumulator = np.zeros((self._height, self._width), dtype=np.float64)
        weights = np.zeros((self._height, self._width), dtype=np.float32)

        return accumulator, weights

    def accumulate_patch(
        self,
        patch_result: npt.NDArray[np.float32],
        row: int,
        col: int,
        accumulator: npt.NDArray[np.float64],
        weights: npt.NDArray[np.float32]
    ) -> None:
        """
        Accumulate a patch result into the output arrays.

        Args:
            patch_result: Patch result array
            row: Starting row position
            col: Starting column position
            accumulator: Accumulator array
            weights: Weight array
        """
        ph, pw = patch_result.shape

        # Compute weights (higher weight at center, lower at edges)
        y, x = np.ogrid[:ph, :pw]
        center_y, center_x = ph / 2, pw / 2

        # Gaussian-like weight
        dist_y = (y - center_y) / (ph / 4)
        dist_x = (x - center_x) / (pw / 4)
        weight = np.exp(-(dist_y ** 2 + dist_x ** 2))
        weight = weight.astype(np.float32)

        # Accumulate
        accumulator[row:row + ph, col:col + pw] += patch_result * weight
        weights[row:row + ph, col:col + pw] += weight

    def finalize(
        self,
        accumulator: npt.NDArray[np.float64],
        weights: npt.NDArray[np.float32],
        valid_mask: Optional[npt.NDArray[np.bool_]] = None
    ) -> npt.NDArray[np.float32]:
        """
        Finalize the accumulated results.

        Args:
            accumulator: Accumulated values
            weights: Weight array
            valid_mask: Optional valid pixel mask

        Returns:
            Final normalized result
        """
        # Normalize by weights
        with np.errstate(divide='ignore', invalid='ignore'):
            result = accumulator / weights

        # Handle invalid pixels
        if valid_mask is not None:
            result[~valid_mask] = 0.0

        # Clip to valid probability range
        result = np.clip(result, 0.0, 1.0)

        return result.astype(np.float32)

    def export_to_geotiff(
        self,
        data: npt.NDArray[np.float32],
        output_path: Path,
        nodata: float = -9999.0,
        compression: str = "lzw"
    ) -> None:
        """
        Export data to GeoTIFF.

        Args:
            data: Data array to export
            output_path: Output file path
            nodata: NoData value
            compression: Compression type
        """
        height, width = data.shape

        # Replace NaN with nodata
        output_data = data.copy()
        output_data[np.isnan(output_data)] = nodata

        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=np.float32,
            crs=self._crs,
            transform=self._transform,
            nodata=nodata,
            compress=compression,
            tiled=True,
            blockysize=256,
            blockxsize=256
        ) as dst:
            dst.write(output_data.astype(np.float32), 1)

        logger.info(f"Exported result to {output_path}")


def create_output_from_results(
    ice_probability: npt.NDArray[np.float32],
    indicators: Dict[str, npt.NDArray[np.float32]],
    reference_paths: List[Path],
    output_dir: Path,
    prefix: str = "ice_probability"
) -> Dict[str, Path]:
    """
    Create all output files from inference results.

    Args:
        ice_probability: Final ice probability map
        indicators: Dictionary of indicator maps
        reference_paths: Reference GeoTIFF paths
        output_dir: Output directory
        prefix: Filename prefix

    Returns:
        Dictionary mapping output names to paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create exporter
    exporter = GeoTIFFExporter.from_paths(reference_paths)

    output_files = {}

    # Main ice probability
    ice_path = output_dir / f"{prefix}.tif"
    exporter.export_ice_probability(ice_probability, ice_path)
    output_files["ice_probability"] = ice_path

    # Export individual indicators if they exist
    indicator_names = [
        ("temperature_indicator", "temp_indicator"),
        ("neutron_indicator", "neutron_indicator"),
        ("radar_indicator", "radar_indicator"),
        ("physics_combined", "physics_combined")
    ]

    for src_name, dst_name in indicator_names:
        if src_name in indicators:
            indicator_path = output_dir / f"{dst_name}.tif"
            exporter.export_ice_probability(
                indicators[src_name],
                indicator_path
            )
            output_files[dst_name] = indicator_path

    # Export reconstruction error if available
    if "reconstruction_error" in indicators:
        error_path = output_dir / "reconstruction_error.tif"
        exporter.export_anomaly_scores(
            indicators["reconstruction_error"],
            error_path
        )
        output_files["reconstruction_error"] = error_path

    logger.info(f"Created {len(output_files)} output files in {output_dir}")

    return output_files
