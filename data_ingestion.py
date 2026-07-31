"""
Data Ingestion Module for Lunar Ice Detection Pipeline
Handles loading and preprocessing of multi-source lunar remote sensing data
"""

import numpy as np
import numpy.typing as npt
from rasterio.windows import Window
from rasterio.io import DatasetReader
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LunarDatasetConfig:
    """Configuration for a single lunar dataset."""
    path: Path
    band_name: str
    physical_unit: str
    valid_range: Tuple[float, float]
    nodata_value: float

    @property
    def exists(self) -> bool:
        return self.path.exists()


@dataclass
class DataStatistics:
    """Statistics for robust scaling normalization."""
    median: float
    q1: float  # 25th percentile
    q3: float  # 75th percentile
    iqr: float  # Interquartile range
    min: float
    max: float
    count: int  # Valid (non-NaN) pixel count


class LunarDataIngestion:
    """
    Memory-efficient data ingestion for multi-modal lunar remote sensing data.

    Handles co-registered GeoTIFFs from:
    - LRO LOLA: elevation, slope, roughness
    - LRO Diviner: max/min temperatures
    - LRO LEND: neutron suppression
    - Mini-RF: CPR radar data

    Supports windowed reading for large datasets (>4GB) and proper NoData handling.
    All data is assumed to be reprojected to EPSG:30120 (Lunar South Pole Stereographic).
    """

    REQUIRED_BANDS = [
        'elevation',      # LOLA elevation (meters)
        'slope',         # LOLA slope (degrees)
        'roughness',     # LOLA RMS roughness (meters)
        'temp_max',      # Diviner maximum temperature (K)
        'temp_min',      # Diviner minimum temperature (K)
        'neutron_suppression',  # LEND neutron suppression (unitless)
        'radar_cpr',     # Mini-RF CPR (circular polarization ratio)
    ]

    # Physics-based constraints for ice detection
    TEMP_ICE_THRESHOLD = 110.0  # K - temperatures above this have near-zero ice probability

    def __init__(
        self,
        data_dir: Path,
        use_windowed_reading: bool = True,
        window_size: Tuple[int, int] = (512, 512),
        max_memory_gb: float = 4.0
    ):
        """
        Initialize the data ingestion pipeline.

        Args:
            data_dir: Directory containing the input GeoTIFF files
            use_windowed_reading: Enable windowed reading for large datasets
            window_size: Size of windows for tiled reading (height, width)
            max_memory_gb: Maximum memory to use for array operations
        """
        self.data_dir = Path(data_dir)
        self.use_windowed_reading = use_windowed_reading
        self.window_size = window_size
        self.max_memory_bytes = int(max_memory_gb * 1024**3)

        self._datasets: Dict[str, DatasetReader] = {}
        self._config: Dict[str, LunarDatasetConfig] = {}
        self._statistics: Dict[str, DataStatistics] = {}
        self._spatial_extent: Optional[Tuple[float, float, float, float]] = None
        self._nodata_masks: Dict[str, npt.NDArray[np.bool_]] = {}

    def register_dataset(
        self,
        band_name: str,
        filename: str,
        physical_unit: str,
        valid_range: Tuple[float, float],
        nodata_value: float = -9999.0
    ) -> None:
        """
        Register a dataset with its metadata.

        Args:
            band_name: Logical name for this band (e.g., 'elevation', 'temp_max')
            filename: Filename of the GeoTIFF in data_dir
            physical_unit: Physical unit string (e.g., 'meters', 'Kelvin')
            valid_range: Valid data range (min, max)
            nodata_value: NoData value in the raster
        """
        if band_name not in self.REQUIRED_BANDS:
            logger.warning(f"Unknown band '{band_name}' - not in standard list")

        filepath = self.data_dir / filename
        config = LunarDatasetConfig(
            path=filepath,
            band_name=band_name,
            physical_unit=physical_unit,
            valid_range=valid_range,
            nodata_value=nodata_value
        )
        self._config[band_name] = config
        logger.info(f"Registered dataset: {band_name} -> {filepath}")

    def _open_dataset(self, band_name: str) -> DatasetReader:
        """Open a raster dataset (lazy loading)."""
        if band_name not in self._datasets:
            if band_name not in self._config:
                raise KeyError(f"Band '{band_name}' not registered")

            config = self._config[band_name]
            if not config.exists:
                raise FileNotFoundError(f"Dataset not found: {config.path}")

            self._datasets[band_name] = DatasetReader(config.path)
            logger.debug(f"Opened dataset: {band_name}")

        return self._datasets[band_name]

    def _compute_statistics(
        self,
        band_name: str,
        sample_fraction: float = 0.1
    ) -> DataStatistics:
        """
        Compute robust statistics for a band using sampling.

        Uses robust statistics (median, IQR) to handle extreme outliers
        in crater depths and temperatures.

        Args:
            band_name: Band to compute statistics for
            sample_fraction: Fraction of pixels to sample (for large datasets)

        Returns:
            DataStatistics object with robust measures
        """
        dataset = self._open_dataset(band_name)
        config = self._config[band_name]

        height, width = dataset.height, dataset.width
        total_pixels = height * width
        sample_size = int(total_pixels * sample_fraction)

        # For smaller datasets, read all; for larger, use windowed sampling
        if not self.use_windowed_reading or total_pixels <= sample_size * 2:
            data = dataset.read(1)
        else:
            # Sample across the dataset using windows
            data = self._sample_dataset(dataset, sample_size)

        # Apply nodata mask
        valid_mask = (data != config.nodata_value) & ~np.isnan(data)
        valid_data = data[valid_mask].astype(np.float64)

        if valid_data.size == 0:
            raise ValueError(f"No valid data found in band '{band_name}'")

        # Compute robust statistics
        median = np.median(valid_data)
        q1 = np.percentile(valid_data, 25)
        q3 = np.percentile(valid_data, 75)
        iqr = q3 - q1

        stats = DataStatistics(
            median=median,
            q1=q1,
            q3=q3,
            iqr=iqr,
            min=float(np.min(valid_data)),
            max=float(np.max(valid_data)),
            count=valid_data.size
        )

        logger.info(
            f"Statistics for {band_name}: median={stats.median:.2f}, "
            f"IQR={stats.iqr:.2f}, valid_pixels={stats.count}"
        )

        return stats

    def _sample_dataset(
        self,
        dataset: DatasetReader,
        sample_size: int
    ) -> npt.NDArray[np.float32]:
        """Sample pixels uniformly across the dataset.

        Returns a flat 1-D array of sampled pixel values. The previous
        implementation reshaped ``sample_size`` elements to ``(H, W)`` which
        silently misbehaved whenever ``H * W != sample_size``.
        """
        height, width = dataset.height, dataset.width

        # Generate random sample coordinates
        np.random.seed(42)  # Reproducible sampling
        row_indices = np.random.randint(0, height, size=sample_size)
        col_indices = np.random.randint(0, width, size=sample_size)

        # Read samples using windowed reading
        samples = np.zeros(sample_size, dtype=np.float32)

        for i, (row, col) in enumerate(zip(row_indices, col_indices)):
            window = Window(col, row, 1, 1)
            data = dataset.read(1, window=window)
            samples[i] = data[0, 0]

        return samples

    def compute_all_statistics(self) -> Dict[str, DataStatistics]:
        """
        Compute statistics for all registered bands.

        Returns:
            Dictionary mapping band_name to DataStatistics
        """
        for band_name in self._config:
            if band_name not in self._statistics:
                self._statistics[band_name] = self._compute_statistics(band_name)

        return self._statistics

    @property
    def spatial_extent(self) -> Tuple[float, float, float, float]:
        """
        Get the spatial extent of the data (xmin, ymin, xmax, ymax).

        Returns:
            Tuple of (xmin, ymin, xmax, ymax) in EPSG:30120 coordinates
        """
        if self._spatial_extent is None:
            # Open first dataset to get extent
            first_band = next(iter(self._config.keys()))
            dataset = self._open_dataset(first_band)

            bounds = dataset.bounds
            self._spatial_extent = (bounds.left, bounds.bottom, bounds.right, bounds.top)

        return self._spatial_extent

    @property
    def shape(self) -> Tuple[int, int]:
        """Get the shape (height, width) of the datasets."""
        first_band = next(iter(self._config.keys()))
        dataset = self._open_dataset(first_band)
        return (dataset.height, dataset.width)

    @property
    def crs(self) -> str:
        """Get the coordinate reference system."""
        first_band = next(iter(self._config.keys()))
        dataset = self._open_dataset(first_band)
        return str(dataset.crs)

    def read_band(
        self,
        band_name: str,
        window: Optional[Window] = None
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
        """
        Read a single band with NoData masking.

        Args:
            band_name: Name of the band to read
            window: Optional window for partial reads

        Returns:
            Tuple of (data_array, valid_mask)
            - data_array: Float32 array with NoData replaced by NaN
            - valid_mask: Boolean mask indicating valid pixels
        """
        dataset = self._open_dataset(band_name)
        config = self._config[band_name]

        if window is not None:
            data = dataset.read(1, window=window).astype(np.float32)
        else:
            data = dataset.read(1).astype(np.float32)

        # Create valid mask and replace NoData with NaN
        valid_mask = (data != config.nodata_value) & ~np.isnan(data)
        data[~valid_mask] = np.nan

        return data, valid_mask

    def read_multi_band_window(
        self,
        bands: Optional[List[str]] = None,
        window: Optional[Window] = None
    ) -> Tuple[npt.NDArray[np.float32], Dict[str, npt.NDArray[np.bool_]]]:
        """
        Read multiple bands into a stacked array.

        Args:
            bands: List of band names to read (default: all registered)
            window: Optional window for partial reads

        Returns:
            Tuple of:
            - stacked_array: Shape (height, width, num_bands)
            - valid_masks: Dict mapping band_name to valid pixel mask
        """
        if bands is None:
            bands = list(self._config.keys())

        arrays = []
        valid_masks = {}

        for band_name in bands:
            data, valid_mask = self.read_band(band_name, window)
            arrays.append(data)
            valid_masks[band_name] = valid_mask

        stacked = np.stack(arrays, axis=-1)

        # Create combined valid mask (pixel is valid if valid in ALL bands)
        combined_valid = np.ones_like(list(valid_masks.values())[0], dtype=bool)
        for mask in valid_masks.values():
            combined_valid &= mask

        return stacked, valid_masks

    def create_aligned_array(
        self,
        bands: Optional[List[str]] = None
    ) -> Tuple[npt.NDArray[np.float32], Dict[str, DataStatistics], Dict[str, npt.NDArray[np.bool_]]]:
        """
        Create a full multi-dimensional feature tensor.

        Args:
            bands: List of band names to include

        Returns:
            Tuple of:
            - feature_tensor: Shape (height, width, channels)
            - statistics: Dict of band statistics
            - valid_masks: Dict of valid masks per band
        """
        # Ensure statistics are computed
        if not self._statistics:
            self.compute_all_statistics()

        # Read all bands
        feature_tensor, valid_masks = self.read_multi_band_window(bands)

        logger.info(
            f"Created feature tensor with shape {feature_tensor.shape}, "
            f"dtype={feature_tensor.dtype}"
        )

        return feature_tensor, self._statistics, valid_masks

    def get_windows(
        self,
        overlap: int = 0
    ) -> List[Window]:
        """
        Generate windowed reading regions for memory-efficient processing.

        Args:
            overlap: Overlap between windows in pixels

        Returns:
            List of rasterio Window objects covering the dataset
        """
        height, width = self.shape
        window_h, window_w = self.window_size

        windows = []

        row = 0
        while row < height:
            col = 0
            row_end = min(row + window_h, height)

            while col < width:
                col_end = min(col + window_w, width)

                window = Window(col, row, col_end - col, row_end - row)
                windows.append(window)

                col = col_end - overlap if col_end < width else col + window_w

            row = row_end - overlap if row_end < height else row + window_h

        logger.info(f"Generated {len(windows)} windows for tiled processing")
        return windows

    def close(self) -> None:
        """Close all open raster datasets."""
        for dataset in self._datasets.values():
            dataset.close()
        self._datasets.clear()
        logger.info("Closed all raster datasets")

    def __enter__(self) -> 'LunarDataIngestion':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
