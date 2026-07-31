"""
Main Pipeline Module for Lunar Ice Detection
Orchestrates data ingestion, model training, inference, and output generation
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
import logging.config
from dataclasses import dataclass, field

try:
    from .data_ingestion import LunarDataIngestion, DataStatistics
    from .normalization import RobustNormalizer, ChannelWiseNormalizer
    from .model import (
        ConvVAE, VAELoss, LunarPatchDataset,
        create_vae_model, train_epoch, evaluate
    )
    from .physics import PhysicsConstraints
    from .output import GeoTIFFExporter, IceProbabilityMapper, create_output_from_results
    from .postprocessing import (
        SmoothingConfig,
        gaussian_smooth,
        smooth_and_summarize,
    )
except ImportError:
    from data_ingestion import LunarDataIngestion, DataStatistics
    from normalization import RobustNormalizer, ChannelWiseNormalizer
    from model import (
        ConvVAE, VAELoss, LunarPatchDataset,
        create_vae_model, train_epoch, evaluate
    )
    from physics import PhysicsConstraints
    from output import GeoTIFFExporter, IceProbabilityMapper, create_output_from_results
    from postprocessing import (
        SmoothingConfig,
        gaussian_smooth,
        smooth_and_summarize,
    )

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the lunar ice detection pipeline."""

    # Data configuration
    data_dir: Path = Path("./data")
    output_dir: Path = Path("./output")

    # Dataset registration (band_name -> filename)
    dataset_files: Dict[str, str] = field(default_factory=lambda: {
        'elevation': 'lola_elevation.tif',
        'slope': 'lola_slope.tif',
        'roughness': 'lola_roughness.tif',
        'temp_max': 'diviner_temp_max.tif',
        'temp_min': 'diviner_temp_min.tif',
        'neutron_suppression': 'lend_neutron_suppression.tif',
        'radar_cpr': 'minirf_cpr.tif',
    })

    # NoData values per band
    nodata_values: Dict[str, float] = field(default_factory=lambda: {
        'elevation': -9999.0,
        'slope': -9999.0,
        'roughness': -9999.0,
        'temp_max': 0.0,
        'temp_min': 0.0,
        'neutron_suppression': -1.0,
        'radar_cpr': -1.0,
    })

    # Valid ranges per band
    valid_ranges: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'elevation': (-5000, 5000),       # meters
        'slope': (0, 90),                  # degrees
        'roughness': (0, 100),              # meters RMS
        'temp_max': (50, 400),              # Kelvin
        'temp_min': (20, 400),              # Kelvin
        'neutron_suppression': (0, 1.5),   # unitless ratio
        'radar_cpr': (0, 2.0),             # unitless ratio
    })

    # Physical units
    physical_units: Dict[str, str] = field(default_factory=lambda: {
        'elevation': 'meters',
        'slope': 'degrees',
        'roughness': 'meters',
        'temp_max': 'Kelvin',
        'temp_min': 'Kelvin',
        'neutron_suppression': 'ratio',
        'radar_cpr': 'ratio',
    })

    # Model configuration
    patch_size: int = 64
    patch_stride: int = 32
    latent_dim: int = 128

    # Training configuration
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-4
    kl_weight: float = 0.1

    # Physics configuration
    temp_threshold: float = 110.0  # Kelvin

    # Memory configuration
    max_memory_gb: float = 4.0
    use_windowed_reading: bool = True
    window_size: Tuple[int, int] = (512, 512)

    # Output configuration
    output_crs: str = "EPSG:30120"
    output_prefix: str = "lunar_ice"

    # Post-processing
    smooth_sigma_pixels: float = 1.5
    generate_report: bool = True
    ice_threshold: float = 0.5


class LunarIceDetectionPipeline:
    """
    Main pipeline for lunar ice detection.

    Orchestrates:
    1. Data ingestion and normalization
    2. VAE model training
    3. Inference with physics constraints
    4. GeoTIFF output generation
    """

    def __init__(
        self,
        config: PipelineConfig,
        device: Optional[str] = None
    ):
        """
        Initialize the pipeline.

        Args:
            config: Pipeline configuration
            device: Device for model training/inference
        """
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize components
        self.data_ingestion: Optional[LunarDataIngestion] = None
        self.normalizer: Optional[RobustNormalizer] = None
        self.model: Optional[ConvVAE] = None
        self.loss_fn: Optional[VAELoss] = None
        self.physics: Optional[PhysicsConstraints] = None
        self.exporter: Optional[GeoTIFFExporter] = None

        # Data storage
        self.feature_tensor: Optional[np.ndarray] = None
        self.statistics: Optional[Dict[str, DataStatistics]] = None

        logger.info(f"Pipeline initialized on device: {self.device}")

    def initialize_data(self) -> None:
        """Initialize data ingestion and register all datasets."""
        logger.info("Initializing data ingestion...")

        self.data_ingestion = LunarDataIngestion(
            data_dir=self.config.data_dir,
            use_windowed_reading=self.config.use_windowed_reading,
            window_size=self.config.window_size,
            max_memory_gb=self.config.max_memory_gb
        )

        # Register all datasets
        for band_name, filename in self.config.dataset_files.items():
            self.data_ingestion.register_dataset(
                band_name=band_name,
                filename=filename,
                physical_unit=self.config.physical_units[band_name],
                valid_range=self.config.valid_ranges[band_name],
                nodata_value=self.config.nodata_values[band_name]
            )

        # Compute statistics
        self.statistics = self.data_ingestion.compute_all_statistics()

        # Create normalizer
        self.normalizer = RobustNormalizer(self.statistics)

        # Initialize physics constraints
        self.physics = PhysicsConstraints()

        logger.info(f"Data initialization complete. Shape: {self.data_ingestion.shape}")

    def load_data(self) -> np.ndarray:
        """
        Load and normalize all data.

        Returns:
            Normalized feature tensor (H, W, channels)
        """
        logger.info("Loading and normalizing data...")

        if self.data_ingestion is None:
            self.initialize_data()

        # Load feature tensor
        self.feature_tensor, self.statistics, valid_masks = (
            self.data_ingestion.create_aligned_array()
        )

        # Get band names in order
        band_names = list(self.config.dataset_files.keys())

        # Normalize
        self.feature_tensor = self.normalizer.fit_transform(
            self.feature_tensor,
            band_names
        )

        logger.info(f"Data loaded: shape={self.feature_tensor.shape}")

        return self.feature_tensor

    def train(
        self,
        feature_tensor: Optional[np.ndarray] = None,
        validation_split: float = 0.2,
        save_path: Optional[Path] = None,
        log_interval: int = 10
    ) -> Dict[str, List[float]]:
        """
        Train the VAE model.

        Args:
            feature_tensor: Feature tensor (loads from pipeline if None)
            validation_split: Fraction of data for validation
            save_path: Path to save best model
            log_interval: Logging interval in batches

        Returns:
            Dictionary of training metrics
        """
        if feature_tensor is None:
            feature_tensor = self.load_data()

        logger.info("Starting VAE training...")

        # Create datasets
        dataset = LunarPatchDataset(
            feature_tensor=feature_tensor,
            patch_size=self.config.patch_size,
            stride=self.config.patch_stride,
            augment=True
        )

        # Split into train/val
        val_size = int(len(dataset) * validation_split)
        train_size = len(dataset) - val_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0
        )

        # Create model
        self.model, self.loss_fn = create_vae_model(
            input_channels=7,  # Number of bands
            patch_size=self.config.patch_size,
            device=self.device
        )

        # Optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate
        )

        # Training loop
        history = {
            'train_loss': [],
            'train_recon': [],
            'train_kl': [],
            'val_loss': [],
            'val_recon': [],
            'val_kl': []
        }

        best_val_loss = float('inf')

        for epoch in range(self.config.num_epochs):
            # Train
            train_loss, train_recon, train_kl = train_epoch(
                self.model, self.loss_fn, train_loader, optimizer, self.device
            )

            # Validate
            val_loss, val_recon, val_kl = evaluate(
                self.model, val_loader, self.loss_fn, self.device
            )

            # Log
            history['train_loss'].append(train_loss)
            history['train_recon'].append(train_recon)
            history['train_kl'].append(train_kl)
            history['val_loss'].append(val_loss)
            history['val_recon'].append(val_recon)
            history['val_kl'].append(val_kl)

            if epoch % log_interval == 0:
                logger.info(
                    f"Epoch {epoch+1}/{self.config.num_epochs}: "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}"
                )

            # Save best model
            if save_path and val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_path)
                logger.info(f"Saved best model to {save_path}")

        logger.info("Training complete!")
        return history

    def infer(
        self,
        feature_tensor: Optional[np.ndarray] = None,
        use_physics: bool = True,
        use_uncertainty: bool = False,
        mc_samples: int = 10
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Run inference to detect ice.

        Args:
            feature_tensor: Feature tensor (loads from pipeline if None)
            use_physics: Whether to apply physics constraints
            use_uncertainty: If True, use Monte Carlo dropout for uncertainty estimation
            mc_samples: Number of MC dropout samples for uncertainty estimation

        Returns:
            Tuple of (ice_probability, indicator_maps)
        """
        if feature_tensor is None:
            feature_tensor = self.load_data()

        if self.model is None:
            raise RuntimeError("Model not trained or loaded")

        logger.info("Running inference..." + (" with uncertainty estimation" if use_uncertainty else ""))

        self.model.eval()

        # Create dataset for inference
        dataset = LunarPatchDataset(
            feature_tensor=feature_tensor,
            patch_size=self.config.patch_size,
            stride=self.config.patch_stride
        )

        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0
        )

        # Accumulate anomaly scores and optionally uncertainty
        height, width, _ = feature_tensor.shape
        accumulator = np.zeros((height, width), dtype=np.float64)
        weights = np.zeros((height, width), dtype=np.float32)

        if use_uncertainty:
            variance_accumulator = np.zeros((height, width), dtype=np.float64)
            variance_weights = np.zeros((height, width), dtype=np.float32)

        with torch.no_grad():
            for patches, (rows, cols) in loader:
                patches = patches.to(self.device)

                # Get anomaly scores (with uncertainty if requested)
                if use_uncertainty:
                    anomaly_scores, uncertainty_maps = self.model.get_anomaly_score(
                        patches, use_mc_dropout=True, mc_samples=mc_samples
                    )

                    # Accumulate std-dev (variance accumulates correctly,
                    # then we sqrt only once after the weighted mean).
                    for i, (row, col) in enumerate(zip(rows, cols)):
                        if uncertainty_maps is not None:
                            unc = uncertainty_maps[i].cpu().numpy()
                            self._accumulate_patch(
                                unc, row, col, variance_accumulator, variance_weights
                            )
                else:
                    anomaly_scores, _ = self.model.get_anomaly_score(patches)

                # Convert to numpy and accumulate
                for i, (row, col) in enumerate(zip(rows, cols)):
                    score = anomaly_scores[i].cpu().numpy()
                    self._accumulate_patch(
                        score, row, col, accumulator, weights
                    )

        # Normalize
        with np.errstate(divide='ignore', invalid='ignore'):
            reconstruction_error = accumulator / weights
        reconstruction_error = reconstruction_error.astype(np.float32)

        # Compute uncertainty map if requested (sqrt of weighted-variance mean)
        uncertainty_map = None
        if use_uncertainty:
            with np.errstate(divide='ignore', invalid='ignore'):
                uncertainty_mean = variance_accumulator / variance_weights
            uncertainty_map = np.sqrt(
                np.clip(uncertainty_mean, 0, None)
            ).astype(np.float32)

        # Apply physics constraints
        if use_physics:
            # Only inverse-transform the bands needed for physics, not the
            # full feature tensor (saves memory + time).
            band_names = list(self.config.dataset_files.keys())
            physics_bands = {
                self.physics.temp_max_band,
                self.physics.temp_min_band,
                self.physics.neutron_band,
                self.physics.radar_band,
            }
            physics_band_indices = [
                i for i, name in enumerate(band_names) if name in physics_bands
            ]
            physics_band_names = [band_names[i] for i in physics_band_indices]

            physics_slice = feature_tensor[:, :, physics_band_indices]
            feature_physical_slice = self.normalizer.inverse_transform(
                physics_slice, physics_band_names
            )

            # Stub any non-physics bands with NaN so PhysicsConstraints can be
            # called with the full band-name list (it indexes by band name,
            # not position).
            feature_physical = np.full(
                feature_tensor.shape, np.nan, dtype=np.float32
            )
            for j, idx in enumerate(physics_band_indices):
                feature_physical[:, :, idx] = feature_physical_slice[:, :, j]

            ice_probability, indicators = self.physics.compute_ice_probability(
                feature_physical,
                band_names,
                reconstruction_error
            )
        else:
            # Use just reconstruction error
            ice_probability = self._normalize_anomaly(reconstruction_error)
            indicators = {'reconstruction_error': reconstruction_error}

        # Add uncertainty to indicators if available
        if uncertainty_map is not None:
            indicators['uncertainty'] = uncertainty_map

        logger.info("Inference complete")

        return ice_probability, indicators

    def _accumulate_patch(
        self,
        patch_result: np.ndarray,
        row: int,
        col: int,
        accumulator: np.ndarray,
        weights: np.ndarray
    ) -> None:
        """Accumulate patch results with Gaussian weighting."""
        ph, pw = patch_result.shape

        # Gaussian weight
        y, x = np.ogrid[:ph, :pw]
        center_y, center_x = ph / 2, pw / 2
        dist_y = (y - center_y) / (ph / 4)
        dist_x = (x - center_x) / (pw / 4)
        weight = np.exp(-(dist_y ** 2 + dist_x ** 2)).astype(np.float32)

        accumulator[row:row + ph, col:col + pw] += patch_result * weight
        weights[row:row + ph, col:col + pw] += weight

    def _normalize_anomaly(self, anomaly_scores: np.ndarray) -> np.ndarray:
        """Normalize anomaly scores to 0-1."""
        p99 = np.percentile(anomaly_scores, 99)
        if p99 > 0:
            normalized = anomaly_scores / p99
        else:
            normalized = anomaly_scores
        return np.clip(normalized, 0, 1).astype(np.float32)

    def run(
        self,
        train: bool = True,
        save_model_path: Optional[Path] = None,
        smooth: Optional[bool] = None,
        smoothing_sigma: Optional[float] = None,
        generate_report: Optional[bool] = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, Path]]:
        """
        Run the complete pipeline.

        Args:
            train: Whether to train the model
            save_model_path: Path to save trained model
            smooth: Override the configured smoothing-sigma on/off behaviour
                (None ⇒ fall back to ``config.smooth_sigma_pixels > 0``).
            smoothing_sigma: Override sigma value (pixels).
            generate_report: Whether to emit a HTML + JSON report.

        Returns:
            Tuple of (ice_probability, indicators, output_files)
        """
        logger.info("Starting complete pipeline run...")

        # Load data
        feature_tensor = self.load_data()

        # Train if requested
        if train:
            self.train(
                feature_tensor=feature_tensor,
                save_path=save_model_path
            )
        elif save_model_path and save_model_path.exists():
            # Load pre-trained model
            self.model, self.loss_fn = create_vae_model(
                input_channels=7,
                patch_size=self.config.patch_size,
                device=self.device
            )
            self.model.load_state_dict(torch.load(save_model_path))
            logger.info(f"Loaded model from {save_model_path}")

        # Run inference
        ice_probability, indicators = self.infer(feature_tensor)

        # Create output
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Get reference paths
        reference_paths = [
            self.config.data_dir / fn
            for fn in self.config.dataset_files.values()
        ]

        output_files = create_output_from_results(
            ice_probability=ice_probability,
            indicators=indicators,
            reference_paths=reference_paths,
            output_dir=self.config.output_dir,
            prefix=self.config.output_prefix
        )

        # Decide smoothing behaviour from explicit args or config defaults.
        if smoothing_sigma is None:
            smoothing_sigma = self.config.smooth_sigma_pixels
        if smooth is None:
            smooth = smoothing_sigma > 0

        if smooth:
            try:
                ice_probability, indicators, output_files = smooth_and_summarize(
                    ice_probability=ice_probability,
                    indicators=indicators,
                    output_files=output_files,
                    output_dir=self.config.output_dir,
                    prefix=self.config.output_prefix,
                    smoothing_sigma=smoothing_sigma,
                    threshold=self.config.ice_threshold,
                    device=self.device,
                )
                # Also export the smoothed map as a GeoTIFF for downstream GIS.
                try:
                    try:
                        from .output import GeoTIFFExporter
                    except ImportError:
                        from output import GeoTIFFExporter
                    smoothed_path = self.config.output_dir / f"{self.config.output_prefix}_ice_probability_smoothed.tif"
                    exporter = GeoTIFFExporter.from_paths(reference_paths)
                    exporter.export_ice_probability(ice_probability, smoothed_path)
                    output_files["ice_probability_smoothed"] = smoothed_path
                except Exception as exc:
                    logger.warning(f"Could not write smoothed GeoTIFF: {exc}")
            except Exception as exc:
                logger.warning(f"Post-processing step failed, returning raw outputs: {exc}")

        logger.info("Pipeline complete!")

        return ice_probability, indicators, output_files

    def close(self) -> None:
        """Clean up resources."""
        if self.data_ingestion is not None:
            self.data_ingestion.close()


def create_pipeline(
    data_dir: Path,
    output_dir: Path,
    device: Optional[str] = None,
    **kwargs
) -> LunarIceDetectionPipeline:
    """
    Factory function to create a configured pipeline.

    Args:
        data_dir: Input data directory
        output_dir: Output directory
        device: Device for model
        **kwargs: Additional configuration overrides

    Returns:
        Configured pipeline instance
    """
    config = PipelineConfig(
        data_dir=data_dir,
        output_dir=output_dir,
        **kwargs
    )

    return LunarIceDetectionPipeline(config, device)
