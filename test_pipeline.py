"""
Unit Tests for Lunar Ice Detection Pipeline
Run with: pytest test_pipeline.py -v
"""

import pytest
import numpy as np
import torch
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

# Import pipeline modules
import sys
sys.path.insert(0, str(Path(__file__).parent))

from model import ConvVAE, VAELoss, LunarPatchDataset, create_vae_model
from physics import PhysicsConstraints, compute_ice_probability_simple
from ice_depth import (
    estimate_ice_depth_from_radar,
    estimate_ice_volume,
    compute_thermal_timescale,
    validate_depth_estimates,
    _cpr_to_penetration_depth,
    _thermal_stability_factor
)


class TestConvVAE:
    """Tests for the VAE model."""

    def test_vae_initialization(self):
        """Test VAE model initialization."""
        model = ConvVAE(
            input_channels=7,
            hidden_channels=[32, 64, 128, 256],
            latent_dim=128
        )

        assert model.input_channels == 7
        assert model.latent_dim == 128
        assert model.hidden_channels == [32, 64, 128, 256]

    def test_vae_forward_pass(self):
        """Test forward pass through VAE."""
        model = ConvVAE(input_channels=7, latent_dim=128)
        model.eval()

        # Create dummy input
        x = torch.randn(4, 7, 64, 64)

        # Forward pass
        reconstruction, mu, logvar = model(x)

        assert reconstruction.shape == x.shape
        assert mu.shape == (4, 128)
        assert logvar.shape == (4, 128)

    def test_vae_encode(self):
        """Test encoding."""
        model = ConvVAE(input_channels=7, latent_dim=128)
        model.eval()

        x = torch.randn(2, 7, 64, 64)
        mu, logvar = model.encode(x)

        assert mu.shape == (2, 128)
        assert logvar.shape == (2, 128)

    def test_vae_decode(self):
        """Test decoding."""
        model = ConvVAE(input_channels=7, latent_dim=128)
        model.eval()

        z = torch.randn(2, 128)
        reconstruction = model.decode(z)

        assert reconstruction.shape[0] == 2
        assert reconstruction.shape[1] == 7

    def test_vae_reparameterize(self):
        """Test reparameterization trick."""
        model = ConvVAE(input_channels=7, latent_dim=128)

        mu = torch.zeros(4, 128)
        logvar = torch.full((4, 128), -100.0)

        z = model.reparameterize(mu, logvar)

        assert z.shape == mu.shape
        # With zero variance, output should equal mean
        assert torch.allclose(z, mu, atol=1e-5)

    def test_anomaly_score(self):
        """Test anomaly score computation."""
        model = ConvVAE(input_channels=7, latent_dim=128)
        model.eval()

        x = torch.randn(2, 7, 64, 64)
        anomaly_score, uncertainty = model.get_anomaly_score(x)

        assert anomaly_score.shape == (2, 64, 64)
        assert uncertainty is None

    def test_anomaly_score_with_uncertainty(self):
        """Test anomaly score with MC dropout."""
        model = ConvVAE(input_channels=7, latent_dim=128)
        model.train()  # Need train mode for dropout

        x = torch.randn(2, 7, 64, 64)
        anomaly_score, uncertainty = model.get_anomaly_score(
            x, use_mc_dropout=True, mc_samples=5
        )

        assert anomaly_score.shape == (2, 64, 64)
        assert uncertainty is not None
        assert uncertainty.shape == (2, 64, 64)


class TestVAELoss:
    """Tests for VAE loss function."""

    def test_vae_loss_computation(self):
        """Test VAE loss calculation."""
        loss_fn = VAELoss(reconstruction_weight=1.0, kl_weight=0.1)

        x = torch.randn(4, 7, 64, 64)
        reconstruction = x + torch.randn_like(x) * 0.1  # Add noise
        mu = torch.randn(4, 128)
        logvar = torch.randn(4, 128)

        total_loss, recon_loss, kl_loss = loss_fn(reconstruction, x, mu, logvar)

        assert total_loss.item() > 0
        assert recon_loss.item() > 0
        assert kl_loss.item() > 0

    def test_kl_weight_zero(self):
        """Test with zero KL weight."""
        loss_fn = VAELoss(reconstruction_weight=1.0, kl_weight=0.0)

        x = torch.randn(2, 7, 64, 64)
        reconstruction = x
        mu = torch.zeros(2, 128)
        logvar = torch.zeros(2, 128)

        total_loss, recon_loss, kl_loss = loss_fn(reconstruction, x, mu, logvar)

        assert recon_loss.item() == 0
        assert kl_loss.item() == 0


class TestLunarPatchDataset:
    """Tests for patch dataset."""

    def test_dataset_initialization(self):
        """Test dataset creation."""
        features = np.random.randn(256, 256, 7).astype(np.float32)
        dataset = LunarPatchDataset(
            feature_tensor=features,
            patch_size=64,
            stride=32
        )

        assert len(dataset) > 0
        assert dataset.patch_size == 64

    def test_dataset_getitem(self):
        """Test getting patches."""
        features = np.random.randn(256, 256, 7).astype(np.float32)
        dataset = LunarPatchDataset(
            feature_tensor=features,
            patch_size=64,
            stride=64  # Non-overlapping
        )

        patch, (row, col) = dataset[0]

        assert patch.shape == (7, 64, 64)  # Channels first
        assert isinstance(row, int)
        assert isinstance(col, int)

    def test_dataset_with_valid_mask(self):
        """Test with valid mask."""
        features = np.random.randn(256, 256, 7).astype(np.float32)
        valid_mask = np.random.rand(256, 256) > 0.3

        dataset = LunarPatchDataset(
            feature_tensor=features,
            patch_size=64,
            stride=32,
            valid_mask=valid_mask
        )

        # Should have fewer patches due to mask filtering
        dataset_no_mask = LunarPatchDataset(
            feature_tensor=features,
            patch_size=64,
            stride=32
        )

        assert len(dataset) <= len(dataset_no_mask)


class TestPhysicsConstraints:
    """Tests for physics constraints."""

    def test_temperature_indicator(self):
        """Test temperature indicator calculation."""
        physics = PhysicsConstraints()

        temp_max = np.array([[100, 120, 80]], dtype=np.float32)
        temp_min = np.array([[90, 110, 70]], dtype=np.float32)

        indicator = physics._temperature_indicator(temp_max, temp_min)

        assert indicator.shape == temp_max.shape
        # Cold region (80K) should have high indicator
        assert indicator[0, 2] > indicator[0, 0]
        # Hot region (120K) should have zero indicator
        assert indicator[0, 1] == 0.0

    def test_neutron_indicator(self):
        """Test neutron indicator."""
        physics = PhysicsConstraints()

        neutron = np.array([[0.5, 0.9, 0.2]], dtype=np.float32)
        indicator = physics._neutron_indicator(neutron)

        assert indicator.shape == neutron.shape
        # Low suppression (0.2) = high hydrogen = high indicator
        assert indicator[0, 2] > indicator[0, 0]

    def test_radar_indicator(self):
        """Test radar CPR indicator."""
        physics = PhysicsConstraints()

        cpr = np.array([[0.05, 0.3, 0.5]], dtype=np.float32)
        indicator = physics._radar_indicator(cpr)

        assert indicator.shape == cpr.shape

    def test_compute_ice_probability(self):
        """Test full ice probability computation."""
        physics = PhysicsConstraints()

        # Create sample feature tensor
        features = np.random.rand(64, 64, 7).astype(np.float32)
        # Set temperature to cold values
        features[:, :, 3] = 80  # temp_max
        features[:, :, 4] = 70  # temp_min

        band_names = ['elevation', 'slope', 'roughness', 'temp_max',
                      'temp_min', 'neutron_suppression', 'radar_cpr']

        prob, indicators = physics.compute_ice_probability(features, band_names)

        assert prob.shape == (64, 64)
        assert 'temperature_indicator' in indicators
        assert 'neutron_indicator' in indicators
        assert 'radar_indicator' in indicators

    def test_simple_ice_probability(self):
        """Test simple ice probability function."""
        temp_min = np.array([80, 100, 120], dtype=np.float32)
        neutron = np.array([0.3, 0.5, 0.7], dtype=np.float32)

        prob = compute_ice_probability_simple(temp_min, neutron)

        assert prob.shape == temp_min.shape
        # Cold + low neutron = high probability
        assert prob[0] > prob[2]


class TestIceDepthEstimation:
    """Tests for ice depth estimation."""

    def test_cpr_to_penetration_depth(self):
        """Test CPR to depth conversion."""
        cpr = np.array([0.1, 0.3, 0.5, 0.7], dtype=np.float32)
        wavelength = 0.12  # 12 cm

        depth = _cpr_to_penetration_depth(cpr, wavelength)

        assert depth.shape == cpr.shape
        # Higher CPR = shallower depth
        assert depth[0] > depth[-1]

    def test_thermal_stability_factor(self):
        """Test thermal stability calculation."""
        temp = np.array([50, 80, 100, 120, 150], dtype=np.float32)

        stability = _thermal_stability_factor(temp)

        assert stability.shape == temp.shape
        # Cold = stable (1.0), hot = unstable (0.0)
        assert stability[0] > stability[-1]
        assert stability[3] < 0.5  # Above threshold

    def test_estimate_ice_depth(self):
        """Test ice depth estimation."""
        # Create test data
        radar_cpr = np.random.rand(64, 64).astype(np.float32) * 0.5
        temp_min = np.random.rand(64, 64).astype(np.float32) * 50 + 60  # 60-110K
        ice_prob = np.random.rand(64, 64).astype(np.float32)

        depth, diagnostics = estimate_ice_depth_from_radar(
            radar_cpr, temp_min, ice_prob
        )

        assert depth.shape == radar_cpr.shape
        assert 'penetration_depth' in diagnostics
        assert 'thermal_stability' in diagnostics
        assert 'confidence_mask' in diagnostics

    def test_estimate_ice_volume(self):
        """Test ice volume estimation."""
        depth = np.random.rand(100, 100).astype(np.float32) * 2  # 0-2m
        probability = np.random.rand(100, 100).astype(np.float32)

        volume = estimate_ice_volume(depth, probability, pixel_size=20)

        assert 'total_volume_m3' in volume
        assert 'total_mass_kg' in volume
        assert volume['valid_pixels'] > 0
        assert volume['mean_depth_m'] > 0

    def test_thermal_timescale(self):
        """Test thermal timescale calculation."""
        temp = np.array([50, 80, 100, 120], dtype=np.float32)
        depth = np.array([0.1, 0.5, 1.0, 2.0], dtype=np.float32)

        timescale = compute_thermal_timescale(temp, depth)

        assert timescale.shape == temp.shape
        # Deeper = longer timescale
        assert timescale[-1] > timescale[0]

    def test_validate_depth_estimates(self):
        """Test depth validation."""
        depth = np.random.rand(64, 64).astype(np.float32) * 3
        cpr = np.random.rand(64, 64).astype(np.float32) * 0.5
        temp = np.random.rand(64, 64).astype(np.float32) * 50 + 60

        validation = validate_depth_estimates(depth, cpr, temp)

        assert 'valid_depth_pixels' in validation
        assert 'mean_depth_m' in validation
        assert validation['mean_depth_m'] > 0


class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_model_training_step(self):
        """Test a single training step."""
        # Create model
        model, loss_fn = create_vae_model(input_channels=7, device='cpu')

        # Create dummy batch
        batch = torch.randn(8, 7, 64, 64)

        # Forward pass
        reconstruction, mu, logvar = model(batch)

        # Compute loss
        loss, recon_loss, kl_loss = loss_fn(reconstruction, batch, mu, logvar)

        # Backward pass
        loss.backward()

        assert loss.item() > 0

    def test_physics_with_realistic_data(self):
        """Test physics with realistic lunar data ranges."""
        physics = PhysicsConstraints()

        # Create realistic feature tensor
        features = np.zeros((100, 100, 7), dtype=np.float32)
        features[:, :, 0] = 1000 + np.random.randn(100, 100) * 500  # elevation
        features[:, :, 1] = np.random.rand(100, 100) * 30  # slope
        features[:, :, 2] = np.random.rand(100, 100) * 10  # roughness
        features[:, :, 3] = 70 + np.random.rand(100, 100) * 40  # temp_max: 70-110K
        features[:, :, 4] = 50 + np.random.rand(100, 100) * 30  # temp_min: 50-80K
        features[:, :, 5] = np.random.rand(100, 100)  # neutron suppression
        features[:, :, 6] = np.random.rand(100, 100) * 0.5  # radar CPR

        band_names = ['elevation', 'slope', 'roughness', 'temp_max',
                      'temp_min', 'neutron_suppression', 'radar_cpr']

        # Add some reconstruction error
        recon_error = np.random.rand(100, 100).astype(np.float32) * 0.5

        prob, indicators = physics.compute_ice_probability(
            features, band_names, recon_error
        )

        # Should have valid probability map
        assert prob.shape == (100, 100)
        assert np.all(prob >= 0)
        assert np.all(prob <= 1)

        # High temp regions should have zero probability
        hot_mask = features[:, :, 3] > 110
        assert np.all(prob[hot_mask] == 0)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_vae_with_different_patch_sizes(self):
        """Test VAE with different input sizes."""
        for patch_size in [32, 64, 128]:
            model = ConvVAE(input_channels=7, hidden_channels=[32, 64, 128], latent_dim=64)
            x = torch.randn(2, 7, patch_size, patch_size)

            reconstruction, mu, logvar = model(x)
            assert reconstruction.shape == x.shape

    def test_physics_with_nan(self):
        """Test physics handling NaN values."""
        physics = PhysicsConstraints()

        features = np.random.rand(50, 50, 7).astype(np.float32)
        features[10:20, 10:20, 3] = np.nan  # Inject NaN in temp_max
        features[10:20, 10:20, 4] = np.nan  # Inject NaN in temp_min

        band_names = ['elevation', 'slope', 'roughness', 'temp_max',
                      'temp_min', 'neutron_suppression', 'radar_cpr']

        # Should handle NaN gracefully
        prob, indicators = physics.compute_ice_probability(features, band_names)

        # NaN regions should remain NaN or be handled
        assert prob.shape == features.shape[:2]

    def test_depth_with_zero_probability(self):
        """Test depth estimation with zero ice probability."""
        radar_cpr = np.ones((50, 50), dtype=np.float32) * 0.3
        temp_min = np.ones((50, 50), dtype=np.float32) * 80
        ice_prob = np.zeros((50, 50), dtype=np.float32)  # No ice

        depth, _ = estimate_ice_depth_from_radar(
            radar_cpr, temp_min, ice_prob
        )

        # Should return zeros when probability is zero
        assert np.all(depth == 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
