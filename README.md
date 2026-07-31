# Lunar Ice Detection Pipeline

AI pipeline for detecting subsurface water ice in the Permanently Shadowed Regions (PSRs) of the lunar south pole using multi-modal remote sensing data.

## Overview

This pipeline implements a Convolutional Variational Autoencoder (VAE) trained to reconstruct lunar terrain from co-registered remote sensing data. Regions with high reconstruction error combined with physics-based constraints (temperature, neutron suppression, radar CPR) indicate potential ice deposits.

## Directory Structure

```
lunar_ice_detection/
├── __init__.py              # Package initialization
├── data_ingestion.py        # Data loading and preprocessing
├── normalization.py         # Robust scaling normalization
├── model.py                 # VAE architecture and training
├── physics.py               # Physics-based ice constraints
├── ice_depth.py             # Ice depth estimation module
├── output.py                # GeoTIFF export functionality
├── pipeline.py              # Main pipeline orchestration
├── run_pipeline.py          # CLI entry point
├── visualization.py         # Streamlit visualization app
├── training_viz.py          # Training history plotting
├── test_pipeline.py         # Unit tests
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## New Features

### Uncertainty Quantification
Monte Carlo dropout provides confidence estimates for ice probability predictions. Enable with `use_uncertainty=True` in inference.

### Ice Depth Estimation
Radar-based ice thickness estimation using CPR and thermal models. Returns depth in meters and volume estimates.

### Interactive Visualization
Streamlit app for exploring results:
```bash
streamlit run visualization.py
```

### Training Visualization
Plot training history and model metrics with `training_viz.py`.

## Data Requirements

### Input Data Format

All input datasets should be:
- GeoTIFF format
- Reprojected to EPSG:30120 (Lunar South Pole Stereographic)
- Co-registered to the same spatial grid
- Aligned with matching dimensions

### Expected Datasets

| Band | Source | Physical Unit | Description |
|------|--------|---------------|-------------|
| `elevation` | LRO LOLA | meters | Lunar elevation |
| `slope` | LRO LOLA | degrees | Surface slope |
| `roughness` | LRO LOLA | meters | RMS surface roughness |
| `temp_max` | LRO Diviner | Kelvin | Maximum temperature |
| `temp_min` | LRO Diviner | Kelvin | Minimum temperature |
| `neutron_suppression` | LRO LEND | ratio | Neutron suppression factor |
| `radar_cpr` | Mini-RF | ratio | Circular polarization ratio |

## Installation

```bash
pip install -r requirements.txt

# For interactive visualization
pip install streamlit
```

## Usage

### Basic Training and Inference

```bash
python run_pipeline.py \
    --data-dir ./data \
    --output-dir ./output \
    --train \
    --epochs 100
```

### Inference with Pre-trained Model

```bash
python run_pipeline.py \
    --data-dir ./data \
    --output-dir ./output \
    --model ./output/model.pth
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--data-dir` | Input data directory | `./data` |
| `--output-dir` | Output directory | `./output` |
| `--train` | Train the model | - |
| `--model` | Path to pre-trained model | - |
| `--epochs` | Number of training epochs | 100 |
| `--batch-size` | Training batch size | 32 |
| `--patch-size` | VAE input patch size | 64 |
| `--device` | Device (cuda/cpu) | cuda |
| `--output-prefix` | Output filename prefix | lunar_ice |
| `--no-physics` | Disable physics constraints | False |

## Output Files

The pipeline generates the following outputs:

| File | Description |
|------|-------------|
| `lunar_ice_ice_probability.tif` | Final ice probability map (0-1) |
| `lunar_ice_temp_indicator.tif` | Temperature-based indicator |
| `lunar_ice_neutron_indicator.tif` | Neutron suppression indicator |
| `lunar_ice_radar_indicator.tif` | Radar CPR indicator |
| `lunar_ice_physics_combined.tif` | Combined physics indicators |
| `lunar_ice_reconstruction_error.tif` | VAE reconstruction error |
| `lunar_ice_uncertainty.tif` | Uncertainty map (if enabled) |
| `model.pth` | Trained model weights |

## Interactive Visualization

Launch the Streamlit visualization app:

```bash
streamlit run visualization.py
```

Features:
- Ice probability map with statistics
- Diagnostic indicator comparison
- Cross-section analysis
- Distribution histograms
- Map comparison tools

## Architecture

### VAE Model
- **Input**: 64x64 pixel patches with 7 channels
- **Encoder**: 4 convolutional layers (32 → 64 → 128 → 256 channels)
- **Latent**: 128-dimensional Gaussian distribution
- **Decoder**: 4 transposed convolutional layers

### Physics Constraints

The pipeline applies hard physics constraints based on:

1. **Temperature (T > 110K = 0 probability)**: Ice is only stable below 110K
2. **Neutron Suppression**: Hydrogen absorbs neutrons, reducing count
3. **Radar CPR**: Ice creates distinctive circular polarization ratio

### Anomaly Detection

Ice probability is computed as:

```
P(ice) = 0.6 × Physics_Score + 0.4 × VAE_Anomaly
```

Where:
- `Physics_Score = 0.4 × T_indicator + 0.35 × Neutron_indicator + 0.25 × Radar_indicator`
- `VAE_Anomaly = Normalized(Reconstruction_Error)`

### Ice Depth Estimation

Uses radar CPR and thermal models to estimate ice thickness:
- Higher CPR → shallower ice
- Lower temperature → more stable ice
- Output in meters with volume estimates

## Memory Efficiency

The pipeline uses:
- Windowed reading for large datasets (>4GB)
- Robust scaling (median/IQR) to handle extreme outliers
- Patch-based processing to limit memory usage

## Testing

Run the test suite:

```bash
pytest test_pipeline.py -v
```

## License

MIT License
