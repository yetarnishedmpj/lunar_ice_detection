#!/usr/bin/env python3
"""
Synthetic Data Generator for Lunar Ice Detection Pipeline.
Generates sample GeoTIFF files in a target directory to test training and inference.
"""

import os
import sys

# Isolate PROJ_DATA / PROJ_LIB to prevent conflicts with external PostGIS installations
for var in ["PROJ_LIB", "PROJ_DATA"]:
    if var in os.environ and "PostgreSQL" in os.environ[var]:
        del os.environ[var]

import site
# Find rasterio's bundled proj_data
import rasterio
try:
    from rasterio._env import get_proj_data
    proj_path = get_proj_data()
    if proj_path:
        os.environ["PROJ_DATA"] = proj_path
        os.environ["PROJ_LIB"] = proj_path
except Exception:
    pass

from pathlib import Path
import numpy as np
from rasterio.transform import from_origin

def generate_synthetic_dataset(data_dir: Path, width: int = 256, height: int = 256):
    """Generate dummy GeoTIFF files for lunar ice detection pipeline."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    transform = from_origin(0, 0, 100, 100)
    
    # Custom CRS string or EPSG to avoid external PROJ issues if needed
    crs = "EPSG:4326"  # Standard WGS84 fallback if 30120 fails, or EPSG:30120
    try:
        rasterio.crs.CRS.from_epsg(30120)
        crs = "EPSG:30120"
    except Exception:
        print("Falling back to WGS84 CRS for synthetic data due to local PROJ database versioning.")
        crs = "EPSG:4326"

    bands = {
        'lola_elevation.tif': (-4000.0, 4000.0, -9999.0, 'float32'),
        'lola_slope.tif': (0.0, 45.0, -9999.0, 'float32'),
        'lola_roughness.tif': (0.1, 15.0, -9999.0, 'float32'),
        'diviner_temp_max.tif': (80.0, 350.0, 0.0, 'float32'),
        'diviner_temp_min.tif': (40.0, 120.0, 0.0, 'float32'),
        'lend_neutron_suppression.tif': (0.1, 1.2, -1.0, 'float32'),
        'minirf_cpr.tif': (0.1, 1.5, -1.0, 'float32'),
    }
    
    np.random.seed(42)
    print(f"Generating synthetic datasets in {data_dir.resolve()}...")
    
    for filename, (val_min, val_max, nodata, dtype) in bands.items():
        filepath = data_dir / filename
        data = np.random.uniform(val_min, val_max, (height, width)).astype(np.float32)
        
        # Add a few nodata pixels
        mask = np.random.rand(height, width) < 0.01
        data[mask] = nodata
        
        with rasterio.open(
            filepath,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=dtype,
            crs=crs,
            transform=transform,
            nodata=nodata
        ) as dst:
            dst.write(data, 1)
        print(f"  Created: {filename}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic data for Lunar Ice Detection")
    parser.add_argument("--data-dir", type=Path, default=Path("./data"), help="Target data directory")
    parser.add_argument("--size", type=int, default=256, help="Raster width/height size in pixels")
    args = parser.parse_args()
    
    generate_synthetic_dataset(args.data_dir, width=args.size, height=args.size)
