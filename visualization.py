"""
Interactive Visualization Module for Lunar Ice Detection Results
Provides Streamlit-based visualization for exploring ice probability maps
"""

import streamlit as st
import numpy as np
import numpy.typing as npt
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import logging

try:
    import rasterio
    from rasterio.plot import show
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    st.warning("rasterio not available - some features may be limited")

logger = logging.getLogger(__name__)


def load_geotiff(path: Path) -> Tuple[npt.NDArray, dict]:
    """
    Load a GeoTIFF file and return its data and metadata.

    Args:
        path: Path to GeoTIFF file

    Returns:
        Tuple of (data array, metadata dict)
    """
    if not RASTERIO_AVAILABLE:
        raise ImportError("rasterio is required for loading GeoTIFF files")

    with rasterio.open(path) as src:
        data = src.read(1)
        metadata = {
            'crs': src.crs,
            'transform': src.transform,
            'bounds': src.bounds,
            'nodata': src.nodata,
            'dtype': src.dtypes[0]
        }

    return data, metadata


def load_results_directory(output_dir: Path) -> Dict[str, npt.NDArray]:
    """
    Load all output files from a results directory.

    Args:
        output_dir: Directory containing output GeoTIFFs

    Returns:
        Dictionary mapping output names to arrays
    """
    if not RASTERIO_AVAILABLE:
        raise ImportError("rasterio is required for loading results")

    results = {}

    # Expected output files
    output_files = {
        'ice_probability': 'lunar_ice_ice_probability.tif',
        'temperature_indicator': 'lunar_ice_temp_indicator.tif',
        'neutron_indicator': 'lunar_ice_neutron_indicator.tif',
        'radar_indicator': 'lunar_ice_radar_indicator.tif',
        'physics_combined': 'lunar_ice_physics_combined.tif',
        'reconstruction_error': 'lunar_ice_reconstruction_error.tif',
        'uncertainty': 'lunar_ice_uncertainty.tif'
    }

    for name, filename in output_files.items():
        filepath = output_dir / filename
        if filepath.exists():
            results[name], _ = load_geotiff(filepath)
            logger.info(f"Loaded {name} from {filename}")

    return results


def compute_statistics(results: Dict[str, npt.NDArray]) -> Dict:
    """
    Compute statistics for result arrays.

    Args:
        results: Dictionary of result arrays

    Returns:
        Dictionary of statistics
    """
    stats = {}

    for name, data in results.items():
        flat = data[np.isfinite(data)]
        if len(flat) > 0:
            stats[name] = {
                'min': float(np.min(flat)),
                'max': float(np.max(flat)),
                'mean': float(np.mean(flat)),
                'median': float(np.median(flat)),
                'std': float(np.std(flat)),
                'p95': float(np.percentile(flat, 95)),
                'p99': float(np.percentile(flat, 99))
            }

    return stats


def create_streamlit_app():
    """
    Create the Streamlit visualization application.

    This function sets up the UI but does not run it.
    Use: streamlit run visualization.py
    """
    st.set_page_config(
        page_title="Lunar Ice Detection Viewer",
        page_icon="🌙",
        layout="wide"
    )

    st.title("🌙 Lunar Ice Detection Results")
    st.markdown("""
    Interactive viewer for exploring ice probability maps and diagnostic indicators
    from the lunar south pole permanently shadowed regions.
    """)

    # Sidebar for data loading
    st.sidebar.header("Data Selection")

    # Option to load from directory or use demo data
    data_source = st.sidebar.radio(
        "Data Source",
        ["Load from Directory", "Use Demo Data"]
    )

    results = {}
    metadata = {}

    if data_source == "Load from Directory":
        output_dir = st.sidebar.text_input(
            "Output Directory",
            value="./output"
        )

        if Path(output_dir).exists():
            try:
                results = load_results_directory(Path(output_dir))
                st.sidebar.success(f"Loaded {len(results)} files")

                # Load metadata from ice probability
                if 'ice_probability' in results:
                    st.session_state['results'] = results
            except Exception as e:
                st.sidebar.error(f"Error loading data: {e}")
        else:
            st.sidebar.warning("Directory not found")

    else:
        # Generate demo data
        st.sidebar.info("Using generated demo data")
        results = generate_demo_data()
        st.session_state['results'] = results

    if not results:
        st.info("👈 Please select a data source to begin")
        return

    # Main visualization
    st.header("Ice Probability Map")

    if 'ice_probability' in results:
        col1, col2 = st.columns([3, 1])

        with col1:
            fig = plot_array(
                results['ice_probability'],
                title="Ice Probability",
                cmap="YlOrRd",
                vmin=0, vmax=1
            )
            st.pyplot(fig)

        with col2:
            st.subheader("Statistics")
            if 'ice_probability' in results:
                prob = results['ice_probability']
                valid = prob[np.isfinite(prob)]

                st.metric("Mean Probability", f"{np.mean(valid):.3f}")
                st.metric("Max Probability", f"{np.max(valid):.3f}")
                st.metric("High Confidence Ice", f"{(valid > 0.5).sum() / len(valid) * 100:.1f}%")

                if 'uncertainty' in results:
                    unc = results['uncertainty']
                    unc_valid = unc[np.isfinite(unc)]
                    st.metric("Mean Uncertainty", f"{np.mean(unc_valid):.3f}")

    # Diagnostic indicators
    st.header("Diagnostic Indicators")

    indicator_tabs = st.tabs([
        "Temperature",
        "Neutron",
        "Radar",
        "Physics Combined",
        "Reconstruction Error"
    ])

    indicators = {
        'temperature_indicator': ('Temperature Indicator', 'coolwarm', None),
        'neutron_indicator': ('Neutron Suppression', 'Blues', None),
        'radar_indicator': ('Radar CPR', 'viridis', None),
        'physics_combined': ('Physics Combined', 'plasma', None),
        'reconstruction_error': ('Reconstruction Error', 'magma', None)
    }

    for i, (key, (title, cmap, vrange)) in enumerate(indicators.items()):
        with indicator_tabs[i]:
            if key in results:
                fig = plot_array(
                    results[key],
                    title=title,
                    cmap=cmap,
                    vmin=vrange[0] if vrange else None,
                    vmax=vrange[1] if vrange else None
                )
                st.pyplot(fig)

                # Stats
                data = results[key]
                valid = data[np.isfinite(data)]
                st.caption(f"Range: [{np.min(valid):.3f}, {np.max(valid):.3f}]")

    # Cross-section analysis
    st.header("Cross-Section Analysis")

    if 'ice_probability' in results:
        col1, col2 = st.columns(2)

        with col1:
            row = st.slider("Select Row", 0, results['ice_probability'].shape[0] - 1, results['ice_probability'].shape[0] // 2)
            profile = results['ice_probability'][row, :]
            fig_profile = plot_profile(profile, title=f"Ice Probability at Row {row}")
            st.pyplot(fig_profile)

        with col2:
            col = st.slider("Select Column", 0, results['ice_probability'].shape[1] - 1, results['ice_probability'].shape[1] // 2)
            profile = results['ice_probability'][:, col]
            fig_profile = plot_profile(profile, title=f"Ice Probability at Column {col}")
            st.pyplot(fig_profile)

    # Histogram analysis
    st.header("Distribution Analysis")

    if 'ice_probability' in results:
        fig_hist = plot_histogram(results['ice_probability'])
        st.pyplot(fig_hist)

    # Comparison view
    st.header("Comparison View")

    compare_options = list(results.keys())
    if len(compare_options) >= 2:
        col1, col2 = st.columns(2)

        with col1:
            map1_name = st.selectbox("Select First Map", compare_options, index=0)
        with col2:
            map2_name = st.selectbox("Select Second Map", compare_options, index=min(1, len(compare_options) - 1))

        if map1_name and map2_name:
            fig_compare = plot_comparison(
                results[map1_name],
                results[map2_name],
                map1_name,
                map2_name
            )
            st.pyplot(fig_compare)

    # Download section
    st.header("Export")

    st.markdown("### Download Results")

    for name, data in results.items():
        if data is not None:
            csv = array_to_csv(data)
            st.download_button(
                label=f"Download {name} as CSV",
                data=csv,
                file_name=f"{name}.csv",
                mime="text/csv"
            )


def generate_demo_data() -> Dict[str, npt.NDArray]:
    """
    Generate demo data for visualization testing.

    Returns:
        Dictionary of demo arrays
    """
    np.random.seed(42)

    # Create a realistic-looking ice detection result
    size = 256

    # Create coordinates
    y, x = np.ogrid[:size, :size]
    center_y, center_x = size // 2, size // 2

    # Distance from center
    dist = np.sqrt((y - center_y)**2 + (x - center_x)**2)

    # Base ice probability (some cold regions)
    ice_prob = np.exp(-dist / 80) * 0.5

    # Add some cold patches (potential ice deposits)
    for _ in range(5):
        patch_y = np.random.randint(50, size - 50)
        patch_x = np.random.randint(50, size - 50)
        patch_dist = np.sqrt((y - patch_y)**2 + (x - patch_x)**2)
        ice_prob += np.exp(-patch_dist / 20) * 0.3

    # Add noise
    ice_prob += np.random.randn(size, size) * 0.1
    ice_prob = np.clip(ice_prob, 0, 1)

    # Temperature indicator (inverse of distance - cold in center)
    temp_indicator = 1 - np.exp(-dist / 60)
    temp_indicator = np.clip(temp_indicator + np.random.randn(size, size) * 0.05, 0, 1)

    # Neutron indicator (similar to ice probability)
    neutron_indicator = ice_prob * 0.8 + np.random.randn(size, size) * 0.1
    neutron_indicator = np.clip(neutron_indicator, 0, 1)

    # Radar indicator
    radar_indicator = ice_prob * 0.6 + np.random.randn(size, size) * 0.15
    radar_indicator = np.clip(radar_indicator, 0, 1)

    # Physics combined
    physics_combined = temp_indicator * 0.4 + neutron_indicator * 0.35 + radar_indicator * 0.25

    # Reconstruction error (anomaly)
    reconstruction_error = np.random.exponential(0.1, (size, size))
    reconstruction_error += ice_prob * 0.5

    # Uncertainty (higher where probability is mid-range)
    uncertainty = np.abs(ice_prob - 0.5) * 0.3 + np.random.randn(size, size) * 0.05
    uncertainty = np.clip(uncertainty, 0, 1)

    return {
        'ice_probability': ice_prob.astype(np.float32),
        'temperature_indicator': temp_indicator.astype(np.float32),
        'neutron_indicator': neutron_indicator.astype(np.float32),
        'radar_indicator': radar_indicator.astype(np.float32),
        'physics_combined': physics_combined.astype(np.float32),
        'reconstruction_error': reconstruction_error.astype(np.float32),
        'uncertainty': uncertainty.astype(np.float32)
    }


def plot_array(
    data: npt.NDArray,
    title: str = "",
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None
):
    """
    Plot a 2D array as an image.

    Args:
        data: 2D array to plot
        title: Plot title
        cmap: Colormap name
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap

    Returns:
        Matplotlib figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        st.error("matplotlib is required for plotting")
        return None

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Value", fontsize=12)

    return fig


def plot_profile(data: npt.NDArray, title: str = ""):
    """
    Plot a 1D profile.

    Args:
        data: 1D array to plot
        title: Plot title

    Returns:
        Matplotlib figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        st.error("matplotlib is required for plotting")
        return None

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(data, linewidth=1.5)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Position (pixels)")
    ax.set_ylabel("Ice Probability")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    return fig


def plot_histogram(data: npt.NDArray, bins: int = 50):
    """
    Plot histogram of data values.

    Args:
        data: Array to analyze
        bins: Number of histogram bins

    Returns:
        Matplotlib figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        st.error("matplotlib is required for plotting")
        return None

    flat = data[np.isfinite(data)]

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.hist(flat, bins=bins, edgecolor='black', alpha=0.7)
    ax.set_title("Value Distribution", fontsize=12)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.3)

    # Add statistics text
    stats_text = f"Mean: {np.mean(flat):.3f}\nStd: {np.std(flat):.3f}\nMedian: {np.median(flat):.3f}"
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    return fig


def plot_comparison(
    data1: npt.NDArray,
    data2: npt.NDArray,
    name1: str = "Map 1",
    name2: str = "Map 2"
):
    """
    Create a side-by-side comparison plot.

    Args:
        data1: First array
        data2: Second array
        name1: Name of first array
        name2: Name of second array

    Returns:
        Matplotlib figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        st.error("matplotlib is required for plotting")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    vmin = min(np.nanmin(data1), np.nanmin(data2))
    vmax = max(np.nanmax(data1), np.nanmax(data2))

    axes[0].imshow(data1, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
    axes[0].set_title(name1, fontsize=12)
    axes[0].axis('off')

    im = axes[1].imshow(data2, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
    axes[1].set_title(name2, fontsize=12)
    axes[1].axis('off')

    plt.colorbar(im, ax=axes, shrink=0.6, label="Value")

    return fig


def array_to_csv(data: npt.NDArray) -> str:
    """
    Convert array to CSV string.

    Args:
        data: Array to convert

    Returns:
        CSV string
    """
    lines = []
    for row in data:
        lines.append(",".join(f"{v:.6f}" if np.isfinite(v) else "NaN" for v in row))
    return "\n".join(lines)


def run_app():
    """Run the Streamlit application."""
    create_streamlit_app()


if __name__ == "__main__":
    run_app()
