"""
Post-processing for lunar ice detection outputs.

Provides:
- Gaussian smoothing of patch-artifact-prone probability maps.
- Confident-extraction at user-tunable thresholds.
- A self-contained summary report (JSON + HTML) that bundles the head-line
  statistics and quality diagnostics for downstream consumers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt

try:
    # scipy is the gold standard for separable Gaussian smoothing. The module
    # tolerates its absence with a slower numpy fallback.
    from scipy.ndimage import gaussian_filter
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover - scipy is in requirements.txt
    _HAS_SCIPY = False


@dataclass
class SmoothingConfig:
    """Smoothing configuration."""

    sigma_pixels: float = 1.5          # Gaussian sigma in pixels
    preserve_range: bool = True        # Clip result back into [0, 1]
    treat_nan_as_invalid: bool = True  # NaN pixels are excluded from smoothing


def _gaussian_filter_nan(array: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing that ignores NaN by treating it as 0 + weight 0.

    For low-NaN inputs (typical of validated lunar rasters) the result is
    indistinguishable from ``scipy.ndimage.gaussian_filter`` with a NaN mask,
    but doesn't require the optional scipy dependency.
    """
    if sigma <= 0:
        return array.copy()

    a = array.astype(np.float64, copy=True)
    nan_mask = np.isnan(a)
    if nan_mask.any():
        filled = np.where(nan_mask, 0.0, a)
        weights = (~nan_mask).astype(np.float64)
    else:
        filled = a
        weights = np.ones_like(a)

    # Approximate the Gaussian with three separable 1-D passes using
    # repeated moving averages; cheap, dependency-free, and good enough for
    # the small sigma values used here.
    radius = max(1, int(np.ceil(3 * sigma)))
    kernel_size = 2 * radius + 1

    def box_blur(img: np.ndarray, k: int) -> np.ndarray:
        # Cumulative-trick box blur, separable into rows and columns.
        pad = k // 2
        padded = np.pad(img, pad, mode="edge")
        cs = padded.cumsum(axis=0)
        cs[k:] = cs[k:] - cs[:-k]
        out = cs[k - 1:-1] / k if k > 1 else cs[k - 1:-1].copy()
        cs = out.cumsum(axis=1)
        cs[:, k:] = cs[:, k:] - cs[:, :-k]
        out = cs[:, k - 1:-1] / k if k > 1 else cs[:, k - 1:-1].copy()
        return out

    # Approximate Gaussian by stacking 3 box blurs (the "box-blur trick").
    def gauss_approx(img: np.ndarray) -> np.ndarray:
        result = box_blur(img, kernel_size)
        result = box_blur(result, kernel_size)
        result = box_blur(result, kernel_size)
        return result

    smoothed_vals = gauss_approx(filled)
    smoothed_weights = gauss_approx(weights)
    out = np.where(
        smoothed_weights > 1e-6,
        smoothed_vals / smoothed_weights,
        np.nan if a.dtype.kind == "f" else 0,
    )
    return out.astype(array.dtype, copy=False)


def gaussian_smooth(
    ice_probability: npt.NDArray[np.float32],
    config: Optional[SmoothingConfig] = None,
) -> npt.NDArray[np.float32]:
    """Apply a small Gaussian blur to an ice probability map.

    Patch-based reconstruction-error inference introduces small artifacts at
    patch boundaries even with weighted accumulation. Light Gaussian
    smoothing removes those without materially shifting the underlying
    probability mass.

    Args:
        ice_probability: HxW probability map (values in [0, 1] or NaN).
        config: Smoothing configuration; defaults to a reasonable value.

    Returns:
        Smoothed probability map of the same shape and dtype.
    """
    cfg = config or SmoothingConfig()
    if cfg.sigma_pixels <= 0:
        return ice_probability.copy()

    if _HAS_SCIPY:
        nan_mask = np.isnan(ice_probability)
        smoothed = gaussian_filter(
            np.nan_to_num(ice_probability, nan=0.0),
            sigma=cfg.sigma_pixels,
            mode="nearest",
        )
        if cfg.treat_nan_as_invalid:
            # Renormalize by the smoothed valid-pixel weight so NaN inputs
            # don't bleed probability into their neighbours.
            weights = gaussian_filter(
                (~nan_mask).astype(np.float64),
                sigma=cfg.sigma_pixels,
                mode="constant",
                cval=0.0,
            )
            with np.errstate(invalid="ignore", divide="ignore"):
                smoothed = np.where(weights > 1e-6, smoothed / weights, np.nan)
        smoothed = smoothed.astype(np.float32)
    else:
        smoothed = _gaussian_filter_nan(ice_probability, cfg.sigma_pixels).astype(np.float32)

    if cfg.preserve_range:
        smoothed = np.clip(smoothed, 0.0, 1.0)
    return smoothed


def threshold_ice(
    ice_probability: npt.NDArray[np.float32],
    threshold: float = 0.5,
) -> npt.NDArray[np.bool_]:
    """Return a boolean mask of pixels considered "ice-bearing"."""
    return (ice_probability >= threshold) & np.isfinite(ice_probability)


def compute_region_stats(
    ice_probability: npt.NDArray[np.float32],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Headline statistics about the ice probability distribution."""
    valid = ice_probability[np.isfinite(ice_probability)]
    if valid.size == 0:
        return {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "fraction_above_threshold": 0.0,
            "valid_pixels": 0,
        }
    mask = threshold_ice(valid, threshold)
    return {
        "mean": float(valid.mean()),
        "median": float(np.median(valid)),
        "std": float(valid.std()),
        "min": float(valid.min()),
        "max": float(valid.max()),
        "fraction_above_threshold": float(mask.sum() / valid.size),
        "valid_pixels": int(valid.size),
    }


@dataclass
class DetectionSummary:
    """A single run's worth of detection statistics."""

    generated_at_utc: str
    valid_pixels: int
    probability_min: float
    probability_max: float
    probability_mean: float
    probability_median: float
    probability_std: float
    fraction_high_confidence: float
    threshold_used: float
    indicator_stats: Dict[str, Dict[str, float]]
    output_files: Dict[str, str]
    device: str
    smoothing_sigma_pixels: float


def build_summary(
    ice_probability: npt.NDArray[np.float32],
    indicators: Dict[str, npt.NDArray[np.float32]],
    output_files: Dict[str, Path],
    threshold: float = 0.5,
    device: str = "unknown",
    smoothing_sigma_pixels: float = 0.0,
) -> DetectionSummary:
    """Build a structured summary suitable for JSON serialization."""
    stat_for = lambda arr: compute_region_stats(arr, threshold=threshold)

    indicator_stats = {
        name: stat_for(arr) for name, arr in indicators.items()
        if isinstance(arr, np.ndarray) and arr.ndim == 2
    }

    summary = DetectionSummary(
        generated_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        valid_pixels=int(np.isfinite(ice_probability).sum()),
        probability_min=float(np.nanmin(ice_probability)) if np.isfinite(ice_probability).any() else 0.0,
        probability_max=float(np.nanmax(ice_probability)) if np.isfinite(ice_probability).any() else 0.0,
        probability_mean=float(np.nanmean(ice_probability)) if np.isfinite(ice_probability).any() else 0.0,
        probability_median=float(np.nanmedian(ice_probability)) if np.isfinite(ice_probability).any() else 0.0,
        probability_std=float(np.nanstd(ice_probability)) if np.isfinite(ice_probability).any() else 0.0,
        fraction_high_confidence=float(
            (threshold_ice(ice_probability, threshold).sum())
            / max(1, np.isfinite(ice_probability).sum())
        ),
        threshold_used=threshold,
        indicator_stats=indicator_stats,
        output_files={k: str(v) for k, v in output_files.items()},
        device=device,
        smoothing_sigma_pixels=smoothing_sigma_pixels,
    )
    return summary


def write_json_report(summary: DetectionSummary, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2, sort_keys=True)
    return output_path


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Lunar Ice Detection Report &mdash; {generated_at}</title>
<style>
  :root {{
    --bg: #0f1115; --fg: #e6e9ef; --muted: #99a0b0; --card: #1a1d24; --accent: #6db5ff;
  }}
  body {{ font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif;
         background: var(--bg); color: var(--fg); margin: 0; padding: 24px; }}
  h1 {{ margin: 0 0 8px; color: var(--accent); }}
  h2 {{ border-bottom: 1px solid #2a2e36; padding-bottom: 4px; margin-top: 32px; }}
  .meta {{ color: var(--muted); margin-bottom: 24px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
          gap: 12px; }}
  .card {{ background: var(--card); padding: 16px; border-radius: 8px;
          box-shadow: 0 1px 0 rgba(255,255,255,0.04); }}
  .card .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase;
                  letter-spacing: 0.06em; }}
  .card .value {{ font-size: 22px; font-weight: 600; margin-top: 6px; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
  th, td {{ padding: 6px 10px; text-align: left; border-bottom: 1px solid #2a2e36; }}
  th {{ color: var(--muted); font-weight: 500; }}
  code {{ background: #20242d; padding: 2px 6px; border-radius: 4px; font-size: 12px; }}
</style>
</head>
<body>
<h1>Lunar Ice Detection Report</h1>
<p class="meta">Generated {generated_at} &middot; device <code>{device}</code>
   &middot; smoothing σ = <code>{smoothing_sigma_pixels}</code> px</p>

<h2>Headline numbers</h2>
<div class="grid">
  <div class="card"><div class="label">Valid pixels</div>
       <div class="value">{valid_pixels:,}</div></div>
  <div class="card"><div class="label">P(ice) mean</div>
       <div class="value">{probability_mean:.4f}</div></div>
  <div class="card"><div class="label">P(ice) max</div>
       <div class="value">{probability_max:.4f}</div></div>
  <div class="card"><div class="label">Fraction &ge; {threshold:.2f}</div>
       <div class="value">{pct_high_confidence:.2f}%</div></div>
</div>

<h2>Indicator breakdown</h2>
<table>
<thead><tr><th>Indicator</th><th>Mean</th><th>Median</th>
       <th>Std</th><th>Min</th><th>Max</th>
       <th>Fraction &ge; {threshold:.2f}</th></tr></thead>
<tbody>{indicator_rows}</tbody>
</table>

<h2>Output files</h2>
<table>
<thead><tr><th>Name</th><th>Path</th></tr></thead>
<tbody>{output_file_rows}</tbody>
</table>

</body>
</html>"""


def write_html_report(summary: DetectionSummary, output_path: Path) -> Path:
    """Write a single-file, dark-themed HTML summary."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def fmt(v: float) -> str:
        return f"{v:.4f}" if v is not None else "—"

    rows = []
    for name, stats in summary.indicator_stats.items():
        rows.append(
            "<tr>"
            f"<td><code>{name}</code></td>"
            f"<td>{fmt(stats.get('mean', float('nan')))}</td>"
            f"<td>{fmt(stats.get('median', float('nan')))}</td>"
            f"<td>{fmt(stats.get('std', float('nan')))}</td>"
            f"<td>{fmt(stats.get('min', float('nan')))}</td>"
            f"<td>{fmt(stats.get('max', float('nan')))}</td>"
            f"<td>{stats.get('fraction_above_threshold', 0.0) * 100:.2f}%</td>"
            "</tr>"
        )
    indicator_rows = "\n".join(rows) if rows else "<tr><td colspan='7'>No indicators</td></tr>"

    of_rows = "\n".join(
        f"<tr><td><code>{name}</code></td><td><code>{path}</code></td></tr>"
        for name, path in summary.output_files.items()
    ) or "<tr><td colspan='2'>No output files recorded</td></tr>"

    html = HTML_TEMPLATE.format(
        generated_at=summary.generated_at_utc,
        device=summary.device,
        smoothing_sigma_pixels=summary.smoothing_sigma_pixels,
        valid_pixels=summary.valid_pixels,
        probability_mean=summary.probability_mean,
        probability_max=summary.probability_max,
        pct_high_confidence=summary.fraction_high_confidence * 100,
        threshold=summary.threshold_used,
        indicator_rows=indicator_rows,
        output_file_rows=of_rows,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path


def smooth_and_summarize(
    ice_probability: npt.NDArray[np.float32],
    indicators: Dict[str, npt.NDArray[np.float32]],
    output_files: Dict[str, Path],
    output_dir: Path,
    prefix: str = "lunar_ice",
    smoothing_sigma: float = 1.5,
    threshold: float = 0.5,
    device: str = "unknown",
) -> Tuple[npt.NDArray[np.float32], Dict[str, npt.NDArray[np.float32]], Dict[str, Path]]:
    """Convenience: smooth the ice probability map, rewrite indicators,
    and emit a JSON + HTML report under ``output_dir``.

    Returns:
        Tuple of (smoothed_ice_probability, new_indicators, extended_output_files).
    """
    smoothed = gaussian_smooth(
        ice_probability,
        config=SmoothingConfig(sigma_pixels=smoothing_sigma),
    )

    # Re-derive the indicators dict so the report reflects the smoothed
    # probability while keeping the raw anomaly maps available for the user.
    new_indicators = dict(indicators)
    new_indicators["ice_probability_smoothed"] = smoothed

    summary = build_summary(
        smoothed,
        new_indicators,
        output_files,
        threshold=threshold,
        device=device,
        smoothing_sigma_pixels=smoothing_sigma,
    )

    new_outputs = dict(output_files)
    new_outputs["report_json"] = write_json_report(
        summary, Path(output_dir) / f"{prefix}_report.json"
    )
    new_outputs["report_html"] = write_html_report(
        summary, Path(output_dir) / f"{prefix}_report.html"
    )

    return smoothed, new_indicators, new_outputs
