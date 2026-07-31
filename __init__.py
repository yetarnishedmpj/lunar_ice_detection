"""
Lunar Ice Detection Pipeline
AI Pipeline for detecting subsurface water ice in Lunar PSRs
"""

from .postprocessing import (
    SmoothingConfig,
    gaussian_smooth,
    smooth_and_summarize,
    build_summary,
    write_json_report,
    write_html_report,
)

__version__ = "1.0.0"
