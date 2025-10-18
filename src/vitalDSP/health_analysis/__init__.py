"""
Health Analysis Module for Physiological Signal Processing

This module provides comprehensive health analysis capabilities for physiological
signals including ECG, PPG, EEG, and other vital signs. It implements automated
health report generation, interpretation engines, and visualization tools for
clinical and research applications.

Author: vitalDSP Team
Date: 2025-01-27
Version: 1.0.0

Key Features:
- Automated health report generation
- Clinical interpretation engines
- Interactive health report visualization
- HTML report templates and rendering
- File I/O utilities for health data
- Multi-threaded processing capabilities
- Comprehensive health metrics analysis

Examples:
--------
Health report generation:
    >>> from vitalDSP.health_analysis import HealthReportGenerator
    >>> generator = HealthReportGenerator()
    >>> report = generator.generate_report(features, output_dir='./reports')

Health visualization:
    >>> from vitalDSP.health_analysis import HealthReportVisualizer
    >>> visualizer = HealthReportVisualizer(config)
    >>> plots = visualizer.create_visualizations(features)
"""

from .health_report_generator import HealthReportGenerator
from .health_report_visualization import HealthReportVisualizer
from .interpretation_engine import InterpretationEngine
from .html_template import render_report
from .file_io import FileIO

__all__ = [
    "HealthReportGenerator",
    "HealthReportVisualizer", 
    "InterpretationEngine",
    "render_report",
    "FileIO",
]
