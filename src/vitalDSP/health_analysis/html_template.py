"""
Health Analysis Module for Physiological Signal Processing

This module provides comprehensive capabilities for physiological
signal processing including ECG, PPG, EEG, and other vital signs.

Author: vitalDSP Team
Date: 2025-01-27
Version: 1.0.0

Key Features:
- Object-oriented design with comprehensive classes
- Multiple processing methods and functions
- NumPy integration for numerical computations

Examples:
--------
Basic usage:
    >>> import numpy as np
    >>> from vitalDSP.health_analysis.html_template import HtmlTemplate
    >>> signal = np.random.randn(1000)
    >>> from vitalDSP.health_analysis.html_template import render_report
    >>> html = render_report(report_data={})
    >>> print(f'Processing result: {result}')
"""

from jinja2 import Template
import numpy as np


def calculate_correlation(feature_values, related_values, thresholds, chunk_size=5):
    counts = {"strongly": 0, "slightly": 0, "no": 0}

    total_chunks = len(feature_values) // chunk_size
    strong_threshold = thresholds.get("strong", 0.5)
    slight_threshold = thresholds.get("slight", 0.2)

    for i in range(total_chunks):
        start = i * chunk_size
        end = start + chunk_size
        current_feature_chunk = feature_values[start:end]
        current_related_chunk = related_values[start:end]

        if len(current_feature_chunk) == len(current_related_chunk):
            correlation_coefficient = np.corrcoef(
                current_feature_chunk, current_related_chunk
            )[0, 1]

            if correlation_coefficient >= strong_threshold:
                counts["strongly"] += 1
            elif correlation_coefficient >= slight_threshold:
                counts["slightly"] += 1
            else:
                counts["no"] += 1

    total_checked_chunks = sum(counts.values())
    if total_checked_chunks > 0:
        percentage_strongly = counts["strongly"] / total_checked_chunks
        percentage_slightly = counts["slightly"] / total_checked_chunks

        if percentage_strongly >= 0.6:
            return "Strong correlation"
        elif percentage_slightly >= 0.6:
            return "Slight correlation"

    return "No significant correlation"


def calculate_contradiction(feature_values, related_values, thresholds, chunk_size=5):
    counts = {"strongly": 0, "slightly": 0, "no": 0}

    total_chunks = len(feature_values) // chunk_size
    strong_threshold = thresholds.get("strong", 0.3)
    slight_threshold = thresholds.get("slight", 0.1)

    for i in range(total_chunks):
        start = i * chunk_size
        end = start + chunk_size
        current_feature_chunk = feature_values[start:end]
        current_related_chunk = related_values[start:end]

        if len(current_feature_chunk) == len(current_related_chunk):
            feature_low = min(current_feature_chunk)
            related_high = max(current_related_chunk)

            if feature_low < strong_threshold and related_high > strong_threshold:
                counts["strongly"] += 1
            elif feature_low < slight_threshold and related_high > slight_threshold:
                counts["slightly"] += 1
            else:
                counts["no"] += 1

    total_checked_chunks = sum(counts.values())
    if total_checked_chunks > 0:
        percentage_strongly = counts["strongly"] / total_checked_chunks
        percentage_slightly = counts["slightly"] / total_checked_chunks

        if percentage_strongly >= 0.6:
            return "Strong contradiction"
        elif percentage_slightly >= 0.6:
            return "Slight contradiction"

    return "No significant contradiction"


def process_interpretations(feature_interpretations):
    for feature, interpretation in feature_interpretations.items():
        # Handle Correlation
        if "correlation" in interpretation:
            for related_feature, explanation in interpretation["correlation"].items():
                feature_values = interpretation.get("value", [])
                related_values = feature_interpretations.get(related_feature, {}).get(
                    "value", []
                )
                thresholds = (
                    interpretation.get("thresholds", {})
                    .get("correlation", {})
                    .get(related_feature, {})
                )

                if feature_values and related_values:
                    correlation_strength = calculate_correlation(
                        feature_values, related_values, thresholds
                    )
                    interpretation["correlation_strength"] = correlation_strength

        # Handle Contradiction
        if "contradiction" in interpretation:
            for related_feature, explanation in interpretation["contradiction"].items():
                feature_values = interpretation.get("value", [])
                related_values = feature_interpretations.get(related_feature, {}).get(
                    "value", []
                )
                thresholds = (
                    interpretation.get("thresholds", {})
                    .get("contradiction", {})
                    .get(related_feature, {})
                )

                if feature_values and related_values:
                    contradiction_strength = calculate_contradiction(
                        feature_values, related_values, thresholds
                    )
                    interpretation["contradiction_strength"] = contradiction_strength

    return feature_interpretations


def _get_base_css():
    base_css = """
        <style>
            :root {
                --bg: #f7f8fa;
                --surface: #ffffff;
                --border: #e2e6ea;
                --accent: #3b6ef0;
                --accent-light: #eef2fd;
                --text: #1a202c;
                --text-muted: #6b7280;
                --green: #16a34a;
                --green-bg: #f0fdf4;
                --amber: #d97706;
                --amber-bg: #fffbeb;
                --red: #dc2626;
                --red-bg: #fef2f2;
                --blue: #2563eb;
                --blue-bg: #eff6ff;
                --shadow-sm: 0 1px 3px rgba(0,0,0,0.07);
                --shadow: 0 2px 8px rgba(0,0,0,0.08);
                --radius: 10px;
                --radius-sm: 6px;
            }

            body {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                padding: 24px 16px;
                background: var(--bg);
                color: var(--text);
                line-height: 1.6;
                margin: 0;
            }

            h1 {
                text-align: center;
                color: var(--text);
                font-size: 2rem;
                font-weight: 700;
                margin: 0 0 8px 0;
                letter-spacing: -0.5px;
            }

            h2 {
                font-size: 1.1rem;
                font-weight: 600;
                color: var(--text);
                margin: 0 0 12px 0;
            }

            /* Dropdowns */
            .filter-dropdown, .plot-dropdown {
                margin-bottom: 16px;
                text-align: center;
            }

            .filter-dropdown select, .plot-dropdown select {
                padding: 8px 12px;
                font-size: 14px;
                border-radius: var(--radius-sm);
                border: 1px solid var(--border);
                outline: none;
                background-color: var(--surface);
                color: var(--text);
                cursor: pointer;
                transition: border-color 0.15s;
            }

            .filter-dropdown select:hover, .plot-dropdown select:hover {
                border-color: var(--accent);
            }

            .filter-dropdown select:focus, .plot-dropdown select:focus {
                border-color: var(--accent);
                box-shadow: 0 0 0 3px rgba(59, 110, 240, 0.12);
            }

            /* Layout */
            .container {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                gap: 16px;
                margin-top: 20px;
                max-width: 1400px;
                margin-left: auto;
                margin-right: auto;
            }

            .column {
                width: 100%;
            }

            .grid-2-cols, .grid-2-cols-modern {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 16px;
                margin: 16px 0;
                align-items: start;
            }

            /* Cards */
            .feature-section {
                padding: 20px;
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: var(--radius);
                box-shadow: var(--shadow-sm);
                margin-bottom: 16px;
            }

            .feature-title {
                font-size: 1.1rem;
                color: var(--text);
                margin-bottom: 16px;
                text-align: center;
                font-weight: 600;
                padding-bottom: 10px;
                border-bottom: 2px solid var(--accent-light);
            }

            .content-block, .content-block-modern {
                padding: 14px;
                margin: 10px 0;
                border-radius: var(--radius-sm);
                background: var(--bg);
                border: 1px solid var(--border);
            }

            /* Dynamic analysis */
            .dynamic-analysis-container {
                margin-bottom: 20px;
                padding: 20px;
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: var(--radius);
                box-shadow: var(--shadow-sm);
            }

            .analysis-grid {
                display: grid;
                grid-template-columns: 2fr 1fr;
                gap: 16px;
                margin-bottom: 16px;
            }

            .main-analysis {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 12px;
            }

            .executive-summary {
                grid-column: 1 / -1;
                padding: 16px;
                background: var(--accent);
                color: #fff;
                border-radius: var(--radius-sm);
                margin-bottom: 12px;
            }

            .executive-summary h2 {
                color: #fff;
                margin-bottom: 6px;
            }

            .risk-assessment, .key-insights {
                padding: 14px;
                background: var(--surface);
                border-radius: var(--radius-sm);
                border: 1px solid var(--border);
                border-left: 3px solid var(--blue);
            }

            .recommendations, .statistics-summary {
                padding: 14px;
                background: var(--surface);
                border-radius: var(--radius-sm);
                border: 1px solid var(--border);
                border-left: 3px solid var(--amber);
            }

            /* Status cards */
            .summary-card, .risk-card, .insights-card, .recommendations-card {
                padding: 16px;
                border-radius: var(--radius-sm);
                margin-top: 12px;
                border: 1px solid var(--border);
                background: var(--surface);
            }

            .summary-card.excellent, .risk-card.low {
                background: var(--green-bg);
                border-left: 4px solid var(--green);
            }

            .summary-card.good {
                background: var(--blue-bg);
                border-left: 4px solid var(--blue);
            }

            .summary-card.fair, .risk-card.moderate {
                background: var(--amber-bg);
                border-left: 4px solid var(--amber);
            }

            .summary-card.poor, .risk-card.high {
                background: var(--red-bg);
                border-left: 4px solid var(--red);
            }

            .health-score, .risk-level {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 14px;
                padding: 14px;
                background: var(--bg);
                border-radius: var(--radius-sm);
                border: 1px solid var(--border);
            }

            .score-value {
                font-size: 1.75rem;
                font-weight: 700;
                color: var(--text);
            }

            .risk-value {
                font-size: 1.1rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }

            .concerns ul, .insights-card ul, .recommendations-card ol {
                margin: 10px 0;
                padding-left: 22px;
            }

            .concerns li, .insights-card li, .recommendations-card li {
                margin: 5px 0;
                line-height: 1.6;
            }

            /* Cross-correlations */
            .cross-correlations {
                margin-bottom: 24px;
                padding: 20px;
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: var(--radius);
                box-shadow: var(--shadow-sm);
            }

            .cross-correlations h2 {
                color: var(--text);
                margin-bottom: 16px;
            }

            .correlations-card {
                display: flex;
                flex-direction: column;
                gap: 12px;
            }

            .correlation-item {
                padding: 14px;
                border-radius: var(--radius-sm);
                background: var(--bg);
                border: 1px solid var(--border);
            }

            .correlation-item.strong {
                border-left: 4px solid var(--green);
                background: var(--green-bg);
            }

            .correlation-item.moderate {
                border-left: 4px solid var(--amber);
                background: var(--amber-bg);
            }

            .correlation-item.weak {
                border-left: 4px solid var(--red);
                background: var(--red-bg);
            }

            .correlation-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 8px;
                flex-wrap: wrap;
                gap: 8px;
            }

            .correlation-features {
                font-weight: 600;
                color: var(--text);
                font-size: 1rem;
            }

            .correlation-strength {
                padding: 3px 10px;
                border-radius: 20px;
                font-size: 0.8rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                border: 1px solid transparent;
            }

            .correlation-item.strong .correlation-strength {
                background: #dcfce7;
                color: var(--green);
                border-color: #bbf7d0;
            }

            .correlation-item.moderate .correlation-strength {
                background: #fef3c7;
                color: var(--amber);
                border-color: #fde68a;
            }

            .correlation-item.weak .correlation-strength {
                background: #fee2e2;
                color: var(--red);
                border-color: #fecaca;
            }

            .correlation-description {
                color: var(--text-muted);
                line-height: 1.6;
                margin: 0;
                font-size: 0.9rem;
            }

            /* Stats grid */
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
                gap: 10px;
                margin-top: 14px;
            }

            .stat-item {
                text-align: center;
                padding: 14px 10px;
                background: var(--surface);
                border-radius: var(--radius-sm);
                border: 1px solid var(--border);
                border-top: 3px solid var(--border);
            }

            .stat-item.in-range {
                border-top-color: var(--green);
            }

            .stat-item.above-range {
                border-top-color: var(--amber);
            }

            .stat-item.below-range {
                border-top-color: var(--red);
            }

            .stat-number {
                display: block;
                font-size: 1.4rem;
                font-weight: 700;
                color: var(--text);
                margin-bottom: 4px;
            }

            .stat-label {
                font-size: 0.72rem;
                color: var(--text-muted);
                text-transform: uppercase;
                letter-spacing: 0.5px;
                font-weight: 500;
            }

            /* Content blocks */
            .description-block {
                padding: 14px;
                border-radius: var(--radius-sm);
                background: var(--green-bg);
                border: 1px solid #bbf7d0;
                border-left: 4px solid var(--green);
            }

            .interpretation-block {
                padding: 14px;
                border-radius: var(--radius-sm);
                background: var(--amber-bg);
                border: 1px solid #fde68a;
                border-left: 4px solid var(--amber);
            }

            .contradiction-block {
                padding: 14px;
                border-radius: var(--radius-sm);
                background: var(--red-bg);
                border: 1px solid #fecaca;
                border-left: 4px solid var(--red);
            }

            .correlation-block {
                padding: 14px;
                border-radius: var(--radius-sm);
                background: var(--blue-bg);
                border: 1px solid #bfdbfe;
                border-left: 4px solid var(--blue);
            }

            .highlight {
                font-weight: 600;
                color: var(--text);
                font-size: 0.8rem;
                text-transform: uppercase;
                letter-spacing: 0.6px;
                margin-bottom: 6px;
                display: block;
            }

            /* Range bar */
            .normal-range-bar, .normal-range-bar-modern {
                position: relative;
                height: 10px;
                width: 100%;
                background: linear-gradient(to right, #93c5fd, #fca5a5);
                border-radius: 5px;
                flex-grow: 1;
                margin: 0 10px;
            }

            .normal-range-bar {
                margin-left: 20px;
                margin-top: 10px;
            }

            .current-value-marker {
                position: absolute;
                top: -10px;
                width: 0;
                height: 0;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-bottom: 10px solid var(--text);
            }

            .current-value-label {
                position: absolute;
                top: -22px;
                font-size: 11px;
                color: var(--text-muted);
            }

            .current-value-marker-modern {
                position: absolute;
                top: -4px;
                width: 8px;
                height: 18px;
                background-color: var(--text);
                border-radius: 2px;
                transition: left 0.2s ease;
            }

            .current-value-label-modern {
                position: absolute;
                top: -26px;
                font-size: 11px;
                font-weight: 600;
                color: var(--text);
                transition: left 0.2s ease;
            }

            /* Visualization */
            .visualization {
                margin: 16px 0;
                text-align: center;
            }

            .value-range-container {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 10px;
            }

            .content-text {
                font-size: 15px;
                color: var(--text);
                margin-right: 16px;
            }

            .value-summary {
                display: flex;
                justify-content: space-between;
            }

            .value-box {
                text-align: center;
            }

            .value-box .label {
                font-size: 13px;
                color: var(--text-muted);
                margin-bottom: 4px;
            }

            .value-box .value {
                font-size: 17px;
                color: var(--text);
                font-weight: 700;
            }

            .range-section {
                text-align: center;
            }

            .range-display {
                display: flex;
                align-items: center;
                justify-content: space-between;
            }

            .range-label {
                font-size: 13px;
                color: var(--text-muted);
                white-space: nowrap;
            }

            /* Responsive */
            @media (max-width: 768px) {
                body { padding: 12px; }

                h1 { font-size: 1.6rem; margin-bottom: 16px; }

                .container {
                    grid-template-columns: 1fr;
                    gap: 12px;
                }

                .feature-section, .dynamic-analysis-container { padding: 14px; }

                .grid-2-cols, .grid-2-cols-modern,
                .analysis-grid, .main-analysis {
                    grid-template-columns: 1fr;
                    gap: 10px;
                }

                .stats-grid {
                    grid-template-columns: repeat(2, 1fr);
                    gap: 8px;
                }

                .stat-item { padding: 10px 8px; }

                .dynamic-analysis-container > div:last-child {
                    grid-template-columns: 1fr !important;
                    gap: 8px !important;
                }
            }

            @media (max-width: 480px) {
                .stats-grid { grid-template-columns: 1fr; }
                .health-score, .risk-level {
                    flex-direction: column;
                    text-align: center;
                    gap: 8px;
                }
            }
        </style>
    """
    return base_css


def _get_filter_dropdown():
    return """
        <div class="filter-dropdown">
            <label for="filter">Filter Features:</label>
            <select id="filter" onchange="filterReport(this)">
                <option value="all" {% if filter_status == 'all' %}selected{% endif %}>All Features</option>
                <option value="in_range" {% if filter_status == 'in_range' %}selected{% endif %}>In Normal Range</option>
                <option value="above_range" {% if filter_status == 'above_range' %}selected{% endif %}>Above Normal Range</option>
                <option value="below_range" {% if filter_status == 'below_range' %}selected{% endif %}>Below Normal Range</option>
            </select>
        </div>
    """


def _get_description_interpretation_template():
    return """
    <!-- Combined Description and Interpretation -->
                <div class="grid-2-cols">
                    <div class="description-block">
                        <p class="highlight">Description:</p>
                        <p>{{ interpretation.get('description', 'Description not available') }}</p>
                    </div>

                    {% if interpretation.get('interpretation') %}
                    <div class="interpretation-block">
                        <p class="highlight">Interpretation:</p>
                        <p>{{ interpretation['interpretation'] }}</p>
                    </div>
                    {% endif %}
                </div>

                <!-- Clinical References Section -->
                {% if interpretation.get('references') %}
                <div class="correlation-block" style="margin-top: 12px;">
                    <p class="highlight">Clinical References:</p>
                    <ul style="margin: 0; padding-left: 20px; font-size: 0.9rem; line-height: 1.7;">
                        {% for reference in interpretation['references'] %}
                        <li style="margin-bottom: 6px;">{{ reference }}</li>
                        {% endfor %}
                    </ul>
                </div>
                {% endif %}

                <!-- Clinical Interpretation Section -->
                {% if interpretation.get('clinical_interpretation') %}
                <div class="interpretation-block" style="margin-top: 12px;">
                    <p class="highlight">Clinical Interpretation:</p>

                    {% if interpretation['clinical_interpretation'].get('pathophysiology') %}
                    <div style="margin-bottom: 10px;">
                        <strong style="font-size: 0.9rem;">Pathophysiology:</strong>
                        <p style="margin: 4px 0 0 0; font-size: 0.9rem; line-height: 1.6;">{{ interpretation['clinical_interpretation']['pathophysiology'] }}</p>
                    </div>
                    {% endif %}

                    {% if interpretation['clinical_interpretation'].get('clinical_significance') %}
                    <div style="margin-bottom: 10px;">
                        <strong style="font-size: 0.9rem;">Clinical Significance:</strong>
                        <p style="margin: 4px 0 0 0; font-size: 0.9rem; line-height: 1.6;">{{ interpretation['clinical_interpretation']['clinical_significance'] }}</p>
                    </div>
                    {% endif %}

                    {% if interpretation['clinical_interpretation'].get('age_factors') %}
                    <div style="margin-bottom: 0;">
                        <strong style="font-size: 0.9rem;">Age & Population Factors:</strong>
                        <p style="margin: 4px 0 0 0; font-size: 0.9rem; line-height: 1.6;">{{ interpretation['clinical_interpretation']['age_factors'] }}</p>
                    </div>
                    {% endif %}
                </div>
                {% endif %}
    """


def _get_correlation_contradiction_template():
    return """
        <!-- Correlation and Contradiction Grid Layout -->
                <div class="grid-2-cols">
                    <!-- Contradiction Section -->
                    {% if interpretation.get('contradiction') %}
                    <div class="contradiction-block">
                        <p class="highlight">Contradiction:</p>
                        <div class="contradiction-content">
                            <p>{{ interpretation['contradiction'] }}</p>
                        </div>
                    </div>
                    {% endif %}

                    <!-- Correlation Section -->
                    {% if interpretation.get('correlation') %}
                    <div class="correlation-block">
                        <p class="highlight">Correlation Analysis:</p>
                        <div class="correlation-content">
                            <p>{{ interpretation['correlation'] }}</p>
                        </div>
                    </div>
                    {% endif %}

                </div>
    """


def _get_range_interpretation_template():
    return """
        <!-- Stylish Summary of Values and Range Display -->
                <div class="grid-2-cols-modern">
                    <div class="content-block-modern">
                        <p class="highlight">Values</p>
                        <div class="value-summary">
                            <div class="value-box">
                                <p class="label">Median</p>
                                <p class="value">{{ '{:.3f}'.format(interpretation.get('median')) if interpretation.get('median') is not none and interpretation.get('median') != 'N/A' else 'N/A' }}</p>
                            </div>
                            <div class="value-box">
                                <p class="label">Standard Deviation</p>
                                <p class="value">{{ '{:.3f}'.format(interpretation.get('stddev')) if interpretation.get('stddev') is not none and interpretation.get('stddev') != 'N/A' else 'N/A' }}</p>
                            </div>
                        </div>
                    </div>

                    <!-- Modern Range Bar with Current Value Marker -->
                    <div class="content-block-modern range-section">
                        <p class="highlight">Range</p>
                        <div class="range-display">
                            <p class="range-label">{{ interpretation.get('normal_range', ['N/A', 'N/A'])[0] }}</p>
                            <div class="normal-range-bar-modern">
                                {% if interpretation.get('value') is not none and interpretation.get('normal_range') %}
                                    {% set value = interpretation.get('median') %}
                                    {% set min_range = interpretation.get('normal_range')[0] %}
                                    {% set max_range = interpretation.get('normal_range')[1] %}
                                    {% set left_position = ((value - min_range) / (max_range - min_range) * 100) %}
                                    {% if value >= min_range and value <= max_range %}
                                        <div class="current-value-marker-modern" style="left: {{ left_position }}%;"></div>
                                        <div class="current-value-label-modern" style="left: {{ left_position }}%;">{{ '{:.3f}'.format(value) if value is not none and value != 'N/A' else 'N/A' }}</div>
                                    {% elif value < min_range %}
                                        <div class="current-value-marker-modern" style="left: 0%;"></div>
                                        <div class="current-value-label-modern" style="left: 0%;">{{ '{:.3f}'.format(value) if value is not none and value != 'N/A' else 'N/A' }}</div>
                                    {% else %}
                                        <div class="current-value-marker-modern" style="left: 100%;"></div>
                                        <div class="current-value-label-modern" style="left: 100%;">{{ '{:.3f}'.format(value) if value is not none and value != 'N/A' else 'N/A' }}</div>
                                    {% endif %}
                                {% else %}
                                    <div class="current-value-marker-modern" style="left: 50%;"></div>
                                    <div class="current-value-label-modern" style="left: 50%;">N/A</div>
                                {% endif %}
                            </div>
                            <p class="range-label">{{ interpretation.get('normal_range', ['N/A', 'N/A'])[1] }}</p>
                        </div>
                    </div>
                </div>
    """


def _get_visualization_template():
    return """
        {# Visualizations section in a 2-column grid with a dropdown for related features #}

            {% if feature in visualizations and visualizations[feature] %}
                <!-- Define available plot types -->
                {% set plot_types = ['gauge_chart', 'violin_plot', 'plot_box_swarm'] %}
                {% set default_plot = plot_types | random %}

                {% set plot_types_2 = ['line_with_rolling_stats', 'trend_sparkline'] %}
                {% set default_plot_2 = plot_types_2 | random %}

                <!-- 2-column grid for visualizations -->
                <div class="grid-2-cols">
                    <!-- First Column: Visualization for the current feature -->
                    <div class="visualization-block">
                        <div class="plot-dropdown">
                            <label for="plot_type_{{ feature }}">Choose Plot Type:</label>
                            <select id="plot_type_{{ feature }}" onchange="changePlot('{{ feature }}', this.value)">
                                <option value="gauge_chart" {% if default_plot == 'gauge_chart' %}selected{% endif %}>Gauge Chart</option>
                                <option value="violin_plot" {% if default_plot == 'violin_plot' %}selected{% endif %}>Violin Plot</option>
                                <option value="plot_box_swarm" {% if default_plot == 'plot_box_swarm' %}selected{% endif %}>Box Swarm Plot</option>
                            </select>
                        </div>
                        <div id="visualization_{{ feature }}" class="visualization">
                            {% if visualizations[feature][default_plot] is not none %}
                                <img src="{{ visualizations[feature][default_plot] }}" alt="Visualization for {{ feature }}" style="max-width: 100%; height: auto;">
                            {% else %}
                                <p>No visualization available for this plot type.</p>
                            {% endif %}
                        </div>
                    </div>

                    <!-- Second Column: Dropdown to select plot type -->
                    <div class="visualization-block">
                        <div class="plot-dropdown">
                            <label for="plot_type_{{ feature }}_second">Choose Plot Type:</label>
                            <select id="plot_type_{{ feature }}_second" onchange="changePlotSecondColumn('{{ feature }}', this.value)">
                                <option value="line_with_rolling_stats" {% if default_plot_2 == 'line_with_rolling_stats' %}selected{% endif %}>Line with Rolling Stats</option>
                                <option value="trend_sparkline" {% if default_plot_2 == 'trend_sparkline' %}selected{% endif %}>Trend Sparkline</option>
                            </select>
                        </div>
                        <div id="visualization_{{ feature }}_second" class="visualization">
                            {% if visualizations[feature][default_plot_2] is not none %}
                                <img src="{{ visualizations[feature][default_plot_2] }}" alt="Visualization for {{ feature }} - {{ default_plot_2 }}" style="max-width: 100%; height: auto;">
                            {% else %}
                                <p>No visualization available for this plot type.</p>
                            {% endif %}
                        </div>
                    </div>

                    <!-- Correlation dropdown removed - now using dynamic string-based correlations -->
                </div>
            {% endif %}
    """


def _get_feature_template():
    return (
        """
        <div class="container">
            {% for feature, interpretation in feature_interpretations.items() %}
            <div class="column feature-section" data-status="{{ interpretation.get('range_status', 'unknown') }}">
                <h2 class="feature-title">
                    {{ feature }}
                </h2>
    """
        + _get_description_interpretation_template()
        + _get_range_interpretation_template()
        + _get_correlation_contradiction_template()
        + _get_visualization_template()
        + """
            </div>
            {% endfor %}
        </div>
    """
    )


def _get_dynamic_analysis_template():
    """Generate template for dynamic analysis sections - Compact & Elegant."""
    return """
        {% if dynamic_analysis %}
        <!-- Dynamic Analysis Section - Compact & Elegant -->
        <div class="dynamic-analysis-container">
            <!-- Executive Summary - Full Width -->
            <div class="executive-summary">
                <h2 style="margin: 0 0 12px 0; font-size: 1.4rem;">📊 Executive Summary</h2>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-size: 0.9rem; opacity: 0.9;">Overall Health Score:</span>
                    <span style="font-size: 1.8rem; font-weight: 700;">{{ "%.1f"|format(dynamic_analysis.overall_health_score) }}/100</span>
                </div>
                <p style="margin: 0; font-size: 0.95rem; line-height: 1.4; opacity: 0.95;">{{ dynamic_analysis.executive_summary.summary }}</p>
            </div>

            <!-- Main Analysis Grid -->
            <div class="analysis-grid">
                <!-- Left Column: Risk & Insights -->
                <div class="main-analysis">
                    <div class="risk-assessment">
                        <h3 style="margin: 0 0 8px 0; font-size: 1.1rem; color: #2c3e50;">⚠️ Risk Assessment</h3>
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <span style="font-size: 0.85rem; color: #7f8c8d;">Risk Level:</span>
                            <span style="font-size: 1rem; font-weight: 600; color: #e74c3c;">{{ dynamic_analysis.risk_assessment.level|title }}</span>
                        </div>
                        <div style="font-size: 0.85rem; color: #555;">
                            <strong>Recommendation:</strong> {{ dynamic_analysis.risk_assessment.recommendation }}
                        </div>
                    </div>

                    <div class="key-insights">
                        <h3 style="margin: 0 0 8px 0; font-size: 1.1rem; color: #2c3e50;">💡 Key Insights</h3>
                        <ul style="margin: 0; padding-left: 16px; font-size: 0.85rem; color: #555;">
                            {% for insight in dynamic_analysis.key_insights[:3] %}
                            <li style="margin-bottom: 4px;">{{ insight }}</li>
                            {% endfor %}
                        </ul>
                    </div>
                </div>

                <!-- Right Column: Statistics -->
                <div class="statistics-summary">
                    <h3 style="margin: 0 0 12px 0; font-size: 1.1rem; color: #2c3e50;">📈 Analysis Summary</h3>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
                        <div style="text-align: center; padding: 8px; background: rgba(255,255,255,0.7); border-radius: 6px;">
                            <div style="font-size: 1.2rem; font-weight: 700; color: #2c3e50;">{{ dynamic_analysis.statistics.total_features }}</div>
                            <div style="font-size: 0.7rem; color: #7f8c8d; text-transform: uppercase;">Total</div>
                        </div>
                        <div style="text-align: center; padding: 8px; background: rgba(40, 167, 69, 0.1); border-radius: 6px; border-top: 2px solid #28a745;">
                            <div style="font-size: 1.2rem; font-weight: 700; color: #28a745;">{{ dynamic_analysis.statistics.in_range }}</div>
                            <div style="font-size: 0.7rem; color: #7f8c8d; text-transform: uppercase;">In Range</div>
                        </div>
                        <div style="text-align: center; padding: 8px; background: rgba(255, 193, 7, 0.1); border-radius: 6px; border-top: 2px solid #ffc107;">
                            <div style="font-size: 1.2rem; font-weight: 700; color: #ffc107;">{{ dynamic_analysis.statistics.above_range }}</div>
                            <div style="font-size: 0.7rem; color: #7f8c8d; text-transform: uppercase;">Above</div>
                        </div>
                        <div style="text-align: center; padding: 8px; background: rgba(220, 53, 69, 0.1); border-radius: 6px; border-top: 2px solid #dc3545;">
                            <div style="font-size: 1.2rem; font-weight: 700; color: #dc3545;">{{ dynamic_analysis.statistics.below_range }}</div>
                            <div style="font-size: 0.7rem; color: #7f8c8d; text-transform: uppercase;">Below</div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Bottom Row: Correlations & Recommendations -->
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 12px;">
                <!-- Cross-Feature Correlations -->
                <div class="cross-correlations" style="margin: 0; padding: 12px; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 10px; border-left: 3px solid #9b59b6;">
                    <h3 style="margin: 0 0 8px 0; font-size: 1.1rem; color: #2c3e50;">🔗 Key Correlations</h3>
                    <div style="font-size: 0.85rem; color: #555;">
                        {% for correlation in dynamic_analysis.cross_correlations[:2] %}
                        <div style="margin-bottom: 6px; padding: 6px; background: rgba(255,255,255,0.7); border-radius: 4px;">
                            <strong>{{ correlation.features[0] }} ↔ {{ correlation.features[1] }}</strong><br>
                            <span style="font-size: 0.8rem;">{{ correlation.description[:80] }}{% if correlation.description|length > 80 %}...{% endif %}</span>
                        </div>
                        {% endfor %}
                    </div>
                </div>

                <!-- Recommendations -->
                <div class="recommendations" style="margin: 0; padding: 12px;">
                    <h3 style="margin: 0 0 8px 0; font-size: 1.1rem; color: #2c3e50;">💊 Recommendations</h3>
                    <ol style="margin: 0; padding-left: 16px; font-size: 0.85rem; color: #555;">
                        {% for recommendation in dynamic_analysis.recommendations[:3] %}
                        <li style="margin-bottom: 4px;">{{ recommendation }}</li>
                        {% endfor %}
                    </ol>
                </div>
            </div>
        </div>
        {% endif %}
    """


def _get_javascript_content():
    return """
        <script>
            const visualizations = {{ visualizations | tojson | safe }};
            function filterReport(select) {
                var filterValue = select.value;
                var featureSections = document.querySelectorAll('.feature-section');

                featureSections.forEach(function(section) {
                    if (filterValue === 'all') {
                        section.style.display = 'block';
                    } else {
                        if (section.getAttribute('data-status') === filterValue) {
                            section.style.display = 'block';
                        } else {
                            section.style.display = 'none';
                        }
                    }
                });
            }
            function changePlot(feature, plotType) {
                const img = document.getElementById(`visualization_${feature}`).querySelector('img');
                const plotUrl = visualizations[feature][plotType];  // Correctly access the plot path

                if (plotUrl) {
                    img.src = plotUrl;  // Set the correct image source based on the selected plot type
                } else {
                    img.src = '';  // Remove image if plot not found
                    img.alt = 'No visualization available for this plot type.';
                    console.error(`Plot type ${plotType} not found for feature ${feature}`);
                }
            }
            function changePlotSecondColumn(feature, plotType) {
                const img = document.getElementById(`visualization_${feature}_second`).querySelector('img');
                const plotUrl = visualizations[feature][plotType];  // Access the plot path for the second column

                if (plotUrl) {
                    img.src = plotUrl;  // Set the correct image source based on the selected plot type
                    img.alt = `Visualization for ${feature} - ${plotType}`; // Update alt text
                } else {
                    img.src = '';  // Remove image if plot not found
                    img.alt = 'No visualization available for this plot type.';
                    console.error(`Plot type ${plotType} not found for feature ${feature}`);
                }
            }

            function changeRelatedFeaturePlot(feature, relatedFeature) {
                const relatedImg = document.getElementById(`related_feature_plot_${feature}`);
                const plotUrl = visualizations[relatedFeature]?.['gauge_chart'];

                if (plotUrl) {
                    relatedImg.src = plotUrl;
                    relatedImg.style.display = 'block';
                } else {
                    relatedImg.style.display = 'none';
                    console.error(`Gauge chart not found for related feature ${relatedFeature}`);
                }
            }
        </script>
    """


def render_report(
    feature_interpretations, visualizations, filter_status="all", dynamic_analysis=None
):
    """
    Renders the health report in HTML format using the given feature interpretations and visualizations.

    Args:
        feature_interpretations (dict): Dictionary containing interpreted feature results.
        visualizations (dict): Dictionary containing paths or embedded links to visualizations.
        filter_status (str): Filter for displaying specific features (in_range, below_range, above_range, or all).
        dynamic_analysis (dict, optional): Dictionary containing dynamic analysis components.

    Returns:
        str: HTML report as a string.
    """
    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        {_get_base_css()}
        <title>Health Report</title>
    </head>
    <body>
        <h1>Health Analysis Report</h1>

        {_get_dynamic_analysis_template()}

        {_get_filter_dropdown()}

        {_get_feature_template()}

        {_get_javascript_content()}
    </body>

    </html>
    """

    # Use Jinja2 template rendering
    try:
        template = Template(html_template)
        rendered_html = template.render(
            feature_interpretations=feature_interpretations,
            visualizations=visualizations,
            filter_status=filter_status,
            dynamic_analysis=dynamic_analysis or {},
        )
        return rendered_html
    except Exception as e:
        # More specific error handling
        print(f"Template rendering error: {e}")
        print(f"Error type: {type(e)}")
        import traceback

        traceback.print_exc()

        # Return a simple error report
        return f"""
        <html>
        <body>
            <h1>Error Generating Report</h1>
            <p>Error: {e}</p>
            <p>Please check the console for more details.</p>
        </body>
        </html>
        """
