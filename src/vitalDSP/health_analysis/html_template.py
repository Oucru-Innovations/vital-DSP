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
                body {
                    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    padding: 16px;
                    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                    color: #2c3e50;
                    line-height: 1.6;
                    margin: 0;
                }
                h1 {
                    text-align: center;
                    color: #2c3e50;
                    font-size: 2.5rem;
                    font-weight: 700;
                    margin: 0 0 24px 0;
                    text-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .filter-dropdown, .plot-dropdown {
                    margin-bottom: 20px;
                    text-align: center;
                }
                .filter-dropdown select, .plot-dropdown select {
                    padding: 10px;
                    font-size: 16px;
                    border-radius: 8px;
                    border: 1px solid #ccc;
                    outline: none;
                    background-color: #fff;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    transition: border-color 0.3s ease;
                }
                .filter-dropdown select:hover, .plot-dropdown select:hover {
                    border-color: #2980b9;
                }
                .filter-dropdown select:focus, .plot-dropdown select:focus {
                    border-color: #3498db;
                    box-shadow: 0 0 5px rgba(52, 152, 219, 0.5);
                }
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
                    margin-bottom: 0;
                }
                .feature-section {
                    padding: 20px;
                    background: rgba(255, 255, 255, 0.95);
                    backdrop-filter: blur(10px);
                    border: 1px solid rgba(255, 255, 255, 0.2);
                    border-radius: 16px;
                    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
                    margin-bottom: 16px;
                    transition: all 0.3s ease;
                    position: relative;
                    overflow: hidden;
                }
                .feature-section:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
                }
                .feature-title {
                    font-size: 1.25rem;
                    color: #2c3e50;
                    margin-bottom: 16px;
                    text-align: center;
                    font-weight: 600;
                    position: relative;
                    padding-bottom: 8px;
                }
                .feature-title::after {
                    content: '';
                    position: absolute;
                    bottom: 0;
                    left: 50%;
                    transform: translateX(-50%);
                    width: 40px;
                    height: 3px;
                    background: linear-gradient(90deg, #3498db, #2ecc71);
                    border-radius: 2px;
                }
                .column.feature-section {
                    width: 100%;
                    margin-bottom: 16px;
                }

                .grid-2-cols {
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 16px;
                    margin: 16px 0;
                }
                @media (max-width: 768px) {
                    .grid-2-cols {
                        grid-template-columns: 1fr;
                        gap: 12px;
                    }
                }
                .content-block {
                    padding: 16px;
                    margin: 12px 0;
                    border-radius: 12px;
                    background: rgba(255, 255, 255, 0.7);
                    backdrop-filter: blur(5px);
                }
                
                /* Dynamic Analysis Styles */
                .dynamic-analysis-container {
                    margin-bottom: 24px;
                    padding: 24px;
                    background: rgba(255, 255, 255, 0.95);
                    backdrop-filter: blur(10px);
                    border-radius: 20px;
                    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
                    border: 1px solid rgba(255, 255, 255, 0.2);
                }
                
                .executive-summary, .risk-assessment, .key-insights, .recommendations, .statistics-summary {
                    margin-bottom: 20px;
                    padding: 20px;
                    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                    border-radius: 16px;
                    border-left: 4px solid #3498db;
                    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05);
                    transition: all 0.3s ease;
                }
                .executive-summary:hover, .risk-assessment:hover, .key-insights:hover, .recommendations:hover, .statistics-summary:hover {
                    transform: translateY(-1px);
                    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1);
                }
                
                .summary-card, .risk-card, .insights-card, .recommendations-card {
                    padding: 20px;
                    border-radius: 12px;
                    margin-top: 12px;
                    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
                    transition: all 0.3s ease;
                }
                .summary-card:hover, .risk-card:hover, .insights-card:hover, .recommendations-card:hover {
                    transform: translateY(-1px);
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
                }
                
                .summary-card.excellent {
                    background-color: #d4edda;
                    border-left: 4px solid #28a745;
                }
                
                .summary-card.good {
                    background-color: #d1ecf1;
                    border-left: 4px solid #17a2b8;
                }
                
                .summary-card.fair {
                    background-color: #fff3cd;
                    border-left: 4px solid #ffc107;
                }
                
                .summary-card.poor {
                    background-color: #f8d7da;
                    border-left: 4px solid #dc3545;
                }
                
                .health-score {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 16px;
                    padding: 16px;
                    background: rgba(255, 255, 255, 0.8);
                    border-radius: 12px;
                    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
                }
                
                .score-value {
                    font-size: 2rem;
                    font-weight: 700;
                    color: #2c3e50;
                    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                }
                
                .risk-card.low {
                    background-color: #d4edda;
                    border-left: 4px solid #28a745;
                }
                
                .risk-card.moderate {
                    background-color: #fff3cd;
                    border-left: 4px solid #ffc107;
                }
                
                .risk-card.high {
                    background-color: #f8d7da;
                    border-left: 4px solid #dc3545;
                }
                
                .risk-level {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 16px;
                    padding: 16px;
                    background: rgba(255, 255, 255, 0.8);
                    border-radius: 12px;
                    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
                }
                
                .risk-value {
                    font-size: 1.25rem;
                    font-weight: 700;
                    text-transform: uppercase;
                    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                }
                
                .concerns ul, .insights-card ul, .recommendations-card ol {
                    margin: 12px 0;
                    padding-left: 24px;
                }
                
                .concerns li, .insights-card li, .recommendations-card li {
                    margin: 6px 0;
                    line-height: 1.6;
                    padding: 4px 0;
                }
                
                .stats-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
                    gap: 12px;
                    margin-top: 16px;
                }
                
                .stat-item {
                    text-align: center;
                    padding: 16px 12px;
                    background: rgba(255, 255, 255, 0.9);
                    border-radius: 12px;
                    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
                    transition: all 0.3s ease;
                }
                .stat-item:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
                }
                
                .stat-item.in-range {
                    border-top: 3px solid #28a745;
                }
                
                .stat-item.above-range {
                    border-top: 3px solid #ffc107;
                }
                
                .stat-item.below-range {
                    border-top: 3px solid #dc3545;
                }
                
                .stat-number {
                    display: block;
                    font-size: 1.5rem;
                    font-weight: 700;
                    color: #2c3e50;
                    margin-bottom: 4px;
                    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                }
                
                .stat-label {
                    font-size: 0.75rem;
                    color: #6c757d;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                    font-weight: 500;
                }
                
                /* Responsive Design */
                @media (max-width: 768px) {
                    body {
                        padding: 12px;
                    }
                    h1 {
                        font-size: 2rem;
                        margin-bottom: 20px;
                    }
                    .container {
                        grid-template-columns: 1fr;
                        gap: 12px;
                    }
                    .feature-section {
                        padding: 16px;
                    }
                    .dynamic-analysis-container {
                        padding: 16px;
                    }
                    .executive-summary, .risk-assessment, .key-insights, .recommendations, .statistics-summary {
                        padding: 16px;
                    }
                    .stats-grid {
                        grid-template-columns: repeat(2, 1fr);
                        gap: 8px;
                    }
                    .stat-item {
                        padding: 12px 8px;
                    }
                }
                
                @media (max-width: 480px) {
                    .stats-grid {
                        grid-template-columns: 1fr;
                    }
                    .health-score, .risk-level {
                        flex-direction: column;
                        text-align: center;
                        gap: 8px;
                    }
                }
                .description-block {
                    padding: 16px;
                    border-radius: 12px;
                    background: linear-gradient(135deg, #e8f5e8 0%, #f0f8f0 100%);
                    border-left: 4px solid #27ae60;
                    box-shadow: 0 2px 8px rgba(39, 174, 96, 0.1);
                }
                .interpretation-block {
                    padding: 16px;
                    border-radius: 12px;
                    background: linear-gradient(135deg, #fff2e8 0%, #ffe8d6 100%);
                    border-left: 4px solid #e67e22;
                    box-shadow: 0 2px 8px rgba(230, 126, 34, 0.1);
                }
                .contradiction-block {
                    padding: 16px;
                    border-radius: 12px;
                    background: linear-gradient(135deg, #fce8f1 0%, #f8d7da 100%);
                    border-left: 4px solid #e74c3c;
                    box-shadow: 0 2px 8px rgba(231, 76, 60, 0.1);
                }
                .correlation-block {
                    padding: 16px;
                    border-radius: 12px;
                    background: linear-gradient(135deg, #e8f4fc 0%, #e3f2fd 100%);
                    border-left: 4px solid #3498db;
                    box-shadow: 0 2px 8px rgba(52, 152, 219, 0.1);
                }
                .correlation-item, .contradiction-item {
                    margin-bottom: 10px;
                }
                .highlight {
                    font-weight: 600;
                    color: #2c3e50;
                    font-size: 0.9rem;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                    margin-bottom: 8px;
                    display: block;
                }
                .normal-range-bar {
                    background: linear-gradient(to right, #85C1E9, #f1948a);
                    height: 12px;
                    width: 100%;
                    border-radius: 5px;
                    position: relative;
                    margin-left: 20px;
                    margin-top: 10px;
                    flex-grow: 1;
                }
                .current-value-marker {
                    position: absolute;
                    top: -12px;
                    width: 0;
                    height: 0;
                    border-left: 6px solid transparent;
                    border-right: 6px solid transparent;
                    border-bottom: 12px solid #2c3e50;
                }
                .current-value-label {
                    position: absolute;
                    top: -25px;
                    font-size: 12px;
                    color: #34495e;
                }
                .visualization {
                    margin: 20px 0;
                    text-align: center;
                }
                .value-range-container {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 10px;
                }
                /* Styling for the content text blocks */
                .content-text {
                    font-size: 16px;
                    color: #2c3e50;
                    margin-right: 20px;
                }
                .grid-2-cols-modern {
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 20px;
                    align-items: center;
                }

                .content-block-modern {
                    padding: 15px;
                    background-color: #ffffff;
                    border-radius: 8px;
                    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
                }

                .value-summary {
                    display: flex;
                    justify-content: space-between;
                }

                .value-box {
                    text-align: center;
                }

                .value-box .label {
                    font-size: 14px;
                    color: #7f8c8d;
                    margin-bottom: 5px;
                }

                .value-box .value {
                    font-size: 18px;
                    color: #2c3e50;
                    font-weight: bold;
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
                    font-size: 14px;
                    color: #7f8c8d;
                }

                .normal-range-bar-modern {
                    position: relative;
                    height: 15px;
                    width: 100%;
                    background: linear-gradient(to right, #85C1E9, #f1948a);
                    border-radius: 7px;
                    margin: 0 10px;
                    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.1);
                }

                .current-value-marker-modern {
                    position: absolute;
                    top: -5px;
                    width: 10px;
                    height: 20px;
                    background-color: #2c3e50;
                    border-radius: 2px;
                    box-shadow: 0 0 5px rgba(0, 0, 0, 0.3);
                    transition: left 0.3s ease-in-out;
                }

                .current-value-label-modern {
                    position: absolute;
                    top: -30px;
                    font-size: 12px;
                    font-weight: bold;
                    color: #34495e;
                    transition: left 0.3s ease-in-out;
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
    """


def _get_correlation_contradiction_template():
    return """
        <!-- Correlation and Contradiction Grid Layout -->
                <div class="grid-2-cols">
                    <!-- Contradiction Section -->
                    {% if interpretation.get('contradiction_strength') %}
                    <div class="contradiction-block">
                        <p class="highlight">Contradiction:</p>
                        <div class="contradiction-content">
                            {% for related_feature, explanation in interpretation['contradiction'].items() %}
                            <p><strong>{{ related_feature }}: </strong></p>
                            <p>
                                {% if interpretation['contradiction_strength'] == 'Strong contradiction' %}
                                    🔴 Strong contradiction. {{ explanation }}
                                {% elif interpretation['contradiction_strength'] == 'Slight contradiction' %}
                                    🟡 Slight contradiction. {{ explanation }}
                                {% else %}
                                    🟢 No significant contradiction. {{ explanation }}
                                {% endif %}
                            </p>
                            {% endfor %}
                        </div>
                    </div>
                    {% endif %}

                    <!-- Correlation Section -->
                    {% if interpretation.get('correlation_strength') %}
                    <div class="correlation-block">
                        <p class="highlight">Correlation:</p>
                        <div class="correlation-content">
                            {% for related_feature, explanation in interpretation['correlation'].items() %}
                            <p><strong>{{ related_feature }}: </strong></p>
                            <p>
                                {% if interpretation['correlation_strength'] == 'Strong correlation' %}
                                    ✅ Strong correlation. {{ explanation }}
                                {% elif interpretation['correlation_strength'] == 'Slight correlation' %}
                                    ⚪ Slight correlation. {{ explanation }}
                                {% else %}
                                    ⭕ No significant correlation. {{ explanation }}
                                {% endif %}
                            </p>
                            {% endfor %}
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
                                <p class="value">{{ '{:.3f}'.format(interpretation.get('median', 'N/A')) }}</p>
                            </div>
                            <div class="value-box">
                                <p class="label">Standard Deviation</p>
                                <p class="value">{{ '{:.3f}'.format(interpretation.get('stddev', 0)) if interpretation.get('stddev') is not none else 'N/A' }}</p>
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
                                        <div class="current-value-label-modern" style="left: {{ left_position }}%;">{{ '{:.3f}'.format(value) }}</div>
                                    {% elif value < min_range %}
                                        <div class="current-value-marker-modern" style="left: 0%;"></div>
                                        <div class="current-value-label-modern" style="left: 0%;">{{ '{:.3f}'.format(value) }}</div>
                                    {% else %}
                                        <div class="current-value-marker-modern" style="left: 100%;"></div>
                                        <div class="current-value-label-modern" style="left: 100%;">{{ '{:.3f}'.format(value) }}</div>
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
                {% set plot_types = ['heatmap', 'bell_plot', 'radar_plot', 'violin_plot'] %}
                {% set default_plot = plot_types | random %}

                {% set plot_types_2 = ['plot_spectrogram', 'plot_spectral_density', 'line_with_rolling_stats','lag_plot'] %}
                {% set default_plot_2 = plot_types_2 | random %}

                <!-- 2-column grid for visualizations -->
                <div class="grid-2-cols">
                    <!-- First Column: Visualization for the current feature -->
                    <div class="visualization-block">
                        <div class="plot-dropdown">
                            <label for="plot_type_{{ feature }}">Choose Plot Type:</label>
                            <select id="plot_type_{{ feature }}" onchange="changePlot('{{ feature }}', this.value)">
                                <option value="heatmap" {% if default_plot == 'heatmap' %}selected{% endif %}>Heat Map</option>
                                <option value="bell_plot" {% if default_plot == 'bell_plot' %}selected{% endif %}>Bell Shape</option>
                                <option value="radar_plot" {% if default_plot == 'radar_plot' %}selected{% endif %}>Radar Plot</option>
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

                    <!-- Second Column: Dropdown to select related feature and show its plot -->
                    <div class="visualization-block">
                        <div class="plot-dropdown">
                            <label for="plot_type_{{ feature }}_second">Choose Plot Type:</label>
                            <select id="plot_type_{{ feature }}_second" onchange="changePlotSecondColumn('{{ feature }}', this.value)">
                                <option value="plot_spectrogram" {% if default_plot_2 == 'plot_spectrogram' %}selected{% endif %}>Spectrogram</option>
                                <option value="lag_plot" {% if default_plot_2 == 'lag_plot' %}selected{% endif %}>Lag Plot</option>
                                <option value="line_with_rolling_stats" {% if default_plot_2 == 'line_with_rolling_stats' %}selected{% endif %}>Enhanced Line Plot</option>
                                <option value="plot_spectral_density" {% if default_plot_2 == 'plot_spectral_density' %}selected{% endif %}>Spectral Density Plot</option>
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

                    <!--
                    <div class="visualization-block">
                        {% if interpretation.get('correlation') %}
                            <div class="plot-dropdown">
                                <label for="related_feature_dropdown_{{ feature }}">Compare with:</label>
                                <select id="related_feature_dropdown_{{ feature }}" onchange="changeRelatedFeaturePlot('{{ feature }}', this.value)">
                                    <option value="">Select a related feature</option>
                                    {% set first_related_feature = interpretation['correlation'].keys() | list | first %}
                                    {% for related_feature in interpretation['correlation'].keys() %}
                                        <option value="{{ related_feature }}" {% if related_feature == first_related_feature %}selected{% endif %}>
                                            {{ related_feature }}
                                        </option>
                                    {% endfor %}
                                </select>
                            </div>

                            <div id="related_visualization_{{ feature }}" class="visualization">
                                {% if visualizations.get(first_related_feature) and visualizations[first_related_feature]['bell_plot'] is not none %}
                                    <img id="related_feature_plot_{{ feature }}" src="{{ visualizations[first_related_feature]['bell_plot'] }}" alt="Related Feature Plot" style="max-width: 100%; height: auto;">
                                {% else %}
                                    <p>No related plot available for the selected feature.</p>
                                {% endif %}
                            </div>
                        {% endif %}
                    </div>
                    -->
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
    """Generate template for dynamic analysis sections."""
    return """
        {% if dynamic_analysis %}
        <!-- Dynamic Analysis Section -->
        <div class="dynamic-analysis-container">
            <!-- Executive Summary -->
            {% if dynamic_analysis.get('executive_summary') %}
            <div class="executive-summary">
                <h2>Executive Summary</h2>
                <div class="summary-card {{ dynamic_analysis.executive_summary.status }}">
                    <div class="health-score">
                        <span class="score-label">Overall Health Score:</span>
                        <span class="score-value">{{ "%.1f"|format(dynamic_analysis.overall_health_score) }}/100</span>
                    </div>
                    <p class="summary-text">{{ dynamic_analysis.executive_summary.summary }}</p>
                </div>
            </div>
            {% endif %}

            <!-- Risk Assessment -->
            {% if dynamic_analysis.get('risk_assessment') %}
            <div class="risk-assessment">
                <h2>Risk Assessment</h2>
                <div class="risk-card {{ dynamic_analysis.risk_assessment.level }}">
                    <div class="risk-level">
                        <span class="risk-label">Risk Level:</span>
                        <span class="risk-value">{{ dynamic_analysis.risk_assessment.level|title }}</span>
                    </div>
                    {% if dynamic_analysis.risk_assessment.concerns %}
                    <div class="concerns">
                        <h4>Areas of Concern:</h4>
                        <ul>
                            {% for concern in dynamic_analysis.risk_assessment.concerns %}
                            <li>{{ concern }}</li>
                            {% endfor %}
                        </ul>
                    </div>
                    {% endif %}
                    <div class="risk-recommendation">
                        <strong>Recommendation:</strong> {{ dynamic_analysis.risk_assessment.recommendation }}
                    </div>
                </div>
            </div>
            {% endif %}

            <!-- Key Insights -->
            {% if dynamic_analysis.get('key_insights') %}
            <div class="key-insights">
                <h2>Key Insights</h2>
                <div class="insights-card">
                    <ul>
                        {% for insight in dynamic_analysis.key_insights %}
                        <li>{{ insight }}</li>
                        {% endfor %}
                    </ul>
                </div>
            </div>
            {% endif %}

            <!-- Recommendations -->
            {% if dynamic_analysis.get('recommendations') %}
            <div class="recommendations">
                <h2>Recommendations</h2>
                <div class="recommendations-card">
                    <ol>
                        {% for recommendation in dynamic_analysis.recommendations %}
                        <li>{{ recommendation }}</li>
                        {% endfor %}
                    </ol>
                </div>
            </div>
            {% endif %}

            <!-- Statistics Summary -->
            {% if dynamic_analysis.get('statistics') %}
            <div class="statistics-summary">
                <h2>Analysis Summary</h2>
                <div class="stats-grid">
                    <div class="stat-item">
                        <span class="stat-number">{{ dynamic_analysis.statistics.total_features }}</span>
                        <span class="stat-label">Total Parameters</span>
                    </div>
                    <div class="stat-item in-range">
                        <span class="stat-number">{{ dynamic_analysis.statistics.in_range }}</span>
                        <span class="stat-label">In Range</span>
                    </div>
                    <div class="stat-item above-range">
                        <span class="stat-number">{{ dynamic_analysis.statistics.above_range }}</span>
                        <span class="stat-label">Above Range</span>
                    </div>
                    <div class="stat-item below-range">
                        <span class="stat-number">{{ dynamic_analysis.statistics.below_range }}</span>
                        <span class="stat-label">Below Range</span>
                    </div>
                </div>
            </div>
            {% endif %}
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
                const plotUrl = visualizations[relatedFeature]?.['bell_plot'];

                if (plotUrl) {
                    relatedImg.src = plotUrl;
                    relatedImg.style.display = 'block';
                } else {
                    relatedImg.style.display = 'none';  // Hide the image if plot not found
                    console.error(`Bell plot not found for related feature ${relatedFeature}`);
                }
            }
        </script>
    """


def render_report(feature_interpretations, visualizations, filter_status="all", dynamic_analysis=None):
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
    template = Template(html_template)
    rendered_html = template.render(
        feature_interpretations=feature_interpretations,
        visualizations=visualizations,
        filter_status=filter_status,
        dynamic_analysis=dynamic_analysis or {},
    )

    return rendered_html
