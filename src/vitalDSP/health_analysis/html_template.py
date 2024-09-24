from jinja2 import Template


def render_report(feature_interpretations, visualizations, filter_status="all"):
    """
    Renders the health report in HTML format using the given feature interpretations and visualizations.

    Args:
        feature_interpretations (dict): Dictionary containing interpreted feature results.
        visualizations (dict): Dictionary containing paths or embedded links to visualizations.
        filter_status (str): Filter for displaying specific features (in_range, below_range, above_range, or all).

    Returns:
        str: HTML report as a string.
    """
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                padding: 20px;
                background-color: #f7f9fa;
                color: #2c3e50;
            }
            h1 {
                text-align: center;
                color: #2c3e50;
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
                display: flex;
                flex-wrap: wrap;
                {#justify-content: space-between;#}
                justify-content: space-around;
                margin-top: 20px;
            }
            .column {
                width: 48%;
                margin-bottom: 20px;
            }
            .feature-section {
                padding: 15px;
                background-color: #ffffff;
                border: 1px solid #ddd;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                margin-bottom: 20px;
            }
            .feature-title {
                font-size: 18px;
                color: #2c3e50;
                margin-bottom: 10px;
                text-align: center;
            }
            .column.feature-section {
                width: 45%;
                margin-bottom: 20px;
            }

            .grid-2-cols {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
            }
            .content-block {
                padding: 15px;
                margin: 15px 0;
                border-radius: 5px;
            }
            .description-block {
                padding: 15px;
                border-radius: 8px;
                background-color: #d1e7dd;
                border-left: 5px solid #0f5132;
            }
            .interpretation-block {
                padding: 15px;
                border-radius: 8px;
                background-color: #ffe5d9;
                border-left: 5px solid #d12e2a;
            }
            .contradiction-block {
                padding: 15px;
                border-radius: 8px;
                background-color: #fce8f1;
                border-left: 5px solid #cc0056;
            }
            .correlation-block {
                padding: 15px;
                border-radius: 8px;
                background-color: #e8f4fc;
                border-left: 5px solid #1a73e8;
            }
            .correlation-item, .contradiction-item {
                margin-bottom: 10px;
            }
            .highlight {
                font-weight: bold;
                color: #2c3e50;
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
        <title>Health Report</title>
    </head>
    <body>
        <h1>Health Analysis Report</h1>

        <div class="filter-dropdown">
            <label for="filter">Filter Features:</label>
            <select id="filter" onchange="filterReport(this)">
                <option value="all" {% if filter_status == 'all' %}selected{% endif %}>All Features</option>
                <option value="in_range" {% if filter_status == 'in_range' %}selected{% endif %}>In Normal Range</option>
                <option value="above_range" {% if filter_status == 'above_range' %}selected{% endif %}>Above Normal Range</option>
                <option value="below_range" {% if filter_status == 'below_range' %}selected{% endif %}>Below Normal Range</option>
            </select>
        </div>

        <div class="container">
            {% for feature, interpretation in feature_interpretations.items() %}
            <div class="column feature-section" data-status="{{ interpretation.get('range_status', 'unknown') }}">
                <h2 class="feature-title">
                    {{ feature }}
                </h2>

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

                <!-- Correlation and Contradiction Grid Layout -->
                <div class="grid-2-cols">
                    {% if interpretation.get('correlation') %}
                    <div class="correlation-block">
                        <p class="highlight">Correlation:</p>
                        <div class="correlation-content">
                            {% for related_feature, explanation in interpretation['correlation'].items() %}
                            {% set correlation_strength = 'no' %}
                            {% set feature_value = interpretation.get('median', 0) %}
                            {% set related_value = feature_interpretations.get(related_feature, {}).get('median', 0) %}
                            {% set thresholds = interpretation.get('thresholds', {}).get('correlation', {}).get(related_feature, {}) %}
                            {% if thresholds %}
                                {% set strong_threshold = thresholds.get('strong', 0.1) %}
                                {% set slight_threshold = thresholds.get('slight', 0.2) %}
                                {% if abs(feature_value - related_value) < strong_threshold * feature_value %}
                                    {% set correlation_strength = 'strongly' %}
                                {% elif abs(feature_value - related_value) < slight_threshold * feature_value %}
                                    {% set correlation_strength = 'slightly' %}
                                {% else %}
                                    {% set correlation_strength = 'no' %}
                                {% endif %}
                            {% endif %}

                            <div class="correlation-item">
                                <p><strong>Related Feature: {{ related_feature }}</strong></p>
                                <p>
                                    {% if correlation_strength == 'strongly' %}
                                        ✅ Strongly correlates with {{ related_feature }}. {{ explanation }}
                                    {% elif correlation_strength == 'slightly' %}
                                        ✔️ Slightly correlates with {{ related_feature }}. {{ explanation }}
                                    {% else %}
                                        ⚡ No significant correlation with {{ related_feature }}. {{ explanation }}
                                    {% endif %}
                                </p>
                            </div>
                            {% endfor %}
                        </div>
                    </div>
                    {% endif %}

                    {% if interpretation.get('contradiction') %}
                    <div class="contradiction-block">
                        <p class="highlight">Contradiction:</p>
                        <div class="contradiction-content">
                            {% for related_feature, explanation in interpretation['contradiction'].items() %}
                            {% set contradiction_strength = 'no' %}
                            {% set feature_value = interpretation.get('median', 0) %}
                            {% set related_value = feature_interpretations.get(related_feature, {}).get('median', 0) %}
                            {% set thresholds = interpretation.get('thresholds', {}).get('contradiction', {}).get(related_feature, {}) %}
                            {% if thresholds %}
                                {% set strong_threshold = thresholds.get('strong', 0.3) %}
                                {% set slight_threshold = thresholds.get('slight', 0.1) %}
                                {% if abs(feature_value - related_value) > strong_threshold * feature_value %}
                                    {% set contradiction_strength = 'strongly' %}
                                {% elif abs(feature_value - related_value) > slight_threshold * feature_value %}
                                    {% set contradiction_strength = 'slightly' %}
                                {% else %}
                                    {% set contradiction_strength = 'no' %}
                                {% endif %}
                            {% endif %}
                            <div class="contradiction-item">
                                <p><strong>Related Feature: {{ related_feature }}</strong></p>
                                <p>
                                    {% if contradiction_strength == 'strongly' %}
                                        ⛔ Strongly contradicts {{ related_feature }}. {{ explanation }}
                                    {% elif contradiction_strength == 'slightly' %}
                                        ⚠️ Slightly contradicts {{ related_feature }}. {{ explanation }}
                                    {% else %}
                                        ⭕ No significant contradiction with {{ related_feature }}. {{ explanation }}
                                    {% endif %}
                                </p>
                            </div>
                            {% endfor %}
                        </div>
                    </div>
                    {% endif %}
                </div>

            {# Final conclusion section #}
            {#
            <div class="content-block final-conclusion-block" style="padding: 20px; margin-top: 30px; background-color: #e2eafc; border-radius: 10px;">
                <h3 style="color: #1c3f91;">Final Conclusion:</h3>
                <p>
                    {% if correlation_strength == 'strongly' %}
                        ✅ Strong correlation overall.
                    {% elif correlation_strength == 'slightly' %}
                        ✔️ Slight correlation overall.
                    {% else %}
                        ⚡ No significant correlation.
                    {% endif %}
                </p>

                <p>
                    {% if contradiction_strength == 'strongly' %}
                        ⛔ Strong contradiction overall. Immediate attention may be required.
                    {% elif contradiction_strength == 'slightly' %}
                        ⚠️ Slight contradiction overall. Monitor for potential issues.
                    {% else %}
                        ⭕ No significant contradiction.
                    {% endif %}
                </p>
            </div>
            #}

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
                                <img src="{{ visualizations[feature][default_plot] }}" alt="Visualization for {{ feature }}" width="400">
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
                                <img src="{{ visualizations[feature][default_plot_2] }}" alt="Visualization for {{ feature }} - {{ default_plot_2 }}" width="400">
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
                                    <img id="related_feature_plot_{{ feature }}" src="{{ visualizations[first_related_feature]['bell_plot'] }}" alt="Related Feature Plot" width="400">
                                {% else %}
                                    <p>No related plot available for the selected feature.</p>
                                {% endif %}
                            </div>
                        {% endif %}
                    </div>
                    -->
                </div>
            {% endif %}

            </div>
            {% endfor %}
        </div>

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
    </body>

    </html>
    """

    # Use Jinja2 template rendering
    template = Template(html_template)
    rendered_html = template.render(
        feature_interpretations=feature_interpretations,
        visualizations=visualizations,
        filter_status=filter_status,
    )

    return rendered_html
