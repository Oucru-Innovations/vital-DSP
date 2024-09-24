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
                justify-content: space-between;
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
            .content-block {
                padding: 15px;
                margin: 15px 0;
                border-radius: 5px;
            }
            .description-block {
                background-color: #d1e7dd;
                border-left: 5px solid #0f5132;
            }
            .interpretation-block {
                background-color: #ffe5d9;
                border-left: 5px solid #d12e2a;
            }
            .contradiction-block {
                background-color: #fce8f1;
                border-left: 5px solid #cc0056;
            }
            .correlation-block {
                background-color: #e8f4fc;
                border-left: 5px solid #1a73e8;
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

        </style>
        <title>Health Report</title>
    </head>
    <body>
        <h1>Health Analysis Report</h1>

        <div class="filter-dropdown">
            <label for="filter">Filter Features:</label>
            <select id="filter" onchange="filterReport(this)">
                <option value="all" {% if filter_status == 'all' or filter_status == interpretation['range_status'] %}selected{% endif %}>All Features</option>
                <option value="in_range" {% if filter_status == 'in_range' %}selected{% endif %}>In Normal Range</option>
                <option value="above_range" {% if filter_status == 'above_range' %}selected{% endif %}>Above Normal Range</option>
                <option value="below_range" {% if filter_status == 'below_range' %}selected{% endif %}>Below Normal Range</option>
            </select>
        </div>

        <div class="container">
            {% for feature, interpretation in feature_interpretations.items() %}
            <div class="column feature-section" data-status="{{ interpretation.get('range_status', 'unknown') }}">
                <h2 class="feature-title" style="
                    font-size: 24px;
                    color: #4a90e2;
                    background-color: #e3f2fd;
                    padding: 10px 20px;
                    border-radius: 8px;
                    text-align: center;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                ">
                    {{ feature }}
                </h2>

                <div class="content-block description-block" style="padding: 10px 15px; margin-top: 15px;">
                    <p class="highlight" style="margin-bottom: 15px;">Description:</p>
                    <p style="padding-left: 10px;">{{ interpretation.get('description', 'Description not available') }}</p>
                </div>

                {% if interpretation.get('interpretation') %}
                <div class="content-block interpretation-block" style="padding: 10px 15px;">
                    <p class="highlight" style="margin-bottom: 15px;">Interpretation:</p>
                    <p style="padding-left: 10px;">{{ interpretation['interpretation'] }}</p>
                </div>
                {% endif %}

                <div class="content-block">
                    <p class="highlight">Value: {{ interpretation.get('value', 'N/A') }}. <br>&nbsp;<br>
                    Normal Range: {{ interpretation.get('normal_range', ['N/A', 'N/A'])[0] }} to {{ interpretation.get('normal_range', ['N/A', 'N/A'])[1] }}</p>
                </div>

                <div class="content-block">
                    <div class="normal-range-bar">
                        {% if interpretation.get('value') is not none and interpretation.get('normal_range') %}
                            {% set value = interpretation['value'] %}
                            {% set min_range = interpretation['normal_range'][0] %}
                            {% set max_range = interpretation['normal_range'][1] %}
                            {% if value >= min_range and value <= max_range %}
                                <div class="current-value-marker" style="left: {{ (value - min_range) / (max_range - min_range) * 100 }}%;"></div>
                                <div class="current-value-label" style="left: {{ (value - min_range) / (max_range - min_range) * 100 }}%;">{{ value }}</div>
                            {% elif value < min_range %}
                                <div class="current-value-marker" style="left: 0%;"></div>
                                <div class="current-value-label" style="left: 0%;">{{ value }}</div>
                            {% else %}
                                <div class="current-value-marker" style="left: 100%;"></div>
                                <div class="current-value-label" style="left: 100%;">{{ value }}</div>
                            {% endif %}
                        {% else %}
                            <div class="current-value-marker" style="left: 50%;"></div>
                            <div class="current-value-label" style="left: 50%;">N/A</div>
                        {% endif %}
                    </div>
                </div>

                {% if interpretation.get('contradiction') %}
                    <div class="content-block contradiction-block" style="padding: 15px; margin-top: 20px;">
                        <p class="highlight" style="margin-bottom: 15px;">Contradiction:</p>
                        <div style="display: flex; flex-wrap: wrap;">
                            {% for related_feature, explanation in interpretation['contradiction'].items() %}
                            <div style="flex: 1 1 100%; margin-bottom: 10px; padding: 15px; background-color: #f8d7da; border-left: 5px solid #721c24; border-radius: 8px;">
                                <p style="font-weight: bold; margin-bottom: 5px;">Related Feature: {{ related_feature }}</p>
                                <p style="margin: 0;">{{ explanation }}</p>
                            </div>
                            {% endfor %}
                        </div>
                    </div>
                {% endif %}

                {% if interpretation.get('correlation') %}
                    <div class="content-block correlation-block" style="padding: 15px; margin-top: 20px;">
                        <p class="highlight" style="margin-bottom: 15px;">Correlation:</p>
                        <div style="display: flex; flex-wrap: wrap;">
                            {% for related_feature, explanation in interpretation['correlation'].items() %}
                            <div style="flex: 1 1 100%; margin-bottom: 10px; padding: 15px; background-color: #d1ecf1; border-left: 5px solid #0c5460; border-radius: 8px;">
                                <p style="font-weight: bold; margin-bottom: 5px;">Related Feature: {{ related_feature }}</p>
                                <p style="margin: 0;">{{ explanation }}</p>
                            </div>
                            {% endfor %}
                        </div>
                    </div>
                {% endif %}

                {% if feature in visualizations and visualizations[feature] %}
                    <!-- Define available plot types -->
                    {% set plot_types = ['heatmap', 'bell_plot', 'radar_plot', 'violin_plot'] %}
                    {% set default_plot = plot_types | random %}

                    <div class="plot-dropdown">
                        <label for="plot_type_{{ feature }}">Choose Plot Type:</label>
                        <select id="plot_type_{{ feature }}" onchange="changePlot('{{ feature }}', this.value)">
                            <option value="heatmap" {% if default_plot == 'heatmap' %}selected{% endif %}>Heat Map</option>
                            <option value="bell_plot" {% if default_plot == 'bell_plot' %}selected{% endif %}>Bell Shape</option>
                            <option value="radar_plot" {% if default_plot == 'radar_plot' %}selected{% endif %}>Radar Plot</option>
                            <option value="violin_plot" {% if default_plot == 'violin_plot' %}selected{% endif %}>Violin Plot</option>
                        </select>
                    </div>

                    <div id="visualization_{{ feature }}" class="visualization">
                        <img src="{{ visualizations[feature][default_plot] }}" alt="Visualization for {{ feature }}" width="400">
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
                    console.error(`Plot type ${plotType} not found for feature ${feature}`);
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
