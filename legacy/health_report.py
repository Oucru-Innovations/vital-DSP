from IPython.display import HTML, display, display_html
import os
import pandas as pd
import numpy as np
from vitalDSP.health_analysis.health_report_generator import HealthReportGenerator

np.random.seed(42)
duration = 60
fs = 100
# Generate mock feature data
feature_data = {
        "sdnn": np.random.normal(
            50, 10, duration
        ).tolist(),  # Normally distributed around 50 with std dev of 10
        "rmssd": np.random.normal(
            30, 8, duration
        ).tolist(),  # Normally distributed around 30 with std dev of 8
        "total_power": np.random.normal(
            1600, 300, duration
        ).tolist(),  # Normally distributed around 1600 with std dev of 300
        "lfnu_power": np.random.normal(
            45, 5, duration
        ).tolist(),  # Normally distributed around 45 with std dev of 5
        "hfnu_power": np.random.normal(
            35, 4, duration
        ).tolist(),  # Normally distributed around 35 with std dev of 4
        "fractal_dimension": np.random.normal(
            1.25, 0.05, duration
        ).tolist(),  # Normally around 1.25 with std dev of 0.05
        "dfa": np.random.normal(
            1.2, 0.08, duration
        ).tolist(),  # Normally distributed around 1.2 with std dev of 0.08
        "poincare_sd1": np.random.normal(
            28.5, 3, duration
        ).tolist(),  # Normally around 28.5 with std dev of 3
        "poincare_sd2": np.random.normal(
            45, 4, duration
        ).tolist(),  # Normally around 45 with std dev of 4
        "sample_entropy": np.random.normal(
            0.85, 0.1, duration
        ).tolist(),  # Normally around 0.85 with std dev of 0.1
        "approximate_entropy": np.random.normal(
            0.72, 0.1, duration
        ).tolist(),  # Normally around 0.72 with std dev of 0.1
        "recurrence_rate": np.random.normal(
            0.65, 0.1, duration
        ).tolist(),  # Normally around 0.65 with std dev of 0.1
        "determinism": np.random.normal(
            0.78, 0.1, duration
        ).tolist(),  # Normally around 0.78 with std dev of 0.1
        "laminarity": np.random.normal(
            0.85, 0.05, duration
        ).tolist(),  # Normally around 0.85 with std dev of 0.05
        "systolic_duration": np.random.normal(
            0.35, 0.05, duration
        ).tolist(),  # Normally around 0.35 with std dev of 0.05
        "diastolic_duration": np.random.normal(
            0.5, 0.05, duration
        ).tolist(),  # Normally around 0.5 with std dev of 0.05
        "systolic_area": np.random.normal(
            200, 20, duration
        ).tolist(),  # Normally around 200 with std dev of 20
        "diastolic_area": np.random.normal(
            180, 20, duration
        ).tolist(),  # Normally around 180 with std dev of 20
        "systolic_slope": np.random.normal(
            1.5, 0.1, duration
        ).tolist(),  # Normally around 1.5 with std dev of 0.1
        "diastolic_slope": np.random.normal(
            1.2, 0.1, duration
        ).tolist(),  # Normally around 1.2 with std dev of 0.1
        "signal_skewness": np.random.normal(
            0.1, 0.02, duration
        ).tolist(),  # Normally around 0.1 with std dev of 0.02
        "peak_trend_slope": np.random.normal(
            0.05, 0.01, duration
        ).tolist(),  # Normally around 0.05 with std dev of 0.01
        "systolic_amplitude_variability": np.random.normal(
            5.0, 0.5, duration
        ).tolist(),  # Normally around 5.0 with std dev of 0.5
        "diastolic_amplitude_variability": np.random.normal(
            4.8, 0.5, duration
        ).tolist(),  # Normally around 4.8 with std dev of 0.5
        # "respiratory_rate": np.random.normal(16, 3, duration).tolist(),  # Normally around 16 breaths/min with std dev of 3
        # "pulse_pressure": np.random.normal(40, 5, duration).tolist(),  # Normally around 40 mmHg with std dev of 5
        # "stroke_volume": np.random.normal(70, 10, duration).tolist(),  # Normally around 70 mL with std dev of 10
        # "cardiac_output": np.random.normal(5.5, 0.7, duration).tolist(),  # Normally around 5.5 L/min with std dev of 0.7
        # "qt_interval": np.random.normal(400, 20, duration).tolist(),  # Normally around 400 ms with std dev of 20
        # "qrs_duration": np.random.normal(90, 10, duration).tolist(),  # Normally around 90 ms with std dev of 10
        # "p_wave_duration": np.random.normal(100, 10, duration).tolist(),  # Normally around 100 ms with std dev of 10
        # "pr_interval": np.random.normal(160, 20, duration).tolist(),  # Normally around 160 ms with std dev of 20
}

def main():
    """Main function to generate health report with proper multiprocessing support."""
    # Initialize the health report generator
    report_generator = HealthReportGenerator(
            feature_data=feature_data, segment_duration="1_min"
    )

    image_dir = "../_static/images"
    os.makedirs(image_dir, exist_ok=True)

    # Generate the report (HTML)
    report_html = report_generator.generate(output_dir=image_dir)
    
    # Write the report to file
    with open('health_report.html', 'w', encoding='utf-8') as file:
        file.write(report_html)
    
    print("Health report generated successfully!")


if __name__ == '__main__':
    main()