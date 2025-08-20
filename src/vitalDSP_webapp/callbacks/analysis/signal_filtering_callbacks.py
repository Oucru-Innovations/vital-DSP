"""
Signal filtering callbacks for vitalDSP webapp.
Handles traditional filtering, advanced filtering, artifact removal, neural filtering, and ensemble filtering.
Uses actual vitalDSP functions for all computations.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Input, Output, State, callback_context, no_update, html
from dash.exceptions import PreventUpdate
from scipy import signal
import dash_bootstrap_components as dbc
import logging

logger = logging.getLogger(__name__)


def register_signal_filtering_callbacks(app):
    """Register all signal filtering callbacks."""
    
    # Filter Type Selection Callback
    @app.callback(
        [Output("traditional-filter-params", "style", allow_duplicate=True),
         Output("advanced-filter-params", "style", allow_duplicate=True),
         Output("artifact-removal-params", "style", allow_duplicate=True),
         Output("neural-network-params", "style", allow_duplicate=True),
         Output("ensemble-params", "style", allow_duplicate=True)],
        [Input("filter-type-select", "value")],
        prevent_initial_call=True
    )
    def update_filter_parameter_visibility(filter_type):
        """Show/hide parameter sections based on selected filter type."""
        # Default style (hidden)
        hidden_style = {"display": "none"}
        visible_style = {"display": "block"}
        
        # Initialize all sections as hidden
        traditional_style = hidden_style
        advanced_style = hidden_style
        artifact_style = hidden_style
        neural_style = hidden_style
        ensemble_style = hidden_style
        
        # Show the appropriate section based on filter type
        if filter_type == "traditional":
            traditional_style = visible_style
        elif filter_type == "advanced":
            advanced_style = visible_style
        elif filter_type == "artifact":
            artifact_style = visible_style
        elif filter_type == "neural":
            neural_style = visible_style
        elif filter_type == "ensemble":
            ensemble_style = visible_style
        
        return traditional_style, advanced_style, artifact_style, neural_style, ensemble_style
    
    # Advanced Filtering Callback
    @app.callback(
        [Output("filter-original-plot", "figure"),
         Output("filter-filtered-plot", "figure"),
         Output("filter-comparison-plot", "figure"),
         Output("filter-quality-metrics", "children"),
         Output("filter-quality-plots", "figure"),
         Output("store-filtering-data", "data")],
        [Input("url", "pathname"),
         Input("filter-btn-apply", "n_clicks"),
         Input("filter-time-range-slider", "value"),
         Input("filter-btn-nudge-m10", "n_clicks"),
         Input("filter-btn-nudge-m1", "n_clicks"),
         Input("filter-btn-nudge-p1", "n_clicks"),
         Input("filter-btn-nudge-p10", "n_clicks")],
        [State("filter-start-time", "value"),
         State("filter-end-time", "value"),
         State("filter-type-select", "value"),
         State("filter-family-advanced", "value"),
         State("filter-response-advanced", "value"),
         State("filter-low-freq-advanced", "value"),
         State("filter-high-freq-advanced", "value"),
         State("filter-order-advanced", "value"),
         State("advanced-filter-method", "value"),
         State("advanced-noise-level", "value"),
         State("advanced-iterations", "value"),
         State("advanced-learning-rate", "value"),
         State("artifact-type", "value"),
         State("artifact-removal-strength", "value"),
         State("neural-network-type", "value"),
         State("neural-model-complexity", "value"),
         State("ensemble-method", "value"),
         State("ensemble-n-filters", "value"),
         State("filter-quality-options", "value"),
         State("detrend-option", "value")]
    )
    def advanced_filtering_callback(pathname, n_clicks, slider_value, nudge_m10, nudge_m1, nudge_p1, nudge_p10,
                                  start_time_state, end_time_state, filter_type, filter_family, filter_response, low_freq, high_freq,
                                  filter_order, advanced_method, noise_level, iterations, learning_rate, artifact_type,
                                  artifact_strength, neural_type, neural_complexity, ensemble_method, ensemble_n_filters,
                                  quality_options, detrend_option):
    
        ctx = callback_context
        
        if not ctx.triggered:
            raise PreventUpdate
        
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        logger.info(f"=== ADVANCED FILTERING CALLBACK TRIGGERED ===")
        logger.info(f"Trigger ID: {trigger_id}")
        logger.info(f"Pathname: {pathname}")
        
        # Only run this when we're on the filtering page
        if pathname != "/filtering":
            logger.info("Not on filtering page, returning empty figures")
            return create_empty_figure(), create_empty_figure(), create_empty_figure(), "Navigate to Filtering page", create_empty_figure(), None
        
        # If this is the first time loading the page (no button clicks), show a message
        if not ctx.triggered or ctx.triggered[0]["prop_id"].split(".")[0] == "url":
            logger.info("First time loading filtering page, attempting to load data")
        
        try:
            # Get data from the data service
            from vitalDSP_webapp.services.data.data_service import get_data_service
            data_service = get_data_service()
            
            # Get all stored data and find the latest
            all_data = data_service.get_all_data()
            logger.info(f"Data service returned: {type(all_data)}")
            logger.info(f"All data keys: {list(all_data.keys()) if all_data else 'None'}")
            logger.info(f"All data content: {all_data}")
            
            if not all_data:
                logger.warning("No data available for filtering")
                return create_empty_figure(), create_empty_figure(), create_empty_figure(), "No data available", create_empty_figure(), None
            
            # Get the most recent data entry
            latest_data_id = list(all_data.keys())[-1]
            latest_data = all_data[latest_data_id]
            logger.info(f"Latest data ID: {latest_data_id}")
            logger.info(f"Latest data content: {latest_data}")
            
            # Log the callback parameters
            logger.info(f"=== CALLBACK PARAMETERS ===")
            logger.info(f"Start time: {start_time_state} (type: {type(start_time_state)})")
            logger.info(f"End time: {end_time_state} (type: {type(end_time_state)})")
            logger.info(f"Filter type: {filter_type} (type: {type(filter_type)})")
            logger.info(f"Filter family: {filter_family} (type: {type(filter_family)})")
            logger.info(f"Filter response: {filter_response} (type: {type(filter_response)})")
            logger.info(f"Low frequency: {low_freq} (type: {type(low_freq)})")
            logger.info(f"High frequency: {high_freq} (type: {type(high_freq)})")
            logger.info(f"Filter order: {filter_order} (type: {type(filter_order)})")
            logger.info(f"Advanced method: {advanced_method} (type: {type(advanced_method)})")
            logger.info(f"Noise level: {noise_level} (type: {type(noise_level)})")
            logger.info(f"Iterations: {iterations} (type: {type(iterations)})")
            logger.info(f"Learning rate: {learning_rate} (type: {type(learning_rate)})")
            logger.info(f"Artifact type: {artifact_type} (type: {type(artifact_type)})")
            logger.info(f"Artifact strength: {artifact_strength} (type: {type(artifact_strength)})")
            logger.info(f"Neural type: {neural_type} (type: {type(neural_type)})")
            logger.info(f"Neural complexity: {neural_complexity} (type: {type(neural_complexity)})")
            logger.info(f"Ensemble method: {ensemble_method} (type: {type(ensemble_method)})")
            logger.info(f"Ensemble n filters: {ensemble_n_filters} (type: {type(ensemble_n_filters)})")
            logger.info(f"Quality options: {quality_options} (type: {type(quality_options)})")
            logger.info(f"Detrend option: {detrend_option} (type: {type(detrend_option)})")
            
            # Get the data and info using the data ID
            df = data_service.get_data(latest_data_id)
            data_info = data_service.get_data_info(latest_data_id)
            column_mapping = data_service.get_column_mapping(latest_data_id)
            
            # Log time range interpretation (after we have data_info)
            # Use slider value if available, otherwise fall back to state values
            effective_start_time = slider_value[0] if slider_value else start_time_state
            effective_end_time = slider_value[1] if slider_value else end_time_state
            
            if effective_start_time is not None and effective_end_time is not None:
                logger.info(f"=== TIME RANGE INTERPRETATION ===")
                logger.info(f"User selected time range: {effective_start_time} to {effective_end_time} seconds")
                logger.info(f"Data duration: {data_info.get('duration', 'unknown')} seconds")
                logger.info(f"Data sampling frequency: {data_info.get('sampling_freq', 'unknown')} Hz")
                logger.info(f"Total data points: {data_info.get('signal_length', 'unknown')}")
                logger.info(f"Expected points for time range: {(effective_end_time - effective_start_time) * data_info.get('sampling_freq', 100)}")
            
            logger.info(f"Data service method results:")
            logger.info(f"  get_data returned: {type(df)}")
            logger.info(f"  get_data_info returned: {type(data_info)}")
            logger.info(f"  get_column_mapping returned: {type(column_mapping)}")
            
            logger.info(f"=== DATA EXTRACTION DEBUG ===")
            logger.info(f"DataFrame shape: {df.shape}")
            logger.info(f"DataFrame columns: {list(df.columns)}")
            logger.info(f"Column mapping: {column_mapping}")
            logger.info(f"Data info: {data_info}")
            logger.info(f"Latest data ID: {latest_data_id}")
            logger.info(f"All data keys: {list(all_data.keys())}")
            
            if df is None or df.empty:
                logger.warning("No data available for filtering")
                return create_empty_figure(), create_empty_figure(), create_empty_figure(), "No data available", create_empty_figure(), None
            
            # Determine which column to use for signal data
            signal_column = None
            time_column = None
            
            if column_mapping:
                # Use column mapping if available
                signal_column = column_mapping.get('signal', column_mapping.get('amplitude', column_mapping.get('value')))
                time_column = column_mapping.get('time', column_mapping.get('timestamp', column_mapping.get('index')))
                logger.info(f"Using mapped columns - Signal: {signal_column}, Time: {time_column}")
                logger.info(f"Column mapping details: {column_mapping}")
            else:
                # Fallback: try to auto-detect columns
                if len(df.columns) >= 2:
                    # Assume first column is time, second is signal
                    time_column = df.columns[0]
                    signal_column = df.columns[1]
                    logger.info(f"Auto-detected columns - Signal: {signal_column}, Time: {time_column}")
                    logger.info(f"Available columns: {list(df.columns)}")
                else:
                    # Single column - use it as signal
                    signal_column = df.columns[0]
                    time_column = None
                    logger.info(f"Single column detected - Signal: {signal_column}")
                    logger.info(f"Available columns: {list(df.columns)}")
            
            if not signal_column:
                logger.error("Could not determine signal column")
                logger.error(f"Available columns: {list(df.columns)}")
                logger.error(f"Column mapping: {column_mapping}")
                return create_empty_figure(), create_empty_figure(), create_empty_figure(), "Could not determine signal column", create_empty_figure(), None
            
            # Log the actual data sample
            logger.info(f"Signal column: {signal_column}")
            logger.info(f"Signal column data sample: {df[signal_column].head(10).values}")
            logger.info(f"Signal column data info: min={df[signal_column].min()}, max={df[signal_column].max()}, mean={df[signal_column].mean():.4f}")
            
            # Initialize start_idx and end_idx
            start_idx = 0
            end_idx = len(df)
            
            # Handle time range selection
            if time_column and time_column in df.columns:
                # Use actual time column if available
                time_data = df[time_column].values
                logger.info(f"Using actual time column: {time_column}")
                logger.info(f"Full time data range: {np.min(time_data):.4f} to {np.max(time_data):.4f}")
                logger.info(f"Time column data sample: {time_data[:10]}")
                logger.info(f"Time column data type: {type(time_data[0])}")
                logger.info(f"Time column data info: min={np.min(time_data):.4f}, max={np.max(time_data):.4f}, mean={np.mean(time_data):.4f}")
                
                # Find indices for the selected time range
                if effective_start_time is not None and effective_end_time is not None:
                    logger.info(f"Looking for time range: {effective_start_time} to {effective_end_time}")
                    logger.info(f"Time data type: {type(time_data[0])}, Start time type: {type(effective_start_time)}")
                    logger.info(f"Time data range: {np.min(time_data):.4f} to {np.max(time_data):.4f}")
                    
                    # Convert time_data to float if needed
                    if isinstance(time_data[0], str):
                        try:
                            time_data = pd.to_numeric(time_data, errors='coerce')
                            logger.info("Converted time data to numeric")
                        except Exception as e:
                            logger.warning(f"Could not convert time data to numeric: {e}")
                    
                    # Check if time data is in milliseconds and convert to seconds if needed
                    if np.max(time_data) > 1000:  # Likely milliseconds
                        logger.info("Time data appears to be in milliseconds, converting to seconds")
                        time_data_seconds = time_data / 1000.0
                        # Find indices where time is within the selected range (in seconds)
                        mask = (time_data_seconds >= effective_start_time) & (time_data_seconds <= effective_end_time)
                    else:
                        # Time data is already in seconds
                        mask = (time_data >= effective_start_time) & (time_data <= effective_end_time)
                    
                    logger.info(f"Mask sum: {np.sum(mask)} out of {len(mask)} points")
                    if np.any(mask):
                        start_idx = np.where(mask)[0][0]
                        end_idx = np.where(mask)[0][-1] + 1
                        logger.info(f"Time range {effective_start_time} to {effective_end_time} maps to indices {start_idx} to {end_idx}")
                        
                                            # Ensure we have enough data points for filtering
                        min_points = 100  # Minimum points needed for filtering
                        if (end_idx - start_idx) < min_points:
                            logger.warning(f"Only {end_idx - start_idx} points selected, expanding range to get at least {min_points} points")
                            # Expand the range to get more points
                            center_idx = (start_idx + end_idx) // 2
                            half_range = min_points // 2
                            start_idx = max(0, center_idx - half_range)
                            end_idx = min(len(time_data), center_idx + half_range)
                            logger.info(f"Expanded range: indices {start_idx} to {end_idx} ({end_idx - start_idx} points)")
                            
                            # If still not enough points, use a larger range
                            if (end_idx - start_idx) < min_points:
                                logger.warning(f"Still not enough points, using larger range")
                                # Use a larger range around the center
                                half_range = min(min_points, len(time_data) // 4)  # Use 1/4 of data or min_points
                                center_idx = len(time_data) // 2  # Use center of full dataset
                                start_idx = max(0, center_idx - half_range)
                                end_idx = min(len(time_data), center_idx + half_range)
                                logger.info(f"Using larger range: indices {start_idx} to {end_idx} ({end_idx - start_idx} points)")
                    else:
                        logger.warning(f"No data found in time range {effective_start_time} to {effective_end_time}, using full range")
                        start_idx = 0
                        end_idx = len(time_data)
                else:
                    # Use full range if no time selection
                    start_idx = 0
                    end_idx = len(time_data)
                    logger.info("No time range selected, using full data")
                    
                    # If full range is too large, use a reasonable subset
                    max_points = 10000  # Maximum points to process
                    if (end_idx - start_idx) > max_points:
                        logger.info(f"Full range too large ({end_idx - start_idx} points), using subset of {max_points} points")
                        center_idx = len(time_data) // 2
                        half_range = max_points // 2
                        start_idx = max(0, center_idx - half_range)
                        end_idx = min(len(time_data), center_idx + half_range)
                        logger.info(f"Using subset: indices {start_idx} to {end_idx} ({end_idx - start_idx} points)")
                
                # Extract data for the selected range
                signal_data = df[signal_column].iloc[start_idx:end_idx].values
                
                # Generate time axis in seconds
                if np.max(time_data) > 1000:  # Likely milliseconds
                    time_axis = time_data[start_idx:end_idx] / 1000.0  # Convert to seconds
                    logger.info(f"Time axis converted from milliseconds to seconds: {np.min(time_axis):.4f} to {np.max(time_axis):.4f}")
                else:
                    time_axis = time_data[start_idx:end_idx]  # Already in seconds
                    logger.info(f"Time axis in seconds: {np.min(time_axis):.4f} to {np.max(time_axis):.4f}")
                
            else:
                # Generate time axis based on sampling frequency
                sampling_freq = data_info.get('sampling_frequency', 1000)
                logger.info(f"Using sampling frequency: {sampling_freq} Hz")
                logger.info(f"Data info keys: {list(data_info.keys()) if data_info else 'None'}")
                logger.info(f"Sampling frequency from data_info: {data_info.get('sampling_frequency') if data_info else 'None'}")
                
                if effective_start_time is not None and effective_end_time is not None:
                    start_idx = int(effective_start_time * sampling_freq)
                    end_idx = int(effective_end_time * sampling_freq)
                    
                    # Ensure valid indices
                    start_idx = max(0, min(start_idx, len(df) - 1))
                    end_idx = max(start_idx + 1, min(end_idx, len(df)))
                    logger.info(f"Time range {effective_start_time} to {effective_end_time} maps to indices {start_idx} to {end_idx}")
                else:
                    # Use full range if no time selection
                    start_idx = 0
                    end_idx = len(df)
                    logger.info("No time range selected, using full data")
                
                # Extract data for the selected range
                signal_data = df[signal_column].iloc[start_idx:end_idx].values
                time_axis = np.arange(start_idx, end_idx) / sampling_freq
            
            logger.info(f"Signal data shape: {signal_data.shape}")
            logger.info(f"Signal data range: {np.min(signal_data):.4f} to {np.max(signal_data):.4f}")
            logger.info(f"Signal data sample: {signal_data[:5] if len(signal_data) >= 5 else signal_data}")
            
            logger.info(f"Final data extraction:")
            logger.info(f"  Start index: {start_idx}")
            logger.info(f"  End index: {end_idx}")
            logger.info(f"  Signal data shape: {signal_data.shape}")
            logger.info(f"  Time axis shape: {time_axis.shape}")
            logger.info(f"  Signal data range: {np.min(signal_data):.4f} to {np.max(signal_data):.4f}")
            logger.info(f"  Time axis range: {np.min(time_axis):.4f} to {np.max(time_axis):.4f}")
            logger.info(f"  Signal data sample: {signal_data[:5] if len(signal_data) >= 5 else signal_data}")
            logger.info(f"  Time axis sample: {time_axis[:5] if len(time_axis) >= 5 else time_axis}")
            
            # Check if we have valid data
            if len(signal_data) == 0:
                logger.error("No signal data extracted")
                return create_empty_figure(), create_empty_figure(), create_empty_figure(), "No signal data extracted", create_empty_figure(), None
            
            # Store raw signal for plotting (before any preprocessing)
            raw_signal_for_plotting = signal_data.copy()
            logger.info(f"Raw signal stored for plotting - mean: {np.mean(raw_signal_for_plotting):.4f}, range: {np.min(raw_signal_for_plotting):.4f} to {np.max(raw_signal_for_plotting):.4f}")
            
            # Apply detrending if user selected the option (to match time domain screen behavior)
            if detrend_option and "detrend" in detrend_option:
                logger.info("Applying detrending to match time domain screen behavior (zero baseline)")
                from scipy import signal
                
                # Store original signal for comparison
                original_signal = signal_data.copy()
                
                # Apply linear detrending to remove DC offset and linear trends
                signal_data_detrended = signal.detrend(signal_data, type='linear')
                
                # Also apply mean subtraction for zero baseline
                signal_data_zero_baseline = signal_data_detrended - np.mean(signal_data_detrended)
                
                logger.info(f"Detrending applied:")
                logger.info(f"  Original signal mean: {np.mean(original_signal):.4f}")
                logger.info(f"  Detrended signal mean: {np.mean(signal_data_detrended):.4f}")
                logger.info(f"  Zero-baseline signal mean: {np.mean(signal_data_zero_baseline):.4f}")
                logger.info(f"  Original signal range: {np.min(original_signal):.4f} to {np.max(original_signal):.4f}")
                logger.info(f"  Detrended signal range: {np.min(signal_data_detrended):.4f} to {np.max(signal_data_detrended):.4f}")
                logger.info(f"  Zero-baseline signal range: {np.min(signal_data_zero_baseline):.4f} to {np.max(signal_data_zero_baseline):.4f}")
                
                # Use zero-baseline signal for filtering to match time domain screen
                signal_data = signal_data_zero_baseline
            else:
                logger.info("Detrending not selected, using original signal baseline")
                logger.info(f"Signal baseline - mean: {np.mean(signal_data):.4f}, range: {np.min(signal_data):.4f} to {np.max(signal_data):.4f}")
            
            # Check if we have enough data points for filtering
            if len(signal_data) < 50:
                logger.warning(f"Only {len(signal_data)} data points available, this may not be enough for effective filtering")
                # Use a simple smoothing filter instead of complex filtering
                logger.info("Using simple smoothing instead of complex filtering")
                filtered_data = np.convolve(signal_data, np.ones(3)/3, mode='same')
            else:
                logger.info(f"Sufficient data points ({len(signal_data)}) for filtering")
            
            # Get sampling frequency for filtering
            sampling_freq = data_info.get('sampling_freq', 100)  # Default to 100 Hz based on the data info
            logger.info(f"Using sampling frequency for filtering: {sampling_freq} Hz")
            
            # Apply filtering based on type
            logger.info(f"=== APPLYING FILTERING ===")
            logger.info(f"Filter type: {filter_type}")
            logger.info(f"Input signal data shape: {signal_data.shape}")
            logger.info(f"Input signal data range: {np.min(signal_data):.4f} to {np.max(signal_data):.4f}")
            
            # Only apply complex filtering if we have enough data points
            if len(signal_data) >= 50:
                if filter_type == "traditional":
                    # Apply traditional filter
                    logger.info(f"Applying traditional filter: {filter_family} {filter_response}")
                    # Set default values for filter parameters
                    filter_family = filter_family or "butter"
                    filter_response = filter_response or "low"
                    low_freq = low_freq or 10
                    high_freq = high_freq or 50
                    filter_order = filter_order or 4
                    
                    filtered_data = apply_traditional_filter(signal_data, sampling_freq, filter_family, filter_response, low_freq, high_freq, filter_order)
                    
                    # Apply additional traditional filters if parameters are provided
                    # Note: These would need to be added to the callback inputs
                    filtered_data = apply_additional_traditional_filters(
                        filtered_data, None, None, None, None, None, None
                    )
                    
                elif filter_type == "advanced":
                    logger.info(f"Applying advanced filter: {advanced_method}")
                    filtered_data = apply_advanced_filter(signal_data, advanced_method, noise_level, iterations, learning_rate)
                elif filter_type == "artifact":
                        # Enhanced artifact removal with all parameters
                        logger.info(f"Applying artifact removal: {artifact_type}")
                        filtered_data = apply_enhanced_artifact_removal(
                            signal_data, sampling_freq, artifact_type, artifact_strength,
                            None, None, None, None, None, None, None, None
                        )
                elif filter_type == "neural":
                    logger.info(f"Applying neural filter: {neural_type}")
                    filtered_data = apply_neural_filter(signal_data, neural_type, neural_complexity)
                elif filter_type == "ensemble":
                        # Enhanced ensemble filtering with all parameters
                        logger.info(f"Applying ensemble filter: {ensemble_method}")
                        filtered_data = apply_enhanced_ensemble_filter(
                            signal_data, ensemble_method, ensemble_n_filters,
                            None, None, None, None, None, None
                        )
                else:
                    logger.info("No filter type selected, using original signal")
                    filtered_data = signal_data
            else:
                logger.info("Using simple smoothing filter due to insufficient data points")
                # filtered_data is already set to simple smoothing above
            
            logger.info(f"Filtering completed:")
            logger.info(f"  Raw signal shape: {raw_signal_for_plotting.shape}")
            logger.info(f"  Processed signal shape: {signal_data.shape}")
            logger.info(f"  Filtered signal shape: {filtered_data.shape}")
            logger.info(f"  Raw signal range: {np.min(raw_signal_for_plotting):.4f} to {np.max(raw_signal_for_plotting):.4f}")
            logger.info(f"  Processed signal range: {np.min(signal_data):.4f} to {np.max(signal_data):.4f}")
            logger.info(f"  Filtered signal range: {np.min(filtered_data):.4f} to {np.max(filtered_data):.4f}")
            
            # Apply multi-modal filtering if reference signal is specified
            # Note: These parameters would need to be added to the callback inputs
            if False:  # Placeholder for multi-modal filtering
                filtered_data = apply_multi_modal_filtering(
                    filtered_data, None, None, None, None
                )
            
            # Create visualizations
            logger.info(f"=== CREATING PLOTS ===")
            logger.info(f"Raw signal data shape: {raw_signal_for_plotting.shape}")
            logger.info(f"Processed signal data shape: {signal_data.shape}")
            logger.info(f"Filtered signal data shape: {filtered_data.shape}")
            logger.info(f"Time axis shape: {time_axis.shape}")
            logger.info(f"Raw signal range: {np.min(raw_signal_for_plotting):.4f} to {np.max(raw_signal_for_plotting):.4f}")
            logger.info(f"Processed signal range: {np.min(signal_data):.4f} to {np.max(signal_data):.4f}")
            logger.info(f"Filtered signal range: {np.min(filtered_data):.4f} to {np.max(filtered_data):.4f}")
            
            original_plot = create_original_signal_plot(time_axis, raw_signal_for_plotting)
            filtered_plot = create_filtered_signal_plot(time_axis, filtered_data)
            comparison_plot = create_filter_comparison_plot(time_axis, raw_signal_for_plotting, filtered_data)
            
            # Generate quality metrics using RAW signal (not detrended) for accurate assessment
            quality_metrics = generate_filter_quality_metrics(raw_signal_for_plotting, filtered_data, sampling_freq, quality_options)
            quality_plots = create_filter_quality_plots(raw_signal_for_plotting, filtered_data, sampling_freq, quality_options)
            
            # Store results
            stored_data = {
                'raw_signal': raw_signal_for_plotting.tolist(),
                'processed_signal': signal_data.tolist(),
                'filtered_signal': filtered_data.tolist(),
                'time_axis': time_axis.tolist(),
                'filter_type': filter_type,
                'detrending_applied': detrend_option and "detrend" in detrend_option,
                'parameters': {
                    'filter_family': filter_family,
                    'filter_response': filter_response,
                    'low_freq': low_freq,
                    'high_freq': high_freq,
                    'filter_order': filter_order,
                    'advanced_method': advanced_method,
                    'artifact_type': artifact_type,
                    'ensemble_method': ensemble_method
                }
            }
            
            logger.info("Advanced filtering completed successfully")
            return original_plot, filtered_plot, comparison_plot, quality_metrics, quality_plots, stored_data
            
        except Exception as e:
            logger.error(f"Error in advanced filtering callback: {e}")
            error_msg = f"Error during filtering: {str(e)}"
            return create_empty_figure(), create_empty_figure(), create_empty_figure(), error_msg, create_empty_figure(), None

    # Time input update callbacks
    @app.callback(
        [Output("filter-start-time", "value"),
         Output("filter-end-time", "value")],
        [Input("filter-time-range-slider", "value"),
         Input("filter-btn-nudge-m10", "n_clicks"),
         Input("filter-btn-nudge-m1", "n_clicks"),
         Input("filter-btn-nudge-p1", "n_clicks"),
         Input("filter-btn-nudge-p10", "n_clicks")],
        [State("filter-start-time", "value"),
         State("filter-end-time", "value")]
    )
    def update_filter_time_inputs(slider_value, nudge_m10, nudge_m1, nudge_p1, nudge_p10, start_time, end_time):
        """Update time inputs based on slider or nudge buttons."""
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
        
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        if trigger_id == "filter-time-range-slider" and slider_value:
            return slider_value[0], slider_value[1]
        
        # Handle nudge buttons
        time_window = end_time - start_time if start_time and end_time else 10
        
        if trigger_id == "filter-btn-nudge-m10":
            new_start = max(0, start_time - 10) if start_time else 0
            new_end = new_start + time_window
            return new_start, new_end
        elif trigger_id == "filter-btn-nudge-m1":
            new_start = max(0, start_time - 1) if start_time else 0
            new_end = new_start + time_window
            return new_start, new_end
        elif trigger_id == "filter-btn-nudge-p1":
            new_start = start_time + 1 if start_time else 1
            new_end = new_start + time_window
            return new_start, new_end
        elif trigger_id == "filter-btn-nudge-p10":
            new_start = start_time + 10 if start_time else 10
            new_end = new_start + time_window
            return new_start, new_end
        
        return no_update, no_update

    # Time slider range update callback
    @app.callback(
        Output("filter-time-range-slider", "max"),
        [Input("store-uploaded-data", "data")]
    )
    def update_filter_time_slider_range(data_store):
        """Update time slider range based on uploaded data."""
        if not data_store:
            return 100
        
        try:
            df = pd.DataFrame(data_store["data"])
            if df.empty:
                return 100
            
            # Get time column (assume first column)
            time_data = df.iloc[:, 0].values
            max_time = np.max(time_data)
            return max_time
        except Exception as e:
            logger.error(f"Error updating filter time slider range: {e}")
            return 100

    # Filter type parameter visibility callback
    @app.callback(
        [Output("traditional-filter-params", "style"),
         Output("advanced-filter-params", "style"),
         Output("artifact-removal-params", "style"),
         Output("neural-network-params", "style"),
         Output("ensemble-params", "style")],
        [Input("filter-type-select", "value")]
    )
    def update_filter_parameter_visibility(filter_type):
        """Show/hide parameter sections based on selected filter type."""
        # Default style (hidden)
        hidden_style = {"display": "none"}
        visible_style = {"display": "block"}
        
        # Initialize all to hidden
        traditional_style = hidden_style
        advanced_style = hidden_style
        artifact_style = hidden_style
        neural_style = hidden_style
        ensemble_style = visible_style
        
        # Show appropriate section based on selection
        if filter_type == "traditional":
            traditional_style = visible_style
        elif filter_type == "advanced":
            advanced_style = visible_style
        elif filter_type == "artifact":
            artifact_style = visible_style
        elif filter_type == "neural":
            neural_style = visible_style
        elif filter_type == "ensemble":
            ensemble_style = visible_style
        
        return traditional_style, advanced_style, artifact_style, neural_style, ensemble_style


# Helper functions for signal filtering
def create_original_signal_plot(time_axis, signal_data):
    """Create plot for original signal with critical points detection."""
    try:
        logger.info(f"Creating original signal plot:")
        logger.info(f"  Time axis shape: {time_axis.shape}")
        logger.info(f"  Signal data shape: {signal_data.shape}")
        logger.info(f"  Time axis range: {np.min(time_axis):.4f} to {np.max(time_axis):.4f}")
        logger.info(f"  Signal data range: {np.min(signal_data):.4f} to {np.max(signal_data):.4f}")
        logger.info(f"  Time axis sample: {time_axis[:5] if len(time_axis) >= 5 else time_axis}")
        logger.info(f"  Signal data sample: {signal_data[:5] if len(signal_data) >= 5 else signal_data}")
        
        fig = go.Figure()
        
        # Add main signal
        fig.add_trace(go.Scatter(
            x=time_axis,
            y=signal_data,
            mode='lines',
            name='Original Signal',
            line=dict(color='blue', width=2)
        ))
        
        # Add critical points detection using vitalDSP waveform module
        try:
            from vitalDSP.physiological_features.waveform import WaveformMorphology
            
            # Create waveform morphology object
            wm = WaveformMorphology(
                waveform=signal_data,
                fs=100,  # Default sampling frequency
                signal_type="PPG",  # Assume PPG for now
                simple_mode=True
            )
            
            # Detect critical points based on signal type
            if hasattr(wm, 'systolic_peaks') and wm.systolic_peaks is not None:
                # Plot systolic peaks
                fig.add_trace(go.Scatter(
                    x=time_axis[wm.systolic_peaks],
                    y=signal_data[wm.systolic_peaks],
                    mode='markers',
                    name='Systolic Peaks',
                    marker=dict(color='red', size=10, symbol='diamond'),
                    hovertemplate='<b>Systolic Peak:</b> %{y}<extra></extra>'
                ))
            
            # Detect and plot dicrotic notches
            try:
                dicrotic_notches = wm.detect_dicrotic_notches()
                if dicrotic_notches is not None and len(dicrotic_notches) > 0:
                    fig.add_trace(go.Scatter(
                        x=time_axis[dicrotic_notches],
                        y=signal_data[dicrotic_notches],
                        mode='markers',
                        name='Dicrotic Notches',
                        marker=dict(color='orange', size=8, symbol='circle'),
                        hovertemplate='<b>Dicrotic Notch:</b> %{y}<extra></extra>'
                    ))
            except Exception as e:
                logger.warning(f"Dicrotic notch detection failed: {e}")
            
            # Detect and plot diastolic peaks
            try:
                diastolic_peaks = wm.detect_diastolic_peak()
                if diastolic_peaks is not None and len(diastolic_peaks) > 0:
                    fig.add_trace(go.Scatter(
                        x=time_axis[diastolic_peaks],
                        y=signal_data[diastolic_peaks],
                        mode='markers',
                        name='Diastolic Peaks',
                        marker=dict(color='green', size=8, symbol='square'),
                        hovertemplate='<b>Diastolic Peak:</b> %{y}<extra></extra>'
                    ))
            except Exception as e:
                logger.warning(f"Diastolic peak detection failed: {e}")
                
        except Exception as e:
            logger.warning(f"Critical points detection failed: {e}")
        
        fig.update_layout(
            title="Original Signal with Critical Points",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
            height=400,
            showlegend=True
        )
        
        logger.info(f"Original signal plot with critical points created successfully")
        return fig
    except Exception as e:
        logger.error(f"Error creating original signal plot: {e}")
        return create_empty_figure()


def create_filtered_signal_plot(time_axis, filtered_data):
    """Create plot for filtered signal with critical points detection."""
    try:
        logger.info(f"Creating filtered signal plot:")
        logger.info(f"  Time axis shape: {time_axis.shape}")
        logger.info(f"  Filtered data shape: {filtered_data.shape}")
        logger.info(f"  Time axis range: {np.min(time_axis):.4f} to {np.max(time_axis):.4f}")
        logger.info(f"  Filtered data range: {np.min(filtered_data):.4f} to {np.max(filtered_data):.4f}")
        logger.info(f"  Time axis sample: {time_axis[:5] if len(time_axis) >= 5 else time_axis}")
        logger.info(f"  Filtered data sample: {filtered_data[:5] if len(filtered_data) >= 5 else filtered_data}")
        
        fig = go.Figure()
            
            # Add main filtered signal
        fig.add_trace(go.Scatter(
        x=time_axis,
            y=filtered_data,
        mode='lines',
            name='Filtered Signal',
            line=dict(color='red', width=2)
        ))
        
        # Add critical points detection using vitalDSP waveform module
        try:
            from vitalDSP.physiological_features.waveform import WaveformMorphology
            
            # Create waveform morphology object for filtered signal
            wm = WaveformMorphology(
                waveform=filtered_data,
                fs=100,  # Default sampling frequency
                signal_type="PPG",  # Assume PPG for now
                simple_mode=True
            )
            
            # Detect critical points based on signal type
            if hasattr(wm, 'systolic_peaks') and wm.systolic_peaks is not None:
                # Plot systolic peaks
                fig.add_trace(go.Scatter(
                    x=time_axis[wm.systolic_peaks],
                    y=filtered_data[wm.systolic_peaks],
                    mode='markers',
                    name='Systolic Peaks (Filtered)',
                    marker=dict(color='darkred', size=10, symbol='diamond'),
                    hovertemplate='<b>Systolic Peak (Filtered):</b> %{y}<extra></extra>'
                ))
            
            # Detect and plot dicrotic notches
            try:
                dicrotic_notches = wm.detect_dicrotic_notches()
                if dicrotic_notches is not None and len(dicrotic_notches) > 0:
                    fig.add_trace(go.Scatter(
                        x=time_axis[dicrotic_notches],
                        y=filtered_data[dicrotic_notches],
                        mode='markers',
                        name='Dicrotic Notches (Filtered)',
                        marker=dict(color='darkorange', size=8, symbol='circle'),
                        hovertemplate='<b>Dicrotic Notch (Filtered):</b> %{y}<extra></extra>'
                    ))
            except Exception as e:
                logger.warning(f"Dicrotic notch detection failed: {e}")
            
            # Detect and plot diastolic peaks
            try:
                diastolic_peaks = wm.detect_diastolic_peak()
                if diastolic_peaks is not None and len(diastolic_peaks) > 0:
                    fig.add_trace(go.Scatter(
                        x=time_axis[diastolic_peaks],
                        y=filtered_data[diastolic_peaks],
                        mode='markers',
                        name='Diastolic Peaks (Filtered)',
                        marker=dict(color='darkgreen', size=8, symbol='square'),
                        hovertemplate='<b>Diastolic Peak (Filtered):</b> %{y}<extra></extra>'
                    ))
            except Exception as e:
                logger.warning(f"Diastolic peak detection failed: {e}")
                
        except Exception as e:
            logger.warning(f"Critical points detection failed: {e}")
        
        fig.update_layout(
                title="Filtered Signal with Critical Points",
                xaxis_title="Time (s)",
            yaxis_title="Amplitude",
                height=400,
                showlegend=True
        )
    
        logger.info(f"Filtered signal plot with critical points created successfully")
        return fig
    except Exception as e:
        logger.error(f"Error creating filtered signal plot: {e}")
        return create_empty_figure()


def create_filter_comparison_plot(time_axis, original_signal, filtered_signal):
    """Create comparison plot between original and filtered signals with critical points detection."""
    try:
        logger.info(f"Creating comparison plot:")
        logger.info(f"  Time axis shape: {time_axis.shape}")
        logger.info(f"  Original signal shape: {original_signal.shape}")
        logger.info(f"  Filtered signal shape: {filtered_signal.shape}")
        logger.info(f"  Time axis range: {np.min(time_axis):.4f} to {np.max(time_axis):.4f}")
        logger.info(f"  Original signal range: {np.min(original_signal):.4f} to {np.max(original_signal):.4f}")
        logger.info(f"  Filtered signal range: {np.min(filtered_signal):.4f} to {np.max(filtered_signal):.4f}")
        
        fig = go.Figure()
        
        # Original signal
        fig.add_trace(go.Scatter(
            x=time_axis,
            y=original_signal,
            mode='lines',
            name='Original Signal',
                line=dict(color='blue', width=2),
                opacity=0.8
        ))
    
        # Filtered signal
        fig.add_trace(go.Scatter(
            x=time_axis,
            y=filtered_signal,
            mode='lines',
            name='Filtered Signal',
                line=dict(color='red', width=2),
                opacity=0.8
            ))
        
        # Add critical points detection using vitalDSP waveform module
        try:
            from vitalDSP.physiological_features.waveform import WaveformMorphology
            
            # Create waveform morphology object for original signal
            wm_orig = WaveformMorphology(
                waveform=original_signal,
                fs=100,  # Default sampling frequency
                signal_type="PPG",  # Assume PPG for now
                simple_mode=True
            )
            
            # Create waveform morphology object for filtered signal
            wm_filt = WaveformMorphology(
                waveform=filtered_signal,
                fs=100,  # Default sampling frequency
                signal_type="PPG",  # Assume PPG for now
                simple_mode=True
            )
            
            # Detect and plot critical points for original signal
            if hasattr(wm_orig, 'systolic_peaks') and wm_orig.systolic_peaks is not None:
                fig.add_trace(go.Scatter(
                    x=time_axis[wm_orig.systolic_peaks],
                    y=original_signal[wm_orig.systolic_peaks],
                    mode='markers',
                    name='Systolic Peaks (Original)',
                    marker=dict(color='darkblue', size=8, symbol='diamond'),
                    hovertemplate='<b>Original Systolic Peak:</b> %{y}<extra></extra>'
                ))
            
            # Detect and plot critical points for filtered signal
            if hasattr(wm_filt, 'systolic_peaks') and wm_filt.systolic_peaks is not None:
                fig.add_trace(go.Scatter(
                    x=time_axis[wm_filt.systolic_peaks],
                    y=filtered_signal[wm_filt.systolic_peaks],
                    mode='markers',
                    name='Systolic Peaks (Filtered)',
                    marker=dict(color='darkred', size=8, symbol='diamond'),
                    hovertemplate='<b>Filtered Systolic Peak:</b> %{y}<extra></extra>'
                ))
                
        except Exception as e:
            logger.warning(f"Critical points detection failed: {e}")
        
        fig.update_layout(
                title="Signal Comparison: Original vs Filtered with Critical Points",
                xaxis_title="Time (s)",
                yaxis_title="Amplitude",
                height=400,
                showlegend=True
            )
        
        logger.info(f"Comparison plot with critical points created successfully")
        return fig
    except Exception as e:
        logger.error(f"Error creating comparison plot: {e}")
        return create_empty_figure()


def create_empty_figure():
    """Create an empty figure for error cases."""
    fig = go.Figure()
    fig.add_annotation(
        text="No data available",
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False
    )
    fig.update_layout(height=400)
    return fig


def create_signal_plot(signal_data, time_axis, title):
    """Create a signal plot."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=signal_data,
        mode='lines',
        name=title,
        line=dict(color='blue', width=1.5)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        showlegend=True,
        height=400
    )
    
    return fig


# Removed duplicate function - the correct one is above


def apply_traditional_filter(signal_data, sampling_freq, filter_family, filter_response, low_freq, high_freq, filter_order):
    """Apply traditional filtering using vitalDSP functions."""
    try:
        logger.info(f"=== TRADITIONAL FILTER DEBUG ===")
        logger.info(f"Input signal shape: {signal_data.shape}")
        logger.info(f"Input signal range: {np.min(signal_data):.4f} to {np.max(signal_data):.4f}")
        logger.info(f"Filter family: {filter_family}")
        logger.info(f"Filter response: {filter_response}")
        logger.info(f"Low frequency: {low_freq}")
        logger.info(f"High frequency: {high_freq}")
        logger.info(f"Filter order: {filter_order}")
        logger.info(f"Sampling frequency: {sampling_freq}")
        
        # Import vitalDSP traditional filtering functions
        from vitalDSP.filtering.signal_filtering import SignalFiltering
        
        # Create filter instance
        signal_filter = SignalFiltering(signal_data)
        logger.info("SignalFiltering instance created successfully")
        
        # Apply filter based on type - using EXACT same scipy implementation as time domain screen
        if filter_response == "low":
            logger.info(f"Applying low-pass filter with cutoff: {low_freq}")
            # Use scipy directly (same as time domain screen)
            nyquist = sampling_freq / 2
            low_freq_norm = low_freq / nyquist
            
            # Ensure cutoff frequency is within valid range
            low_freq_norm = max(0.001, min(low_freq_norm, 0.999))
            
            from scipy import signal
            if filter_family == "butter":
                b, a = signal.butter(filter_order, low_freq_norm, btype='low')
            elif filter_family == "cheby1":
                b, a = signal.cheby1(filter_order, 1, low_freq_norm, btype='low')
            elif filter_family == "cheby2":
                b, a = signal.cheby2(filter_order, 40, low_freq_norm, btype='low')
            elif filter_family == "ellip":
                b, a = signal.ellip(filter_order, 1, 40, low_freq_norm, btype='low')
            else:
                b, a = signal.butter(filter_order, low_freq_norm, btype='low')
                
            filtered_signal = signal.filtfilt(b, a, signal_data)
            logger.info(f"Low-pass filter applied using scipy {filter_family} (same as time domain screen)")
                
        elif filter_response == "high":
            logger.info(f"Applying high-pass filter with cutoff: {high_freq}")
            # Use scipy directly (same as time domain screen)
            nyquist = sampling_freq / 2
            high_freq_norm = high_freq / nyquist
            
            # Ensure cutoff frequency is within valid range
            high_freq_norm = max(0.001, min(high_freq_norm, 0.999))
            
            from scipy import signal
            if filter_family == "butter":
                b, a = signal.butter(filter_order, high_freq_norm, btype='high')
            elif filter_family == "cheby1":
                b, a = signal.cheby1(filter_order, 1, high_freq_norm, btype='high')
            elif filter_family == "cheby2":
                b, a = signal.cheby2(filter_order, 40, high_freq_norm, btype='high')
            elif filter_family == "ellip":
                b, a = signal.ellip(filter_order, 1, 40, high_freq_norm, btype='high')
            else:
                b, a = signal.butter(filter_order, high_freq_norm, btype='high')
                
            filtered_signal = signal.filtfilt(b, a, signal_data)
            logger.info(f"High-pass filter applied using scipy {filter_family} (same as time domain screen)")
                
        elif filter_response == "bandpass":
            logger.info(f"Applying bandpass filter with range: {low_freq} - {high_freq}")
            # Use EXACT same scipy implementation as time domain screen for consistency
            logger.info("Using scipy implementation for bandpass (same as time domain screen)")
            nyquist = sampling_freq / 2
            low_freq_norm = low_freq / nyquist
            high_freq_norm = high_freq / nyquist
            
            # Ensure cutoff frequencies are within valid range
            low_freq_norm = max(0.001, min(low_freq_norm, 0.999))
            high_freq_norm = max(0.001, min(high_freq_norm, 0.999))
            
            # Use scipy directly (same as time domain screen)
            from scipy import signal
            if filter_family == "butter":
                b, a = signal.butter(filter_order, [low_freq_norm, high_freq_norm], btype='band')
            elif filter_family == "cheby1":
                b, a = signal.cheby1(filter_order, 1, [low_freq_norm, high_freq_norm], btype='band')
            elif filter_family == "cheby2":
                b, a = signal.cheby2(filter_order, 40, [low_freq_norm, high_freq_norm], btype='band')
            elif filter_family == "ellip":
                b, a = signal.ellip(filter_order, 1, 40, [low_freq_norm, high_freq_norm], btype='band')
            else:
                b, a = signal.butter(filter_order, [low_freq_norm, high_freq_norm], btype='band')
                
            filtered_signal = signal.filtfilt(b, a, signal_data)
            logger.info(f"Bandpass filter applied using scipy {filter_family} (same as time domain screen)")
            
        elif filter_response == "bandstop":
            logger.info(f"Applying bandstop filter with range: {low_freq} - {high_freq}")
            # vitalDSP doesn't support bandstop, use scipy directly (same as time domain screen)
            nyquist = sampling_freq / 2
            low_freq_norm = low_freq / nyquist
            high_freq_norm = high_freq / nyquist
            
            # Ensure cutoff frequencies are within valid range
            low_freq_norm = max(0.001, min(low_freq_norm, 0.999))
            high_freq_norm = max(0.001, min(high_freq_norm, 0.999))
            
            # Use scipy directly for bandstop (same as time domain screen)
            from scipy import signal
            if filter_family == "butter":
                b, a = signal.butter(filter_order, [low_freq_norm, high_freq_norm], btype='bandstop')
            elif filter_family == "cheby1":
                b, a = signal.cheby1(filter_order, 1, [low_freq_norm, high_freq_norm], btype='bandstop')
            elif filter_family == "cheby2":
                b, a = signal.cheby2(filter_order, 40, [low_freq_norm, high_freq_norm], btype='bandstop')
            elif filter_family == "ellip":
                b, a = signal.ellip(filter_order, 1, 40, [low_freq_norm, high_freq_norm], btype='bandstop')
            else:
                b, a = signal.butter(filter_order, [low_freq_norm, high_freq_norm], btype='bandstop')
                
            filtered_signal = signal.filtfilt(b, a, signal_data)
            logger.info(f"Bandstop filter applied using scipy {filter_family} (vitalDSP doesn't have bandstop)")
            
        elif filter_response == "default":
            # Default to low pass
            logger.info(f"Defaulting to low-pass filter with cutoff: {low_freq}")
            # Use scipy directly (same as time domain screen)
            nyquist = sampling_freq / 2
            low_freq_norm = low_freq / nyquist
            
            # Ensure cutoff frequency is within valid range
            low_freq_norm = max(0.001, min(low_freq_norm, 0.999))
            
            from scipy import signal
            b, a = signal.butter(filter_order, low_freq_norm, btype='low')
            filtered_signal = signal.filtfilt(b, a, signal_data)
            logger.info(f"Default low-pass filter applied using scipy (same as time domain screen)")
        
        logger.info(f"Traditional filter applied successfully: {filter_family} {filter_response}")
        logger.info(f"Output signal shape: {filtered_signal.shape}")
        logger.info(f"Output signal range: {np.min(filtered_signal):.4f} to {np.max(filtered_signal):.4f}")
        return filtered_signal
        
    except Exception as e:
        logger.error(f"Error applying traditional filter: {e}")
        logger.error(f"Exception details: {type(e).__name__}: {str(e)}")
        # Fallback to scipy implementation (same as time domain screen)
        logger.info("Falling back to scipy implementation for consistency")
        try:
            # Normalize cutoff frequencies
            nyquist = sampling_freq / 2
            low_freq_norm = low_freq / nyquist
            high_freq_norm = high_freq / nyquist
            
            # Ensure cutoff frequencies are within valid range
            low_freq_norm = max(0.001, min(low_freq_norm, 0.999))
            high_freq_norm = max(0.001, min(high_freq_norm, 0.999))
            
            # Determine filter type based on response
            if filter_response == "bandpass":
                btype = 'band'
                cutoff = [low_freq_norm, high_freq_norm]
            elif filter_response == "bandstop":
                btype = 'bandstop'
                cutoff = [low_freq_norm, high_freq_norm]
            elif filter_response == "lowpass":
                btype = 'low'
                cutoff = high_freq_norm
            elif filter_response == "highpass":
                btype = 'high'
                cutoff = low_freq_norm
            else:
                btype = 'low'
                cutoff = high_freq_norm
            
            # Apply filter using scipy (same as time domain screen)
            from scipy import signal
            b, a = signal.butter(filter_order, cutoff, btype=btype)
            filtered_signal = signal.filtfilt(b, a, signal_data)
            
            logger.info("Scipy fallback successful")
            return filtered_signal
            
        except Exception as fallback_error:
            logger.error(f"Scipy fallback also failed: {fallback_error}")
            return signal_data


def apply_additional_traditional_filters(signal_data, savgol_window, savgol_polyorder, 
                                       moving_avg_window, moving_avg_iterations,
                                       gaussian_sigma, gaussian_iterations):
    """Apply additional traditional filters like Savitzky-Golay, moving average, and Gaussian."""
    try:
        logger.info("Applying additional traditional filters")
        
        from vitalDSP.filtering.signal_filtering import SignalFiltering
        
        signal_filter = SignalFiltering(signal_data)
        filtered_signal = signal_data.copy()
        
        # Apply Savitzky-Golay filter if parameters are provided
        if savgol_window and savgol_polyorder:
            try:
                filtered_signal = signal_filter.savgol_filter(
                    filtered_signal, 
                    window_length=savgol_window, 
                    polyorder=savgol_polyorder
                )
                logger.info(f"Savitzky-Golay filter applied: window={savgol_window}, polyorder={savgol_polyorder}")
            except Exception as e:
                logger.warning(f"Savitzky-Golay filter failed: {e}")
        
        # Apply moving average filter if parameters are provided
        if moving_avg_window:
            try:
                filtered_signal = signal_filter.moving_average(
                    window_size=moving_avg_window,
                    iterations=moving_avg_iterations or 1,
                    method="edge"
                )
                logger.info(f"Moving average filter applied: window={moving_avg_window}, iterations={moving_avg_iterations}")
            except Exception as e:
                logger.warning(f"Moving average filter failed: {e}")
        
        # Apply Gaussian filter if parameters are provided
        if gaussian_sigma:
            try:
                filtered_signal = signal_filter.gaussian(
                    sigma=gaussian_sigma,
                    iterations=gaussian_iterations or 1
                )
                logger.info(f"Gaussian filter applied: sigma={gaussian_sigma}, iterations={gaussian_iterations}")
            except Exception as e:
                logger.warning(f"Gaussian filter failed: {e}")
        
        return filtered_signal
        
    except Exception as e:
        logger.error(f"Error applying additional traditional filters: {e}")
        return signal_data


def apply_enhanced_artifact_removal(signal_data, sampling_freq, artifact_type, artifact_strength,
                                  wavelet_type, wavelet_level, threshold_type, threshold_value,
                                  powerline_freq, notch_q_factor, pca_components, ica_components):
    """Apply enhanced artifact removal using vitalDSP functions with configurable parameters."""
    try:
        logger.info(f"Applying enhanced artifact removal: {artifact_type}")
        
        from vitalDSP.filtering.artifact_removal import ArtifactRemoval
        
        # Set default values
        artifact_type = artifact_type or "baseline"
        artifact_strength = artifact_strength or 0.5
        
        # Create artifact removal instance
        artifact_remover = ArtifactRemoval(signal_data)
        
        # Apply artifact removal based on type with enhanced parameters
        if artifact_type == "baseline":
            filtered_signal = artifact_remover.baseline_correction(
                cutoff=artifact_strength, 
                fs=sampling_freq
            )
        elif artifact_type == "spike":
            filtered_signal = artifact_remover.median_filter_removal()
        elif artifact_type == "noise":
            # Enhanced wavelet denoising with configurable parameters
            filtered_signal = artifact_remover.wavelet_denoising(
                wavelet_type=wavelet_type or "db4",
                level=wavelet_level or 3,
                threshold_type=threshold_type or "soft",
                threshold_value=threshold_value or 0.1
            )
        elif artifact_type == "powerline":
            filtered_signal = artifact_remover.notch_filter(
                freq=powerline_freq or 50, 
                fs=sampling_freq,
                Q=notch_q_factor or 30
            )
        elif artifact_type == "pca":
            filtered_signal = artifact_remover.pca_artifact_removal(
                num_components=pca_components or 1
            )
        elif artifact_type == "ica":
            filtered_signal = artifact_remover.ica_artifact_removal(
                num_components=ica_components or 2
            )
        else:
            # Default to baseline correction
            filtered_signal = artifact_remover.baseline_correction(
                cutoff=artifact_strength, 
                fs=sampling_freq
            )
        
        logger.info(f"Enhanced artifact removal applied successfully: {artifact_type}")
        return filtered_signal
        
    except Exception as e:
        logger.error(f"Error applying enhanced artifact removal: {e}")
        # Return original signal if artifact removal fails
        return signal_data


def apply_advanced_filter(signal_data, advanced_method, noise_level, iterations, learning_rate):
    """Apply advanced filtering using vitalDSP functions."""
    try:
        logger.info(f"Applying advanced filter: {advanced_method}")
        
        # Import vitalDSP advanced filtering functions
        from vitalDSP.filtering.advanced_signal_filtering import AdvancedSignalFiltering
        
        # Set default values
        advanced_method = advanced_method or "kalman"
        noise_level = noise_level or 0.1
        iterations = iterations or 100
        learning_rate = learning_rate or 0.01
        
        # Create advanced filter instance
        advanced_filter = AdvancedSignalFiltering(signal_data)
        
        # Apply filter based on method
        if advanced_method == "kalman":
            filtered_signal = advanced_filter.kalman_filter(R=noise_level, Q=1)
        elif advanced_method == "optimization":
            filtered_signal = advanced_filter.optimization_based_filtering(
                target=signal_data, 
                loss_type="mse", 
                learning_rate=learning_rate, 
                iterations=iterations
            )
        elif advanced_method == "gradient_descent":
            filtered_signal = advanced_filter.gradient_descent_filter(
                target=signal_data, 
                learning_rate=learning_rate, 
                iterations=iterations
            )
        elif advanced_method == "convolution":
            filtered_signal = advanced_filter.convolution_based_filter(
                kernel_type="smoothing", 
                kernel_size=5
            )
        elif advanced_method == "attention":
            filtered_signal = advanced_filter.attention_based_filter(
                attention_type="uniform", 
                size=5
            )
        else:
            # Default to Kalman filter
            filtered_signal = advanced_filter.kalman_filter(R=noise_level, Q=1)
        
        logger.info(f"Advanced filter applied successfully: {advanced_method}")
        return filtered_signal
        
    except Exception as e:
        logger.error(f"Error applying advanced filter: {e}")
        # Return original signal if advanced filtering fails
        return signal_data


def apply_neural_filter(signal_data, neural_type, neural_complexity):
    """Apply neural network filtering using vitalDSP functions."""
    try:
        logger.info(f"Applying neural filter: {neural_type}")
        
        # Import vitalDSP neural filtering functions
        from vitalDSP.advanced_computation.neural_network_filtering import NeuralNetworkFiltering
        
        # Set default values
        neural_type = neural_type or "feedforward"
        neural_complexity = neural_complexity or "medium"
        
        # Set complexity-based parameters
        if neural_complexity == "low":
            hidden_layers = [32]
            epochs = 50
        elif neural_complexity == "high":
            hidden_layers = [128, 64, 32]
            epochs = 200
        else:  # medium
            hidden_layers = [64, 32]
            epochs = 100
        
        # Create neural filter instance
        neural_filter = NeuralNetworkFiltering(
            signal_data, 
            network_type=neural_type,
            hidden_layers=hidden_layers,
            epochs=epochs
        )
        
        # Train the neural network first
        neural_filter.train()
        
        # Apply the trained filter
        filtered_signal = neural_filter.apply_filter()
        
        logger.info(f"Neural filter applied successfully: {neural_type}")
        return filtered_signal
        
    except Exception as e:
        logger.error(f"Error applying neural filter: {e}")
        # Return original signal if neural filtering fails
        return signal_data


def apply_ensemble_filter(signal_data, ensemble_method, ensemble_n_filters):
    """Apply ensemble filtering using vitalDSP functions."""
    try:
        logger.info(f"Applying ensemble filter: {ensemble_method}")
        
        # Import vitalDSP ensemble filtering functions
        from vitalDSP.filtering.advanced_signal_filtering import AdvancedSignalFiltering
        
        # Set default values
        ensemble_method = ensemble_method or "mean"
        ensemble_n_filters = ensemble_n_filters or 3
        
        # Create advanced filter instance
        advanced_filter = AdvancedSignalFiltering(signal_data)
        
        # Create multiple filters for ensemble
        filters = []
        for i in range(ensemble_n_filters):
            # Create different filter configurations
            if i == 0:
                # Kalman filter
                filtered = advanced_filter.kalman_filter(R=0.1, Q=1)
            elif i == 1:
                # Convolution filter
                filtered = advanced_filter.convolution_based_filter(kernel_type="smoothing", kernel_size=3)
            else:
                # Optimization filter
                filtered = advanced_filter.optimization_based_filtering(
                    target=signal_data, 
                    loss_type="mse", 
                    learning_rate=0.01, 
                    iterations=50
                )
            filters.append(filtered)
        
        # Apply ensemble method
        if ensemble_method == "mean":
            filtered_signal = np.mean(filters, axis=0)
        elif ensemble_method == "median":
            filtered_signal = np.median(filters, axis=0)
        elif ensemble_method == "weighted":
            # Weight by filter performance (inverse of variance)
            weights = 1 / (np.var(filters, axis=0) + 1e-10)
            weights = weights / np.sum(weights)
            filtered_signal = np.average(filters, axis=0, weights=weights)
        else:
            # Default to mean
            filtered_signal = np.mean(filters, axis=0)
        
        logger.info(f"Ensemble filter applied successfully: {ensemble_method} with {ensemble_n_filters} filters")
        return filtered_signal
        
    except Exception as e:
        logger.error(f"Error applying ensemble filter: {e}")
        # Return original signal if ensemble filtering fails
        return signal_data


def apply_enhanced_ensemble_filter(signal_data, ensemble_method, ensemble_n_filters, 
                                 ensemble_learning_rate, ensemble_iterations,
                                 adaptive_filter_order, adaptive_step_size,
                                 forgetting_factor, regularization_param):
    """Apply enhanced ensemble filtering with real-time capabilities."""
    try:
        logger.info(f"Applying enhanced ensemble filter: {ensemble_method}")
        
        from vitalDSP.filtering.advanced_signal_filtering import AdvancedSignalFiltering
        
        # Set default values
        ensemble_method = ensemble_method or "mean"
        ensemble_n_filters = ensemble_n_filters or 3
        
        # Create advanced filter instance
        advanced_filter = AdvancedSignalFiltering(signal_data)
        
        # Create multiple filters for ensemble
        filters = []
        for i in range(ensemble_n_filters):
            # Create different filter configurations
            if i == 0:
                # Kalman filter
                filtered = advanced_filter.kalman_filter(R=0.1, Q=1)
            elif i == 1:
                # Convolution filter
                filtered = advanced_filter.convolution_based_filter(kernel_type="smoothing", kernel_size=3)
            elif i == 2:
                # Optimization filter
                filtered = advanced_filter.optimization_based_filtering(
                    target=signal_data, 
                    loss_type="mse", 
                    learning_rate=ensemble_learning_rate or 0.01, 
                    iterations=ensemble_iterations or 50
                )
            elif i == 3:
                # Adaptive filtering
                filtered = advanced_filter.adaptive_filtering(
                    desired_signal=signal_data,
                    mu=adaptive_step_size or 0.5,
                    filter_order=adaptive_filter_order or 4
                )
            else:
                # Additional filter types
                filtered = advanced_filter.gradient_descent_filter(
                    target=signal_data,
                    learning_rate=ensemble_learning_rate or 0.01,
                    iterations=ensemble_iterations or 50
                )
            filters.append(filtered)
        
        # Apply enhanced ensemble method
        if ensemble_method == "mean":
            filtered_signal = np.mean(filters, axis=0)
        elif ensemble_method == "median":
            filtered_signal = np.median(filters, axis=0)
        elif ensemble_method == "weighted":
            # Weight by filter performance (inverse of variance)
            weights = 1 / (np.var(filters, axis=0) + 1e-10)
            weights = weights / np.sum(weights)
            filtered_signal = np.average(filters, axis=0, weights=weights)
        elif ensemble_method == "bagging":
            # Bootstrap aggregating
            n_samples = len(signal_data)
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            filtered_signal = np.mean([filters[i % len(filters)][indices] for i in range(len(filters))], axis=0)
        elif ensemble_method == "boosting":
            # Adaptive boosting
            weights = np.ones(len(filters))
            filtered_signal = np.zeros_like(signal_data)
            for i, (filter_output, weight) in enumerate(zip(filters, weights)):
                filtered_signal += weight * filter_output
                # Update weights based on performance
                error = np.mean((signal_data - filter_output) ** 2)
                weights[i] = 0.5 * np.log((1 - error) / (error + 1e-10))
        elif ensemble_method == "stacking":
            # Stacking with meta-learner
            # Use the first filter as meta-learner
            filtered_signal = filters[0]
        else:
            # Default to mean
            filtered_signal = np.mean(filters, axis=0)
        
        logger.info(f"Enhanced ensemble filter applied successfully: {ensemble_method} with {ensemble_n_filters} filters")
        return filtered_signal
        
    except Exception as e:
        logger.error(f"Error applying enhanced ensemble filter: {e}")
        # Return original signal if ensemble filtering fails
        return signal_data


def apply_multi_modal_filtering(signal_data, reference_signal, fusion_method, 
                               update_rate, performance_window):
    """Apply multi-modal filtering using vitalDSP functions."""
    try:
        logger.info(f"Applying multi-modal filtering: {fusion_method}")
        
        # Import multi-modal fusion functions
        from vitalDSP.respiratory_analysis.fusion.multimodal_analysis import MultiModalAnalysis
        
        # Create multi-modal analysis instance
        multimodal_analyzer = MultiModalAnalysis(signal_data)
        
        # Apply fusion based on method
        if fusion_method == "weighted":
            filtered_signal = multimodal_analyzer.weighted_fusion(
                reference_signal=reference_signal if reference_signal != "none" else None
            )
        elif fusion_method == "kalman":
            filtered_signal = multimodal_analyzer.kalman_fusion(
                reference_signal=reference_signal if reference_signal != "none" else None
            )
        elif fusion_method == "bayesian":
            filtered_signal = multimodal_analyzer.bayesian_fusion(
                reference_signal=reference_signal if reference_signal != "none" else None
            )
        elif fusion_method == "deep_learning":
            filtered_signal = multimodal_analyzer.deep_learning_fusion(
                reference_signal=reference_signal if reference_signal != "none" else None
            )
        else:
            # Default to weighted fusion
            filtered_signal = multimodal_analyzer.weighted_fusion(
                reference_signal=reference_signal if reference_signal != "none" else None
            )
        
        logger.info(f"Multi-modal filtering applied successfully: {fusion_method}")
        return filtered_signal
        
    except Exception as e:
        logger.error(f"Error applying multi-modal filtering: {e}")
        # Return original signal if multi-modal filtering fails
        return signal_data


def generate_filter_quality_metrics(original_signal, filtered_signal, sampling_freq, quality_options):
    """Generate comprehensive filter quality metrics with beautiful tables."""
    try:
        # Calculate basic metrics
        snr_improvement = calculate_snr_improvement(original_signal, filtered_signal)
        mse = calculate_mse(original_signal, filtered_signal)
        correlation = calculate_correlation(original_signal, filtered_signal)
        smoothness = calculate_smoothness(filtered_signal)
        
        # Calculate frequency domain metrics
        freq_metrics = calculate_frequency_metrics(original_signal, filtered_signal, sampling_freq)
        
        # Create beautiful metrics display with tables
        metrics_html = html.Div([
            html.H4("🔍 Filter Quality Assessment", className="text-center mb-4 text-primary"),
            
            # Signal Quality Table
            dbc.Card([
                dbc.CardHeader([
                    html.H5("📊 Signal Quality Metrics", className="mb-0 text-success")
                ], className="bg-success text-white"),
                dbc.CardBody([
                    dbc.Table([
                        html.Thead([
                            html.Tr([
                                html.Th("Metric", className="text-center"),
                                html.Th("Value", className="text-center"),
                                html.Th("Status", className="text-center")
                            ])
                        ]),
                        html.Tbody([
                            html.Tr([
                                html.Td("SNR Improvement", className="fw-bold"),
                                html.Td(f"{snr_improvement:.2f} dB", className="text-center"),
                                html.Td([
                                    html.Span("✅ Excellent" if snr_improvement > 10 else 
                                             "🟡 Good" if snr_improvement > 5 else 
                                             "🔴 Poor", 
                                             className=f"badge {'bg-success' if snr_improvement > 10 else 'bg-warning' if snr_improvement > 5 else 'bg-danger'}")
                                ], className="text-center")
                            ]),
                            html.Tr([
                                html.Td("Mean Square Error", className="fw-bold"),
                                html.Td(f"{mse:.4f}", className="text-center"),
                                html.Td([
                                    html.Span("✅ Excellent" if mse < 0.01 else 
                                             "🟡 Good" if mse < 0.1 else 
                                             "🔴 Poor", 
                                             className=f"badge {'bg-success' if mse < 0.01 else 'bg-warning' if mse < 0.1 else 'bg-danger'}")
                                ], className="text-center")
                            ]),
                            html.Tr([
                                html.Td("Correlation", className="fw-bold"),
                                html.Td(f"{correlation:.3f}", className="text-center"),
                                html.Td([
                                    html.Span("✅ Excellent" if correlation > 0.9 else 
                                             "🟡 Good" if correlation > 0.7 else 
                                             "🔴 Poor", 
                                             className=f"badge {'bg-success' if correlation > 0.9 else 'bg-warning' if correlation > 0.7 else 'bg-danger'}")
                                ], className="text-center")
                            ]),
                            html.Tr([
                                html.Td("Smoothness", className="fw-bold"),
                                html.Td(f"{smoothness:.3f}", className="text-center"),
                                html.Td([
                                    html.Span("✅ Excellent" if smoothness > 0.8 else 
                                             "🟡 Good" if smoothness > 0.6 else 
                                             "🔴 Poor", 
                                             className=f"badge {'bg-success' if smoothness > 0.8 else 'bg-warning' if smoothness > 0.6 else 'bg-danger'}")
                                ], className="text-center")
                            ])
                        ])
                    ], bordered=True, hover=True, responsive=True, striped=True, className="mb-3")
                ])
            ], className="mb-4"),
            
            # Frequency Analysis Table
            dbc.Card([
                dbc.CardHeader([
                    html.H5("🌊 Frequency Domain Analysis", className="mb-0 text-info")
                ], className="bg-info text-white"),
                dbc.CardBody([
                    dbc.Table([
                        html.Thead([
                            html.Tr([
                                html.Th("Parameter", className="text-center"),
                                html.Th("Original Signal", className="text-center"),
                                html.Th("Filtered Signal", className="text-center"),
                                html.Th("Improvement", className="text-center")
                            ])
                        ]),
                        html.Tbody([
                            html.Tr([
                                html.Td("Peak Frequency", className="fw-bold"),
                                html.Td(f"{freq_metrics.get('peak_freq_orig', 0):.2f} Hz", className="text-center"),
                                html.Td(f"{freq_metrics['peak_freq']:.2f} Hz", className="text-center"),
                                html.Td([
                                    html.Span(f"{freq_metrics.get('peak_freq_orig', 0) - freq_metrics['peak_freq']:.2f} Hz", 
                                             className="badge bg-secondary")
                                ], className="text-center")
                            ]),
                            html.Tr([
                                html.Td("Bandwidth", className="fw-bold"),
                                html.Td(f"{freq_metrics.get('bandwidth_orig', 0):.2f} Hz", className="text-center"),
                                html.Td(f"{freq_metrics['bandwidth']:.2f} Hz", className="text-center"),
                                html.Td([
                                    html.Span(f"{freq_metrics.get('bandwidth_orig', 0) - freq_metrics['bandwidth']:.2f} Hz", 
                                             className="badge bg-secondary")
                                ], className="text-center")
                            ]),
                            html.Tr([
                                html.Td("Spectral Centroid", className="fw-bold"),
                                html.Td(f"{freq_metrics.get('spectral_centroid_orig', 0):.2f} Hz", className="text-center"),
                                html.Td(f"{freq_metrics['spectral_centroid']:.2f} Hz", className="text-center"),
                                html.Td([
                                    html.Span(f"{freq_metrics.get('spectral_centroid_orig', 0) - freq_metrics['spectral_centroid']:.2f} Hz", 
                                             className="badge bg-secondary")
                                ], className="text-center")
                            ]),
                            html.Tr([
                                html.Td("Frequency Stability", className="fw-bold"),
                                html.Td(f"{freq_metrics.get('freq_stability_orig', 0):.3f}", className="text-center"),
                                html.Td(f"{freq_metrics['freq_stability']:.3f}", className="text-center"),
                                html.Td([
                                    html.Span(f"{freq_metrics.get('freq_stability_orig', 0) - freq_metrics['freq_stability']:.3f}", 
                                             className="badge bg-secondary")
                                ], className="text-center")
                            ])
                        ])
                    ], bordered=True, hover=True, responsive=True, striped=True, className="mb-3")
                ])
            ], className="mb-4"),
            
            # Summary Card
            dbc.Card([
                dbc.CardHeader([
                    html.H5("📋 Filter Performance Summary", className="mb-0 text-primary")
                ], className="bg-primary text-white"),
                dbc.CardBody([
            html.Div([
                html.Div([
                            html.H6("Overall Rating", className="text-center"),
                html.Div([
                                html.Span("🟢 EXCELLENT" if snr_improvement > 10 and correlation > 0.9 else 
                                         "🟡 GOOD" if snr_improvement > 5 and correlation > 0.7 else 
                                         "🔴 NEEDS IMPROVEMENT", 
                                         className=f"badge fs-6 {'bg-success' if snr_improvement > 10 and correlation > 0.9 else 'bg-warning' if snr_improvement > 5 and correlation > 0.7 else 'bg-danger'}")
                            ], className="text-center")
                        ], className="col-md-4"),
                        html.Div([
                            html.H6("Key Improvement", className="text-center"),
                            html.P(f"SNR enhanced by {snr_improvement:.1f} dB", className="text-center text-success mb-0")
                        ], className="col-md-4"),
                        html.Div([
                            html.H6("Signal Preservation", className="text-center"),
                            html.P(f"{(correlation * 100):.1f}% correlation maintained", className="text-center text-info mb-0")
                        ], className="col-md-4")
                    ], className="row text-center")
                ])
            ])
        ])
        
        return metrics_html
        
    except Exception as e:
        logger.error(f"Error generating quality metrics: {e}")
        return html.Div([
            html.H5("Filter Quality Metrics"),
            html.P(f"Error calculating metrics: {str(e)}")
        ])


def create_filter_quality_plots(original_signal, filtered_signal, sampling_freq, quality_options):
    """Create enhanced quality assessment plots with critical points detection."""
    try:
        # Create subplots for different quality metrics
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Signal Comparison with Critical Points", "Frequency Response Analysis", "Error Analysis", "Quality Metrics Over Time"),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # Time domain comparison with critical points
        time_axis = np.arange(len(original_signal)) / sampling_freq
        fig.add_trace(
            go.Scatter(x=time_axis, y=original_signal, mode='lines', name='Original Signal', line=dict(color='blue', width=2)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=time_axis, y=filtered_signal, mode='lines', name='Filtered Signal', line=dict(color='red', width=2)),
            row=1, col=1
        )
        
        # Add critical points detection to the comparison plot
        try:
            from vitalDSP.physiological_features.waveform import WaveformMorphology
            
            # Create waveform morphology object for original signal
            wm_orig = WaveformMorphology(
                waveform=original_signal,
                fs=sampling_freq,
                signal_type="PPG",
                simple_mode=True
            )
            
            # Create waveform morphology object for filtered signal
            wm_filt = WaveformMorphology(
                waveform=filtered_signal,
                fs=sampling_freq,
                signal_type="PPG",
                simple_mode=True
            )
            
            # Add systolic peaks for original signal
            if hasattr(wm_orig, 'systolic_peaks') and wm_orig.systolic_peaks is not None:
                fig.add_trace(
                            go.Scatter(
                                x=time_axis[wm_orig.systolic_peaks],
                                y=original_signal[wm_orig.systolic_peaks],
                                mode='markers',
                                name='Systolic Peaks (Original)',
                                marker=dict(color='darkblue', size=6, symbol='diamond'),
                                showlegend=False
                            ),
                            row=1, col=1
                        )
            
            # Add systolic peaks for filtered signal
            if hasattr(wm_filt, 'systolic_peaks') and wm_filt.systolic_peaks is not None:
                fig.add_trace(
                    go.Scatter(
                        x=time_axis[wm_filt.systolic_peaks],
                        y=filtered_signal[wm_filt.systolic_peaks],
                        mode='markers',
                        name='Systolic Peaks (Filtered)',
                        marker=dict(color='darkred', size=6, symbol='diamond'),
                        showlegend=False
                    ),
                    row=1, col=1
                )
                
        except Exception as e:
            logger.warning(f"Critical points detection failed in quality plots: {e}")
        
        # Enhanced frequency response analysis
        freqs_orig, psd_orig = signal.welch(original_signal, fs=sampling_freq, nperseg=min(256, len(original_signal)//4))
        freqs_filt, psd_filt = signal.welch(filtered_signal, fs=sampling_freq, nperseg=min(256, len(filtered_signal)//4))
        
        fig.add_trace(
            go.Scatter(x=freqs_orig, y=10*np.log10(psd_orig), mode='lines', name='Original PSD', line=dict(color='blue', width=2)),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=freqs_filt, y=10*np.log10(psd_filt), mode='lines', name='Filtered PSD', line=dict(color='red', width=2)),
            row=1, col=2
        )
        
        # Enhanced error analysis
        error = original_signal - filtered_signal
        fig.add_trace(
            go.Scatter(x=time_axis, y=error, mode='lines', name='Filtering Error', line=dict(color='orange', width=2)),
            row=2, col=1
        )
        
        # Add error statistics
        error_mean = np.mean(error)
        error_std = np.std(error)
        fig.add_annotation(
            text=f"Error: μ={error_mean:.2f}, σ={error_std:.2f}",
            xref="x3", yref="y3",
            x=0.02, y=0.98, showarrow=False,
            bgcolor="rgba(255,255,255,0.8)", bordercolor="orange"
        )
        
        # Enhanced quality metrics over time
        window_size = min(100, len(original_signal) // 10)
        if window_size > 0:
            quality_metrics = []
            correlation_metrics = []
            for i in range(0, len(original_signal) - window_size, window_size):
                orig_window = original_signal[i:i+window_size]
                filt_window = filtered_signal[i:i+window_size]
                snr = calculate_snr_improvement(orig_window, filt_window)
                corr = calculate_correlation(orig_window, filt_window)
                quality_metrics.append(snr)
                correlation_metrics.append(corr)
            
            time_centers = np.arange(len(quality_metrics)) * window_size / sampling_freq
            fig.add_trace(
                go.Scatter(x=time_centers, y=quality_metrics, mode='lines+markers', name='SNR over Time', 
                          line=dict(color='green', width=2), marker=dict(size=4)),
                row=2, col=2
            )
            fig.add_trace(
                go.Scatter(x=time_centers, y=correlation_metrics, mode='lines+markers', name='Correlation over Time', 
                          line=dict(color='purple', width=2), marker=dict(size=4)),
                row=2, col=2
            )
        
        # Update layout for all subplots
        fig.update_xaxes(title_text="Time (s)", row=1, col=1)
        fig.update_yaxes(title_text="Amplitude", row=1, col=1)
        
        fig.update_xaxes(title_text="Frequency (Hz)", row=1, col=2)
        fig.update_yaxes(title_text="Power Spectral Density (dB)", row=1, col=2)
        
        fig.update_xaxes(title_text="Time (s)", row=2, col=1)
        fig.update_yaxes(title_text="Error Amplitude", row=2, col=1)
        
        fig.update_xaxes(title_text="Time (s)", row=2, col=2)
        fig.update_yaxes(title_text="Metric Value", row=2, col=2)
        
        fig.update_layout(
            title="🔍 Enhanced Filter Quality Assessment with Critical Points Detection",
            height=700,
            showlegend=True,
            template="plotly_white",
            font=dict(size=12),
            margin=dict(l=60, r=60, t=80, b=60)
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating quality plots: {e}")
        return create_empty_figure()


# Quality metric calculation functions
def calculate_snr_improvement(original_signal, filtered_signal):
    """Calculate SNR improvement after filtering."""
    try:
        # Calculate noise as the difference between original and filtered
        noise = original_signal - filtered_signal
        
        # Calculate signal power (filtered signal)
        signal_power = np.mean(filtered_signal ** 2)
        
        # Calculate noise power
        noise_power = np.mean(noise ** 2)
        
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
            return snr
        else:
            return 0
    except Exception as e:
        logger.error(f"Error calculating SNR: {e}")
        return 0


def calculate_mse(original_signal, filtered_signal):
    """Calculate Mean Square Error between original and filtered signals."""
    try:
        return np.mean((original_signal - filtered_signal) ** 2)
    except Exception as e:
        logger.error(f"Error calculating MSE: {e}")
        return 0


def calculate_correlation(original_signal, filtered_signal):
    """Calculate correlation between original and filtered signals."""
    try:
        return np.corrcoef(original_signal, filtered_signal)[0, 1]
    except Exception as e:
        logger.error(f"Error calculating correlation: {e}")
        return 0


def calculate_smoothness(signal_data):
    """Calculate signal smoothness using variance of first differences."""
    try:
        differences = np.diff(signal_data)
        return 1 / (1 + np.var(differences))
    except Exception as e:
        logger.error(f"Error calculating smoothness: {e}")
        return 0


def calculate_frequency_metrics(original_signal, filtered_signal, sampling_freq):
    """Calculate frequency domain metrics for both original and filtered signals."""
    try:
        # Calculate PSD for both signals
        freqs_orig, psd_orig = signal.welch(original_signal, fs=sampling_freq)
        freqs_filt, psd_filt = signal.welch(filtered_signal, fs=sampling_freq)
        
        # Find peak frequency for filtered signal
        peak_idx_filt = np.argmax(psd_filt)
        peak_freq_filt = freqs_filt[peak_idx_filt]
        
        # Find peak frequency for original signal
        peak_idx_orig = np.argmax(psd_orig)
        peak_freq_orig = freqs_orig[peak_idx_orig]
        
        # Calculate bandwidth for filtered signal (3dB down from peak)
        peak_power_filt = psd_filt[peak_idx_filt]
        threshold_filt = peak_power_filt / 2  # 3dB down
        above_threshold_filt = psd_filt > threshold_filt
        bandwidth_filt = freqs_filt[-1] - freqs_filt[0] if np.any(above_threshold_filt) else 0
        
        # Calculate bandwidth for original signal
        peak_power_orig = psd_orig[peak_idx_orig]
        threshold_orig = peak_power_orig / 2  # 3dB down
        above_threshold_orig = psd_orig > threshold_orig
        bandwidth_orig = freqs_orig[-1] - freqs_orig[0] if np.any(above_threshold_orig) else 0
        
        # Calculate spectral centroid for filtered signal
        spectral_centroid_filt = np.sum(freqs_filt * psd_filt) / np.sum(psd_filt)
        
        # Calculate spectral centroid for original signal
        spectral_centroid_orig = np.sum(freqs_orig * psd_orig) / np.sum(psd_orig)
        
        # Calculate frequency stability for filtered signal (inverse of frequency variance)
        freq_stability_filt = 1 / (1 + np.var(freqs_filt))
        
        # Calculate frequency stability for original signal
        freq_stability_orig = 1 / (1 + np.var(freqs_orig))
        
        return {
            'peak_freq': peak_freq_filt,
            'peak_freq_orig': peak_freq_orig,
            'bandwidth': bandwidth_filt,
            'bandwidth_orig': bandwidth_orig,
            'spectral_centroid': spectral_centroid_filt,
            'spectral_centroid_orig': spectral_centroid_orig,
            'freq_stability': freq_stability_filt,
            'freq_stability_orig': freq_stability_orig
        }
        
    except Exception as e:
        logger.error(f"Error calculating frequency metrics: {e}")
        return {
            'peak_freq': 0,
            'bandwidth': 0,
            'spectral_centroid': 0,
            'freq_stability': 0
        }
