"""
Enhanced Signal Filtering Callbacks with Heavy Data Processing Integration

This module extends the existing signal filtering callbacks to integrate with the
HeavyDataFilteringService for efficient processing of large datasets.

Features:
- Integration with HeavyDataFilteringService
- Progressive filtering with real-time updates
- Lazy loading for large datasets
- Memory-optimized processing
- WebSocket integration for real-time progress updates

Author: vitalDSP Development Team
Date: January 11, 2025
Phase: 3B - Heavy Data Processing Integration
"""

import os
import sys
import time
import threading
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable, Generator, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
import uuid
from datetime import datetime

# Add vitalDSP to path for imports
current_dir = Path(__file__).parent
vitaldsp_path = current_dir.parent.parent.parent / "vitalDSP"
if str(vitaldsp_path) not in sys.path:
    sys.path.insert(0, str(vitaldsp_path))

try:
    # Import heavy data filtering service
    from vitalDSP_webapp.services.filtering.heavy_data_filtering_service import (
        HeavyDataFilteringService,
        FilteringRequest,
        FilteringResult,
        FilteringStrategy,
        FilteringMode,
        get_heavy_data_filtering_service,
        create_filtering_request
    )
    
    # Import WebSocket manager for real-time updates
    from vitalDSP_webapp.services.async_services.websocket_manager import get_websocket_manager
    
    HEAVY_DATA_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Heavy data filtering service not available: {e}")
    HEAVY_DATA_AVAILABLE = False

logger = logging.getLogger(__name__)


class EnhancedFilteringCallback:
    """
    Enhanced filtering callback that integrates heavy data processing strategies.
    
    This class provides a bridge between the existing webapp filtering callbacks
    and the new heavy data processing capabilities.
    """
    
    def __init__(self):
        """Initialize enhanced filtering callback."""
        self.heavy_data_service = get_heavy_data_filtering_service() if HEAVY_DATA_AVAILABLE else None
        self.websocket_manager = get_websocket_manager() if HEAVY_DATA_AVAILABLE else None
        
        # Processing state tracking
        self.active_requests = {}
        self.processing_stats = {
            "total_requests": 0,
            "heavy_data_requests": 0,
            "standard_requests": 0,
            "average_processing_time": 0.0,
            "memory_usage_peak": 0
        }
        
        logger.info("EnhancedFilteringCallback initialized")
    
    def should_use_heavy_data_processing(
        self, 
        signal_data: np.ndarray, 
        filter_params: Dict[str, Any]
    ) -> bool:
        """
        Determine if heavy data processing should be used.
        
        Args:
            signal_data: Signal data to process
            filter_params: Filtering parameters
            
        Returns:
            True if heavy data processing should be used
        """
        if not HEAVY_DATA_AVAILABLE or self.heavy_data_service is None:
            return False
        
        # Calculate data size in MB
        data_size_mb = signal_data.nbytes / (1024 * 1024)
        
        # Use heavy data processing for:
        # - Large datasets (> 50MB)
        # - Complex filtering operations
        # - Memory-constrained environments
        use_heavy_processing = (
            data_size_mb > 50 or  # Large datasets
            filter_params.get("complexity", "medium") == "high" or  # Complex operations
            filter_params.get("use_heavy_processing", False)  # Explicit request
        )
        
        logger.info(f"Data size: {data_size_mb:.1f}MB, Use heavy processing: {use_heavy_processing}")
        return use_heavy_processing
    
    def process_filtering_request_enhanced(
        self,
        signal_data: np.ndarray,
        sampling_freq: float,
        filter_params: Dict[str, Any],
        signal_type: str = "unknown",
        progress_callback: Optional[Callable[[float, str], None]] = None,
        completion_callback: Optional[Callable[[FilteringResult], None]] = None
    ) -> Union[FilteringResult, Generator[FilteringResult, None, None]]:
        """
        Process filtering request with enhanced heavy data processing.
        
        Args:
            signal_data: Signal data to filter
            sampling_freq: Sampling frequency
            filter_params: Filtering parameters
            signal_type: Type of signal
            progress_callback: Progress callback function
            completion_callback: Completion callback function
            
        Returns:
            FilteringResult or Generator of FilteringResults
        """
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        self.processing_stats["total_requests"] += 1
        
        try:
            # Determine processing strategy
            use_heavy_processing = self.should_use_heavy_data_processing(signal_data, filter_params)
            
            if use_heavy_processing and self.heavy_data_service:
                self.processing_stats["heavy_data_requests"] += 1
                return self._process_with_heavy_data_service(
                    request_id, signal_data, sampling_freq, filter_params, 
                    signal_type, progress_callback, completion_callback
                )
            else:
                self.processing_stats["standard_requests"] += 1
                return self._process_with_standard_method(
                    request_id, signal_data, sampling_freq, filter_params, 
                    signal_type, progress_callback, completion_callback
                )
                
        except Exception as e:
            logger.error(f"Error in enhanced filtering: {e}")
            return FilteringResult(
                request_id=request_id,
                success=False,
                error_message=str(e),
                processing_time=time.time() - start_time
            )
    
    def _process_with_heavy_data_service(
        self,
        request_id: str,
        signal_data: np.ndarray,
        sampling_freq: float,
        filter_params: Dict[str, Any],
        signal_type: str,
        progress_callback: Optional[Callable] = None,
        completion_callback: Optional[Callable] = None
    ) -> Union[FilteringResult, Generator[FilteringResult, None, None]]:
        """Process using heavy data service."""
        try:
            # Create filtering request
            request = create_filtering_request(
                request_id=request_id,
                signal_data=signal_data,
                sampling_freq=sampling_freq,
                filter_params=filter_params,
                signal_type=signal_type
            )
            
            # Track active request
            self.active_requests[request_id] = {
                "start_time": time.time(),
                "status": "processing",
                "progress": 0.0
            }
            
            # Enhanced progress callback with WebSocket updates
            def enhanced_progress_callback(progress: float, message: str):
                if progress_callback:
                    progress_callback(progress, message)
                
                # Update WebSocket if available
                if self.websocket_manager:
                    try:
                        import asyncio
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        
                        async def broadcast_update():
                            from vitalDSP_webapp.services.async_services.websocket_manager import WebSocketMessage
                            ws_message = WebSocketMessage(
                                type="filtering_progress",
                                data={
                                    "request_id": request_id,
                                    "progress": progress,
                                    "message": message,
                                    "timestamp": datetime.now().isoformat()
                                }
                            )
                            return await self.websocket_manager.broadcast_to_all(ws_message)
                        
                        loop.run_until_complete(broadcast_update())
                        loop.close()
                    except Exception as e:
                        logger.warning(f"Failed to broadcast WebSocket update: {e}")
                
                # Update active request
                if request_id in self.active_requests:
                    self.active_requests[request_id]["progress"] = progress
                    self.active_requests[request_id]["message"] = message
            
            # Process request
            result = self.heavy_data_service.process_filtering_request(
                request, enhanced_progress_callback
            )
            
            # Handle completion
            if completion_callback:
                if isinstance(result, Generator):
                    # For progressive processing, call completion callback for each result
                    for chunk_result in result:
                        completion_callback(chunk_result)
                        yield chunk_result
                else:
                    completion_callback(result)
                    return result
            
            return result
            
        except Exception as e:
            logger.error(f"Error in heavy data service processing: {e}")
            raise
    
    def _process_with_standard_method(
        self,
        request_id: str,
        signal_data: np.ndarray,
        sampling_freq: float,
        filter_params: Dict[str, Any],
        signal_type: str,
        progress_callback: Optional[Callable] = None,
        completion_callback: Optional[Callable] = None
    ) -> FilteringResult:
        """Process using standard method (fallback)."""
        try:
            if progress_callback:
                progress_callback(0.1, "Initializing standard processing...")
            
            # Apply standard filtering (using existing webapp logic)
            filtered_signal = self._apply_standard_filter(signal_data, sampling_freq, filter_params)
            
            if progress_callback:
                progress_callback(1.0, "Standard processing completed")
            
            result = FilteringResult(
                request_id=request_id,
                success=True,
                filtered_signal=filtered_signal,
                original_signal=signal_data,
                strategy_used=FilteringStrategy.STANDARD,
                metadata={"processing_method": "standard_fallback"}
            )
            
            if completion_callback:
                completion_callback(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in standard method processing: {e}")
            return FilteringResult(
                request_id=request_id,
                success=False,
                error_message=str(e)
            )
    
    def _apply_standard_filter(
        self, 
        signal_data: np.ndarray, 
        sampling_freq: float, 
        filter_params: Dict[str, Any]
    ) -> np.ndarray:
        """Apply standard filtering using scipy."""
        try:
            from scipy import signal
            
            filtered_data = signal_data.copy()
            
            # Apply filter based on parameters
            filter_type = filter_params.get("filter_type", "lowpass")
            
            if filter_type == "lowpass":
                nyquist = sampling_freq / 2
                cutoff = filter_params.get("low_freq", 10) / nyquist
                b, a = signal.butter(filter_params.get("filter_order", 4), cutoff, btype='low')
                filtered_data = signal.filtfilt(b, a, filtered_data)
            
            elif filter_type == "highpass":
                nyquist = sampling_freq / 2
                cutoff = filter_params.get("high_freq", 1) / nyquist
                b, a = signal.butter(filter_params.get("filter_order", 4), cutoff, btype='high')
                filtered_data = signal.filtfilt(b, a, filtered_data)
            
            elif filter_type == "bandpass":
                nyquist = sampling_freq / 2
                low_cutoff = filter_params.get("low_freq", 1) / nyquist
                high_cutoff = filter_params.get("high_freq", 10) / nyquist
                b, a = signal.butter(filter_params.get("filter_order", 4), [low_cutoff, high_cutoff], btype='band')
                filtered_data = signal.filtfilt(b, a, filtered_data)
            
            return filtered_data
            
        except Exception as e:
            logger.error(f"Error applying standard filter: {e}")
            return signal_data
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        stats = self.processing_stats.copy()
        
        # Add heavy data service stats if available
        if self.heavy_data_service:
            heavy_stats = self.heavy_data_service.get_statistics()
            stats["heavy_data_service"] = heavy_stats
        
        # Add active requests info
        stats["active_requests"] = len(self.active_requests)
        stats["active_request_details"] = {
            req_id: {
                "status": req_info["status"],
                "progress": req_info["progress"],
                "duration": time.time() - req_info["start_time"]
            }
            for req_id, req_info in self.active_requests.items()
        }
        
        return stats
    
    def cancel_request(self, request_id: str) -> bool:
        """Cancel an active processing request."""
        try:
            if request_id in self.active_requests:
                self.active_requests[request_id]["status"] = "cancelled"
                
                # Cancel in heavy data service if available
                if self.heavy_data_service:
                    # Note: Heavy data service cancellation would need to be implemented
                    pass
                
                logger.info(f"Cancelled request {request_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error cancelling request {request_id}: {e}")
            return False
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.heavy_data_service:
            self.heavy_data_service.cleanup()
        
        self.active_requests.clear()
        logger.info("EnhancedFilteringCallback cleaned up")


# Global enhanced filtering callback instance
_enhanced_filtering_callback = None


def get_enhanced_filtering_callback() -> EnhancedFilteringCallback:
    """Get global enhanced filtering callback instance."""
    global _enhanced_filtering_callback
    if _enhanced_filtering_callback is None:
        _enhanced_filtering_callback = EnhancedFilteringCallback()
    return _enhanced_filtering_callback


def integrate_heavy_data_filtering_with_callbacks():
    """
    Integrate heavy data filtering with existing webapp filtering callbacks.
    
    This function modifies the existing filtering callbacks to use the enhanced
    heavy data processing capabilities when appropriate.
    """
    try:
        # Import the existing filtering callbacks
        from vitalDSP_webapp.callbacks.analysis.signal_filtering_callbacks import (
            apply_traditional_filter,
            apply_advanced_filter,
            apply_neural_filter,
            apply_ensemble_filter
        )
        
        # Get enhanced filtering callback
        enhanced_callback = get_enhanced_filtering_callback()
        
        # Create wrapper functions that use heavy data processing when appropriate
        def enhanced_apply_traditional_filter(
            signal_data: np.ndarray,
            sampling_freq: float,
            filter_family: str,
            filter_response: str,
            low_freq: Optional[float] = None,
            high_freq: Optional[float] = None,
            filter_order: int = 4
        ) -> np.ndarray:
            """Enhanced traditional filter with heavy data processing."""
            filter_params = {
                "filter_type": filter_response,
                "filter_family": filter_family,
                "low_freq": low_freq,
                "high_freq": high_freq,
                "filter_order": filter_order,
                "sampling_freq": sampling_freq
            }
            
            result = enhanced_callback.process_filtering_request_enhanced(
                signal_data, sampling_freq, filter_params, "unknown"
            )
            
            if isinstance(result, Generator):
                # For progressive processing, collect all chunks
                filtered_chunks = []
                for chunk_result in result:
                    if chunk_result.success and chunk_result.filtered_signal is not None:
                        filtered_chunks.append(chunk_result.filtered_signal)
                
                # Concatenate chunks
                if filtered_chunks:
                    return np.concatenate(filtered_chunks)
                else:
                    return signal_data
            else:
                return result.filtered_signal if result.success else signal_data
        
        def enhanced_apply_advanced_filter(
            signal_data: np.ndarray,
            advanced_method: str,
            noise_level: Optional[float] = None,
            iterations: Optional[int] = None,
            learning_rate: Optional[float] = None
        ) -> np.ndarray:
            """Enhanced advanced filter with heavy data processing."""
            filter_params = {
                "filter_type": "advanced",
                "advanced_method": advanced_method,
                "noise_level": noise_level,
                "iterations": iterations,
                "learning_rate": learning_rate,
                "complexity": "high"  # Advanced filters are complex
            }
            
            result = enhanced_callback.process_filtering_request_enhanced(
                signal_data, 100.0, filter_params, "unknown"
            )
            
            if isinstance(result, Generator):
                # For progressive processing, collect all chunks
                filtered_chunks = []
                for chunk_result in result:
                    if chunk_result.success and chunk_result.filtered_signal is not None:
                        filtered_chunks.append(chunk_result.filtered_signal)
                
                # Concatenate chunks
                if filtered_chunks:
                    return np.concatenate(filtered_chunks)
                else:
                    return signal_data
            else:
                return result.filtered_signal if result.success else signal_data
        
        # Replace the original functions with enhanced versions
        import vitalDSP_webapp.callbacks.analysis.signal_filtering_callbacks as filtering_module
        
        # Monkey patch the functions
        filtering_module.apply_traditional_filter = enhanced_apply_traditional_filter
        filtering_module.apply_advanced_filter = enhanced_apply_advanced_filter
        
        logger.info("Successfully integrated heavy data filtering with existing callbacks")
        
        return True
        
    except Exception as e:
        logger.error(f"Error integrating heavy data filtering: {e}")
        return False


def register_enhanced_filtering_callbacks(app):
    """
    Register enhanced filtering callbacks with heavy data processing.
    
    Args:
        app: Dash application instance
    """
    try:
        # Get enhanced filtering callback
        enhanced_callback = get_enhanced_filtering_callback()
        
        # Register WebSocket endpoint for real-time progress updates
        if enhanced_callback.websocket_manager:
            @app.server.route('/ws/filtering-progress')
            def filtering_progress_websocket():
                """WebSocket endpoint for filtering progress updates."""
                return enhanced_callback.websocket_manager.handle_websocket()
        
        # Register API endpoint for processing statistics
        @app.server.route('/api/filtering/statistics')
        def get_filtering_statistics():
            """Get filtering processing statistics."""
            return enhanced_callback.get_processing_statistics()
        
        # Register API endpoint for cancelling requests
        @app.server.route('/api/filtering/cancel/<request_id>', methods=['POST'])
        def cancel_filtering_request(request_id):
            """Cancel a filtering request."""
            success = enhanced_callback.cancel_request(request_id)
            return {"success": success, "request_id": request_id}
        
        logger.info("Enhanced filtering callbacks registered successfully")
        
    except Exception as e:
        logger.error(f"Error registering enhanced filtering callbacks: {e}")


# Auto-integrate when module is imported
if HEAVY_DATA_AVAILABLE:
    integrate_heavy_data_filtering_with_callbacks()
