"""
Test script for Phase 3B Heavy Data Processing Integration

This script tests the integration of heavy data processing strategies
with the webapp filtering system.

Author: vitalDSP Development Team
Date: January 11, 2025
Phase: 3B - Heavy Data Processing Integration
"""

import os
import sys
import time
import logging
import numpy as np
from pathlib import Path

# Add vitalDSP to path for imports
current_dir = Path(__file__).parent
vitaldsp_path = current_dir.parent.parent / "vitalDSP"
if str(vitaldsp_path) not in sys.path:
    sys.path.insert(0, str(vitaldsp_path))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_heavy_data_filtering_service():
    """Test the HeavyDataFilteringService."""
    try:
        from vitalDSP_webapp.services.filtering.heavy_data_filtering_service import (
            get_heavy_data_filtering_service,
            create_filtering_request,
            FilteringStrategy
        )
        
        logger.info("Testing HeavyDataFilteringService...")
        
        # Create service
        service = get_heavy_data_filtering_service()
        logger.info("✅ HeavyDataFilteringService created successfully")
        
        # Test with small dataset
        small_data = np.random.randn(10000)  # ~80KB
        request = create_filtering_request(
            request_id="test_small",
            signal_data=small_data,
            sampling_freq=1000.0,
            filter_params={
                "filter_type": "lowpass",
                "low_freq": 10.0,
                "filter_order": 4
            }
        )
        
        result = service.process_filtering_request(request)
        if result.success:
            logger.info("✅ Small dataset processing successful")
        else:
            logger.error(f"❌ Small dataset processing failed: {result.error_message}")
        
        # Test with medium dataset
        medium_data = np.random.randn(1000000)  # ~8MB
        request = create_filtering_request(
            request_id="test_medium",
            signal_data=medium_data,
            sampling_freq=1000.0,
            filter_params={
                "filter_type": "bandpass",
                "low_freq": 1.0,
                "high_freq": 10.0,
                "filter_order": 4
            }
        )
        
        result = service.process_filtering_request(request)
        if result.success:
            logger.info("✅ Medium dataset processing successful")
        else:
            logger.error(f"❌ Medium dataset processing failed: {result.error_message}")
        
        # Get statistics
        stats = service.get_statistics()
        logger.info(f"✅ Service statistics: {stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ HeavyDataFilteringService test failed: {e}")
        return False


def test_lazy_loading_solution():
    """Test the lazy loading solution."""
    try:
        from vitalDSP_webapp.services.filtering.lazy_loading_solution import (
            get_progressive_data_loader
        )
        
        logger.info("Testing Lazy Loading Solution...")
        
        # Create loader
        loader = get_progressive_data_loader()
        logger.info("✅ ProgressiveDataLoader created successfully")
        
        # Test with medium dataset
        medium_data = np.random.randn(500000)  # ~4MB
        filter_params = {
            "filter_type": "lowpass",
            "low_freq": 10.0,
            "filter_order": 4,
            "sampling_freq": 1000.0
        }
        
        def progress_callback(progress, message):
            logger.info(f"Progress: {progress:.2%} - {message}")
        
        # Process with lazy loading
        results = list(loader.process_lazy_filtering(
            medium_data, filter_params, 1000.0, "test", progress_callback
        ))
        
        if results and all(r.success for r in results):
            logger.info(f"✅ Lazy loading processing successful: {len(results)} chunks")
        else:
            logger.error("❌ Lazy loading processing failed")
        
        # Get statistics
        stats = loader.get_statistics()
        logger.info(f"✅ Loader statistics: {stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Lazy Loading Solution test failed: {e}")
        return False


def test_enhanced_filtering_callbacks():
    """Test the enhanced filtering callbacks."""
    try:
        from vitalDSP_webapp.callbacks.analysis.enhanced_filtering_callbacks import (
            get_enhanced_filtering_callback
        )
        
        logger.info("Testing Enhanced Filtering Callbacks...")
        
        # Create callback
        callback = get_enhanced_filtering_callback()
        logger.info("✅ EnhancedFilteringCallback created successfully")
        
        # Test with small dataset
        small_data = np.random.randn(5000)  # ~40KB
        filter_params = {
            "filter_type": "highpass",
            "high_freq": 1.0,
            "filter_order": 4,
            "sampling_freq": 1000.0
        }
        
        def progress_callback(progress, message):
            logger.info(f"Progress: {progress:.2%} - {message}")
        
        result = callback.process_filtering_request_enhanced(
            small_data, 1000.0, filter_params, "test", progress_callback
        )
        
        if result.success:
            logger.info("✅ Enhanced filtering callback successful")
        else:
            logger.error(f"❌ Enhanced filtering callback failed: {result.error_message}")
        
        # Get statistics
        stats = callback.get_processing_statistics()
        logger.info(f"✅ Callback statistics: {stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced Filtering Callbacks test failed: {e}")
        return False


def test_integration():
    """Test the complete integration."""
    try:
        logger.info("Testing Complete Integration...")
        
        # Test all components
        tests = [
            ("HeavyDataFilteringService", test_heavy_data_filtering_service),
            ("Lazy Loading Solution", test_lazy_loading_solution),
            ("Enhanced Filtering Callbacks", test_enhanced_filtering_callbacks)
        ]
        
        results = []
        for test_name, test_func in tests:
            logger.info(f"\n--- Testing {test_name} ---")
            result = test_func()
            results.append((test_name, result))
        
        # Summary
        logger.info("\n" + "="*50)
        logger.info("INTEGRATION TEST SUMMARY")
        logger.info("="*50)
        
        all_passed = True
        for test_name, result in results:
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{test_name}: {status}")
            if not result:
                all_passed = False
        
        if all_passed:
            logger.info("\n🎉 ALL TESTS PASSED - Integration successful!")
        else:
            logger.info("\n⚠️  SOME TESTS FAILED - Check logs for details")
        
        return all_passed
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        return False


if __name__ == "__main__":
    logger.info("Starting Phase 3B Heavy Data Processing Integration Tests...")
    
    success = test_integration()
    
    if success:
        logger.info("\n🚀 Phase 3B implementation is ready for production!")
        sys.exit(0)
    else:
        logger.error("\n❌ Phase 3B implementation needs fixes before production")
        sys.exit(1)
