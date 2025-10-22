"""
Comprehensive vitalDSP Webapp Performance Evaluation Script

This script evaluates the performance of the vitalDSP webapp integration,
including:
- Core infrastructure integration (processing_pipeline, quality_screener)
- Enhanced data service performance
- Heavy data filtering service performance
- Memory usage and optimization
- Webapp responsiveness and throughput

Author: vitalDSP Development Team
Date: October 16, 2025
"""

import sys
import time
import psutil
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add vitalDSP to path
vitaldsp_path = Path(__file__).parent / "src" / "vitalDSP"
webapp_path = Path(__file__).parent / "src" / "vitalDSP_webapp"
if str(vitaldsp_path) not in sys.path:
    sys.path.insert(0, str(vitaldsp_path))
if str(webapp_path) not in sys.path:
    sys.path.insert(0, str(webapp_path))


class PerformanceMetrics:
    """Container for performance metrics."""

    def __init__(self, test_name: str):
        self.test_name = test_name
        self.start_time = None
        self.end_time = None
        self.memory_before = None
        self.memory_after = None
        self.memory_peak = None
        self.success = False
        self.error_message = None
        self.additional_metrics = {}

    def start(self):
        """Start performance tracking."""
        self.start_time = time.time()
        self.memory_before = psutil.Process().memory_info().rss / (1024**2)  # MB

    def end(self, success: bool = True, error_message: str = None):
        """End performance tracking."""
        self.end_time = time.time()
        self.memory_after = psutil.Process().memory_info().rss / (1024**2)  # MB
        self.success = success
        self.error_message = error_message

    def get_duration(self) -> float:
        """Get test duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0

    def get_memory_increase(self) -> float:
        """Get memory increase in MB."""
        if self.memory_before and self.memory_after:
            return self.memory_after - self.memory_before
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_name": self.test_name,
            "duration_seconds": self.get_duration(),
            "memory_before_mb": self.memory_before,
            "memory_after_mb": self.memory_after,
            "memory_increase_mb": self.get_memory_increase(),
            "success": self.success,
            "error_message": self.error_message,
            **self.additional_metrics
        }


class WebappPerformanceEvaluator:
    """Main webapp performance evaluator."""

    def __init__(self):
        """Initialize evaluator."""
        self.results = []
        self.core_available = False
        self.webapp_available = False

        # Try importing core infrastructure
        try:
            from vitalDSP.utils.core_infrastructure.processing_pipeline import StandardProcessingPipeline
            from vitalDSP.utils.core_infrastructure.quality_screener import QualityScreener
            from vitalDSP.utils.core_infrastructure.data_loaders import ChunkedDataLoader
            self.core_available = True
            self.StandardProcessingPipeline = StandardProcessingPipeline
            self.QualityScreener = QualityScreener
            self.ChunkedDataLoader = ChunkedDataLoader
            logger.info("✅ Core vitalDSP infrastructure available")
        except ImportError as e:
            logger.warning(f"❌ Core vitalDSP infrastructure not available: {e}")

        # Try importing webapp services
        try:
            from services.data.enhanced_data_service import EnhancedDataService
            from services.filtering.heavy_data_filtering_service import HeavyDataFilteringService
            self.webapp_available = True
            self.EnhancedDataService = EnhancedDataService
            self.HeavyDataFilteringService = HeavyDataFilteringService
            logger.info("✅ Webapp services available")
        except ImportError as e:
            logger.warning(f"❌ Webapp services not available: {e}")

    def evaluate_all(self) -> Dict[str, Any]:
        """Run all performance evaluations."""
        logger.info("="*70)
        logger.info("VITALDSP WEBAPP PERFORMANCE EVALUATION")
        logger.info("="*70)

        # Test 1: Core Infrastructure - Processing Pipeline Stage 3
        if self.core_available:
            self.test_processing_pipeline_stage3()
            self.test_quality_screener_parallel()
            self.test_chunked_data_loader()

        # Test 2: Webapp Integration
        if self.webapp_available:
            self.test_enhanced_data_service()
            self.test_heavy_data_filtering_service()

        # Test 3: End-to-End Performance
        if self.core_available and self.webapp_available:
            self.test_end_to_end_pipeline()

        # Generate report
        return self.generate_report()

    def test_processing_pipeline_stage3(self):
        """Test processing pipeline Stage 3 (multi-path processing)."""
        logger.info("\n[TEST 1] Processing Pipeline Stage 3 - Multi-Path Processing")
        logger.info("-"*70)

        metrics = PerformanceMetrics("processing_pipeline_stage3")
        metrics.start()

        try:
            # Create test signal
            signal_length = 10000  # 10k samples
            sampling_freq = 100  # 100 Hz
            signal = np.sin(2 * np.pi * 1.0 * np.linspace(0, signal_length/sampling_freq, signal_length))
            signal += np.random.normal(0, 0.1, signal_length)

            logger.info(f"  Signal: {signal_length} samples @ {sampling_freq} Hz ({signal_length/sampling_freq:.1f}s)")

            # Initialize pipeline
            pipeline = self.StandardProcessingPipeline()

            # Process signal through all 8 stages
            result = pipeline.process_signal(
                signal=signal,
                fs=sampling_freq,
                signal_type="ECG"
            )

            # Verify Stage 3 completed
            assert "parallel_processing" in result.get("processing_results", {}), "Stage 3 not in results"
            stage3_result = result["processing_results"]["parallel_processing"]

            # Verify all 4 paths processed
            paths = stage3_result.get("data", {}).get("paths", {})
            assert len(paths) == 4, f"Expected 4 paths, got {len(paths)}"
            assert "raw" in paths, "Raw path missing"
            assert "filtered" in paths, "Filtered path missing"
            assert "preprocessed" in paths, "Preprocessed path missing"
            assert "full" in paths, "Full path missing"

            # Get best path
            best_path = stage3_result.get("data", {}).get("comparison", {}).get("best_path", "unknown")

            logger.info(f"  ✅ Stage 3 completed successfully")
            logger.info(f"  Paths processed: {list(paths.keys())}")
            logger.info(f"  Best path selected: {best_path}")

            metrics.additional_metrics = {
                "signal_length": signal_length,
                "sampling_freq": sampling_freq,
                "paths_processed": len(paths),
                "best_path": best_path
            }

            metrics.end(success=True)

        except Exception as e:
            logger.error(f"  ❌ Stage 3 test failed: {e}")
            metrics.end(success=False, error_message=str(e))

        self.results.append(metrics)

    def test_quality_screener_parallel(self):
        """Test quality screener parallel processing."""
        logger.info("\n[TEST 2] Quality Screener - Parallel Processing")
        logger.info("-"*70)

        metrics = PerformanceMetrics("quality_screener_parallel")
        metrics.start()

        try:
            # Create test signal
            signal_length = 60000  # 60k samples (10 minutes @ 100Hz)
            signal = np.sin(2 * np.pi * 1.0 * np.linspace(0, 600, signal_length))
            signal += np.random.normal(0, 0.1, signal_length)

            logger.info(f"  Signal: {signal_length} samples (10 minutes @ 100 Hz)")

            # Initialize screener with parallel processing
            screener = self.QualityScreener(
                signal_type="ECG",
                sampling_rate=100,
                enable_parallel=True
            )

            # Screen signal
            results = screener.screen_signal(signal)

            # Verify results
            assert len(results) > 0, "No screening results returned"

            logger.info(f"  ✅ Parallel screening completed")
            logger.info(f"  Segments screened: {len(results)}")
            logger.info(f"  Quality scores: {[r.quality_metrics.overall_quality for r in results[:3]]}...")

            metrics.additional_metrics = {
                "signal_length": signal_length,
                "segments_screened": len(results),
                "average_quality": np.mean([r.quality_metrics.overall_quality for r in results])
            }

            metrics.end(success=True)

        except Exception as e:
            logger.error(f"  ❌ Quality screener test failed: {e}")
            metrics.end(success=False, error_message=str(e))

        self.results.append(metrics)

    def test_chunked_data_loader(self):
        """Test chunked data loader performance."""
        logger.info("\n[TEST 3] Chunked Data Loader - Performance")
        logger.info("-"*70)

        metrics = PerformanceMetrics("chunked_data_loader")
        metrics.start()

        try:
            # Create temporary test data file
            test_file = Path("test_data_large.csv")

            # Generate large dataset (simulating 50MB file)
            data_size = 500000  # 500k rows
            df = pd.DataFrame({
                'time': np.arange(data_size),
                'signal': np.sin(2 * np.pi * 0.01 * np.arange(data_size)) + np.random.normal(0, 0.1, data_size)
            })
            df.to_csv(test_file, index=False)

            file_size_mb = test_file.stat().st_size / (1024**2)
            logger.info(f"  Test file: {file_size_mb:.1f} MB, {data_size} rows")

            # Load using chunked loader
            loader = self.ChunkedDataLoader(
                file_path=str(test_file),
                chunk_size="auto"
            )

            chunks = list(loader.load_chunks(max_chunks=10))

            logger.info(f"  ✅ Chunked loading completed")
            logger.info(f"  Chunks loaded: {len(chunks)}")
            logger.info(f"  Chunk sizes: {[len(c) for c in chunks]}")

            metrics.additional_metrics = {
                "file_size_mb": file_size_mb,
                "total_rows": data_size,
                "chunks_loaded": len(chunks),
                "average_chunk_size": np.mean([len(c) for c in chunks])
            }

            metrics.end(success=True)

            # Cleanup
            test_file.unlink()

        except Exception as e:
            logger.error(f"  ❌ Chunked data loader test failed: {e}")
            metrics.end(success=False, error_message=str(e))

        self.results.append(metrics)

    def test_enhanced_data_service(self):
        """Test enhanced data service."""
        logger.info("\n[TEST 4] Enhanced Data Service - Webapp Integration")
        logger.info("-"*70)

        metrics = PerformanceMetrics("enhanced_data_service")
        metrics.start()

        try:
            # Initialize service
            service = self.EnhancedDataService(max_memory_mb=500)

            logger.info(f"  Service initialized with 500MB memory limit")

            # Get service stats
            stats = service.get_service_stats()

            logger.info(f"  ✅ Enhanced data service functional")
            logger.info(f"  Total requests: {stats.get('total_requests', 0)}")
            logger.info(f"  Memory usage: {stats.get('memory_usage_mb', 0):.1f} MB")

            metrics.additional_metrics = {
                "total_requests": stats.get('total_requests', 0),
                "memory_usage_mb": stats.get('memory_usage_mb', 0)
            }

            metrics.end(success=True)

            # Cleanup
            service.cleanup()

        except Exception as e:
            logger.error(f"  ❌ Enhanced data service test failed: {e}")
            metrics.end(success=False, error_message=str(e))

        self.results.append(metrics)

    def test_heavy_data_filtering_service(self):
        """Test heavy data filtering service."""
        logger.info("\n[TEST 5] Heavy Data Filtering Service - Webapp Integration")
        logger.info("-"*70)

        metrics = PerformanceMetrics("heavy_data_filtering_service")
        metrics.start()

        try:
            # Initialize service
            from services.filtering.heavy_data_filtering_service import (
                HeavyDataFilteringService,
                FilteringRequest
            )

            service = HeavyDataFilteringService(max_memory_mb=1000)

            logger.info(f"  Service initialized with 1000MB memory limit")

            # Create test filtering request
            signal = np.sin(2 * np.pi * 1.0 * np.linspace(0, 10, 10000)) + np.random.normal(0, 0.1, 10000)

            request = FilteringRequest(
                request_id="test_001",
                signal_data=signal,
                sampling_freq=100.0,
                signal_type="ECG",
                filter_params={
                    "filter_type": "bandpass",
                    "low_freq": 0.5,
                    "high_freq": 40.0,
                    "filter_order": 4,
                    "sampling_freq": 100.0
                }
            )

            # Process request
            result = service.process_filtering_request(request)

            # Get stats
            stats = service.get_statistics()

            logger.info(f"  ✅ Heavy data filtering service functional")
            logger.info(f"  Total requests: {stats['service_stats']['total_requests']}")
            logger.info(f"  Successful: {stats['service_stats']['successful_requests']}")
            logger.info(f"  Average time: {stats['service_stats']['average_processing_time']:.3f}s")

            metrics.additional_metrics = {
                "total_requests": stats['service_stats']['total_requests'],
                "successful_requests": stats['service_stats']['successful_requests'],
                "average_processing_time": stats['service_stats']['average_processing_time']
            }

            metrics.end(success=True)

        except Exception as e:
            logger.error(f"  ❌ Heavy data filtering service test failed: {e}")
            metrics.end(success=False, error_message=str(e))

        self.results.append(metrics)

    def test_end_to_end_pipeline(self):
        """Test end-to-end pipeline integration."""
        logger.info("\n[TEST 6] End-to-End Pipeline - Full Integration")
        logger.info("-"*70)

        metrics = PerformanceMetrics("end_to_end_pipeline")
        metrics.start()

        try:
            # Create test signal
            signal_length = 100000  # 100k samples (16.67 minutes @ 100Hz)
            sampling_freq = 100
            signal = np.sin(2 * np.pi * 1.0 * np.linspace(0, signal_length/sampling_freq, signal_length))
            signal += np.random.normal(0, 0.1, signal_length)

            logger.info(f"  Signal: {signal_length} samples ({signal_length/sampling_freq/60:.1f} minutes @ {sampling_freq} Hz)")

            # Step 1: Quality Screening
            screener = self.QualityScreener(signal_type="ECG", sampling_rate=sampling_freq, enable_parallel=True)
            quality_results = screener.screen_signal(signal)

            logger.info(f"  Step 1: Quality screening - {len(quality_results)} segments")

            # Step 2: Processing Pipeline
            pipeline = self.StandardProcessingPipeline()
            processing_result = pipeline.process_signal(
                signal=signal,
                fs=sampling_freq,
                signal_type="ECG"
            )

            logger.info(f"  Step 2: Processing pipeline - 8 stages completed")

            # Step 3: Get processing stats
            stats = pipeline.get_processing_statistics()

            logger.info(f"  ✅ End-to-end pipeline completed")
            logger.info(f"  Total processing time: {stats['pipeline_stats']['total_processing_time']:.3f}s")
            logger.info(f"  Stages completed: {stats['pipeline_stats']['stages_completed']}")
            logger.info(f"  Cache hit rate: {stats['cache_stats']['hit_rate']:.2%}")

            metrics.additional_metrics = {
                "signal_length": signal_length,
                "quality_segments": len(quality_results),
                "stages_completed": stats['pipeline_stats']['stages_completed'],
                "total_processing_time": stats['pipeline_stats']['total_processing_time'],
                "cache_hit_rate": stats['cache_stats']['hit_rate']
            }

            metrics.end(success=True)

        except Exception as e:
            logger.error(f"  ❌ End-to-end pipeline test failed: {e}")
            metrics.end(success=False, error_message=str(e))

        self.results.append(metrics)

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        logger.info("\n" + "="*70)
        logger.info("PERFORMANCE EVALUATION REPORT")
        logger.info("="*70)

        # Calculate summary statistics
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - successful_tests

        total_duration = sum(r.get_duration() for r in self.results)
        total_memory = sum(r.get_memory_increase() for r in self.results)

        # Print summary
        logger.info(f"\nSummary:")
        logger.info(f"  Total tests: {total_tests}")
        logger.info(f"  Successful: {successful_tests} ({successful_tests/total_tests*100:.1f}%)")
        logger.info(f"  Failed: {failed_tests}")
        logger.info(f"  Total duration: {total_duration:.3f}s")
        logger.info(f"  Total memory increase: {total_memory:.1f} MB")

        # Print individual test results
        logger.info(f"\nIndividual Test Results:")
        logger.info("-"*70)

        for result in self.results:
            status = "✅ PASS" if result.success else "❌ FAIL"
            logger.info(f"  {result.test_name}:")
            logger.info(f"    Status: {status}")
            logger.info(f"    Duration: {result.get_duration():.3f}s")
            logger.info(f"    Memory: {result.get_memory_increase():.1f} MB")
            if result.error_message:
                logger.info(f"    Error: {result.error_message}")
            logger.info("")

        # Generate report dict
        report = {
            "summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "failed_tests": failed_tests,
                "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
                "total_duration_seconds": total_duration,
                "total_memory_increase_mb": total_memory
            },
            "tests": [r.to_dict() for r in self.results],
            "core_available": self.core_available,
            "webapp_available": self.webapp_available,
            "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        # Save report to file
        report_file = Path("webapp_performance_report.json")
        import json
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Report saved to: {report_file}")

        return report


def main():
    """Main evaluation function."""
    evaluator = WebappPerformanceEvaluator()
    report = evaluator.evaluate_all()

    # Print final assessment
    logger.info("\n" + "="*70)
    logger.info("FINAL ASSESSMENT")
    logger.info("="*70)

    success_rate = report['summary']['success_rate']

    if success_rate == 1.0:
        logger.info("✅ EXCELLENT: All tests passed!")
        logger.info("   Webapp integration is production-ready.")
    elif success_rate >= 0.8:
        logger.info("✅ GOOD: Most tests passed.")
        logger.info("   Webapp integration is functional with minor issues.")
    elif success_rate >= 0.5:
        logger.info("⚠️  FAIR: Some tests failed.")
        logger.info("   Webapp integration needs improvements.")
    else:
        logger.info("❌ POOR: Many tests failed.")
        logger.info("   Webapp integration has significant issues.")

    logger.info("="*70)


if __name__ == "__main__":
    main()
