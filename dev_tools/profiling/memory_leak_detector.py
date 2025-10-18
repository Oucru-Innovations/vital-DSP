"""
Memory Leak Detector for vitalDSP Phase 1-3 Components

Detects memory leaks by running components repeatedly and monitoring
memory growth patterns.

Author: vitalDSP Team
Date: October 17, 2025
Phase: 4 (Optimization & Testing)
"""

import gc
import psutil
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Callable, List, Dict, Any, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class MemorySnapshot:
    """Single memory measurement."""
    iteration: int
    rss_mb: float
    vms_mb: float
    timestamp: float
    objects_count: int


@dataclass
class LeakDetectionResult:
    """Result of leak detection analysis."""
    component_name: str
    iterations: int
    snapshots: List[MemorySnapshot]
    leak_detected: bool
    growth_rate_mb_per_iter: float
    total_growth_mb: float
    recommendations: List[str]

    def plot_memory_usage(self, output_path: Path = None):
        """Plot memory usage over iterations."""
        iterations = [s.iteration for s in self.snapshots]
        rss = [s.rss_mb for s in self.snapshots]

        plt.figure(figsize=(10, 6))
        plt.plot(iterations, rss, marker='o', linewidth=2, markersize=4)
        plt.xlabel('Iteration')
        plt.ylabel('Memory Usage (MB)')
        plt.title(f'Memory Usage: {self.component_name}')
        plt.grid(True, alpha=0.3)

        # Add trend line
        z = np.polyfit(iterations, rss, 1)
        p = np.poly1d(z)
        plt.plot(iterations, p(iterations), "r--", alpha=0.8, label=f'Trend: {z[0]:.2f} MB/iter')

        # Annotate leak detection
        if self.leak_detected:
            plt.text(
                0.5, 0.95,
                f'⚠️ LEAK DETECTED: {self.growth_rate_mb_per_iter:.2f} MB/iter',
                transform=plt.gca().transAxes,
                ha='center',
                va='top',
                bbox=dict(boxstyle='round', facecolor='red', alpha=0.3),
                fontsize=12,
                fontweight='bold'
            )
        else:
            plt.text(
                0.5, 0.95,
                '✅ No leak detected',
                transform=plt.gca().transAxes,
                ha='center',
                va='top',
                bbox=dict(boxstyle='round', facecolor='green', alpha=0.3),
                fontsize=12,
                fontweight='bold'
            )

        plt.legend()
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150)
            print(f"📊 Memory plot saved: {output_path}")
        else:
            plt.show()

        plt.close()


class MemoryLeakDetector:
    """
    Detect memory leaks by repeated execution.

    Strategy:
    1. Run function multiple times
    2. Force garbage collection between runs
    3. Monitor memory growth
    4. Detect linear growth pattern (indicates leak)
    """

    def __init__(self, output_dir: Path = None):
        """
        Initialize leak detector.

        Args:
            output_dir: Directory for reports and plots
        """
        self.output_dir = output_dir or Path('./memory_reports')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.process = psutil.Process(os.getpid())

    def detect_leaks(
        self,
        func: Callable,
        component_name: str,
        iterations: int = 20,
        warmup_iterations: int = 3,
        gc_between_runs: bool = True,
        *args,
        **kwargs
    ) -> LeakDetectionResult:
        """
        Run leak detection on a function.

        Args:
            func: Function to test
            component_name: Name of component
            iterations: Number of test iterations
            warmup_iterations: Initial iterations to skip (JIT warmup)
            gc_between_runs: Force garbage collection between iterations
            *args, **kwargs: Arguments to func

        Returns:
            LeakDetectionResult with analysis
        """
        print(f"\n{'='*60}")
        print(f"Memory Leak Detection: {component_name}")
        print(f"{'='*60}")
        print(f"Running {iterations} iterations...")

        snapshots: List[MemorySnapshot] = []

        # Warmup phase
        print(f"Warmup ({warmup_iterations} iterations)...")
        for i in range(warmup_iterations):
            func(*args, **kwargs)
            if gc_between_runs:
                gc.collect()

        # Test phase
        print(f"Testing ({iterations} iterations)...")
        for i in range(iterations):
            # Run function
            func(*args, **kwargs)

            # Force garbage collection if requested
            if gc_between_runs:
                gc.collect()

            # Take memory snapshot
            mem_info = self.process.memory_info()
            snapshot = MemorySnapshot(
                iteration=i,
                rss_mb=mem_info.rss / (1024 * 1024),
                vms_mb=mem_info.vms / (1024 * 1024),
                timestamp=time.time(),
                objects_count=len(gc.get_objects())
            )
            snapshots.append(snapshot)

            # Progress
            if (i + 1) % 5 == 0:
                print(f"  Iteration {i+1}/{iterations}: {snapshot.rss_mb:.1f}MB RSS")

        # Analyze results
        result = self._analyze_memory_growth(component_name, snapshots, iterations)

        # Print summary
        self._print_summary(result)

        # Generate plot
        plot_path = self.output_dir / f'{component_name.replace(" ", "_")}_memory.png'
        result.plot_memory_usage(plot_path)

        return result

    def _analyze_memory_growth(
        self,
        component_name: str,
        snapshots: List[MemorySnapshot],
        iterations: int
    ) -> LeakDetectionResult:
        """Analyze memory growth pattern."""
        # Extract RSS values
        rss_values = [s.rss_mb for s in snapshots]
        iterations_list = [s.iteration for s in snapshots]

        # Linear regression to detect growth trend
        z = np.polyfit(iterations_list, rss_values, 1)
        growth_rate = z[0]  # MB per iteration

        # Total growth
        total_growth = rss_values[-1] - rss_values[0]

        # Detect leak (threshold: >0.5 MB/iteration growth)
        leak_detected = growth_rate > 0.5

        # Generate recommendations
        recommendations = []
        if leak_detected:
            recommendations.append(
                f"⚠️ Memory leak detected: {growth_rate:.2f} MB/iteration"
            )
            recommendations.append(
                "Potential causes:"
                "\n  - Unreleased references to large objects"
                "\n  - Accumulating cache without bounds"
                "\n  - Circular references preventing GC"
                "\n  - Unclosed file handles or database connections"
            )
            recommendations.append(
                "Investigation steps:"
                "\n  1. Use tracemalloc to identify growing allocations"
                "\n  2. Check for circular references with gc.get_referrers()"
                "\n  3. Verify all resources are explicitly closed"
                "\n  4. Review caching strategies for size limits"
            )
        else:
            if abs(growth_rate) < 0.1:
                recommendations.append(
                    f"✅ Excellent: Stable memory usage ({growth_rate:.3f} MB/iter)"
                )
            else:
                recommendations.append(
                    f"✅ Good: Minor growth detected ({growth_rate:.2f} MB/iter)"
                )
                recommendations.append(
                    "This is typically acceptable and may be due to:"
                    "\n  - Python memory allocator fragmentation"
                    "\n  - Cached compiled code (JIT)"
                    "\n  - Normal runtime optimization"
                )

        return LeakDetectionResult(
            component_name=component_name,
            iterations=iterations,
            snapshots=snapshots,
            leak_detected=leak_detected,
            growth_rate_mb_per_iter=growth_rate,
            total_growth_mb=total_growth,
            recommendations=recommendations
        )

    def _print_summary(self, result: LeakDetectionResult):
        """Print leak detection summary."""
        print(f"\n{'─'*60}")
        print(f"LEAK DETECTION SUMMARY: {result.component_name}")
        print(f"{'─'*60}")
        print(f"Iterations: {result.iterations}")
        print(f"Growth Rate: {result.growth_rate_mb_per_iter:.3f} MB/iteration")
        print(f"Total Growth: {result.total_growth_mb:.1f} MB")

        if result.leak_detected:
            print(f"\n⚠️  LEAK DETECTED")
        else:
            print(f"\n✅ No leak detected")

        print(f"\n💡 Recommendations:")
        for rec in result.recommendations:
            print(f"   {rec}")

        print(f"{'─'*60}\n")

    def detect_leaks_in_loop(
        self,
        func: Callable,
        component_name: str,
        iterations: int = 100,
        check_interval: int = 10,
        *args,
        **kwargs
    ) -> LeakDetectionResult:
        """
        Detect leaks in a loop-based function.

        Useful for testing components that process data in batches.

        Args:
            func: Function that runs a loop internally
            component_name: Component name
            iterations: How many times to call the function
            check_interval: How often to check memory
            *args, **kwargs: Arguments to func

        Returns:
            LeakDetectionResult
        """
        print(f"\n{'='*60}")
        print(f"Loop-based Leak Detection: {component_name}")
        print(f"{'='*60}")

        snapshots: List[MemorySnapshot] = []

        for i in range(iterations):
            # Run function
            func(*args, **kwargs)

            # Check memory at intervals
            if i % check_interval == 0:
                gc.collect()
                mem_info = self.process.memory_info()
                snapshot = MemorySnapshot(
                    iteration=i,
                    rss_mb=mem_info.rss / (1024 * 1024),
                    vms_mb=mem_info.vms / (1024 * 1024),
                    timestamp=time.time(),
                    objects_count=len(gc.get_objects())
                )
                snapshots.append(snapshot)
                print(f"  Check {len(snapshots)}: {snapshot.rss_mb:.1f}MB RSS")

        # Analyze
        result = self._analyze_memory_growth(component_name, snapshots, len(snapshots))
        self._print_summary(result)

        # Plot
        plot_path = self.output_dir / f'{component_name.replace(" ", "_")}_loop_memory.png'
        result.plot_memory_usage(plot_path)

        return result

    def compare_memory_profiles(
        self,
        func1: Callable,
        func2: Callable,
        name1: str,
        name2: str,
        iterations: int = 20,
        *args,
        **kwargs
    ):
        """
        Compare memory profiles of two implementations.

        Args:
            func1: First implementation
            func2: Second implementation
            name1: Name of first
            name2: Name of second
            iterations: Test iterations
            *args, **kwargs: Arguments to both functions
        """
        print(f"\n{'='*60}")
        print(f"Memory Profile Comparison")
        print(f"{'='*60}")

        # Test both
        result1 = self.detect_leaks(func1, name1, iterations, *args, **kwargs)
        result2 = self.detect_leaks(func2, name2, iterations, *args, **kwargs)

        # Compare
        print(f"\n{'─'*60}")
        print("COMPARISON")
        print(f"{'─'*60}")

        growth_diff = result1.growth_rate_mb_per_iter - result2.growth_rate_mb_per_iter
        if abs(growth_diff) < 0.1:
            print(f"Memory growth: Similar ({abs(growth_diff):.3f} MB/iter difference)")
        elif growth_diff > 0:
            print(f"💾 {name2} has {abs(growth_diff):.3f} MB/iter less growth")
        else:
            print(f"💾 {name1} has {abs(growth_diff):.3f} MB/iter less growth")

        total_diff = result1.total_growth_mb - result2.total_growth_mb
        if abs(total_diff) < 1:
            print(f"Total growth: Similar ({abs(total_diff):.1f} MB difference)")
        elif total_diff > 0:
            print(f"💾 {name2} grew {abs(total_diff):.1f} MB less overall")
        else:
            print(f"💾 {name1} grew {abs(total_diff):.1f} MB less overall")

        # Combined plot
        plt.figure(figsize=(12, 6))

        iterations1 = [s.iteration for s in result1.snapshots]
        rss1 = [s.rss_mb for s in result1.snapshots]
        iterations2 = [s.iteration for s in result2.snapshots]
        rss2 = [s.rss_mb for s in result2.snapshots]

        plt.plot(iterations1, rss1, marker='o', label=name1, linewidth=2, markersize=4)
        plt.plot(iterations2, rss2, marker='s', label=name2, linewidth=2, markersize=4)

        plt.xlabel('Iteration')
        plt.ylabel('Memory Usage (MB)')
        plt.title('Memory Usage Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_path = self.output_dir / 'memory_comparison.png'
        plt.savefig(plot_path, dpi=150)
        print(f"\n📊 Comparison plot saved: {plot_path}")
        plt.close()

        print(f"{'─'*60}\n")


if __name__ == '__main__':
    # Example usage
    detector = MemoryLeakDetector()

    # Example 1: Function with no leak
    def no_leak_function():
        data = np.random.randn(1000000)
        return data.sum()

    result1 = detector.detect_leaks(
        no_leak_function,
        "No Leak Example",
        iterations=20
    )

    # Example 2: Function with intentional leak
    global_cache = []

    def leaking_function():
        # Intentionally leak memory
        data = np.random.randn(100000)
        global_cache.append(data)  # Never released!
        return data.sum()

    result2 = detector.detect_leaks(
        leaking_function,
        "Leaking Example",
        iterations=20
    )

    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)
    print(f"Check {detector.output_dir} for plots and reports")
