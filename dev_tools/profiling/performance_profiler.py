"""
Performance Profiler for vitalDSP Phase 1-3 Components

Profiles CPU usage, memory usage, and execution time for each component
to identify bottlenecks and optimization opportunities.

Author: vitalDSP Team
Date: October 17, 2025
Phase: 4 (Optimization & Testing)
"""

import cProfile
import pstats
import io
import time
import psutil
import os
import tracemalloc
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class ProfileResult:
    """Container for profiling results."""
    component_name: str
    execution_time_sec: float
    cpu_time_sec: float
    peak_memory_mb: float
    memory_increase_mb: float
    function_stats: List[Dict[str, Any]] = field(default_factory=list)
    hotspots: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'component': self.component_name,
            'execution_time_sec': self.execution_time_sec,
            'cpu_time_sec': self.cpu_time_sec,
            'peak_memory_mb': self.peak_memory_mb,
            'memory_increase_mb': self.memory_increase_mb,
            'hotspots': self.hotspots,
            'recommendations': self.recommendations,
        }


class PerformanceProfiler:
    """
    Comprehensive performance profiler for vitalDSP components.

    Profiles:
    - CPU time per function
    - Memory allocation per function
    - Peak memory usage
    - Execution time
    - Identifies hot paths and bottlenecks
    """

    def __init__(self, output_dir: Path = None):
        """
        Initialize profiler.

        Args:
            output_dir: Directory to save profiling reports (default: ./profiling_reports)
        """
        self.output_dir = output_dir or Path('./profiling_reports')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.process = psutil.Process(os.getpid())
        self.results: List[ProfileResult] = []

    def profile_function(
        self,
        func: Callable,
        component_name: str,
        *args,
        **kwargs
    ) -> Tuple[ProfileResult, Any]:
        """
        Profile a function with comprehensive metrics.

        Args:
            func: Function to profile
            component_name: Name of component being profiled
            *args: Positional arguments to func
            **kwargs: Keyword arguments to func

        Returns:
            Tuple of (ProfileResult, function return value)
        """
        print(f"\n{'='*60}")
        print(f"Profiling: {component_name}")
        print(f"{'='*60}")

        # Start memory tracking
        tracemalloc.start()
        initial_memory = self.process.memory_info().rss / (1024 * 1024)  # MB

        # Start CPU profiling
        profiler = cProfile.Profile()
        profiler.enable()

        # Execute function
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        # Stop profiling
        profiler.disable()

        # Get memory stats
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        final_memory = self.process.memory_info().rss / (1024 * 1024)

        # Parse profiling stats
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.strip_dirs()
        stats.sort_stats('cumulative')

        # Extract top functions
        function_stats = []
        stats.print_stats(20)  # Top 20 functions
        stream.seek(0)
        lines = stream.readlines()

        # Parse stats output
        for line in lines:
            if line.strip() and not line.startswith(('ncalls', '---', ' ')):
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        function_stats.append({
                            'ncalls': parts[0],
                            'tottime': float(parts[1]),
                            'cumtime': float(parts[3]),
                            'function': ' '.join(parts[5:])
                        })
                    except (ValueError, IndexError):
                        pass

        # Identify hotspots (functions taking >5% of total time)
        total_time = end_time - start_time
        hotspots = []
        for stat in function_stats[:10]:
            if stat['cumtime'] > total_time * 0.05:
                hotspots.append(
                    f"{stat['function']}: {stat['cumtime']:.3f}s ({stat['cumtime']/total_time*100:.1f}%)"
                )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            execution_time=total_time,
            memory_increase=final_memory - initial_memory,
            peak_memory=peak / (1024 * 1024),
            function_stats=function_stats
        )

        # Create result
        profile_result = ProfileResult(
            component_name=component_name,
            execution_time_sec=total_time,
            cpu_time_sec=sum(s['cumtime'] for s in function_stats[:5]),
            peak_memory_mb=peak / (1024 * 1024),
            memory_increase_mb=final_memory - initial_memory,
            function_stats=function_stats[:10],
            hotspots=hotspots,
            recommendations=recommendations
        )

        self.results.append(profile_result)

        # Print summary
        self._print_profile_summary(profile_result)

        return profile_result, result

    def _generate_recommendations(
        self,
        execution_time: float,
        memory_increase: float,
        peak_memory: float,
        function_stats: List[Dict]
    ) -> List[str]:
        """Generate optimization recommendations based on profiling."""
        recommendations = []

        # Time-based recommendations
        if execution_time > 10.0:
            recommendations.append(
                f"⚠️ Long execution time ({execution_time:.1f}s). Consider:"
                "\n  - Parallel processing if not already enabled"
                "\n  - Algorithmic optimizations for hot paths"
                "\n  - Caching intermediate results"
            )

        # Memory-based recommendations
        if memory_increase > 500:
            recommendations.append(
                f"⚠️ High memory increase ({memory_increase:.0f}MB). Consider:"
                "\n  - Streaming/chunked processing"
                "\n  - In-place operations where possible"
                "\n  - Memory-mapped arrays for large data"
            )

        if peak_memory > 1000:
            recommendations.append(
                f"⚠️ High peak memory ({peak_memory:.0f}MB). Consider:"
                "\n  - Generator-based processing"
                "\n  - Smaller chunk sizes"
                "\n  - Explicit garbage collection"
            )

        # Function-based recommendations
        if function_stats:
            top_func = function_stats[0]
            if top_func['cumtime'] > execution_time * 0.5:
                recommendations.append(
                    f"🔥 Hot path detected: {top_func['function']}"
                    f"\n  - Takes {top_func['cumtime']/execution_time*100:.1f}% of execution time"
                    "\n  - Priority target for optimization"
                )

        if not recommendations:
            recommendations.append("✅ Performance looks good! No major issues detected.")

        return recommendations

    def _print_profile_summary(self, result: ProfileResult):
        """Print profiling summary."""
        print(f"\n{'─'*60}")
        print(f"PROFILING SUMMARY: {result.component_name}")
        print(f"{'─'*60}")
        print(f"⏱️  Execution Time: {result.execution_time_sec:.3f}s")
        print(f"💾 Memory Increase: {result.memory_increase_mb:.1f}MB")
        print(f"📊 Peak Memory: {result.peak_memory_mb:.1f}MB")

        if result.hotspots:
            print(f"\n🔥 Hotspots:")
            for hotspot in result.hotspots[:5]:
                print(f"   {hotspot}")

        if result.recommendations:
            print(f"\n💡 Recommendations:")
            for rec in result.recommendations:
                print(f"   {rec}")

        print(f"{'─'*60}\n")

    def generate_report(self, filename: str = None) -> Path:
        """
        Generate comprehensive profiling report.

        Args:
            filename: Report filename (default: profiling_report_TIMESTAMP.md)

        Returns:
            Path to generated report
        """
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'profiling_report_{timestamp}.md'

        report_path = self.output_dir / filename

        report = []
        report.append("# vitalDSP Performance Profiling Report")
        report.append(f"\n**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**System:** {psutil.cpu_count()} cores, {psutil.virtual_memory().total / (1024**3):.1f}GB RAM")
        report.append(f"**Python:** {os.sys.version.split()[0]}")
        report.append("\n---\n")

        # Executive summary
        report.append("## Executive Summary")
        report.append(f"\n**Components Profiled:** {len(self.results)}")

        total_time = sum(r.execution_time_sec for r in self.results)
        total_memory = sum(r.peak_memory_mb for r in self.results)

        report.append(f"**Total Execution Time:** {total_time:.2f}s")
        report.append(f"**Peak Memory Usage:** {total_memory:.1f}MB")

        # Find bottlenecks
        slowest = max(self.results, key=lambda r: r.execution_time_sec)
        most_memory = max(self.results, key=lambda r: r.peak_memory_mb)

        report.append(f"\n**Slowest Component:** {slowest.component_name} ({slowest.execution_time_sec:.2f}s)")
        report.append(f"**Highest Memory:** {most_memory.component_name} ({most_memory.peak_memory_mb:.1f}MB)")

        report.append("\n---\n")

        # Detailed results per component
        report.append("## Detailed Component Analysis\n")

        for result in sorted(self.results, key=lambda r: r.execution_time_sec, reverse=True):
            report.append(f"### {result.component_name}\n")
            report.append(f"**Execution Time:** {result.execution_time_sec:.3f}s")
            report.append(f"**Memory Increase:** {result.memory_increase_mb:.1f}MB")
            report.append(f"**Peak Memory:** {result.peak_memory_mb:.1f}MB\n")

            if result.hotspots:
                report.append("**Hotspots:**")
                for hotspot in result.hotspots:
                    report.append(f"- {hotspot}")
                report.append("")

            if result.function_stats:
                report.append("**Top Functions:**")
                report.append("| Function | Calls | Time (s) | Cumulative (s) |")
                report.append("|----------|-------|----------|----------------|")
                for stat in result.function_stats[:5]:
                    func_name = stat['function'][:50]  # Truncate long names
                    report.append(
                        f"| {func_name} | {stat['ncalls']} | "
                        f"{stat['tottime']:.3f} | {stat['cumtime']:.3f} |"
                    )
                report.append("")

            if result.recommendations:
                report.append("**Recommendations:**")
                for rec in result.recommendations:
                    # Format multiline recommendations
                    lines = rec.split('\n')
                    for line in lines:
                        report.append(f"{line}")
                report.append("")

            report.append("---\n")

        # Overall recommendations
        report.append("## Overall Recommendations\n")

        # Identify cross-component patterns
        avg_time = total_time / len(self.results)
        slow_components = [r for r in self.results if r.execution_time_sec > avg_time * 1.5]

        if slow_components:
            report.append("### Priority Optimization Targets\n")
            for comp in slow_components:
                report.append(f"**{comp.component_name}**")
                report.append(f"- {comp.execution_time_sec / total_time * 100:.1f}% of total time")
                report.append(f"- {comp.peak_memory_mb:.1f}MB peak memory")
                report.append("")

        # Memory optimization opportunities
        high_memory = [r for r in self.results if r.peak_memory_mb > 100]
        if high_memory:
            report.append("### Memory Optimization Opportunities\n")
            for comp in high_memory:
                report.append(f"- **{comp.component_name}**: {comp.peak_memory_mb:.1f}MB peak")

        report.append("\n---\n")
        report.append(f"\n*Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

        # Save report
        report_path.write_text('\n'.join(report))

        # Also save JSON for programmatic access
        json_path = self.output_dir / filename.replace('.md', '.json')
        json_data = {
            'timestamp': datetime.now().isoformat(),
            'system': {
                'cpu_count': psutil.cpu_count(),
                'total_memory_gb': psutil.virtual_memory().total / (1024**3),
            },
            'results': [r.to_dict() for r in self.results]
        }
        json_path.write_text(json.dumps(json_data, indent=2))

        print(f"\n✅ Profiling report saved:")
        print(f"   📄 {report_path}")
        print(f"   📊 {json_path}")

        return report_path

    def compare_implementations(
        self,
        func1: Callable,
        func2: Callable,
        name1: str,
        name2: str,
        *args,
        **kwargs
    ):
        """
        Compare two implementations of the same functionality.

        Args:
            func1: First implementation
            func2: Second implementation
            name1: Name of first implementation
            name2: Name of second implementation
            *args, **kwargs: Arguments to both functions
        """
        print(f"\n{'='*60}")
        print(f"COMPARING: {name1} vs {name2}")
        print(f"{'='*60}")

        # Profile both
        result1, output1 = self.profile_function(func1, name1, *args, **kwargs)
        result2, output2 = self.profile_function(func2, name2, *args, **kwargs)

        # Compare
        print(f"\n{'─'*60}")
        print("COMPARISON RESULTS")
        print(f"{'─'*60}")

        time_diff = result1.execution_time_sec - result2.execution_time_sec
        time_pct = abs(time_diff) / result1.execution_time_sec * 100

        if time_diff > 0:
            print(f"⚡ {name2} is {time_pct:.1f}% faster ({abs(time_diff):.3f}s)")
        else:
            print(f"⚡ {name1} is {time_pct:.1f}% faster ({abs(time_diff):.3f}s)")

        mem_diff = result1.peak_memory_mb - result2.peak_memory_mb
        mem_pct = abs(mem_diff) / result1.peak_memory_mb * 100 if result1.peak_memory_mb > 0 else 0

        if mem_diff > 0:
            print(f"💾 {name2} uses {mem_pct:.1f}% less memory ({abs(mem_diff):.1f}MB)")
        else:
            print(f"💾 {name1} uses {mem_pct:.1f}% less memory ({abs(mem_diff):.1f}MB)")

        print(f"{'─'*60}\n")


if __name__ == '__main__':
    # Example usage
    profiler = PerformanceProfiler()

    # Example: Profile a simple function
    def example_function(n):
        return np.random.randn(n).sum()

    result, output = profiler.profile_function(
        example_function,
        "Example Computation",
        10000000
    )

    # Generate report
    profiler.generate_report()
