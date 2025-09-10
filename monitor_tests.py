#!/usr/bin/env python3
"""
Test performance monitoring script for vitalDSP project.
Helps identify slow tests and optimize execution.
"""

import subprocess
import time
import json
import os
from datetime import datetime

def run_test_analysis():
    """Run comprehensive test analysis."""
    print("🔍 Analyzing test performance...")
    
    # Get test collection info
    print("\n📊 Test Collection Analysis:")
    result = subprocess.run([
        "python", "-m", "pytest", "--collect-only", "-q"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        lines = result.stdout.split('\n')
        test_count = len([line for line in lines if 'test_' in line and '::' in line])
        print(f"   Total test functions: {test_count}")
        
        # Count by category
        unit_tests = len([line for line in lines if 'test_' in line and '::' in line and 'unit' in line.lower()])
        webapp_tests = len([line for line in lines if 'test_' in line and '::' in line and 'webapp' in line.lower()])
        core_tests = len([line for line in lines if 'test_' in line and '::' in line and 'vitaldsp/' in line])
        
        print(f"   Unit tests: {unit_tests}")
        print(f"   Webapp tests: {webapp_tests}")
        print(f"   Core tests: {core_tests}")
    else:
        print("   ❌ Failed to collect tests")
    
    # Run duration analysis
    print("\n⏱️  Test Duration Analysis:")
    result = subprocess.run([
        "python", "-m", "pytest", "--durations=20", "--durations-min=1.0", "-q"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("   Slowest tests (>1 second):")
        lines = result.stdout.split('\n')
        for line in lines:
            if 'seconds' in line and 'test_' in line:
                print(f"     {line.strip()}")
    else:
        print("   ❌ Failed to analyze durations")

def run_parallel_test():
    """Run a quick parallel test to measure performance."""
    print("\n🚀 Running parallel test execution...")
    
    start_time = time.time()
    
    result = subprocess.run([
        "python", "-m", "pytest", 
        "-n", "4",  # Use 4 workers
        "--timeout=60",
        "--maxfail=3",
        "-m", "unit",  # Only unit tests for speed
        "--durations=5"
    ], capture_output=True, text=True)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"   ⏱️  Execution time: {duration:.2f} seconds")
    
    if result.returncode == 0:
        print("   ✅ Parallel execution successful")
        if result.stdout:
            print("   📊 Results summary:")
            lines = result.stdout.split('\n')
            for line in lines[-10:]:  # Last 10 lines
                if line.strip():
                    print(f"     {line}")
    else:
        print("   ❌ Parallel execution failed")
        if result.stderr:
            print(f"   Error: {result.stderr[-200:]}")

def generate_test_report():
    """Generate a comprehensive test report."""
    print("\n📋 Generating test report...")
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "test_analysis": {},
        "recommendations": []
    }
    
    # Check for slow tests
    result = subprocess.run([
        "python", "-m", "pytest", "--durations=10", "--durations-min=5.0", "-q"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        slow_tests = [line.strip() for line in result.stdout.split('\n') 
                     if 'seconds' in line and 'test_' in line]
        report["slow_tests"] = slow_tests
        
        if slow_tests:
            report["recommendations"].append(
                f"Found {len(slow_tests)} slow tests (>5 seconds). Consider optimizing or marking as @pytest.mark.slow"
            )
    
    # Check test coverage
    result = subprocess.run([
        "python", "-m", "pytest", "--cov=src/vitalDSP", "--cov-report=term-missing", "-q"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        lines = result.stdout.split('\n')
        coverage_line = [line for line in lines if 'TOTAL' in line]
        if coverage_line:
            report["coverage"] = coverage_line[0].strip()
    
    # Save report
    with open("test_performance_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("   📄 Report saved to test_performance_report.json")
    
    # Print recommendations
    if report["recommendations"]:
        print("\n💡 Recommendations:")
        for rec in report["recommendations"]:
            print(f"   • {rec}")

def main():
    print("🧪 vitalDSP Test Performance Monitor")
    print("=" * 50)
    
    # Check if pytest-xdist is installed
    try:
        import xdist
        print("✅ pytest-xdist is available for parallel execution")
    except ImportError:
        print("⚠️  pytest-xdist not installed. Install with: pip install pytest-xdist")
    
    # Run analysis
    run_test_analysis()
    run_parallel_test()
    generate_test_report()
    
    print("\n🎯 Next Steps:")
    print("   1. Run 'python run_tests.py parallel' for maximum speed")
    print("   2. Run 'python run_tests.py full' for full coverage with optimizations")
    print("   3. Check test_performance_report.json for detailed analysis")

if __name__ == "__main__":
    main()
