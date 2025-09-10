#!/usr/bin/env python3
"""
Test runner script for vitalDSP project.
Provides different test configurations for different scenarios.
"""

import subprocess
import sys
import argparse
import os

def run_command(cmd, description):
    """Run a command and return the result."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ SUCCESS")
        if result.stdout:
            print("STDOUT:", result.stdout[-500:])  # Last 500 chars
    else:
        print("❌ FAILED")
        if result.stderr:
            print("STDERR:", result.stderr[-500:])  # Last 500 chars
        if result.stdout:
            print("STDOUT:", result.stdout[-500:])  # Last 500 chars
    
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(description="Run different test suites for vitalDSP")
    parser.add_argument(
        "suite", 
        choices=["fast", "ci", "unit", "core", "webapp", "all", "coverage", "full", "parallel"],
        help="Test suite to run"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Base pytest command
    base_cmd = ["python", "-m", "pytest"]
    
    if args.verbose:
        base_cmd.append("-v")
    
    # Test suite configurations
    if args.suite == "fast":
        # Fast tests - only unit tests, no comprehensive tests
        cmd = base_cmd + [
            "-m", "unit",
            "--maxfail=3",
            "--durations=5",
            "--ignore=tests/vitalDSP_webapp/callbacks/features/test_physiological_callbacks_comprehensive.py",
            "--ignore=tests/vitalDSP_webapp/callbacks/analysis/test_advanced_callbacks_comprehensive.py",
            "--ignore=tests/vitalDSP_webapp/callbacks/analysis/test_signal_filtering_callbacks_comprehensive.py",
            "--ignore=tests/vitalDSP_webapp/callbacks/features/test_physiological_callbacks_added.py",
        ]
        success = run_command(cmd, "Fast Tests (Unit tests only)")
        
    elif args.suite == "ci":
        # CI tests - use CI configuration
        cmd = base_cmd + ["-c", "pytest-ci.ini"]
        success = run_command(cmd, "CI Tests (Fast CI configuration)")
        
    elif args.suite == "unit":
        # Unit tests only
        cmd = base_cmd + ["-m", "unit", "--maxfail=5"]
        success = run_command(cmd, "Unit Tests Only")
        
    elif args.suite == "core":
        # Core library tests only
        cmd = base_cmd + ["tests/vitalDSP/", "--maxfail=5"]
        success = run_command(cmd, "Core Library Tests")
        
    elif args.suite == "webapp":
        # Webapp tests only
        cmd = base_cmd + ["tests/vitalDSP_webapp/", "--maxfail=5"]
        success = run_command(cmd, "Webapp Tests")
        
    elif args.suite == "all":
        # All tests
        cmd = base_cmd + ["--maxfail=10", "--durations=20"]
        success = run_command(cmd, "All Tests (Full Suite)")
        
    elif args.suite == "coverage":
        # Coverage tests
        cmd = base_cmd + [
            "--cov=src/vitalDSP",
            "--cov-report=html:cov_html",
            "--cov-report=term-missing",
            "--cov-fail-under=70",
            "-m", "unit or core"
        ]
        success = run_command(cmd, "Coverage Tests")
        
    elif args.suite == "full":
        # Full test suite with all optimizations
        cmd = base_cmd + [
            "-c", "pytest-full.ini",
            "--cov=src/vitalDSP",
            "--cov-report=html:cov_html",
            "--cov-report=term-missing",
            "--cov-fail-under=80"
        ]
        success = run_command(cmd, "Full Test Suite (All Tests with Optimizations)")
        
    elif args.suite == "parallel":
        # Parallel execution for maximum speed
        cmd = base_cmd + [
            "-n", "auto",
            "--dist=worksteal",
            "--timeout=300",
            "--cov=src/vitalDSP",
            "--cov-report=html:cov_html",
            "--cov-report=term-missing",
            "--cov-fail-under=80",
            "--durations=10"
        ]
        success = run_command(cmd, "Parallel Test Execution (Maximum Speed)")
    
    if success:
        print(f"\n🎉 {args.suite.upper()} tests completed successfully!")
        sys.exit(0)
    else:
        print(f"\n💥 {args.suite.upper()} tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
