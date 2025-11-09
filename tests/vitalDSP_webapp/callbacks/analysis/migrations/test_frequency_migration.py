"""Test script to verify Frequency page migration."""
import sys

def test_frequency_migration():
    """Test all imports and functionality for Frequency migration."""
    print("=" * 80)
    print("FREQUENCY PAGE MIGRATION - VERIFICATION")
    print("=" * 80)
    print()
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Direct import
    try:
        from vitalDSP_webapp.layout.pages.frequency_page import frequency_layout
        print("[PASS] Direct frequency_page import")
        tests_passed += 1
    except Exception as e:
        print(f"[FAIL] Direct frequency_page import: {e}")
        tests_failed += 1
    
    # Test 2: Package import
    try:
        from vitalDSP_webapp.layout.pages import frequency_layout
        print("[PASS] Package frequency_layout import")
        tests_passed += 1
    except Exception as e:
        print(f"[FAIL] Package frequency_layout import: {e}")
        tests_failed += 1
    
    # Test 3: Layout callable
    try:
        from vitalDSP_webapp.layout.pages.frequency_page import frequency_layout
        layout = frequency_layout()
        print("[PASS] frequency_layout() callable")
        tests_passed += 1
    except Exception as e:
        print(f"[FAIL] frequency_layout() callable: {e}")
        tests_failed += 1
    
    # Test 4: Page routing
    try:
        from vitalDSP_webapp.callbacks.core.page_routing_callbacks import display_page
        print("[PASS] Page routing imports")
        tests_passed += 1
    except Exception as e:
        print(f"[FAIL] Page routing imports: {e}")
        tests_failed += 1
    
    # Test 5: Frequency callbacks exist
    try:
        from vitalDSP_webapp.callbacks.analysis.frequency_filtering_callbacks import (
            register_frequency_filtering_callbacks
        )
        print("[PASS] Frequency filtering callbacks import")
        tests_passed += 1
    except Exception as e:
        print(f"[FAIL] Frequency filtering callbacks import: {e}")
        tests_failed += 1
    
    # Summary
    print()
    print("=" * 80)
    print(f"RESULTS: {tests_passed} passed, {tests_failed} failed")
    print("=" * 80)
    
    if tests_failed == 0:
        print("SUCCESS: All tests passed!")
        return 0
    else:
        print(f"FAILURE: {tests_failed} tests failed")
        return 1

