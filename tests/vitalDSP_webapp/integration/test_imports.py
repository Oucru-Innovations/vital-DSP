"""
Test script to verify all imports for Time Domain migration.
"""
import sys

def test_imports():
    """Test all imports affected by the Time Domain migration."""
    print("=" * 80)
    print("TIME DOMAIN MIGRATION - IMPORT VERIFICATION")
    print("=" * 80)
    print()
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Analysis callbacks
    try:
        from vitalDSP_webapp.callbacks.analysis import (
            register_vitaldsp_callbacks,
            register_time_domain_callbacks,
        )
        print("[PASS] Analysis callbacks imports")
        tests_passed += 1
    except Exception as e:
        print(f"[FAIL] Analysis callbacks imports: {e}")
        tests_failed += 1
    
    # Test 2: Main callbacks
    try:
        from vitalDSP_webapp.callbacks import (
            register_vitaldsp_callbacks,
            register_time_domain_callbacks,
        )
        print("[PASS] Main callbacks imports")
        tests_passed += 1
    except Exception as e:
        print(f"[FAIL] Main callbacks imports: {e}")
        tests_failed += 1
    
    # Test 3: Page layouts
    try:
        from vitalDSP_webapp.layout.pages import time_domain_layout
        print("[PASS] time_domain_layout import")
        tests_passed += 1
    except Exception as e:
        print(f"[FAIL] time_domain_layout import: {e}")
        tests_failed += 1
    
    # Test 4: Page routing
    try:
        from vitalDSP_webapp.callbacks.core.page_routing_callbacks import display_page
        print("[PASS] page routing imports")
        tests_passed += 1
    except Exception as e:
        print(f"[FAIL] page routing imports: {e}")
        tests_failed += 1
    
    # Test 5: App module
    try:
        from vitalDSP_webapp.app import create_dash_app
        print("[PASS] App module import")
        tests_passed += 1
    except Exception as e:
        print(f"[FAIL] App module import: {e}")
        tests_failed += 1
    
    # Test 6: Layout function callable
    try:
        from vitalDSP_webapp.layout.pages.time_domain_page import time_domain_layout
        layout = time_domain_layout()
        print("[PASS] time_domain_layout() callable")
        tests_passed += 1
    except Exception as e:
        print(f"[FAIL] time_domain_layout() callable: {e}")
        tests_failed += 1
    
    # Test 7: Callback registration functions callable
    try:
        from vitalDSP_webapp.callbacks.analysis import (
            register_vitaldsp_callbacks,
            register_time_domain_callbacks,
        )
        assert callable(register_vitaldsp_callbacks)
        assert callable(register_time_domain_callbacks)
        print("[PASS] Callback registration functions callable")
        tests_passed += 1
    except Exception as e:
        print(f"[FAIL] Callback registration functions callable: {e}")
        tests_failed += 1
    
    # Summary
    print()
    print("=" * 80)
    print(f"RESULTS: {tests_passed} passed, {tests_failed} failed")
    print("=" * 80)
    
    if tests_failed == 0:
        print("SUCCESS: All imports verified!")
        return 0
    else:
        print(f"FAILURE: {tests_failed} tests failed")
        return 1

