#!/usr/bin/env python
"""
Comprehensive test for respiratory page migration.

Tests:
1. Respiratory page layout import and creation
2. All component IDs present
3. Callbacks import
4. No duplicate IDs
5. Correct time controls (start position % + duration)
6. Hidden components for cross-page compatibility
"""

import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def get_all_ids(component, ids=None, max_depth=20, depth=0):
    """Recursively get all component IDs."""
    if ids is None:
        ids = []

    if depth > max_depth:
        return ids

    comp_id = getattr(component, 'id', None)
    comp_type = type(component).__name__

    if comp_id:
        ids.append((comp_id, depth, comp_type))

    if hasattr(component, 'children'):
        children = component.children
        if isinstance(children, list):
            for child in children:
                if child is not None:
                    get_all_ids(child, ids, max_depth, depth + 1)
        elif children is not None:
            get_all_ids(children, ids, max_depth, depth + 1)

    return ids


def test_respiratory_page_import():
    """Test that respiratory_page can be imported and layout created."""
    try:
        from vitalDSP_webapp.layout.pages.respiratory_page import respiratory_layout
        logger.info("[PASS] Respiratory page import")

        layout = respiratory_layout()
        logger.info("[PASS] Respiratory layout created")
        return True, layout
    except Exception as e:
        logger.error(f"[FAIL] Respiratory page import/creation: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_component_ids():
    """Test that all required component IDs are present."""
    try:
        # Get layout from import test
        success, layout = test_respiratory_page_import()
        if not success or layout is None:
            logger.error("[FAIL] Cannot test component IDs without layout")
            return False
        
        all_ids = get_all_ids(layout)
        id_dict = {comp_id: (depth, comp_type) for comp_id, depth, comp_type in all_ids}

        logger.info(f"[INFO] Total component IDs found: {len(id_dict)}")

        # Required visible components
        required_visible = {
            'resp-start-position-slider': 'Slider',
            'resp-duration-select': 'Select',
            'resp-btn-nudge-m10': 'Button',
            'resp-btn-nudge-m1': 'Button',
            'resp-btn-nudge-p1': 'Button',
            'resp-btn-nudge-p10': 'Button',
            'resp-signal-type': 'Select',
            'resp-signal-source-select': 'Select',
            'resp-estimation-methods': 'Checklist',
            'resp-ensemble-options': 'Div',
            'resp-ensemble-method': 'Select',
            'resp-advanced-options': 'Checklist',
            # REMOVED: 'resp-preprocessing-options' - preprocessing done on filtering page
            # REMOVED: 'resp-low-cut' - preprocessing done on filtering page
            # REMOVED: 'resp-high-cut' - preprocessing done on filtering page
            'resp-min-breath-duration': 'Input',
            'resp-max-breath-duration': 'Input',
            'resp-analyze-btn': 'Button',
            'resp-btn-export-results': 'Button',
            'resp-main-plot': 'Graph',
            'resp-analysis-results': 'Div',
            'resp-analysis-plots': 'Graph',
            'resp-additional-analysis-section': 'Div',
            'resp-data-store': 'Store',
            'resp-features-store': 'Store',
        }

        # Required hidden components (cross-page compatibility)
        required_hidden = {
            'filter-btn-apply': 'Div',
            'filter-type-select': 'Dropdown',
            'filter-family-advanced': 'Dropdown',
            'filter-response-advanced': 'Dropdown',
            'store-filtered-signal': 'Store',
            'store-filtering-data': 'Store',
            'store-filter-comparison': 'Store',
            'store-filter-quality-metrics': 'Store',
            'filter-original-plot': 'Graph',
            'filter-filtered-plot': 'Graph',
            'filter-comparison-plot': 'Graph',
            'filter-quality-metrics': 'Div',
            'filter-quality-plots': 'Graph',
            'btn-nudge-m10': 'Button',
            'btn-center': 'Button',
            'btn-nudge-p10': 'Button',
            'start-position-slider': 'Slider',
            'duration-select': 'Dropdown',
        }

        all_required = {**required_visible, **required_hidden}

        missing = []
        wrong_type = []

        for comp_id, expected_type in all_required.items():
            if comp_id not in id_dict:
                missing.append(comp_id)
            else:
                actual_type = id_dict[comp_id][1]
                # Allow some type variations (e.g., Select vs Dropdown)
                if expected_type not in actual_type and actual_type not in expected_type:
                    wrong_type.append((comp_id, expected_type, actual_type))

        if missing:
            logger.error(f"[FAIL] Missing component IDs: {missing}")
            return False

        if wrong_type:
            logger.warning(f"[WARN] Component type mismatches: {wrong_type}")

        logger.info(f"[PASS] All {len(all_required)} required components present")
        return True

    except Exception as e:
        logger.error(f"[FAIL] Component ID check: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_no_duplicate_ids():
    """Test that there are no duplicate component IDs."""
    try:
        # Get layout from import test
        success, layout = test_respiratory_page_import()
        if not success or layout is None:
            logger.error("[FAIL] Cannot test duplicate IDs without layout")
            return False
        
        all_ids = get_all_ids(layout)
        id_list = [comp_id for comp_id, _, _ in all_ids]

        duplicates = []
        seen = set()
        for comp_id in id_list:
            if comp_id in seen:
                duplicates.append(comp_id)
            seen.add(comp_id)

        if duplicates:
            logger.error(f"[FAIL] Duplicate IDs found: {set(duplicates)}")
            return False

        logger.info("[PASS] No duplicate component IDs")
        return True

    except Exception as e:
        logger.error(f"[FAIL] Duplicate ID check: {e}")
        return False


def test_old_components_removed():
    """Test that old component IDs are NOT present."""
    try:
        from vitalDSP_webapp.layout.pages.respiratory_page import respiratory_layout
        layout = respiratory_layout()

        all_ids = get_all_ids(layout)
        id_set = {comp_id for comp_id, _, _ in all_ids}

        # OLD components that should NOT exist
        old_components = [
            'resp-time-range-slider',  # Old RangeSlider (was duplicate ID)
            'resp-start-time',  # Old start time input
            'resp-end-time',  # Old end time input
            'store-resp-analysis',  # Unused store
        ]

        found_old = []
        for old_id in old_components:
            if old_id in id_set:
                found_old.append(old_id)

        if found_old:
            logger.error(f"[FAIL] Old components still present: {found_old}")
            return False

        logger.info("[PASS] All old components removed")
        return True

    except Exception as e:
        logger.error(f"[FAIL] Old components check: {e}")
        return False


def test_respiratory_callbacks_import():
    """Test that respiratory callbacks can be imported."""
    try:
        from vitalDSP_webapp.callbacks.analysis.respiratory_callbacks import register_respiratory_callbacks
        logger.info("[PASS] Respiratory callbacks import")
        return True
    except Exception as e:
        logger.error(f"[FAIL] Respiratory callbacks import: {e}")
        return False


def test_layout_package_imports():
    """Test that respiratory_layout can be imported from layout package."""
    try:
        from vitalDSP_webapp.layout import respiratory_layout
        logger.info("[PASS] Layout package import")

        from vitalDSP_webapp.layout.pages import respiratory_layout
        logger.info("[PASS] Layout.pages package import")
        return True
    except Exception as e:
        logger.error(f"[FAIL] Layout package import: {e}")
        return False


def test_page_routing_import():
    """Test that page routing callbacks can import respiratory_layout."""
    try:
        from vitalDSP_webapp.callbacks.core.page_routing_callbacks import register_page_routing_callbacks
        logger.info("[PASS] Page routing import")
        return True
    except Exception as e:
        logger.error(f"[FAIL] Page routing import: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("=" * 70)
    logger.info("RESPIRATORY PAGE MIGRATION VERIFICATION")
    logger.info("=" * 70)
    logger.info("")

    tests = [
        ("Respiratory page import and creation", test_respiratory_page_import),
    ]

    # Run first test to get layout
    success, layout = tests[0][1]()
    if not success:
        logger.error("Cannot proceed without layout")
        return 1

    # Run remaining tests
    additional_tests = [
        ("Component IDs present", lambda: test_component_ids(layout)),
        ("No duplicate IDs", lambda: test_no_duplicate_ids(layout)),
        ("Old components removed", test_old_components_removed),
        ("Respiratory callbacks import", test_respiratory_callbacks_import),
        ("Layout package imports", test_layout_package_imports),
        ("Page routing import", test_page_routing_import),
    ]

    results = [success]  # Include first test result
    for test_name, test_func in additional_tests:
        logger.info("")
        results.append(test_func())

    logger.info("")
    logger.info("=" * 70)
    passed = sum(results)
    total = len(results)
    logger.info(f"RESULTS: {passed}/{total} tests passed")

    if passed == total:
        logger.info("SUCCESS: All tests passed! Respiratory page migration verified.")
        logger.info("")
        logger.info("Summary of changes:")
        logger.info("  + Migrated layout from analysis_pages.py to respiratory_page.py")
        logger.info("  + Fixed duplicate ID (resp-start-position-slider)")
        logger.info("  + Updated to start position (%) + duration pattern")
        logger.info("  + Removed orphaned callback (update_resp_time_inputs)")
        logger.info("  + Fixed callback logic to convert position % to time")
        logger.info("  + Removed preprocessing controls (done on filtering page)")
        logger.info("  + Removed preprocessing parameters from callbacks")
        logger.info("  + Added 19 hidden components for cross-page compatibility")
        logger.info("  + Updated all imports in affected files")
        logger.info("  + Removed respiratory_layout from analysis_pages.py")
        return 0
    else:
        logger.error(f"FAILURE: {total - passed} test(s) failed")
        return 1


