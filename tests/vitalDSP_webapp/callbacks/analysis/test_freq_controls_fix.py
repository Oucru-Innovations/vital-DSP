#!/usr/bin/env python
"""Test that frequency page has correct start position + duration controls."""

import sys

def get_all_ids(component, ids=None, max_depth=20, depth=0):
    """Recursively get all component IDs."""
    if ids is None:
        ids = set()

    if depth > max_depth:
        return ids

    comp_id = getattr(component, 'id', None)
    if comp_id:
        ids.add(comp_id)

    if hasattr(component, 'children'):
        children = component.children
        if isinstance(children, list):
            for child in children:
                if child is not None:
                    get_all_ids(child, ids, max_depth, depth + 1)
        elif children is not None:
            get_all_ids(children, ids, max_depth, depth + 1)

    return ids

def main():
    print("=" * 60)
    print("FREQUENCY PAGE CONTROLS FIX VERIFICATION")
    print("=" * 60)
    print()

    # Test 1: Import with forced reload
    try:
        # Clear all cached modules
        import importlib
        modules_to_clear = [m for m in sys.modules.keys() if 'vitalDSP_webapp.layout' in m]
        for mod in modules_to_clear:
            del sys.modules[mod]

        from vitalDSP_webapp.layout.pages import frequency_page
        importlib.reload(frequency_page)
        frequency_layout = frequency_page.frequency_layout
        print("[PASS] Import successful (with reload)")
    except Exception as e:
        print(f"[FAIL] Import failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Test 2: Create layout
    try:
        layout = frequency_layout()
        print("[PASS] Layout created")
    except Exception as e:
        print(f"[FAIL] Layout creation failed: {e}")
        return 1

    # Test 3: Get all IDs
    all_ids = get_all_ids(layout)
    print(f"[INFO] Total component IDs: {len(all_ids)}")
    print()

    # Test 4: Check for correct controls
    print("Checking for NEW controls (start position % + duration):")
    new_controls = {
        'freq-start-position-slider': 'Start Position Slider (0-100%)',
        'freq-duration-select': 'Duration Dropdown (30s, 1m, 2m, 5m)',
    }

    all_passed = True
    for comp_id, description in new_controls.items():
        found = comp_id in all_ids
        status = "FOUND" if found else "NOT FOUND"
        symbol = "+" if found else "-"
        print(f"  {symbol} {comp_id}: {status}")
        print(f"     ({description})")
        if not found:
            all_passed = False

    print()
    print("Checking for OLD controls (should NOT exist):")
    old_controls = {
        'freq-start-time-input': 'Old Start Time Input',
    }

    for comp_id, description in old_controls.items():
        found = comp_id in all_ids
        status = "FOUND (BAD!)" if found else "NOT FOUND (GOOD!)"
        symbol = "-" if found else "+"
        print(f"  {symbol} {comp_id}: {status}")
        print(f"     ({description})")
        if found:
            all_passed = False

    print()
    print("Checking for hidden cross-page components:")
    hidden_components = {
        'filter-original-plot': 'Filter Original Plot',
        'btn-nudge-m10': 'Nudge -10% Button',
        'btn-center': 'Center Button',
        'btn-nudge-p10': 'Nudge +10% Button',
        'start-position-slider': 'Start Position Slider (non-freq)',
        'duration-select': 'Duration Select (non-freq)',
    }

    for comp_id, description in hidden_components.items():
        found = comp_id in all_ids
        status = "FOUND" if found else "NOT FOUND"
        symbol = "+" if found else "-"
        print(f"  {symbol} {comp_id}: {status}")
        if not found:
            all_passed = False

    print()
    print("Checking navigation button labels:")
    # We can't check labels directly, but we can verify the IDs are correct
    nav_buttons = ['freq-btn-nudge-m10', 'freq-btn-nudge-m1', 'freq-btn-nudge-p1', 'freq-btn-nudge-p10']
    for btn_id in nav_buttons:
        found = btn_id in all_ids
        status = "FOUND" if found else "NOT FOUND"
        symbol = "+" if found else "-"
        print(f"  {symbol} {btn_id}: {status}")
        if not found:
            all_passed = False

    print()
    print("=" * 60)
    if all_passed:
        print("SUCCESS: All controls correctly updated!")
        print()
        print("Summary of changes:")
        print("  + Replaced Start Time (s) -> Start Position (%)")
        print("  + Replaced End Time (s) -> Duration (dropdown)")
        print("  + Changed RangeSlider -> Slider (0-100% with marks)")
        print("  + Updated button labels: -10s/-1s/+1s/+10s -> -10%/-5%/+5%/+10%")
        print("  + Added filter-original-plot hidden component")
        return 0
    else:
        print("FAILURE: Some components missing or incorrect")
        return 1

