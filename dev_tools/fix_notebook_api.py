"""
Fix notebook API calls to match updated QualityScreener API.

Updates screen_signal() calls to use the correct API:
- Old: screener.screen_signal(signal, fs, signal_type)
- New: screener.sampling_rate = fs; screener.signal_type = signal_type; screener.screen_signal(signal)
"""

import json
import re
from pathlib import Path

def fix_screen_signal_calls(source_lines):
    """Fix screen_signal() API calls in code."""
    source = ''.join(source_lines)

    # Pattern: screener.screen_signal(signal, fs, 'signal_type')
    # or: screener_xxx.screen_signal(signal, fs, signal_type_var)
    pattern = r'([\w_]+)\.screen_signal\((\w+),\s*(\w+),\s*([\'"]?\w+[\'"]?)\)'

    def replacement(match):
        screener_var = match.group(1)
        signal_var = match.group(2)
        fs_var = match.group(3)
        signal_type = match.group(4)

        # Build replacement
        return (
            f"{screener_var}.sampling_rate = {fs_var}\n"
            f"{screener_var}.signal_type = {signal_type}\n"
            f"results = {screener_var}.screen_signal({signal_var})"
        )

    fixed = re.sub(pattern, replacement, source)

    # Also update result variable references if needed
    # Change result.overall_pass_rate to calculated pass rate
    if 'overall_pass_rate' in fixed or 'segment_results' in fixed:
        # Need to add pass rate calculation
        if 'results = ' in fixed and 'passed = ' not in fixed:
            # Find where results is assigned and add calculation after
            lines = fixed.split('\n')
            new_lines = []
            for i, line in enumerate(lines):
                new_lines.append(line)
                if 'results = ' in line and '.screen_signal(' in line:
                    # Add pass rate calculation after
                    indent = len(line) - len(line.lstrip())
                    new_lines.append(' ' * indent + f"passed = sum(1 for r in results if r.passed_screening)")
                    new_lines.append(' ' * indent + f"pass_rate = passed / len(results) if results else 0.0")
            fixed = '\n'.join(new_lines)

    # Update references to result.overall_pass_rate -> pass_rate
    fixed = fixed.replace('result.overall_pass_rate', 'pass_rate')
    fixed = fixed.replace('result.segment_results', 'results')
    fixed = fixed.replace('result_seq.overall_pass_rate', 'pass_rate_seq')
    fixed = fixed.replace('result_par.overall_pass_rate', 'pass_rate_par')
    fixed = fixed.replace('result_custom.overall_pass_rate', 'pass_rate_custom')
    fixed = fixed.replace('quality_result.overall_pass_rate', 'quality_pass_rate')

    return fixed.split('\n') if '\n' in fixed else [fixed]

def main():
    notebook_path = Path('examples/notebooks/01_Large_File_Processing_Tutorial.ipynb')

    print(f"Fixing notebook: {notebook_path}")

    # Load notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Fix cells
    fixed_count = 0
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = cell.get('source', [])
            if any('screen_signal(' in line for line in source):
                # Check if it's already fixed
                if not any('screener.sampling_rate' in line or 'screener_seq.sampling_rate' in line or 'screener_par.sampling_rate' in line for line in source):
                    print(f"\n  Fixing cell")
                    cell['source'] = fix_screen_signal_calls(source)
                    # Clear outputs
                    cell['outputs'] = []
                    cell['execution_count'] = None
                    fixed_count += 1

    print(f"\nFixed {fixed_count} cells")

    # Save notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

    print(f"OK Notebook saved: {notebook_path}")

if __name__ == '__main__':
    main()
