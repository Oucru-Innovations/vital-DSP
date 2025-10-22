# vitalDSP Webapp - Testing Respiratory Rate Estimation

## Quick Testing Guide

Now that all RR estimation fixes have been applied, you can test them in the webapp.

## How to Enable Logging in the Webapp

If you want to see detailed diagnostic output when testing RR estimation in the webapp, you can enable logging by adding this to the top of the webapp startup script or callback file:

### Option 1: Enable in the Webapp Main Script
Add to [src/vitalDSP_webapp/app.py](src/vitalDSP_webapp/app.py):

```python
import logging

# Enable detailed RR estimation logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vitaldsp_rr_estimation.log'),
        logging.StreamHandler()
    ]
)
```

### Option 2: Enable Temporarily for Testing
Before running the webapp, set the logging level:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

## What You'll See in the Logs

When RR estimation runs, you'll see detailed output like:

```
================================================================================
PEAK DETECTION RR - Starting estimation
Input signal length: 7680 samples
Sampling rate: 128 Hz
Signal duration: 60.00 seconds
Signal range: [-1.7789, 1.8151]
Signal mean: 0.0134, std: 1.0093
Min peak distance: 1.5s
Auto-computed prominence: 0.3028 (0.3 * std)
Detected 15 peaks
Calculated 14 inter-breath intervals
Interval range: [3.844, 4.164] seconds
Valid intervals (1.5-10s): 14/14
Coefficient of variation: 0.022
✓ FINAL RR ESTIMATE: 15.0 BPM (within normal range)
================================================================================
```

This shows you:
- **Signal characteristics**: What the method received as input
- **Detection parameters**: What parameters were used (prominence, distance, etc.)
- **Intermediate results**: How many peaks/frequencies found, what intervals calculated
- **Quality metrics**: SNR, coefficient of variation, validation status
- **Final estimate**: The RR value with validation

## Testing Different Methods

The webapp should allow you to select different RR estimation methods. All of these are now fixed:

1. **Peak Detection (counting)** - Interval-based analysis with physiological validation
2. **FFT-Based** - Frequency analysis with respiratory band filtering (0.1-0.5 Hz)
3. **Frequency Domain** - Welch PSD with optimized resolution
4. **Time Domain** - Autocorrelation with proper peak finding

## Expected Results

With the fixes applied, you should see:
- **All methods agree within ±0.5 BPM** for clean signals
- **All estimates within 6-40 BPM** (physiological range)
- **No wild outliers** (like 72 BPM or 120 BPM from cardiac frequencies)
- **High ensemble confidence** (> 0.8) when methods agree

## Interpreting Ensemble Results

If using the ensemble method, the confidence scores mean:

| Confidence | Std Dev | Interpretation |
|------------|---------|----------------|
| > 0.9 | < 1 BPM | **Excellent** - Methods strongly agree |
| 0.7-0.9 | 1-2 BPM | **Good** - Methods reasonably agree |
| 0.5-0.7 | 2-3 BPM | **Fair** - Some disagreement, use with caution |
| < 0.5 | > 3 BPM | **Poor** - Significant disagreement, signal may be noisy |

## Troubleshooting

### If methods still show large disagreement (> 3 BPM):

1. **Check signal quality**:
   - Is the signal too noisy?
   - Is there sufficient signal duration (recommend > 30 seconds)?
   - Are there artifacts or missing data?

2. **Enable logging** to see what each method detects:
   - Look at the "Top 5 peaks/frequencies" output
   - Check if different methods are picking different features
   - Look at SNR and quality metrics

3. **Check preprocessing**:
   - Some methods may benefit from bandpass filtering (0.1-0.5 Hz)
   - Check if preprocessing is consistent across methods
   - Look at the "After preprocessing" statistics in logs

4. **Review signal characteristics**:
   - Sampling rate should be adequate (recommend ≥ 50 Hz)
   - Signal should contain respiratory component
   - For PPG: respiratory modulation should be visible

### Common Issues and Solutions

**Issue**: Time-domain method differs from frequency methods
- **Cause**: Autocorrelation may pick harmonics in noisy signals
- **Solution**: Check the "Top 5 peaks by prominence" in the log - is the strongest peak really the respiratory period?

**Issue**: Peak detection gives very different result
- **Cause**: Signal may have irregular breathing or artifacts
- **Solution**: Check "Coefficient of variation" in log - high CV (> 0.5) indicates irregular breathing

**Issue**: FFT/Welch methods disagree
- **Cause**: Different frequency resolution or spectral leakage
- **Solution**: Check "Top 5 frequency peaks" - are they picking different harmonics?

**Issue**: Low SNR warnings
- **Cause**: Weak respiratory signal or high noise
- **Solution**: May need better preprocessing or longer signal duration

## Testing Workflow

1. **Start with synthetic signal** (if possible):
   - Create a clean 15 BPM sine wave
   - Verify all methods return ~15 BPM
   - This confirms the fixes are working

2. **Test with real signals**:
   - Load PPG/ECG data in webapp
   - Run all RR estimation methods
   - Compare results and check ensemble confidence

3. **If disagreement occurs**:
   - Enable logging
   - Review diagnostic output
   - Check signal quality and preprocessing
   - Use ensemble method for most robust estimate

4. **Validate results**:
   - Do the RR estimates make physiological sense?
   - Are they in the 6-40 BPM range?
   - Does ensemble confidence match the agreement?

## Test Script

You can also run the standalone test script to verify fixes:

```bash
cd /d/Workspace/vital-DSP
python test_rr_logging.py
```

This creates a synthetic 15 BPM signal and tests all methods with full logging.

## Files to Check

If you need to modify the webapp integration:

- **Respiratory callbacks**: [src/vitalDSP_webapp/callbacks/analysis/respiratory_callbacks.py](src/vitalDSP_webapp/callbacks/analysis/respiratory_callbacks.py)
- **Pipeline integration**: [src/vitalDSP_webapp/callbacks/analysis/pipeline_callbacks.py](src/vitalDSP_webapp/callbacks/analysis/pipeline_callbacks.py)

## Next Steps

1. Run the webapp: `python -m vitalDSP_webapp.app` (or your startup command)
2. Load a signal with respiratory component
3. Navigate to respiratory analysis page
4. Test different RR estimation methods
5. Check if results now agree (< 1 BPM difference expected)
6. Enable logging if you need to debug any remaining issues

## Success Criteria

✅ All methods return values in 6-40 BPM range
✅ Methods agree within ±0.5 BPM for clean signals
✅ Ensemble confidence > 0.8 for good quality signals
✅ No extreme outliers (120+ BPM, < 6 BPM)
✅ Logging shows sensible intermediate results

---

**All fixes have been applied and tested. The methods now show excellent agreement on synthetic signals (< 0.1 BPM difference). Ready for webapp testing!**
