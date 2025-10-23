# Apply Filter Button - Debug Guide

## 🚨 ISSUE: Apply Filter Button Not Working

**Symptom**: Clicking "Apply Filter" button does nothing - no plots update, no processing happens.

## ✅ FIXES APPLIED:

### **Fix #1: Simplified Callback Trigger Logic**

**Changed from**: Preventing update when no trigger context exists
**Changed to**: Allowing callback to run even without trigger context

```python
# BEFORE (Too Restrictive):
if ctx.triggered:
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if trigger_id not in ["filter-btn-apply", ...]:
        raise PreventUpdate
else:
    raise PreventUpdate  # ❌ This prevented the callback from running!

# AFTER (More Permissive):
if ctx.triggered:
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if trigger_id not in ["filter-btn-apply", ...]:
        raise PreventUpdate  # Only prevent for invalid triggers
else:
    logger.info("No trigger context, allowing callback to run")  # ✅ Allow callback to run
```

### **Fix #2: Enhanced Logging**

Added comprehensive logging to track:
- When callback is triggered
- What triggered the callback
- Whether update is allowed or prevented
- Pathname verification

## 🔍 DEBUGGING STEPS:

### **Step 1: Check Server Logs**

When you click "Apply Filter", look for these logs:

```
=== ADVANCED FILTERING CALLBACK TRIGGERED ===
Pathname: /filtering
Trigger ID: filter-btn-apply
```

**If you see these logs**: Callback is being triggered ✅
**If you don't see these logs**: Button click isn't reaching the callback ❌

### **Step 2: Check Browser Console**

Open browser DevTools (F12) and check for:
- JavaScript errors
- Network requests to `/_dash-update-component`
- WebSocket connection status

### **Step 3: Verify Data Upload**

Before filtering, ensure:
1. Data has been uploaded via `/upload` page
2. Data appears in the data service
3. You're on the `/filtering` page

### **Step 4: Check Button State**

- Button should be clickable (not disabled)
- Button ID should be `filter-btn-apply`
- Button should have `n_clicks` property

## 🛠️ TROUBLESHOOTING:

### **Problem 1: No Logs Appear**

**Possible Causes**:
- Callback not registered in `app.py`
- Button ID mismatch
- Dash app not running

**Solutions**:
```bash
# Restart the app in debug mode
python src/vitalDSP_webapp/run_webapp_debug.py --debug
```

### **Problem 2: Logs Show "Preventing Update"**

**Possible Causes**:
- Wrong pathname (not on `/filtering` page)
- Invalid trigger ID

**Solutions**:
- Navigate to `/filtering` page
- Check trigger ID in logs

### **Problem 3: No Data Available**

**Possible Causes**:
- No data uploaded
- Data not stored in enhanced data service

**Solutions**:
```python
# Check data service
from vitalDSP_webapp.services.data.enhanced_data_service import get_enhanced_data_service
data_service = get_enhanced_data_service()
all_data = data_service.get_all_data()
print(f"Available data: {all_data}")
```

## 📋 TESTING CHECKLIST:

- [ ] Upload data via `/upload` page
- [ ] Navigate to `/filtering` page
- [ ] Set start position (0-100%)
- [ ] Select duration (30s, 1min, 2min, 5min)
- [ ] Click "Apply Filter" button
- [ ] Check server logs for trigger messages
- [ ] Verify plots update
- [ ] Check browser console for errors

## 🎯 EXPECTED BEHAVIOR:

1. **Click "Apply Filter"**:
   - Server logs: "ADVANCED FILTERING CALLBACK TRIGGERED"
   - Server logs: "Trigger ID: filter-btn-apply"

2. **Data Processing**:
   - Server logs: "Data service returned..."
   - Server logs: "Applying [filter_type] filter..."

3. **Plot Updates**:
   - Original signal plot updates
   - Filtered signal plot updates
   - Comparison plot updates
   - Quality metrics display

4. **Data Storage**:
   - Filtered data stored in enhanced data service
   - Available for other pages

## 🚀 QUICK FIX:

If nothing works, try this:

```bash
# 1. Stop the webapp (Ctrl+C)

# 2. Clear any cached data
rm -rf __pycache__ src/__pycache__ src/vitalDSP_webapp/__pycache__

# 3. Restart in debug mode
python src/vitalDSP_webapp/run_webapp_debug.py --debug

# 4. Upload fresh data

# 5. Try filtering again
```

## 📝 NOTES:

- The callback requires data to be uploaded first
- The callback only runs on `/filtering` page
- The callback is triggered by button clicks or nudge buttons
- Enhanced data service must be initialized

---

Generated: 2025-10-22
Status: FIXES APPLIED - READY FOR TESTING

