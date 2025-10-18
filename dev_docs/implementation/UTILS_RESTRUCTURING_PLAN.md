# Utils Directory Restructuring Plan

## Overview
This document outlines the reorganization of the `src/vitalDSP/utils/` directory to improve maintainability and logical organization.

## New Directory Structure

### 1. Core Infrastructure (`core_infrastructure/`)
**Purpose**: Phase 1 core infrastructure components for large-scale data processing
- `advanced_data_loaders.py` → `core_infrastructure/data_loaders.py`
- `optimized_advanced_data_loaders.py` → `core_infrastructure/optimized_data_loaders.py`
- `parallel_pipeline.py` → `core_infrastructure/parallel_pipeline.py`
- `optimized_parallel_pipeline.py` → `core_infrastructure/optimized_parallel_pipeline.py`
- `quality_screener.py` → `core_infrastructure/quality_screener.py`
- `optimized_quality_screener.py` → `core_infrastructure/optimized_quality_screener.py`

### 2. Data Processing (`data_processing/`)
**Purpose**: Data loading, validation, and synthesis utilities
- `data_loader.py` → `data_processing/data_loader.py`
- `validation.py` → `data_processing/validation.py`
- `synthesize_data.py` → `data_processing/synthesize_data.py`

### 3. Signal Processing (`signal_processing/`)
**Purpose**: Signal processing utilities, transforms, and features
- `peak_detection.py` → `signal_processing/peak_detection.py`
- `interpolations.py` → `signal_processing/interpolations.py`
- `normalization.py` → `signal_processing/normalization.py`
- `scaler.py` → `signal_processing/scaler.py`
- `mother_wavelets.py` → `signal_processing/mother_wavelets.py`
- `convolutional_kernels.py` → `signal_processing/convolutional_kernels.py`
- `loss_functions.py` → `signal_processing/loss_functions.py`
- `attention_weights.py` → `signal_processing/attention_weights.py`

### 4. Quality & Performance (`quality_performance/`)
**Purpose**: Quality assessment and performance monitoring
- `performance_monitoring.py` → `quality_performance/performance_monitoring.py`

### 5. Configuration & Utilities (`config_utilities/`)
**Purpose**: Configuration management and common utilities
- `dynamic_config.py` → `config_utilities/dynamic_config.py`
- `common.py` → `config_utilities/common.py`
- `error_recovery.py` → `config_utilities/error_recovery.py`
- `adaptive_parameters.py` → `config_utilities/adaptive_parameters.py`

## Files to Update

### Import Updates Required
Based on dependency analysis, the following files need import updates:

1. **Core Infrastructure Files**:
   - `core_infrastructure/quality_screener.py` - imports from `advanced_data_loaders`
   - `core_infrastructure/parallel_pipeline.py` - imports from `advanced_data_loaders` and `quality_screener`
   - `core_infrastructure/optimized_quality_screener.py` - imports from `dynamic_config` and `optimized_advanced_data_loaders`
   - `core_infrastructure/optimized_parallel_pipeline.py` - imports from `dynamic_config`, `optimized_advanced_data_loaders`, and `optimized_quality_screener`

2. **External Dependencies** (24 files):
   - Various files in `src/vitalDSP/` that import from utils
   - Files in `src/vitalDSP_webapp/` that import from utils

## Migration Steps

1. **Create new directory structure** ✅
2. **Move files to new locations** (with renaming)
3. **Update internal imports** within utils
4. **Update external imports** from other modules
5. **Update __init__.py** files
6. **Test all imports** work correctly
7. **Update documentation**

## Backward Compatibility

To maintain backward compatibility during transition:
- Keep original files temporarily with import redirects
- Update imports gradually
- Remove redirects after all dependencies are updated

## Risk Mitigation

- **Backup**: Keep original files until migration is complete
- **Testing**: Test each import update individually
- **Documentation**: Track all changes for rollback if needed
- **Gradual Migration**: Update imports in batches to minimize risk
