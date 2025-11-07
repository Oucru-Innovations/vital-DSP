# 📚 vital-DSP Documentation Index

**Last Updated**: November 4, 2025  
**Status**: Organized & Current (Second Reorganization Complete)

Welcome to the vital-DSP documentation hub! This index provides quick access to all project documentation.

---

## 🚀 Quick Start

**New to vital-DSP?** Start here:
- 📖 [README.md](../README.md) - Project overview and installation
- ⚡ [QUICK_REFERENCE_PHASES_1_4.md](reference/QUICK_REFERENCE_PHASES_1_4.md) - Quick reference guide
- 🎉 [ALL_PHASES_COMPLETE_SUMMARY.md](current/refactoring/ALL_PHASES_COMPLETE_SUMMARY.md) - Latest achievements (Phase 1-4)

---

## 📁 Documentation Structure

```
vital-DSP/
├── 📄 README.md                                    # Start here! (ONLY doc in root)
├── 📄 Changelog.md                                 # Version history
├── 📄 CONTRIBUTING.md                              # How to contribute
│
├── 📂 dev_docs/                                    # All developer documentation
│   ├── 📄 DOCUMENTATION_INDEX.md                   # This file - master index
│   ├── 📄 README.md                                # Dev docs overview
│   │
│   ├── 📂 current/                                 # ⭐ Current & active work
│   │   ├── refactoring/                            # Phase 1-4 refactoring docs
│   │   ├── integration/                            # Integration reports
│   │   ├── pipeline/                               # Pipeline documentation
│   │   ├── planning/                               # Enhancement plans & designs
│   │   └── testing/                                # Testing guides
│   │
│   ├── 📂 fix_reports/                             # Bug fix reports
│   ├── 📂 analysis_reports/                        # Analysis & evaluation reports
│   ├── 📂 implementation/                          # Implementation reports
│   ├── 📂 guides/                                  # How-to guides
│   │   ├── user_guides/                            # End-user guides
│   │   ├── developer_guides/                       # Developer guides
│   │   └── testing/                                # Testing guides
│   │
│   ├── 📂 reference/                               # Reference materials
│   │   ├── architecture/                           # Architecture docs
│   │   └── analysis/                               # Analysis reports
│   │
│   ├── 📂 archive/                                 # Historical dev docs
│   │   ├── 2025_q4/                                # Q4 2025 archived docs
│   │   └── ...                                     # Other quarters
│   │
│   └── ... (other specialized folders)
│
└── 📂 archive_docs/                                # Old root documents (archived)
    ├── page_fixes/                                 # Old page-specific fixes
    ├── bug_fixes/                                  # Old bug fixes
    ├── migrations/                                 # Old migrations
    ├── old_phases/                                 # Old phase completions
    └── enhancements/                               # Old enhancements
```

---

## ⭐ Current & Active Documentation

### Integration & Architecture
📍 **Location**: `dev_docs/current/integration/`

- [WEBAPP_VITALDSP_INTEGRATION_AUDIT_REPORT.md](current/integration/WEBAPP_VITALDSP_INTEGRATION_AUDIT_REPORT.md)
  - Integration status: 90% (A-)
  - Component-by-component analysis
  - Recommendations and next steps

### Refactoring (Phase 1-4)
📍 **Location**: `dev_docs/current/refactoring/`

**Phase 1: Transform Functions** ✅
- [TRANSFORM_FUNCTIONS_REFACTORING_REPORT.md](current/refactoring/TRANSFORM_FUNCTIONS_REFACTORING_REPORT.md)
  - FFT, STFT, Wavelet, Hilbert, MFCC refactoring
  - Changed from scipy/pywt to vitalDSP library

**Phase 2: Feature Extraction** ✅
- [FEATURE_EXTRACTION_REFACTORING_REPORT.md](current/refactoring/FEATURE_EXTRACTION_REFACTORING_REPORT.md)
  - Temporal and morphological features refactored
  - Peak detection now uses vitalDSP

**Phase 3 & 4: Filtering and ML/DL** ✅
- [PHASE_3_4_COMPLETION_REPORT.md](current/refactoring/PHASE_3_4_COMPLETION_REPORT.md)
- [PHASE_3_4_ANALYSIS_REPORT.md](current/refactoring/PHASE_3_4_ANALYSIS_REPORT.md)
  - Filtering consolidation (98% integration)
  - ML/DL already complete (90% integration)

**Session Summaries**:
- [COMPLETE_SESSION_SUMMARY_NOV_3_2025.md](current/refactoring/COMPLETE_SESSION_SUMMARY_NOV_3_2025.md)
  - Comprehensive summary of all phases
- [SESSION_PROGRESS_SUMMARY.md](current/refactoring/SESSION_PROGRESS_SUMMARY.md)
  - Detailed progress tracking

**Pipeline & Quality Alignment**:
- [PIPELINE_QUALITY_ALIGNMENT_COMPLETE.md](current/refactoring/PIPELINE_QUALITY_ALIGNMENT_COMPLETE.md)
  - Stage 2 & 4 quality alignment
- [PIPELINE_STAGE3_FILTERING_ALIGNMENT_COMPLETE.md](current/refactoring/PIPELINE_STAGE3_FILTERING_ALIGNMENT_COMPLETE.md)
  - Stage 3 filtering alignment
- [PIPELINE_ERROR_HANDLING_IMPROVEMENTS.md](current/refactoring/PIPELINE_ERROR_HANDLING_IMPROVEMENTS.md)
  - Error handling improvements

**Completion Summaries**:
- [ALL_PHASES_COMPLETE_SUMMARY.md](current/refactoring/ALL_PHASES_COMPLETE_SUMMARY.md)
- [PHASE_A_B_PIPELINE_GUI_COMPLETE.md](current/refactoring/PHASE_A_B_PIPELINE_GUI_COMPLETE.md)
- [PIPELINE_GUI_ENHANCEMENTS_COMPLETE.md](current/refactoring/PIPELINE_GUI_ENHANCEMENTS_COMPLETE.md)
- [SESSION_SUMMARY_STAGE6_COMPLETION.md](current/refactoring/SESSION_SUMMARY_STAGE6_COMPLETION.md)
- [STAGE6_FEATURE_EXTRACTION_ENHANCEMENT_COMPLETE.md](current/refactoring/STAGE6_FEATURE_EXTRACTION_ENHANCEMENT_COMPLETE.md)

### Planning & Design Documents
📍 **Location**: `dev_docs/current/planning/`

- [PIPELINE_GUI_ENHANCEMENT_DESIGN.md](current/planning/PIPELINE_GUI_ENHANCEMENT_DESIGN.md)
  - Pipeline GUI enhancement design
- [STAGE6_FEATURE_EXTRACTION_ENHANCEMENT_PLAN.md](current/planning/STAGE6_FEATURE_EXTRACTION_ENHANCEMENT_PLAN.md)
  - Stage 6 feature extraction plan
- [ENHANCEMENT_EXPORT_AND_REPORT.md](current/planning/ENHANCEMENT_EXPORT_AND_REPORT.md)
  - Export and report enhancements
- [STAGE6_VISUALIZATION_ENHANCEMENT.md](current/planning/STAGE6_VISUALIZATION_ENHANCEMENT.md)
  - Visualization enhancements

### Pipeline Processing
📍 **Location**: `dev_docs/current/pipeline/`

- [PIPELINE_COMPREHENSIVE_REVIEW_REPORT_UPDATED.md](current/pipeline/PIPELINE_COMPREHENSIVE_REVIEW_REPORT_UPDATED.md)
  - Pipeline architecture and optimization
  - Processing workflow

### Testing
📍 **Location**: `dev_docs/current/testing/`

- [TRANSFORM_REFACTORING_TESTING_GUIDE.md](current/testing/TRANSFORM_REFACTORING_TESTING_GUIDE.md)
  - Unit and integration testing guide
- [TEST_ORGANIZATION_SUMMARY.md](current/testing/TEST_ORGANIZATION_SUMMARY.md)
  - Test organization and structure

---

## 🐛 Fix Reports

📍 **Location**: `dev_docs/fix_reports/`

**Critical Fixes**:
- [CRITICAL_FIX_API_ERRORS.md](fix_reports/CRITICAL_FIX_API_ERRORS.md)
- [CRITICAL_FIX_PLOT_NOT_UPDATING.md](fix_reports/CRITICAL_FIX_PLOT_NOT_UPDATING.md)
- [CRITICAL_FIX_UI_ISSUES.md](fix_reports/CRITICAL_FIX_UI_ISSUES.md)
- [CRITICAL_FIX_WAVEFORM_MORPHOLOGY_API.md](fix_reports/CRITICAL_FIX_WAVEFORM_MORPHOLOGY_API.md)

**Quality & Score Fixes**:
- [FIX_QUALITY_SCORES_IDENTICAL.md](fix_reports/FIX_QUALITY_SCORES_IDENTICAL.md)
- [DEBUG_QUALITY_SCORES_SAME.md](fix_reports/DEBUG_QUALITY_SCORES_SAME.md)

**Feature & UI Fixes**:
- [FIX_EXPORT_REPORT_BUTTONS.md](fix_reports/FIX_EXPORT_REPORT_BUTTONS.md)
- [STAGE6_MORPHOLOGICAL_FEATURES_WAVEFORM_FIX.md](fix_reports/STAGE6_MORPHOLOGICAL_FEATURES_WAVEFORM_FIX.md)
- [SESSION_FIXES_SUMMARY.md](fix_reports/SESSION_FIXES_SUMMARY.md)
- [PIPELINE_GUI_FIXES_COMPLETE.md](fix_reports/PIPELINE_GUI_FIXES_COMPLETE.md)

**Performance & Service Fixes**:
- [WEBAPP_SLOWNESS_ROOT_CAUSE_ANALYSIS.md](fix_reports/WEBAPP_SLOWNESS_ROOT_CAUSE_ANALYSIS.md)
- [WEBAPP_SERVICE_MANAGER_FIX.md](fix_reports/WEBAPP_SERVICE_MANAGER_FIX.md)
- [WEBAPP_RR_INTEGRATION_FIXES.md](fix_reports/WEBAPP_RR_INTEGRATION_FIXES.md)

---

## 📊 Analysis Reports

📍 **Location**: `dev_docs/analysis_reports/`

- [HEALTH_REPORT_PAGE_ANALYSIS.md](analysis_reports/HEALTH_REPORT_PAGE_ANALYSIS.md)
- [VITALDSP_WEBAPP_GAP_ANALYSIS_REPORT.md](analysis_reports/VITALDSP_WEBAPP_GAP_ANALYSIS_REPORT.md)
- [COMPLETE_ANALYSIS_SUMMARY.md](analysis_reports/COMPLETE_ANALYSIS_SUMMARY.md)
- [VITALDSP_COMPREHENSIVE_ANALYSIS_REPORT.md](analysis_reports/VITALDSP_COMPREHENSIVE_ANALYSIS_REPORT.md)
- [VITALDSP_PERFORMANCE_ANALYSIS_REPORT.md](analysis_reports/VITALDSP_PERFORMANCE_ANALYSIS_REPORT.md)
- [VITALDSP_EFFECTIVENESS_AND_ACCURACY.md](analysis_reports/VITALDSP_EFFECTIVENESS_AND_ACCURACY.md)
- [VITALDSP_EDGE_CASE_ANALYSIS_REPORT.md](analysis_reports/VITALDSP_EDGE_CASE_ANALYSIS_REPORT.md)
- [VITALDSP_COMPREHENSIVE_ACCURACY_ANALYSIS_REPORT.md](analysis_reports/VITALDSP_COMPREHENSIVE_ACCURACY_ANALYSIS_REPORT.md)
- [SIGNAL_PROCESSING_ANALYSIS_REPORT.md](analysis_reports/SIGNAL_PROCESSING_ANALYSIS_REPORT.md)
- [ENHANCED_DATA_SERVICE_MIGRATION_STATUS.md](analysis_reports/ENHANCED_DATA_SERVICE_MIGRATION_STATUS.md)

---

## 📖 Developer Guides

### User Guides
📍 **Location**: `dev_docs/user_guides/`

- [README.md](user_guides/README.md) - User guides index
- [FEATURE_EXTRACTION_PIPELINE_USER_GUIDE.md](user_guides/FEATURE_EXTRACTION_PIPELINE_USER_GUIDE.md)
- [PIPELINE_QUICK_REFERENCE.md](user_guides/PIPELINE_QUICK_REFERENCE.md)
- [PIPELINE_VISUAL_WORKFLOW.md](user_guides/PIPELINE_VISUAL_WORKFLOW.md)
- [LARGE_FILE_PROCESSING_GUIDE.md](user_guides/LARGE_FILE_PROCESSING_GUIDE.md)
- [USER_GUIDE_COMPLETE.md](user_guides/USER_GUIDE_COMPLETE.md)
- [WEBAPP_RUN_MODES.md](user_guides/WEBAPP_RUN_MODES.md)

### Developer Guides
📍 **Location**: `dev_docs/developer_guides/`

- [PERFORMANCE_TUNING_GUIDE.md](developer_guides/PERFORMANCE_TUNING_GUIDE.md)
- [PHASE1_3_API_REFERENCE.md](developer_guides/PHASE1_3_API_REFERENCE.md)

### Testing Guides
📍 **Location**: `dev_docs/guides/testing/`

- [WEBAPP_TESTING_GUIDE.md](guides/testing/WEBAPP_TESTING_GUIDE.md)
  - Comprehensive testing guide for webapp

---

## 📚 Reference Materials

### Architecture
📍 **Location**: `dev_docs/reference/architecture/`

- [ADVANCED_FEATURES_GUIDE.md](reference/architecture/ADVANCED_FEATURES_GUIDE.md)
- [LARGE_DATA_PROCESSING_ARCHITECTURE.md](reference/architecture/LARGE_DATA_PROCESSING_ARCHITECTURE.md)
- [LARGE_DATA_PROCESSING_PIPELINE_DESIGN.md](reference/architecture/LARGE_DATA_PROCESSING_PIPELINE_DESIGN.md)

### Reference Materials
📍 **Location**: `dev_docs/reference/`

- [QUICK_REFERENCE_PHASES_1_4.md](reference/QUICK_REFERENCE_PHASES_1_4.md)
- [PIPELINE_QUALITY_ALIGNMENT_ANALYSIS.md](reference/PIPELINE_QUALITY_ALIGNMENT_ANALYSIS.md)
- [PIPELINE_QUALITY_ALIGNMENT_COMPLETE.md](reference/PIPELINE_QUALITY_ALIGNMENT_COMPLETE.md)
- [DOCUMENTATION_REORGANIZATION_PLAN.md](reference/DOCUMENTATION_REORGANIZATION_PLAN.md)
- [And more...](reference/)

### Planning Documents (in reference)
📍 **Location**: `dev_docs/reference/planning/`

- [VITALDSP_ENHANCEMENT_PLAN_2025.md](reference/planning/VITALDSP_ENHANCEMENT_PLAN_2025.md)
- [VITALDSP_ENHANCEMENT_VERIFICATION_REPORT.md](reference/planning/VITALDSP_ENHANCEMENT_VERIFICATION_REPORT.md)

---

## 🗄️ Archive

### Old Page Fixes
📍 **Location**: `archive_docs/page_fixes/`
- ADVANCED_PAGE_*.md (10 files)
- QUALITY_PAGE_*.md (7 files)
- TRANSFORM_PAGE_*.md (3 files)
- RESPIRATORY_*.md (4+ files)

### Old Bug Fixes
📍 **Location**: `archive_docs/bug_fixes/`
- FILTER_TYPE_*.md
- FILTERED_SIGNAL_*.md
- CALLBACK_FIX_SUMMARY.md
- And more...

### Old Migrations
📍 **Location**: `archive_docs/migrations/`
- DATA_SERVICE_MIGRATION_STATUS_REPORT.md
- FREQUENCY_MIGRATION_SUMMARY.txt
- TIME_DOMAIN_MIGRATION_SUMMARY.txt

### Old Phase Completions
📍 **Location**: `archive_docs/old_phases/`
- PHASE1_COMPLETION_SUMMARY.md (old)
- PHASE2_COMPLETION_SUMMARY.md (old)
- PHASE3_COMPLETION_SUMMARY.md (old)
- PHASE4_5_COMPLETION_SUMMARY.md (old)
- PHASE_B/C/D_*.md

### Old Enhancements
📍 **Location**: `archive_docs/enhancements/`
- Enhancement plans and analysis
- Old session reports
- Filtering improvements

### Historical Dev Docs
📍 **Location**: `dev_docs/archive/`
- Organized by year and quarter
- Old implementation docs
- Superseded documents
- [DOCUMENTATION_REORGANIZATION_COMPLETE.md](archive/2025_q4/DOCUMENTATION_REORGANIZATION_COMPLETE.md)
- [DOCUMENTATION_REORGANIZATION_PLAN.md](archive/2025_q4/DOCUMENTATION_REORGANIZATION_PLAN.md)

---

## 🔍 Finding What You Need

### I want to...

**Get started with vital-DSP**
→ Start with [README.md](../README.md) and [QUICK_REFERENCE_PHASES_1_4.md](reference/QUICK_REFERENCE_PHASES_1_4.md)

**Understand the latest changes (Phase 1-4)**
→ Read [ALL_PHASES_COMPLETE_SUMMARY.md](current/refactoring/ALL_PHASES_COMPLETE_SUMMARY.md)

**Learn about vitalDSP integration status**
→ Check [WEBAPP_VITALDSP_INTEGRATION_AUDIT_REPORT.md](current/integration/WEBAPP_VITALDSP_INTEGRATION_AUDIT_REPORT.md)

**Understand specific refactoring (Phase 1-4)**
→ See `current/refactoring/` for detailed reports

**Run tests**
→ Read [TRANSFORM_REFACTORING_TESTING_GUIDE.md](current/testing/TRANSFORM_REFACTORING_TESTING_GUIDE.md)

**Use the pipeline**
→ Check [PIPELINE_QUICK_REFERENCE.md](user_guides/PIPELINE_QUICK_REFERENCE.md)

**Process large files**
→ See [LARGE_FILE_PROCESSING_GUIDE.md](user_guides/LARGE_FILE_PROCESSING_GUIDE.md)

**Optimize performance**
→ Read [PERFORMANCE_TUNING_GUIDE.md](developer_guides/PERFORMANCE_TUNING_GUIDE.md)

**Find bug fix reports**
→ Check `fix_reports/` for all fix documentation

**Find analysis reports**
→ Check `analysis_reports/` for evaluation reports

**Find old documentation**
→ Check `archive_docs/` and `archive/` for historical documents

**Understand architecture**
→ See `reference/architecture/` for architecture docs

---

## 📊 Documentation Statistics

**Total markdown files**: ~200+  
**Files in root**: 3 (README.md, Changelog.md, CONTRIBUTING.md) - **100% clean!** ✅  
**Current active docs**: ~20 key documents  
**Fix reports**: ~15 documents  
**Analysis reports**: ~10 documents  
**Archived docs**: ~70+ files  
**Developer guides**: 10+ guides  
**Reference materials**: 30+ documents

---

## 🤝 Contributing to Documentation

Found an issue or want to improve documentation?
→ See [CONTRIBUTING.md](CONTRIBUTING.md)

---

## 📝 Document Organization

**Current (dev_docs/current/)**: Active development documentation  
**Guides (dev_docs/guides/)**: How-to guides for users and developers  
**Reference (dev_docs/reference/)**: Reference materials and analysis  
**Archive (archive_docs/ & dev_docs/archive/)**: Historical documents

---

**Need help?** Check the relevant guide in `dev_docs/guides/` or open an issue!

**Last Reorganized**: November 4, 2025 (Second Pass)  
**Reorganization Status**: ✅ Complete - Root folder clean (only README.md, Changelog.md, CONTRIBUTING.md)

