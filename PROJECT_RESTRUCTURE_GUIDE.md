# Project Restructure Guide

## Overview
This document summarizes the restructuring of the RT-DETR Polyp Detection project to follow ML/CV project standards.

## What Changed

### Directory Structure

**Before:**
```
project/
├── classification_vid_ann_saver.py (root level)
├── s3_sync.py (root level)
├── s3_sync_dataset.py (root level)
├── cmds.txt (root level)
├── data_pipelines/ (separate folder)
├── s3_tools/ (separate folder)
├── tracker/ (separate folder)
├── references/deploy/ (separate folder)
├── scripts/ (mixed content)
└── rtdetr_polyp/ (unused duplicate)
```

**After:**
```
project/
├── configs/              (unchanged - model configurations)
├── src/                  (unchanged - core training code)
│   └── tracker/         (moved from root tracker/)
├── tools/                (unchanged - training/export tools)
├── data/                 (unchanged - datasets)
├── notebooks/            (unchanged - jupyter notebooks)
├── outputs/              (unchanged - model outputs)
├── scripts/              (reorganized)
│   ├── inference/       (all inference scripts)
│   │   ├── core/        (inference engines)
│   │   ├── deploy/      (deployment scripts)
│   │   ├── classification_vid_ann_saver.py
│   │   ├── run_inference.py
│   │   ├── test_inference.py
│   │   └── config*.yaml
│   ├── data/            (data pipeline - formerly data_pipelines/)
│   │   ├── cleaners/
│   │   ├── core/
│   │   ├── downloaders/
│   │   ├── organizers/
│   │   ├── pipelines/
│   │   ├── preparers/
│   │   ├── cli_classification.py
│   │   ├── cli_detection.py
│   │   └── prepare_dataset_classification.py
│   ├── cloud/           (cloud sync scripts)
│   │   ├── s3_tools/    (s3 management utilities)
│   │   ├── s3_sync_models.py
│   │   └── s3_sync_datasets.py
│   └── training_commands.txt
└── requirements/         (consolidated requirements)
    ├── requirements.txt
    ├── desktop-requirements.txt
    ├── mac-requirements.txt
    ├── inference-requirements.txt
    └── cloud-requirements.txt
```

### Key Changes

1. **Tracker Module**: Moved from `tracker/` to `src/tracker/` (integrated with core source)
2. **Data Pipelines**: Moved from `data_pipelines/` to `scripts/data/`
3. **S3 Tools**: Moved from `s3_tools/` to `scripts/cloud/s3_tools/`
4. **Inference Scripts**: Consolidated in `scripts/inference/`
5. **Deploy Scripts**: Moved from `references/deploy/` to `scripts/inference/deploy/`
6. **Root Scripts**: Moved to appropriate subdirectories
7. **Removed**: `rtdetr_polyp/` (unused duplicate structure)

### Import Changes

All imports have been updated to reflect the new structure:

**Tracker imports:**
- Old: `from tracker.byte_tracker import BYTETracker`
- New: `from src.tracker.byte_tracker import BYTETracker`

**Data pipeline imports:**
- Relative imports within `scripts/data/` remain unchanged
- Package structure maintained

## How to Run

### 1. Training

```bash
# Classification training
python tools/train.py -c configs/rtdetr/rtdetr_r18vd_6x_coco_classification.yml -t weights/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth --use-amp

# Resume training
python tools/train.py -c configs/rtdetr/rtdetr_r18vd_6x_coco_classification.yml -t output/rtdetr_r18vd_6x_classification_v1/checkpoint0099.pth --use-amp
```

See `scripts/training_commands.txt` for more training commands.

### 2. Model Export

```bash
# Export to ONNX
python tools/export_onnx.py -c configs/rtdetr/rtdetr_r18vd_6x_coco_classification.yml -r output/rtdetr_r18vd_6x_classification_v1/best.pth -o output/rtdetr_r18vd_6x_classification_v1/polyp_classifier.onnx -s 640 --check --simplify
```

### 3. Inference

```bash
# Run inference with tracking
python scripts/inference/run_inference.py

# Classification with video annotation
python scripts/inference/classification_vid_ann_saver.py

# Test inference
python scripts/inference/test_inference.py
```

### 4. Data Pipeline

```bash
# Classification pipeline
python -m scripts.data.cli_classification

# Detection pipeline
python -m scripts.data.cli_detection

# Prepare classification dataset
python scripts/data/prepare_dataset_classification.py
```

### 5. Cloud Sync (S3)

```bash
# Upload models to S3
python scripts/cloud/s3_sync_models.py upload --local_dir output --s3-prefix polyp_data_ml/models/classification/custom_rtdetr_r18vd_polyp/

# Download models from S3
python scripts/cloud/s3_sync_models.py download --local_dir output --s3-prefix polyp_data_ml/models/classification/custom_rtdetr_r18vd_polyp/

# Upload dataset to S3
python scripts/cloud/s3_sync_datasets.py upload --local_dir classification_dataset --s3-prefix polyp_data_ml/dataset_versions/v1_2_1/dataset_classification/

# Download dataset from S3
python scripts/cloud/s3_sync_datasets.py download --local_dir classification_dataset --s3-prefix polyp_data_ml/dataset_versions/v1_2_1/dataset_classification/
```

### 6. S3 Tools (Advanced)

```bash
# Navigate to s3_tools
cd scripts/cloud/s3_tools

# Run s3 management CLI
python -m s3_tools --help
```

## Installation

### Base Requirements
```bash
pip install -r requirements/requirements.txt
```

### Desktop/GUI Requirements (for classification_vid_ann_saver.py)
```bash
pip install -r requirements/desktop-requirements.txt
```

### Inference Requirements
```bash
pip install -r requirements/inference-requirements.txt
```

### Cloud/S3 Requirements
```bash
pip install -r requirements/cloud-requirements.txt
```

### Mac-specific Requirements
```bash
pip install -r requirements/mac-requirements.txt
```

## Environment Setup

1. Copy `.env.example` to `.env`
2. Configure AWS credentials (if using S3 sync):
   ```
   AWS_ACCESS_KEY_ID=your_key
   AWS_SECRET_ACCESS_KEY=your_secret
   AWS_REGION=your_region
   ```

## Testing

All modules have been tested for:
- Import correctness
- Python syntax validation
- Path resolution

Run diagnostics on any file:
```bash
python -m py_compile <file_path>
```

## Notes

- All relative imports within packages remain unchanged
- Absolute imports updated to reflect new structure
- No logical code changes - only structural reorganization
- All functionality preserved
- Backward compatibility maintained where possible

## Migration Checklist

- [x] Move tracker to src/
- [x] Consolidate data pipelines to scripts/data/
- [x] Organize inference scripts to scripts/inference/
- [x] Move cloud tools to scripts/cloud/
- [x] Update all imports
- [x] Test all modules
- [x] Remove unused directories
- [x] Consolidate requirements
- [x] Verify functionality

## Verification

Run the verification script to ensure the restructure is complete:

```bash
python verify_structure.py
```

This will check:
- All new files are in the correct locations
- All old files have been removed
- Python syntax is valid for all modules
- Import paths are correct

## Summary of Changes

### Files Moved
- `classification_vid_ann_saver.py` → `scripts/inference/classification_vid_ann_saver.py`
- `s3_sync.py` → `scripts/cloud/s3_sync_models.py`
- `s3_sync_dataset.py` → `scripts/cloud/s3_sync_datasets.py`
- `cmds.txt` → `scripts/training_commands.txt`
- `data_pipelines/*` → `scripts/data/*`
- `s3_tools/*` → `scripts/cloud/s3_tools/*`
- `tracker/*` → `src/tracker/*`
- `references/deploy/*` → `scripts/inference/deploy/*`
- `scripts/core/*` → `scripts/inference/core/*`
- `scripts/*.yaml` → `scripts/inference/*.yaml`
- `scripts/run_inference.py` → `scripts/inference/run_inference.py`
- `scripts/test_*.py` → `scripts/inference/test_*.py`

### Directories Removed
- `data_pipelines/` (moved to `scripts/data/`)
- `s3_tools/` (moved to `scripts/cloud/s3_tools/`)
- `tracker/` (moved to `src/tracker/`)
- `references/` (moved to `scripts/inference/deploy/`)
- `rtdetr_polyp/` (unused duplicate - deleted)

### Import Updates
- `from tracker.byte_tracker` → `from src.tracker.byte_tracker`
- All relative imports within packages maintained

### Requirements Consolidated
- `scripts/requirements.txt` → `requirements/inference-requirements.txt`
- `s3_tools/requirements.txt` → `requirements/cloud-requirements.txt`
- Existing requirements files maintained

## Support

For issues or questions about the restructure, refer to:
- `QUICK_START.md` - Quick reference for common commands
- `scripts/data/README.md` - Data pipeline documentation
- `scripts/inference/README.md` - Inference documentation
- `scripts/data/MIGRATION_GUIDE.md` - Data pipeline migration guide
- `verify_structure.py` - Automated verification script
