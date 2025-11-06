# Complete Usage Guide for Data Pipelines

## Overview

This guide provides step-by-step instructions for using the new data pipeline system for both **Detection** and **Classification** workflows.

## Your Original Workflow (Now Automated)

### Detection Workflow
1. **Download**: `dataset_versions_downloader.py` → Downloads to `client_data/`
2. **Organize**: Manual aggregation → Creates `detection_dataset/` with single `images/` and `masks/` folders
3. **Clean**: `clean_up_dataset.py` → Removes unmatched/corrupted files
4. **Prepare**: `prepare_dataset.py` → Creates COCO format in `coco/`

### Classification Workflow
1. **Download**: `dataset_versions_downloader.py` → Downloads to `client_data/`
2. **Organize**: `clean_and_organize_classification_dataset.py` → Creates `classification_dataset/` with colored masks
3. **Clean**: Manual or script-based cleaning
4. **Prepare**: `prepare_dataset_classification.py` → Creates COCO format in `coco_classification/`

---

## New Pipeline System

### Directory Structure

After running the full pipeline, you'll have:

```
your_project/
├── client_data/                    # Downloaded from API
│   ├── positive_samples/
│   │   ├── adenoma/
│   │   │   ├── images/
│   │   │   └── masks/
│   │   ├── hyperplastic/
│   │   └── benign/
│   └── negative_samples/
├── detection_dataset/              # For detection (binary masks)
│   ├── images/                     # All images aggregated
│   ├── masks/                      # All masks aggregated
│   └── negative_samples/
├── classification_dataset/         # For classification (colored masks)
│   ├── images/                     # Images with class suffix
│   ├── masks/                      # Colored masks
│   └── negative_samples/
├── coco/                           # Detection COCO format
│   ├── train2017/
│   ├── val2017/
│   └── annotations/
└── coco_classification/            # Classification COCO format
    ├── train2017/
    ├── val2017/
    └── annotations/
```

---

## Detection Pipeline Usage

### Scenario 1: Complete Detection Pipeline (Download → Organize → Clean → Prepare)

```bash
python -m data-pipelines.cli_detection \
    --base-dir ./my_detection_project \
    --api-url http://your-server:8000/api/v1/dataset-versions/list \
    --dataset-version-ids 1 2 3 \
    --step full
```

**What this does:**
1. Downloads dataset versions 1, 2, 3 from API to `my_detection_project/client_data/`
2. Organizes into `my_detection_project/detection_dataset/` (aggregates all classes)
3. Cleans unmatched/corrupted files
4. Prepares COCO format in `my_detection_project/coco/`

### Scenario 2: Already Downloaded, Just Organize + Clean + Prepare

```bash
python -m data-pipelines.cli_detection \
    --base-dir ./my_detection_project \
    --step full \
    --skip-download
```

### Scenario 3: Run Individual Steps

#### Step 1: Download Only
```bash
python -m data-pipelines.cli_detection \
    --base-dir ./my_detection_project \
    --api-url http://your-server:8000/api/v1/dataset-versions/list \
    --dataset-version-ids 1 2 3 \
    --step download
```

#### Step 2: Organize Only (after download)
```bash
python -m data-pipelines.cli_detection \
    --base-dir ./my_detection_project \
    --step organize
```

#### Step 3: Clean Only
```bash
python -m data-pipelines.cli_detection \
    --base-dir ./my_detection_project \
    --step clean
```

#### Step 4: Prepare COCO Only
```bash
python -m data-pipelines.cli_detection \
    --base-dir ./my_detection_project \
    --step prepare
```

### Scenario 4: Dry Run (Test Without Making Changes)

```bash
python -m data-pipelines.cli_detection \
    --base-dir ./my_detection_project \
    --api-url http://your-server:8000/api/v1/dataset-versions/list \
    --dataset-version-ids 1 2 3 \
    --step full \
    --dry-run
```

---

## Classification Pipeline Usage

### Scenario 1: Complete Classification Pipeline

```bash
python -m data-pipelines.cli_classification \
    --base-dir ./my_classification_project \
    --api-url http://your-server:8000/api/v1/dataset-versions/list \
    --dataset-version-ids 1 2 3 \
    --step full
```

**What this does:**
1. Downloads dataset versions to `my_classification_project/client_data/`
2. Organizes into `my_classification_project/classification_dataset/` with:
   - Images renamed with class suffix (e.g., `image_123_adenoma.jpg`)
   - Masks converted to colored format:
     - **Adenoma**: Red (255, 0, 0)
     - **Hyperplastic**: Green (0, 255, 0)
     - **Benign**: Purple (157, 0, 255)
     - **No Pathology**: White (255, 255, 255)
3. Cleans unmatched/corrupted files
4. Prepares COCO format in `my_classification_project/coco_classification/`

### Scenario 2: Already Downloaded, Skip Download

```bash
python -m data-pipelines.cli_classification \
    --base-dir ./my_classification_project \
    --step full \
    --skip-download
```

### Scenario 3: Run Individual Steps

#### Step 1: Download Only
```bash
python -m data-pipelines.cli_classification \
    --base-dir ./my_classification_project \
    --api-url http://your-server:8000/api/v1/dataset-versions/list \
    --dataset-version-ids 1 2 3 \
    --step download
```

#### Step 2: Organize Only (converts masks to colored format)
```bash
python -m data-pipelines.cli_classification \
    --base-dir ./my_classification_project \
    --step organize
```

#### Step 3: Clean Only
```bash
python -m data-pipelines.cli_classification \
    --base-dir ./my_classification_project \
    --step clean
```

#### Step 4: Prepare COCO Only
```bash
python -m data-pipelines.cli_classification \
    --base-dir ./my_classification_project \
    --step prepare
```

### Scenario 4: Dry Run

```bash
python -m data-pipelines.cli_classification \
    --base-dir ./my_classification_project \
    --api-url http://your-server:8000/api/v1/dataset-versions/list \
    --dataset-version-ids 1 2 3 \
    --step full \
    --dry-run
```

---

## Python API Usage

### Detection Pipeline (Programmatic)

```python
from pathlib import Path
from data_pipelines.pipelines import DetectionPipeline

# Initialize pipeline
pipeline = DetectionPipeline(
    base_dir=Path("./my_detection_project"),
    dataset_version_ids=[1, 2, 3],
    api_url="http://your-server:8000/api/v1/dataset-versions/list",
    dry_run=False
)

# Option 1: Run full pipeline
results = pipeline.run_full_pipeline()

# Option 2: Run individual steps
download_stats = pipeline.run_download()
organize_stats = pipeline.run_organize()
clean_stats = pipeline.run_clean()
prepare_stats = pipeline.run_prepare()

# Option 3: Skip download if data already exists
results = pipeline.run_full_pipeline(skip_download=True)
```

### Classification Pipeline (Programmatic)

```python
from pathlib import Path
from data_pipelines.pipelines import ClassificationPipeline

# Initialize pipeline
pipeline = ClassificationPipeline(
    base_dir=Path("./my_classification_project"),
    dataset_version_ids=[1, 2, 3],
    api_url="http://your-server:8000/api/v1/dataset-versions/list",
    dry_run=False
)

# Run full pipeline
results = pipeline.run_full_pipeline()

# Or run individual steps
download_stats = pipeline.run_download()
organize_stats = pipeline.run_organize()  # Converts to colored masks
clean_stats = pipeline.run_clean()
prepare_stats = pipeline.run_prepare()
```

---

## Common Use Cases

### Use Case 1: Testing with Dry Run First

```bash
# Test detection pipeline
python -m data-pipelines.cli_detection \
    --base-dir ./test_project \
    --api-url http://your-server:8000/api/v1/dataset-versions/list \
    --dataset-version-ids 1 \
    --step full \
    --dry-run

# If looks good, run for real
python -m data-pipelines.cli_detection \
    --base-dir ./test_project \
    --api-url http://your-server:8000/api/v1/dataset-versions/list \
    --dataset-version-ids 1 \
    --step full
```

### Use Case 2: Re-organize Existing Downloaded Data

If you already have `client_data/` from the old script:

```bash
# For detection
python -m data-pipelines.cli_detection \
    --base-dir . \
    --step organize

# For classification
python -m data-pipelines.cli_classification \
    --base-dir . \
    --step organize
```

### Use Case 3: Re-prepare COCO Format with Different Split

Currently, you need to modify the pipeline code to change the split ratio. In Python:

```python
from pathlib import Path
from data_pipelines.pipelines import DetectionPipeline

pipeline = DetectionPipeline(base_dir=Path("."))

# Modify the preparer config
pipeline.preparer_config.train_split = 0.9  # 90% train, 10% val
pipeline.preparer_config.seed = 123  # Different random seed

# Run only the prepare step
pipeline.run_prepare()
```

### Use Case 4: Multiple Dataset Versions

```bash
# Download and process multiple versions
python -m data-pipelines.cli_detection \
    --base-dir ./combined_dataset \
    --api-url http://your-server:8000/api/v1/dataset-versions/list \
    --dataset-version-ids 1 2 3 4 5 \
    --step full
```

---

## Troubleshooting

### Issue 1: "client_data directory not found"

**Cause**: You're running organize/clean/prepare without downloading first.

**Solution**: Either run download step first, or use `--skip-download` if data exists:
```bash
python -m data-pipelines.cli_detection --step download --api-url ... --dataset-version-ids ...
```

### Issue 2: "No matching mask for image"

**Cause**: Image and mask filenames don't match (different stems).

**Solution**: Check your `client_data/` structure. The organizer matches by filename stem (without extension).

### Issue 3: Import errors

**Cause**: Running CLI scripts incorrectly.

**Solution**: Always use `-m` flag:
```bash
# Correct
python -m data-pipelines.cli_detection

# Incorrect
python data-pipelines/cli_detection.py
```

### Issue 4: "Unknown class" warning

**Cause**: A class in your data isn't in the CLASS_COLORS mapping.

**Solution**: The pipeline will skip unknown classes. Check your data or add the class to the configuration.

---

## Key Differences from Old Scripts

| Aspect | Old Scripts | New Pipeline |
|--------|-------------|--------------|
| **Commands** | 4 separate scripts | 1 command with steps |
| **Paths** | Hardcoded | Dynamic via `--base-dir` |
| **Configuration** | Edit script constants | Command-line arguments |
| **Dry Run** | Not available | `--dry-run` flag |
| **Validation** | Manual | Automatic after each step |
| **Logging** | Basic | Comprehensive with progress bars |
| **Error Handling** | Limited | Robust with retries |

---

## Migration from Old Scripts

### If you were using:

**Old Detection Workflow:**
```bash
python dataset_versions_downloader.py --dataset_version_ids 1 2
# Manual organization
python clean_up_dataset.py
python prepare_dataset.py
```

**New Detection Workflow:**
```bash
python -m data-pipelines.cli_detection \
    --dataset-version-ids 1 2 \
    --api-url http://your-server:8000/api/v1/dataset-versions/list \
    --step full
```

**Old Classification Workflow:**
```bash
python dataset_versions_downloader.py --dataset_version_ids 1 2
python data-pipelines/clean_and_organize_classification_dataset.py
python prepare_dataset_classification.py
```

**New Classification Workflow:**
```bash
python -m data-pipelines.cli_classification \
    --dataset-version-ids 1 2 \
    --api-url http://your-server:8000/api/v1/dataset-versions/list \
    --step full
```

---

## Next Steps

1. **Test with dry-run**: Always test with `--dry-run` first
2. **Start small**: Test with 1 dataset version before processing many
3. **Check outputs**: Verify each step's output before proceeding
4. **Customize**: Use Python API for advanced customization

For more details, see:
- `README.md` - Architecture overview
- `MIGRATION_GUIDE.md` - Detailed migration instructions
- Source code in `data-pipelines/` - Implementation details
