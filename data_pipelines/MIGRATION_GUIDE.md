# Migration Guide

Guide for migrating from old scripts to the new generic pipeline structure.

## Overview

The new pipeline structure provides:
- **Modularity**: Each step is independent and reusable
- **Configurability**: Easy to customize via config objects
- **Extensibility**: Simple to add new downloaders, organizers, etc.
- **Type Safety**: Clear interfaces with abstract base classes
- **Testability**: Each component can be tested independently

## Old vs New Structure

### Old Scripts (Root Directory)

```
dataset_versions_downloader.py    # Monolithic download script
prepare_dataset.py                 # Detection COCO prep
prepare_dataset_classification.py  # Classification COCO prep
clean_up_dataset.py               # Dataset cleaning
s3_download_data.py               # S3 downloads
```

### New Structure (data-pipelines/)

```
data-pipelines/
├── core/              # Shared utilities
├── downloaders/       # Download modules
├── organizers/        # Organization modules
├── cleaners/          # Cleaning modules
├── preparers/         # COCO preparation
└── pipelines/         # End-to-end workflows
```

## Migration Examples

### 1. Dataset Version Downloader

**Old:**
```bash
python dataset_versions_downloader.py \
    --dataset_version_ids 1 2 3 \
    --api_url http://server:8000/api/v1/dataset-versions/list \
    --output_root . \
    --dry_run
```

**New:**
```bash
python data-pipelines/cli_detection.py \
    --step download \
    --dataset-version-ids 1 2 3 \
    --api-url http://server:8000/api/v1/dataset-versions/list \
    --base-dir . \
    --dry-run
```

**Or in Python:**
```python
from data_pipelines.core import DownloaderConfig
from data_pipelines.downloaders import APIDownloader

config = DownloaderConfig(
    api_url="http://server:8000/api/v1/dataset-versions/list",
    dataset_version_ids=[1, 2, 3],
    output_dir=".",
    dry_run=True
)

downloader = APIDownloader(config)
stats = downloader.download()
```

### 2. Detection Dataset Preparation

**Old:**
```python
# prepare_dataset.py
prepare_data(
    dataset_root="detection_dataset",
    output_dir="coco",
    train_split=0.8,
    seed=42
)
```

**New:**
```python
from pathlib import Path
from data_pipelines.pipelines import DetectionPipeline

pipeline = DetectionPipeline(base_dir=Path("."))
pipeline.run_prepare()
```

**Or with custom config:**
```python
from data_pipelines.core import PreparerConfig
from data_pipelines.preparers import DetectionPreparer

config = PreparerConfig(
    input_dir=Path("detection_dataset"),
    output_dir=Path("coco"),
    train_split=0.8,
    seed=42,
    categories=[{"id": 1, "name": "polyp", "supercategory": "none"}]
)

preparer = DetectionPreparer(config)
stats = preparer.prepare()
```

### 3. Classification Dataset Preparation

**Old:**
```python
# prepare_dataset_classification.py
prepare_data(
    dataset_root="classification_dataset",
    output_dir="coco_classification",
    train_split=0.8,
    seed=42
)
```

**New:**
```python
from pathlib import Path
from data_pipelines.pipelines import ClassificationPipeline

pipeline = ClassificationPipeline(base_dir=Path("."))
pipeline.run_prepare()
```

### 4. Dataset Cleaning

**Old:**
```bash
python clean_up_dataset.py --count-only --overall
```

**New:**
```bash
python data-pipelines/cli_detection.py --step clean --base-dir .
```

**Or in Python:**
```python
from data_pipelines.core import CleanerConfig
from data_pipelines.cleaners import DatasetCleaner

config = CleanerConfig(
    input_dir=Path("detection_dataset"),
    remove_unmatched=True,
    check_corrupted=True
)

cleaner = DatasetCleaner(config)
analysis = cleaner.analyze()  # Just analyze
stats = cleaner.clean()       # Clean dataset
```

### 5. S3 Direct Download

**Old:**
```python
# s3_download_data.py
download_folder('images/', os.path.join(local_dir, 'images'))
download_folder('masks/', os.path.join(local_dir, 'masks'))
```

**New:**
```python
from data_pipelines.core import DownloaderConfig
from data_pipelines.downloaders import S3Downloader

config = DownloaderConfig(
    bucket_name="bucket-name",
    s3_prefix="polyp_data_ml/data/",
    output_dir=Path("dataset")
)

downloader = S3Downloader(config)
stats = downloader.download()
```

## Full Pipeline Examples

### Detection Pipeline (Complete Workflow)

**Old approach (multiple scripts):**
```bash
# Step 1: Download
python dataset_versions_downloader.py --dataset_version_ids 1 2

# Step 2: Organize (manual)
# ... manual file organization ...

# Step 3: Clean
python clean_up_dataset.py

# Step 4: Prepare COCO
python prepare_dataset.py
```

**New approach (single command):**
```bash
python data-pipelines/cli_detection.py \
    --step full \
    --dataset-version-ids 1 2 \
    --api-url http://server:8000/api/v1/dataset-versions/list \
    --base-dir ./my_project
```

**Or in Python:**
```python
from pathlib import Path
from data_pipelines.pipelines import DetectionPipeline

pipeline = DetectionPipeline(
    base_dir=Path("./my_project"),
    dataset_version_ids=[1, 2],
    api_url="http://server:8000/api/v1/dataset-versions/list"
)

# Run everything
results = pipeline.run_full_pipeline()

# Or run steps individually
pipeline.run_download()
pipeline.run_organize()
pipeline.run_clean()
pipeline.run_prepare()
```

### Classification Pipeline (Complete Workflow)

**New approach:**
```python
from pathlib import Path
from data_pipelines.pipelines import ClassificationPipeline

pipeline = ClassificationPipeline(
    base_dir=Path("./my_project"),
    dataset_version_ids=[1, 2, 3],
    api_url="http://server:8000/api/v1/dataset-versions/list"
)

results = pipeline.run_full_pipeline()
```

## Key Differences

### 1. Configuration

**Old:** Hardcoded constants at top of scripts
```python
DEFAULT_API_URL = "http://..."
CLIENT_DATA_DIR = "client_data"
MAX_WORKERS = 20
```

**New:** Configuration objects
```python
config = DownloaderConfig(
    api_url="http://...",
    output_dir="client_data",
    max_workers=20
)
```

### 2. Modularity

**Old:** Monolithic scripts with all logic in one file

**New:** Separated concerns
- `downloaders/` - Data acquisition
- `organizers/` - Dataset organization
- `cleaners/` - Dataset cleaning
- `preparers/` - COCO format conversion

### 3. Reusability

**Old:** Copy-paste code between scripts

**New:** Shared utilities in `core/`
- `FileUtils` - File operations
- `S3Utils` - S3 operations
- Config classes - Standardized configuration

### 4. Extensibility

**Old:** Modify existing scripts

**New:** Extend base classes
```python
from data_pipelines.downloaders import BaseDownloader

class MyCustomDownloader(BaseDownloader):
    def download(self):
        # Custom logic
        pass
```

## Benefits of New Structure

1. **Easier Testing**: Each component can be tested independently
2. **Better Maintainability**: Clear separation of concerns
3. **Flexible Configuration**: Easy to customize without code changes
4. **Dry Run Support**: Test pipelines without making changes
5. **Better Logging**: Consistent logging across all components
6. **Error Handling**: Robust retry logic and error recovery
7. **Type Hints**: Better IDE support and code clarity

## Backward Compatibility

The old scripts are still available in the root directory and will continue to work. However, we recommend migrating to the new structure for:
- New projects
- Complex workflows
- Production deployments
- Team collaboration

## Next Steps

1. Try the new CLI tools with `--dry-run` flag
2. Test individual pipeline steps
3. Gradually migrate existing workflows
4. Extend the pipeline for custom needs
5. Remove old scripts once migration is complete
