# Data Pipelines

Generic, reusable data pipelines for detection and classification datasets.

## Structure

```
data-pipelines/
├── core/                   # Core utilities
│   ├── config.py          # Configuration classes
│   ├── file_utils.py      # File operations
│   ├── s3_utils.py        # S3 operations
│   └── logger.py          # Logging setup
├── downloaders/           # Data acquisition
│   ├── base_downloader.py
│   ├── api_downloader.py  # API + S3 downloads
│   └── s3_downloader.py   # Direct S3 downloads
├── organizers/            # Dataset organization
│   ├── base_organizer.py
│   ├── detection_organizer.py
│   └── classification_organizer.py
├── cleaners/              # Dataset cleaning
│   ├── base_cleaner.py
│   └── dataset_cleaner.py
├── preparers/             # COCO format preparation
│   ├── base_preparer.py
│   ├── detection_preparer.py
│   └── classification_preparer.py
└── pipelines/             # End-to-end workflows
    ├── detection_pipeline.py
    └── classification_pipeline.py
```

## Quick Start

### Detection Pipeline

Run the complete detection pipeline:

```bash
python data-pipelines/cli_detection.py \
    --base-dir ./my_project \
    --api-url http://your-api:8000/api/v1/dataset-versions/list \
    --dataset-version-ids 1 2 3 \
    --step full
```

Run individual steps:

```bash
# Download only
python data-pipelines/cli_detection.py --step download --api-url ... --dataset-version-ids 1 2

# Organize only (assumes data already downloaded)
python data-pipelines/cli_detection.py --step organize --base-dir ./my_project

# Clean only
python data-pipelines/cli_detection.py --step clean --base-dir ./my_project

# Prepare COCO format only
python data-pipelines/cli_detection.py --step prepare --base-dir ./my_project
```

### Classification Pipeline

Run the complete classification pipeline:

```bash
python data-pipelines/cli_classification.py \
    --base-dir ./my_project \
    --api-url http://your-api:8000/api/v1/dataset-versions/list \
    --dataset-version-ids 1 2 3 \
    --step full
```

### Dry Run Mode

Test the pipeline without making changes:

```bash
python data-pipelines/cli_detection.py --dry-run --step full
```

## Pipeline Steps

### 1. Download
- Downloads data from API with S3 integration
- Organizes into `client_data/` structure
- Supports parallel downloads with retries
- Verifies downloaded files

### 2. Organize
- **Detection**: Aggregates all classes into single `images/` and `masks/` folders
- **Classification**: Converts masks to colored format with class-specific colors
- Handles negative samples
- Validates image-mask pairs

### 3. Clean
- Removes unmatched masks
- Detects and removes corrupted files
- Provides dataset analysis
- Counts samples by class

### 4. Prepare
- Converts to COCO format
- Splits into train/val sets
- Extracts bounding boxes from masks
- Includes negative samples (optional)

## Using as a Library

### Detection Pipeline

```python
from pathlib import Path
from data_pipelines.pipelines import DetectionPipeline

# Initialize pipeline
pipeline = DetectionPipeline(
    base_dir=Path("./my_project"),
    dataset_version_ids=[1, 2, 3],
    api_url="http://your-api:8000/api/v1/dataset-versions/list",
    dry_run=False
)

# Run full pipeline
results = pipeline.run_full_pipeline(skip_download=False)

# Or run individual steps
download_stats = pipeline.run_download()
organize_stats = pipeline.run_organize()
clean_stats = pipeline.run_clean()
prepare_stats = pipeline.run_prepare()
```

### Classification Pipeline

```python
from pathlib import Path
from data_pipelines.pipelines import ClassificationPipeline

pipeline = ClassificationPipeline(
    base_dir=Path("./my_project"),
    dataset_version_ids=[1, 2, 3],
    api_url="http://your-api:8000/api/v1/dataset-versions/list"
)

results = pipeline.run_full_pipeline()
```

### Custom Configuration

```python
from data_pipelines.core import PreparerConfig
from data_pipelines.preparers import DetectionPreparer

# Custom configuration
config = PreparerConfig(
    input_dir=Path("./detection_dataset"),
    output_dir=Path("./coco"),
    train_split=0.85,
    min_area_threshold=100,
    add_negative_samples=True,
    categories=[{"id": 1, "name": "polyp", "supercategory": "none"}]
)

preparer = DetectionPreparer(config)
stats = preparer.prepare()
```

## Configuration

All pipeline components use configuration classes from `core/config.py`:

- `DownloaderConfig`: Download settings (API, S3, retries, etc.)
- `OrganizerConfig`: Organization settings (paths, class mappings, colors)
- `CleanerConfig`: Cleaning settings (validation, thresholds)
- `PreparerConfig`: COCO preparation settings (splits, categories)

## Extending the Pipeline

### Add a New Downloader

```python
from data_pipelines.downloaders import BaseDownloader

class MyDownloader(BaseDownloader):
    def download(self):
        # Your download logic
        pass
    
    def verify_downloads(self):
        # Your verification logic
        pass
```

### Add a New Organizer

```python
from data_pipelines.organizers import BaseOrganizer

class MyOrganizer(BaseOrganizer):
    def organize(self):
        # Your organization logic
        pass
    
    def validate(self):
        # Your validation logic
        pass
```

## Output Structure

### Detection Pipeline

```
my_project/
├── client_data/              # Downloaded data
│   ├── positive_samples/
│   └── negative_samples/
├── detection_dataset/        # Organized dataset
│   ├── images/
│   ├── masks/
│   └── negative_samples/
└── coco/                     # COCO format
    ├── train2017/
    ├── val2017/
    └── annotations/
        ├── instances_train2017.json
        └── instances_val2017.json
```

### Classification Pipeline

```
my_project/
├── client_data/              # Downloaded data
│   ├── positive_samples/
│   │   ├── adenoma/
│   │   └── hyperplastic/
│   └── negative_samples/
├── classification_dataset/   # Organized dataset
│   ├── images/              # Images with class suffix
│   ├── masks/               # Colored masks
│   └── negative_samples/
└── coco_classification/      # COCO format
    ├── train2017/
    ├── val2017/
    └── annotations/
```

## Requirements

- Python 3.7+
- boto3
- opencv-python
- numpy
- Pillow
- tqdm
- requests

Install with:
```bash
pip install boto3 opencv-python numpy Pillow tqdm requests
```
