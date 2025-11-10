# Data Pipelines - Complete Guide

Generic, reusable pipelines for detection and classification datasets.

---

## 🚀 Quick Start

### Detection Pipeline
```bash
python -m data_pipelines.cli_detection \
    --base-dir ./my_project \
    --api-url http://dev.api.domain-name.ai/dataset/api/v1/dataset-versions/detail \
    --dataset-version-ids 42 \
    --step full
```

### Classification Pipeline
```bash
python -m data_pipelines.cli_classification \
    --base-dir ./my_project \
    --api-url http://dev.api.domain-name.ai/dataset/api/v1/dataset-versions/detail \
    --dataset-version-ids 42 \
    --step full
```

### Test First (Dry Run)
```bash
# Add --dry-run to any command
python -m data_pipelines.cli_detection --base-dir . --api-url ... --dataset-version-ids 42 --step full --dry-run
```

---

## 📁 Output Structure

```
your_project/
├── client_data/              # Downloaded from API
│   ├── positive_samples/
│   │   ├── adenoma/
│   │   ├── hyperplastic/
│   │   └── benign/
│   └── negative_samples/
├── detection_dataset/        # For detection (binary masks)
│   ├── images/
│   ├── masks/
│   └── negative_samples/
├── classification_dataset/   # For classification (colored masks)
│   ├── images/
│   ├── masks/
│   └── negative_samples/
├── coco/                     # Detection COCO format
└── coco_classification/      # Classification COCO format
```

---

## 🔧 Pipeline Steps

### Step 1: Download
Downloads data from API with S3 integration.

```bash
python -m data_pipelines.cli_detection \
    --step download \
    --api-url http://dev.api.domain-name.ai/dataset/api/v1/dataset-versions/detail \
    --dataset-version-ids 42
```

### Step 2: Organize

**Detection**: Aggregates all classes into single folders
```bash
python -m data_pipelines.cli_detection --step organize --base-dir ./my_project
```

**Classification**: Converts masks to colored format with class-specific colors
```bash
python -m data_pipelines.cli_classification --step organize --base-dir ./my_project
```

### Step 3: Clean
Removes unmatched/corrupted files
```bash
python -m data_pipelines.cli_detection --step clean --base-dir ./my_project
```

### Step 4: Prepare
Converts to COCO format
```bash
python -m data_pipelines.cli_detection --step prepare --base-dir ./my_project
```

---

## 🎯 Common Scenarios

### Already Downloaded Data
```bash
python -m data_pipelines.cli_detection --base-dir . --step full --skip-download
```

### Multiple Dataset Versions
```bash
python -m data_pipelines.cli_detection \
    --base-dir ./combined \
    --api-url http://dev.api.domain-name.ai/dataset/api/v1/dataset-versions/detail \
    --dataset-version-ids 40 41 42 43 \
    --step full
```

### Re-organize Existing Data
```bash
python -m data_pipelines.cli_detection --base-dir . --step organize
```

---

## 🐍 Python API

```python
from pathlib import Path
from data_pipelines.pipelines import DetectionPipeline

# Initialize
pipeline = DetectionPipeline(
    base_dir=Path("./my_project"),
    dataset_version_ids=[42],
    api_url="http://dev.api.domain-name.ai/dataset/api/v1/dataset-versions/detail"
)

# Run full pipeline
results = pipeline.run_full_pipeline()

# Or run individual steps
pipeline.run_download()
pipeline.run_organize()
pipeline.run_clean()
pipeline.run_prepare()
```

---

## 📊 Data Flow

### Detection Pipeline
```
API → client_data/ → detection_dataset/ → coco/
      (download)     (organize)           (prepare)
                     (clean)
```

**Key Logic**:
- Binary masks (any non-black pixel = foreground)
- Connected components → Bounding boxes
- Single category: "polyp"
- 80/20 train/val split

### Classification Pipeline
```
API → client_data/ → classification_dataset/ → coco_classification/
      (download)     (organize + color)        (prepare)
                     (clean)
```

**Key Logic**:
- Colored masks:
  - Adenoma: Red (255, 0, 0)
  - Hyperplastic: Green (0, 255, 0)
  - Benign: Purple (157, 0, 255)
  - No Pathology: White (255, 255, 255)
- Multi-class bounding boxes
- 80/20 train/val split

---

## 🏗️ Architecture

```
data_pipelines/
├── core/              # Shared utilities
│   ├── config.py      # Configuration classes
│   ├── file_utils.py  # File operations
│   ├── s3_utils.py    # S3 operations
│   └── logger.py      # Logging
├── downloaders/       # Data acquisition
│   ├── api_downloader.py
│   └── s3_downloader.py
├── organizers/        # Dataset organization
│   ├── detection_organizer.py
│   └── classification_organizer.py
├── cleaners/          # Dataset cleaning
│   └── dataset_cleaner.py
├── preparers/         # COCO format
│   ├── detection_preparer.py
│   └── classification_preparer.py
└── pipelines/         # End-to-end workflows
    ├── detection_pipeline.py
    └── classification_pipeline.py
```

---

## ⚙️ Configuration

All components use configuration classes:

```python
from data_pipelines.core import PreparerConfig

config = PreparerConfig(
    input_dir=Path("./detection_dataset"),
    output_dir=Path("./coco"),
    train_split=0.85,
    min_area_threshold=100,
    add_negative_samples=True,
    seed=42
)
```

---

## 🔌 Extending the Pipeline

### Add Custom Downloader
```python
from data_pipelines.downloaders import BaseDownloader

class MyDownloader(BaseDownloader):
    def download(self):
        # Your logic
        pass
```

### Add Custom Organizer
```python
from data_pipelines.organizers import BaseOrganizer

class MyOrganizer(BaseOrganizer):
    def organize(self):
        # Your logic
        pass
```

---

## 📝 Migration from Old Scripts

| Old Script | New Command |
|------------|-------------|
| `dataset_versions_downloader.py` | `--step download` |
| `prepare_dataset.py` | `--step prepare` (detection) |
| `prepare_dataset_classification.py` | `--step prepare` (classification) |
| `clean_and_organize_classification_dataset.py` | `--step organize` (classification) |
| `clean_up_dataset.py` | `--step clean` |

**One command replaces all**:
```bash
python -m data_pipelines.cli_detection --step full
```

---

## ⚠️ Important Notes

1. **Always use `-m` flag**: `python -m data_pipelines.cli_detection`
2. **Test with `--dry-run`** before running on production data
3. **Module name**: `data_pipelines` (underscore, not hyphen)
4. **Path structure**: Files go to `base_dir/client_data/` (simplified)

---

## 🐛 Troubleshooting

### "client_data directory not found"
Run download step first or check `--base-dir` path.

### "No module named 'data_pipelines'"
Use `python -m data_pipelines.cli_detection` (with `-m` flag).

### "No matching mask for image"
Check that image and mask filenames match (same stem, different extensions).

---

## 📚 Additional Documentation

- **MIGRATION_GUIDE.md** - Detailed migration from old scripts
- **BUGFIXES.md** - Technical details of fixes applied
- **WORKFLOW_DIAGRAM.md** - Visual diagrams

---

## ✅ Features

- ✅ Modular architecture
- ✅ Type-safe configurations
- ✅ Dry run support
- ✅ Parallel downloads
- ✅ Automatic validation
- ✅ Progress bars
- ✅ Comprehensive logging
- ✅ Error recovery with retries

---

## 📦 Requirements

```bash
pip install boto3 opencv-python numpy Pillow tqdm requests
```

---

## 🎉 Ready to Use!

```bash
# Start here
python -m data_pipelines.cli_detection \
    --base-dir . \
    --api-url http://<dev.api.domain-name.ai>/dataset/api/v1/dataset-versions/detail \
    --dataset-version-ids 42 \
    --step full \
    --dry-run
```

Then remove `--dry-run` to run for real!
