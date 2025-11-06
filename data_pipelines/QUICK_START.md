# Quick Start Guide

## Detection Pipeline - One Command

```bash
# Full pipeline (download → organize → clean → prepare)
python -m data-pipelines.cli_detection \
    --base-dir ./my_project \
    --api-url http://your-server:8000/api/v1/dataset-versions/list \
    --dataset-version-ids 1 2 3 \
    --step full
```

## Classification Pipeline - One Command

```bash
# Full pipeline (download → organize with colored masks → clean → prepare)
python -m data-pipelines.cli_classification \
    --base-dir ./my_project \
    --api-url http://your-server:8000/api/v1/dataset-versions/list \
    --dataset-version-ids 1 2 3 \
    --step full
```

## Already Have Downloaded Data?

```bash
# Detection: Skip download
python -m data-pipelines.cli_detection \
    --base-dir ./my_project \
    --step full \
    --skip-download

# Classification: Skip download
python -m data-pipelines.cli_classification \
    --base-dir ./my_project \
    --step full \
    --skip-download
```

## Test First (Dry Run)

```bash
# Add --dry-run to any command
python -m data-pipelines.cli_detection \
    --base-dir ./my_project \
    --api-url http://your-server:8000/api/v1/dataset-versions/list \
    --dataset-version-ids 1 \
    --step full \
    --dry-run
```

## Individual Steps

```bash
# Download only
python -m data-pipelines.cli_detection --step download --api-url ... --dataset-version-ids ...

# Organize only
python -m data-pipelines.cli_detection --step organize --base-dir ./my_project

# Clean only
python -m data-pipelines.cli_detection --step clean --base-dir ./my_project

# Prepare COCO only
python -m data-pipelines.cli_detection --step prepare --base-dir ./my_project
```

## Output Structure

```
my_project/
├── client_data/              # Downloaded data
├── detection_dataset/        # Organized for detection
│   ├── images/
│   ├── masks/
│   └── negative_samples/
├── classification_dataset/   # Organized for classification
│   ├── images/              # With class suffix
│   ├── masks/               # Colored masks
│   └── negative_samples/
├── coco/                     # Detection COCO format
└── coco_classification/      # Classification COCO format
```

## Key Points

1. **Always use `-m` flag**: `python -m data-pipelines.cli_detection`
2. **Test with `--dry-run`** before running for real
3. **Use `--base-dir`** to specify where everything goes
4. **Use `--skip-download`** if you already have `client_data/`
5. **Run individual steps** if you need more control

## Need Help?

- Full usage guide: `USAGE_GUIDE.md`
- Migration from old scripts: `MIGRATION_GUIDE.md`
- Architecture details: `README.md`
