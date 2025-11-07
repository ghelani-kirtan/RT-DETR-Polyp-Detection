# Data Pipelines

Automated pipelines for preparing detection and classification datasets.

## Quick Start

```bash
# Detection
python -m data_pipelines.cli_detection \
    --base-dir . \
    --api-url http://dev.api.seekiq.ai/dataset/api/v1/dataset-versions/detail \
    --dataset-version-ids 42 \
    --step full

# Classification
python -m data_pipelines.cli_classification \
    --base-dir . \
    --api-url http://dev.api.seekiq.ai/dataset/api/v1/dataset-versions/detail \
    --dataset-version-ids 42 \
    --step full
```

## Documentation

See **[data_pipelines/README.md](data_pipelines/README.md)** for complete documentation.

## Key Features

- One command replaces 4+ old scripts
- Dry run mode for testing
- Automatic validation and cleaning
- Parallel downloads with retries
- COCO format output

## Individual Steps

```bash
--step download   # Download from API
--step organize   # Organize dataset
--step clean      # Remove unmatched/corrupted files
--step prepare    # Convert to COCO format
--step full       # Run all steps
```

Add `--dry-run` to test without making changes.
