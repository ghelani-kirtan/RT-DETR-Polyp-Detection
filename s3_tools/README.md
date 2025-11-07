# S3 Tools

Simple CLI tool to sync ML models to/from S3 with auto-detection.

## Quick Start

```bash
# Install
pip install boto3 PyYAML python-dotenv tqdm

# Configure AWS
aws configure

# Use
python -m s3_tools list      # List local models
python -m s3_tools upload    # Upload latest model
```

## Commands

```bash
# List
python -m s3_tools list                    # List local models
python -m s3_tools list --s3               # List S3 models

# Upload
python -m s3_tools upload                  # Auto-upload latest
python -m s3_tools upload --model NAME     # Upload specific
python -m s3_tools upload --dry-run        # Preview only

# Download
python -m s3_tools download NAME           # Download model

# Delete
python -m s3_tools delete NAME             # Delete from S3

# Interactive
python -m s3_tools                         # Menu interface

# Help
python -m s3_tools --help                  # Show all options
```

## Configuration

Edit `config.yaml`:

```yaml
s3:
  bucket: your-bucket-name
  region: us-east-1
  base_prefix: models
  
  prefixes:
    classification: classification/custom_models
    detection: detection/custom_models

local:
  output_dir: output
  download_dir: models_downloaded

detection:
  model_patterns:
    - "rtdetr_*"
    - "rtdetrv2_*"
  
  required_files:
    - "*.onnx"
    - "*.pth"
```

## How It Works

1. Scans `output/` directory for models matching patterns
2. Detects type from folder name (classification/detection)
3. Validates by checking for `.onnx` or `.pth` files
4. Selects latest by modification time
5. Builds S3 path automatically

## Example

```bash
# Check models
$ python -m s3_tools list
📁 Local Models (2 models):
  1. rtdetr_r34vd_6x_coco_classification_v1_3_2 (1.0 GB)
  2. rtdetr_r18vd_6x_classification_v1_2_5 (1.6 GB)

# Preview upload
$ python -m s3_tools upload --dry-run
📤 Upload Plan:
  Local:  output\rtdetr_r34vd_6x_coco_classification_v1_3_2
  S3:     s3://your-bucket/models/classification/...
  Size:   1.0 GB (7 files)

# Upload
$ python -m s3_tools upload
Proceed? [Y/n]: y
✅ Upload complete!
```

## AWS Credentials

**Option 1**: AWS CLI (recommended)
```bash
aws configure
```

**Option 2**: Environment variables
```bash
cp .env.example .env
# Edit .env with your credentials
```

**Option 3**: IAM role (for EC2/ECS)
No setup needed.

## Troubleshooting

**No models found**: Check models are in `output/` and match patterns (`rtdetr_*`)

**AWS error**: Run `aws configure` to set credentials

**Model exists**: Tool will ask to overwrite

## Features

- ✅ Auto-detection of models and types
- ✅ Parallel uploads/downloads with progress bars
- ✅ Dry-run mode for previews
- ✅ Interactive menu interface
- ✅ Safety confirmations
- ✅ Configurable via YAML

---

**Version**: 1.0.0  
**Status**: Production Ready
