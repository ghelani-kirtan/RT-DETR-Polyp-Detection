# S3 Tools

CLI tool to sync ML models to/from S3 with auto-detection.

## Quick Start

```bash
# Install dependencies
pip install boto3 PyYAML python-dotenv tqdm

# Configure AWS credentials
aws configure

# Use the tool
python -m s3_tools list      # List local models
python -m s3_tools upload    # Upload latest model
```

## Commands

```bash
# List models
python -m s3_tools list                    # List local models
python -m s3_tools list --s3               # List S3 models
python -m s3_tools list --s3 --type classification

# Upload models
python -m s3_tools upload                  # Auto-upload latest
python -m s3_tools upload --model NAME     # Upload specific model
python -m s3_tools upload --dry-run        # Preview without uploading

# Download models
python -m s3_tools download NAME           # Download model
python -m s3_tools download NAME --dry-run # Preview without downloading

# Delete models
python -m s3_tools delete NAME             # Delete from S3
python -m s3_tools delete NAME --dry-run   # Preview without deleting

# Interactive mode
python -m s3_tools                         # Menu-driven interface

# Help
python -m s3_tools --help                  # Show all options
```

## Configuration

Edit `config.yaml` to match your S3 structure:

```yaml
s3:
  bucket: your-bucket-name
  region: us-east-1
  base_prefix: models
  
  # Point to the folder that CONTAINS your model folders
  # Structure: bucket/base_prefix/type_prefix/MODEL_NAME/files
  prefixes:
    classification: classification/custom_models
    detection: detection/custom_models

local:
  output_dir: output              # Where trained models are
  download_dir: models_downloaded # Where downloads go

detection:
  model_patterns:
    - "rtdetr_*"      # Model folder name patterns
    - "rtdetrv2_*"
  
  required_files:
    - "*.onnx"        # Must have .onnx OR .pth
    - "*.pth"
```

### Example S3 Structure

```
your-bucket/
  models/                           # base_prefix
    classification/                 
      custom_models/                # type_prefix
        rtdetr_r34vd_v1_3_2/        # MODEL_NAME (auto-detected)
          best.pth
          config.yaml
          ...
        rtdetr_r18vd_v1_2_5/
          ...
    detection/
      custom_models/
        ...
```

## How It Works

1. **Scans** `output/` directory for folders matching patterns
2. **Detects** model type from folder name (classification/detection)
3. **Validates** by checking for `.onnx` or `.pth` files
4. **Selects** latest model by modification time
5. **Builds** S3 path automatically using config

## Example Usage

```bash
# Check what models you have locally
$ python -m s3_tools list
📁 Local Models (2 models):
  1. rtdetr_r34vd_6x_coco_classification_v1_3_2 (1.0 GB, 7 files)
  2. rtdetr_r18vd_6x_classification_v1_2_5 (1.6 GB, 11 files)

# Check what's in S3
$ python -m s3_tools list --s3 --type classification
☁️  S3 Classification Models:
  📦 rtdetr_r34vd_6x_coco_classification_v1_3_2 (7 files, 1.0 GB)

# Preview upload
$ python -m s3_tools upload --dry-run
📤 Upload Plan:
  Local:  output\rtdetr_r34vd_6x_coco_classification_v1_3_2
  S3:     s3://your-bucket/models/classification/custom_models/rtdetr_r34vd_6x_coco_classification_v1_3_2
  Size:   1.0 GB (7 files)
🔍 Dry run - no files will be uploaded

# Actually upload
$ python -m s3_tools upload
Proceed with upload? [Y/n]: y
📤 Uploading 7 files...
✅ Upload complete!
```

## AWS Credentials

### Option 1: AWS CLI (Recommended)
```bash
aws configure
# Enter: Access Key ID, Secret Access Key, Region
```

### Option 2: Environment Variables
```bash
cp .env.example .env
# Edit .env with your credentials:
# AWS_ACCESS_KEY_ID=your_key
# AWS_SECRET_ACCESS_KEY=your_secret
```

### Option 3: IAM Role
No setup needed - automatically uses instance role (for EC2/ECS).

## Troubleshooting

**"No models found"**
- Check models are in `output/` directory
- Verify folder names match patterns (`rtdetr_*`)
- Ensure models have `.onnx` or `.pth` files

**"AWS credentials error"**
```bash
aws configure  # Set up credentials
```

**"No objects found at ..."**
- Check `config.yaml` prefixes match your S3 structure
- Ensure prefix points to folder CONTAINING model folders, not the models themselves
- Example: If models are at `bucket/models/classification/custom_models/model_v1/`
  - Set `base_prefix: models`
  - Set `classification: classification/custom_models`

**"Model already exists"**
- Tool will ask if you want to overwrite
- Type `y` to replace or `n` to cancel

## Features

- ✅ Auto-detection of models and types
- ✅ Parallel uploads/downloads with progress bars
- ✅ Dry-run mode for safe previews
- ✅ Interactive menu interface
- ✅ Safety confirmations for destructive operations
- ✅ Configurable via YAML
- ✅ Supports multiple model types

## Tips

1. **Always preview first**: Use `--dry-run` to see what will happen
2. **Use interactive mode**: Run `python -m s3_tools` for easiest experience
3. **Let it auto-detect**: Just run `upload` without specifying model name
4. **Check config**: Make sure prefixes match your S3 structure

---

**Version**: 1.0.1  
**Status**: Production Ready ✅
