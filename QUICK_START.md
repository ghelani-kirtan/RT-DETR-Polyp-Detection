# Quick Start Guide

## Installation

```bash
# Install base requirements
pip install -r requirements/requirements.txt

# For desktop GUI applications (classification_vid_ann_saver.py)
pip install -r requirements/desktop-requirements.txt

# For inference scripts
pip install -r requirements/inference-requirements.txt

# For cloud/S3 operations
pip install -r requirements/cloud-requirements.txt
```

## Common Commands

### Training
```bash
# Train classification model
python tools/train.py -c configs/rtdetr/rtdetr_r18vd_6x_coco_classification.yml -t weights/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth --use-amp

# Resume training
python tools/train.py -c configs/rtdetr/rtdetr_r18vd_6x_coco_classification.yml -t output/rtdetr_r18vd_6x_classification_v1/checkpoint0099.pth --use-amp
```

### Export Model
```bash
python tools/export_onnx.py -c configs/rtdetr/rtdetr_r18vd_6x_coco_classification.yml -r output/rtdetr_r18vd_6x_classification_v1/best.pth -o output/rtdetr_r18vd_6x_classification_v1/polyp_classifier.onnx -s 640 --check --simplify
```

### Inference
```bash
# Standard inference
python scripts/inference/run_inference.py

# Classification with video annotation and tracking
python scripts/inference/classification_vid_ann_saver.py
```

### Data Pipeline
```bash
# Classification dataset preparation
python -m scripts.data.cli_classification

# Detection dataset preparation
python -m scripts.data.cli_detection
```

### Cloud Sync
```bash
# Upload model to S3
python scripts/cloud/s3_sync_models.py upload --local_dir output --s3-prefix polyp_data_ml/models/classification/custom_rtdetr_r18vd_polyp/

# Download model from S3
python scripts/cloud/s3_sync_models.py download --local_dir output --s3-prefix polyp_data_ml/models/classification/custom_rtdetr_r18vd_polyp/

# Upload dataset to S3
python scripts/cloud/s3_sync_datasets.py upload --local_dir classification_dataset --s3-prefix polyp_data_ml/dataset_versions/v1_2_1/dataset_classification/

# Download dataset from S3
python scripts/cloud/s3_sync_datasets.py download --local_dir classification_dataset --s3-prefix polyp_data_ml/dataset_versions/v1_2_1/dataset_classification/
```

## Project Structure

```
project/
├── configs/              # Model configurations
├── src/                  # Core training code & tracker
├── tools/                # Training/export tools
├── data/                 # Datasets
├── notebooks/            # Jupyter notebooks
├── outputs/              # Model outputs
├── scripts/
│   ├── inference/       # Inference scripts
│   ├── data/            # Data pipeline
│   └── cloud/           # Cloud sync tools
└── requirements/         # Dependencies
```

## Environment Setup

1. Copy `.env.example` to `.env`
2. Configure AWS credentials (for S3):
   ```
   AWS_ACCESS_KEY_ID=your_key
   AWS_SECRET_ACCESS_KEY=your_secret
   AWS_REGION=your_region
   ```

## Documentation

- `PROJECT_RESTRUCTURE_GUIDE.md` - Complete restructure documentation
- `scripts/data/README.md` - Data pipeline documentation
- `scripts/inference/README.md` - Inference documentation
- `scripts/training_commands.txt` - Training command examples
