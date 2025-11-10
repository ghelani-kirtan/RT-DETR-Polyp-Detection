# RT-DETR Polyp Detection & Classification

Real-time polyp detection and classification system based on RT-DETR with modular data pipelines and inference system.

---

## 🚀 Quick Start

### 1. Setup
```bash
pip install -r requirements.txt
```

### 2. Run Inference
```bash
# Classification with tracker
python scripts/run_inference.py --config scripts/config.yaml

# High-resolution mode
python scripts/run_inference.py --config scripts/config_highres.yaml
```

### 3. Prepare Dataset
```bash
# Detection dataset
python -m data_pipelines.cli_detection \
    --base-dir ./my_project \
    --api-url http://your-server:8000/api/v1/dataset-versions/detail \
    --dataset-version-ids 42 \
    --step full

# Classification dataset
python -m data_pipelines.cli_classification \
    --base-dir ./my_project \
    --api-url http://your-server:8000/api/v1/dataset-versions/detail \
    --dataset-version-ids 42 \
    --step full
```

---

## 📁 Project Structure

```
RT-DETR-Polyp-Detection/
├── scripts/                    # Inference system
│   ├── config.yaml            # Configurations
│   ├── run_inference.py       # Main entry point
│   ├── core/                  # Core modules
│   └── README.md              # Complete guide
├── data_pipelines/            # Dataset preparation
│   ├── cli_detection.py       # Detection pipeline CLI
│   ├── cli_classification.py  # Classification pipeline CLI
│   ├── core/                  # Core utilities
│   └── README.md              # Pipeline guide
├── tracker/                   # ByteTrack module
├── src/                       # Training code
└── configs/                   # Training configs
```

---

## 🎯 Features

### Inference System
- ✅ Real-time detection & classification
- ✅ ByteTrack integration
- ✅ EMA smoothing for stable classification
- ✅ High-resolution support (letterbox)
- ✅ YAML-based configuration
- ✅ GPU acceleration (CUDA)

### Data Pipelines
- ✅ Automated dataset preparation
- ✅ API + S3 integration
- ✅ COCO format conversion
- ✅ Dataset cleaning & validation
- ✅ Configurable via YAML

---

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| **scripts/README.md** | Complete inference guide |
| **data_pipelines/README.md** | Dataset pipeline guide |
| **data_pipelines/MIGRATION_GUIDE.md** | Migration from old scripts |
| **PERFORMANCE_AND_FIXES_ANALYSIS.md** | Technical analysis |

---

## 🔧 Configuration

### Inference
Edit `scripts/config.yaml`:
```yaml
model:
  path: "./output/model.onnx"
  score_threshold: 0.70

video:
  source: 2  # Camera index or video path

tracker:
  enabled: true
  track_thresh: 0.45

smoothing:
  enabled: true
  ema_alpha: 0.65
```

### Data Pipeline
Edit pipeline configs or use CLI arguments:
```bash
python -m data_pipelines.cli_detection \
    --base-dir ./project \
    --step full \
    --dry-run  # Test first
```

---

## 🎓 Use Cases

### Inference

| Use Case | Command |
|----------|---------|
| **Classification + Tracker** | `python scripts/run_inference.py --config scripts/config.yaml` |
| **High-Resolution** | `python scripts/run_inference.py --config scripts/config_highres.yaml` |
| **Detection Only** | `python scripts/run_inference.py --config scripts/config_detection.yaml` |
| **No Tracker** | `python scripts/run_inference.py --config scripts/config_no_tracker.yaml` |

### Data Preparation

| Task | Command |
|------|---------|
| **Detection Dataset** | `python -m data_pipelines.cli_detection --step full` |
| **Classification Dataset** | `python -m data_pipelines.cli_classification --step full` |
| **Download Only** | `--step download` |
| **Organize Only** | `--step organize` |

---

## 🧪 Testing

```bash
# Test inference system
python scripts/test_inference.py

# Test data pipelines
python -m data_pipelines.cli_detection --step full --dry-run
```

---

## 📊 Model Zoo

### Base Models

| Model | Dataset | Input Size | AP<sup>val</sup> | #Params(M) | FPS |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **RT-DETRv2-S** | COCO | 640 | 48.1 | 20 | 217 |
| **RT-DETRv2-M** | COCO | 640 | 49.9 | 31 | 161 |
| **RT-DETRv2-L** | COCO | 640 | 53.4 | 42 | 108 |

---

## 🛠️ Requirements

```
Python >= 3.8
PyTorch >= 2.0
CUDA >= 11.0 (for GPU)
```

See `requirements.txt` for complete list.

---

## 📝 License

This project is based on [RT-DETR](https://github.com/lyuwenyu/RT-DETR).

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

---

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Ready to use!** 🎉

See detailed documentation in `scripts/README.md` and `data_pipelines/README.md`.
