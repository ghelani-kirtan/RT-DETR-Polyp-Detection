# Inference Scripts - Complete Guide

Modular inference system for polyp detection and classification with configurable tracking and smoothing.

---

## 🚀 Quick Start

### Basic Usage
```bash
cd scripts
python run_inference.py --config config.yaml
```

### All Use Cases
```bash
# Classification with tracker and smoothing (default)
python run_inference.py --config config.yaml

# Detection only (single class)
python run_inference.py --config config_detection.yaml

# High-resolution mode
python run_inference.py --config config_highres.yaml

# Without tracker (raw frame detection)
python run_inference.py --config config_no_tracker.yaml
```

---

## 📁 Structure

```
scripts/
├── config.yaml                  # Default: Classification with tracker
├── config_detection.yaml        # Detection only (single class)
├── config_highres.yaml          # High-resolution mode
├── config_no_tracker.yaml       # No tracker (raw detection)
├── run_inference.py             # Main entry point
├── core/                        # Core modules
│   ├── __init__.py
│   ├── config_loader.py         # YAML configuration loader
│   ├── inference_engine.py      # Standard inference engine
│   ├── inference_engine_highres.py  # High-res with letterbox
│   ├── video_processor.py       # Tracking and postprocessing
│   └── visualizer.py            # Visualization (boxes, labels, trails)
└── README.md                    # This file
```

---

## ⚙️ Configuration Guide

### Model Configuration
```yaml
model:
  path: "./output/model.onnx"    # Path to ONNX model
  input_size: [640, 640]         # Model input size [H, W]
  task: "classification"         # "detection" or "classification"
  
  classes:
    names: ["adenoma", "hyperplastic"]
    colors: [[0, 0, 255], [0, 255, 0]]  # BGR format
  
  score_threshold: 0.70          # Display threshold
  tracker_input_threshold: 0.10  # Tracker input threshold
```

### Video Configuration
```yaml
video:
  source: 2                      # Camera index or video path
  fps: 60                        # Frame rate for tracker
  high_resolution: false         # Enable letterbox preprocessing
```

### Tracker Configuration
```yaml
tracker:
  enabled: true                  # Enable/disable tracker
  track_thresh: 0.45             # Min score to start track
  track_buffer: 30               # Frames to keep lost track
  match_thresh: 0.75             # IoU threshold for matching
  mot20: false                   # MOT20 mode (keep false)
```

### Smoothing Configuration
```yaml
smoothing:
  enabled: true                  # Enable class smoothing
  ema_alpha: 0.65                # EMA weight (0-1)
  low_prob_threshold: 0.75       # Hold previous if below
  history_length: 10             # Frames in history
```

### Visualization Configuration
```yaml
visualization:
  show_original: false           # Show side-by-side
  show_trails: true              # Show tracking trails
  
  trail:
    max_length: 40
    color: [255, 0, 0]           # BGR
    fading: true
  
  latency:
    show_overall: true
    show_details: false
  
  window:
    title: "Polyp Detection"
    width: 800
    height: 600
```

---

## 🎯 Use Cases

### 1. Classification with Tracker (Default)
**Config**: `config.yaml`  
**Features**: 2-class classification, tracking, smoothing, trails  
**Best for**: Production use, stable classification

```bash
python run_inference.py --config config.yaml
```

**Key Settings**:
- Tracker enabled
- Smoothing enabled (EMA α=0.65)
- Score threshold: 0.70

---

### 2. Detection Only (Single Class)
**Config**: `config_detection.yaml`  
**Features**: Single class (polyp), tracking, no smoothing  
**Best for**: Testing detection models

```bash
python run_inference.py --config config_detection.yaml
```

**Key Settings**:
- Single class: "polyp"
- Tracker enabled
- No smoothing (not needed for single class)

---

### 3. High-Resolution Mode
**Config**: `config_highres.yaml`  
**Features**: Letterbox preprocessing, maintains aspect ratio  
**Best for**: High-resolution videos, production deployment

```bash
python run_inference.py --config config_highres.yaml
```

**Key Settings**:
- `high_resolution: true`
- Letterbox preprocessing
- Tighter thresholds (score=0.80, track=0.35)
- Lower EMA α=0.15 for stability

---

### 4. No Tracker (Raw Detection)
**Config**: `config_no_tracker.yaml`  
**Features**: Frame-by-frame detection, no temporal consistency  
**Best for**: Quick model testing, debugging

```bash
python run_inference.py --config config_no_tracker.yaml
```

**Key Settings**:
- Tracker disabled
- Smoothing disabled
- No trails
- Same threshold for display and detection

---

## 🔧 Customization

### Change Model
Edit `model.path` in config file:
```yaml
model:
  path: "./output/my_model/model.onnx"
```

### Change Video Source
```yaml
video:
  source: 0                    # Webcam
  source: 2                    # USB camera
  source: "video.mp4"          # Video file
```

### Adjust Tracker Sensitivity
```yaml
tracker:
  track_thresh: 0.35           # Lower = more sensitive
  track_buffer: 90             # Higher = longer memory
  match_thresh: 0.55           # Lower = more lenient matching
```

### Adjust Smoothing
```yaml
smoothing:
  ema_alpha: 0.30              # Lower = more stable, less responsive
  low_prob_threshold: 0.95     # Higher = hold previous class more often
```

### Add New Classes
```yaml
model:
  classes:
    names: ["adenoma", "hyperplastic", "benign"]
    colors: [[0, 0, 255], [0, 255, 0], [157, 0, 255]]
```

---

## 📊 Performance Tuning

### GPU Acceleration
Automatically uses CUDA if available. Check output:
```
✓ ONNX Runtime using: CUDAExecutionProvider
```

### Latency Optimization
```yaml
# Show detailed timing
visualization:
  latency:
    show_overall: true
    show_details: true    # Shows Pre/Inf/Post times
```

### High-Resolution Optimization
```yaml
video:
  high_resolution: true   # Use letterbox preprocessing
  
performance:
  frame_queue_size: 1     # Smaller = lower latency
  result_queue_size: 1
```

---

## 🐛 Troubleshooting

### Model Not Found
```
✗ Model not found: ./output/model.onnx
```
**Solution**: Check `model.path` in config file

### Video Source Error
```
✗ Error: Could not open video source: 2
```
**Solution**: Try different camera index (0, 1, 2) or check video file path

### Import Error
```
ModuleNotFoundError: No module named 'tracker'
```
**Solution**: Run from scripts/ directory or check tracker/ module exists

### Low FPS
- Reduce `input_size` in config
- Disable `show_original`
- Disable `show_trails`
- Use GPU (CUDA)

---

## 🔄 Migration from Old Scripts

| Old Script | New Command |
|------------|-------------|
| `app_pyqt6_classification.py` | `python run_inference.py --config config_no_tracker.yaml` |
| `smoothening_inference.py` | `python run_inference.py --config config.yaml` |
| `classification_smoother_infer.py` | `python run_inference.py --config config.yaml` |
| `infer_high_resolution.py` | `python run_inference.py --config config_highres.yaml` |

**Benefits**:
- Single codebase
- No code changes needed
- Just edit YAML config
- All logic preserved

---

## 📝 Configuration Templates

### Minimal Config (Detection)
```yaml
model:
  path: "./model.onnx"
  input_size: [640, 640]
  task: "detection"
  classes:
    names: ["polyp"]
    colors: [[0, 255, 0]]
  score_threshold: 0.70

video:
  source: 0

tracker:
  enabled: false

visualization:
  show_original: false
  show_trails: false
```

### Full Config (Classification + Tracking)
See `config.yaml` for complete example with all options.

---

## 🎓 Key Concepts

### Tracker Thresholds
- **track_thresh**: Minimum confidence to START a new track
- **score_threshold**: Minimum confidence to DISPLAY a track
- **tracker_input_threshold**: Minimum confidence to FEED to tracker (gating)

**Relationship**: `tracker_input_threshold < track_thresh < score_threshold`

### Smoothing (EMA)
Reduces class flickering using Exponential Moving Average:
```
smoothed = α × current + (1-α) × previous
```
- Higher α (0.8): More responsive, less stable
- Lower α (0.2): More stable, less responsive

### High-Resolution Mode
Uses letterbox preprocessing to:
- Preserve aspect ratio
- Maintain original resolution
- Avoid distortion
- Better for production

---

## ✅ Testing Checklist

- [ ] Test with webcam (`source: 0`)
- [ ] Test with video file (`source: "video.mp4"`)
- [ ] Test detection config
- [ ] Test classification config
- [ ] Test high-res mode
- [ ] Test without tracker
- [ ] Verify GPU usage
- [ ] Check latency overlay
- [ ] Test different models

---

## 🚀 Next Steps

1. **Start with default config**:
   ```bash
   python run_inference.py
   ```

2. **Test your model**:
   - Edit `model.path` in `config.yaml`
   - Run inference

3. **Tune parameters**:
   - Adjust thresholds
   - Enable/disable features
   - Optimize for your use case

4. **Create custom config**:
   - Copy existing config
   - Modify for your needs
   - Save with descriptive name

---

## 📚 Additional Resources

- **Tracker Module**: `../tracker/` (ByteTrack implementation)
- **Model Training**: `../src/` (RT-DETR training code)
- **Data Pipelines**: `../data_pipelines/` (Dataset preparation)

---

## 💡 Tips

1. **Always test with `--dry-run` first** (if implemented)
2. **Start with default config** before customizing
3. **Use high-res mode for production**
4. **Disable tracker for quick model testing**
5. **Lower EMA α for more stable classification**
6. **Check GPU usage** for performance

---

**Ready to test!** 🎉

```bash
cd scripts
python run_inference.py --config config.yaml
```
