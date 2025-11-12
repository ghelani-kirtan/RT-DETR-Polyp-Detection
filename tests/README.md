# Tests Directory

## Test Scripts

### `test_detection_annotations.py`
Visualize detection annotations before preparing COCO dataset.
```bash
python tests/test_detection_annotations.py \
    --images-dir data/temp/detection_dataset/images \
    --masks-dir data/temp/detection_dataset/masks \
    --remove-subsets
```

### `test_v2_integration.py`
Integration tests for DetectionPreparerV2.
```bash
python tests/test_v2_integration.py
```

### `compare_preparers.py`
Compare V1 vs V2 preparers.
```bash
python tests/compare_preparers.py --base-dir data/temp
```

### `test_bbox_processor.py`
Test bbox processor functionality.
```bash
python tests/test_bbox_processor.py
```
