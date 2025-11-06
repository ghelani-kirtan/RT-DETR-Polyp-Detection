# Bug Fixes Applied to Data Pipeline

## Summary

This document lists all bugs identified and fixed in the new data pipeline system.

---

## Bug #1: Missing `class_mappings` in DownloaderConfig

**Location**: `data-pipelines/core/config.py`

**Issue**: The `DownloaderConfig` class was missing the `class_mappings` field, which is required by the API downloader to map pathology names to directory names.

**Original Code**:
```python
@dataclass
class DownloaderConfig(PipelineConfig):
    # ... other fields ...
    retry_delay: int = 5
    # Missing class_mappings!
```

**Fixed Code**:
```python
@dataclass
class DownloaderConfig(PipelineConfig):
    # ... other fields ...
    retry_delay: int = 5
    
    # Class mappings for organizing downloads
    class_mappings: Dict[str, str] = field(default_factory=dict)
```

**Impact**: Without this fix, the downloader couldn't properly organize downloaded files by class.

---

## Bug #2: Incorrect Path Resolution in Detection Organizer

**Location**: `data-pipelines/organizers/detection_organizer.py`

**Issue**: The organizer was looking for `client_data` in the wrong location and didn't handle missing directories gracefully.

**Original Code**:
```python
def organize(self) -> Dict[str, any]:
    stats = {"images": 0, "masks": 0, "negatives": 0}
    
    client_data_path = self.config.input_dir / "client_data"
    positive_path = client_data_path / self.config.positive_subdir
    negative_path = client_data_path / self.config.negative_subdir
    # No error handling if client_data doesn't exist
```

**Fixed Code**:
```python
def organize(self) -> Dict[str, any]:
    stats = {"images": 0, "masks": 0, "negatives": 0}
    
    # Look for client_data in input_dir (base_dir)
    client_data_path = self.config.input_dir / "client_data"
    if not client_data_path.exists():
        self.logger.error(f"client_data directory not found at {client_data_path}")
        return stats
        
    positive_path = client_data_path / self.config.positive_subdir
    negative_path = client_data_path / self.config.negative_subdir
```

**Impact**: Without this fix, the organizer would fail silently or crash when `client_data` didn't exist.

---

## Bug #3: Incorrect Path Resolution in Classification Organizer

**Location**: `data-pipelines/organizers/classification_organizer.py`

**Issue**: Same as Bug #2, but for classification organizer.

**Fixed**: Added proper path resolution and error handling (same fix as Bug #2).

---

## Bug #4: Incorrect Import Paths in CLI Scripts

**Location**: `data-pipelines/cli_detection.py` and `data-pipelines/cli_classification.py`

**Issue**: The CLI scripts used absolute imports instead of relative imports, which would fail when running as a module.

**Original Code**:
```python
from pipelines import DetectionPipeline
```

**Fixed Code**:
```python
from .pipelines import DetectionPipeline
```

**Impact**: Without this fix, running `python -m data-pipelines.cli_detection` would fail with import errors.

---

## Bug #5: Incomplete Color Channel Handling in Classification Preparer

**Location**: `data-pipelines/preparers/classification_preparer.py`

**Issue**: The classification preparer only handled `adenoma` (red) and `hyperplastic` (green) classes, but your original script also supports `benign` (purple) and `no_pathology` (white).

**Original Code**:
```python
def _extract_annotations(...):
    for category in self.config.categories:
        cat_name = category["name"]
        
        if cat_name == "adenoma":
            channel = mask_img[:, :, 2]  # Red
        elif cat_name == "hyperplastic":
            channel = mask_img[:, :, 1]  # Green
        else:
            continue  # Skips benign and no_pathology!
```

**Fixed Code**:
```python
def _extract_annotations(...):
    for category in self.config.categories:
        cat_name = category["name"].lower()
        
        if cat_name == "adenoma":
            channel = mask_img[:, :, 2]  # Red
        elif cat_name == "hyperplastic":
            channel = mask_img[:, :, 1]  # Green
        elif cat_name == "benign":
            # Purple: high red + high blue, low green
            red_channel = mask_img[:, :, 2]
            blue_channel = mask_img[:, :, 0]
            green_channel = mask_img[:, :, 1]
            channel = np.where(
                (red_channel > 127) & (blue_channel > 127) & (green_channel < 128),
                255, 0
            ).astype(np.uint8)
        elif cat_name == "no_pathology" or cat_name == "no pathology":
            # White: all channels high
            channel = np.where(
                (mask_img[:, :, 0] > 200) & 
                (mask_img[:, :, 1] > 200) & 
                (mask_img[:, :, 2] > 200),
                255, 0
            ).astype(np.uint8)
        else:
            continue
```

**Impact**: Without this fix, benign and no_pathology classes would be completely ignored during COCO preparation.

---

## Bug #6: Missing Categories in Classification Pipeline

**Location**: `data-pipelines/pipelines/classification_pipeline.py`

**Issue**: The classification pipeline only defined 2 categories (adenoma, hyperplastic) but should support all 4 classes.

**Original Code**:
```python
self.preparer_config = PreparerConfig(
    categories=[
        {"id": 1, "name": "adenoma", "supercategory": "none"},
        {"id": 2, "name": "hyperplastic", "supercategory": "none"}
    ]
)
```

**Fixed Code**:
```python
self.preparer_config = PreparerConfig(
    categories=[
        {"id": 1, "name": "adenoma", "supercategory": "none"},
        {"id": 2, "name": "hyperplastic", "supercategory": "none"},
        {"id": 3, "name": "benign", "supercategory": "none"},
        {"id": 4, "name": "no_pathology", "supercategory": "none"}
    ]
)
```

**Impact**: Without this fix, benign and no_pathology annotations wouldn't be created in the COCO JSON.

---

## Verification Checklist

- [x] **Bug #1**: DownloaderConfig has class_mappings field
- [x] **Bug #2**: Detection organizer handles missing client_data
- [x] **Bug #3**: Classification organizer handles missing client_data
- [x] **Bug #4**: CLI scripts use relative imports
- [x] **Bug #5**: Classification preparer handles all 4 color classes
- [x] **Bug #6**: Classification pipeline defines all 4 categories

---

## Testing Recommendations

### Test 1: Detection Pipeline
```bash
python -m data-pipelines.cli_detection \
    --base-dir ./test_detection \
    --api-url http://your-server:8000/api/v1/dataset-versions/list \
    --dataset-version-ids 1 \
    --step full \
    --dry-run
```

### Test 2: Classification Pipeline
```bash
python -m data-pipelines.cli_classification \
    --base-dir ./test_classification \
    --api-url http://your-server:8000/api/v1/dataset-versions/list \
    --dataset-version-ids 1 \
    --step full \
    --dry-run
```

### Test 3: Individual Steps
```bash
# Test organize step with existing client_data
python -m data-pipelines.cli_detection \
    --base-dir ./existing_project \
    --step organize \
    --dry-run
```

### Test 4: Error Handling
```bash
# Test with non-existent client_data (should show error message)
python -m data-pipelines.cli_detection \
    --base-dir ./empty_dir \
    --step organize
```

---

## Additional Improvements Made

Beyond bug fixes, the following improvements were also made:

1. **Better Error Messages**: All path-related errors now show the exact path that was expected
2. **Consistent Logging**: All components use the same logging format
3. **Validation**: Each step validates its inputs before processing
4. **Progress Bars**: tqdm progress bars for long-running operations
5. **Dry Run Support**: Test mode for all operations

---

## Original Logic Preserved

The following core logic from your original scripts was preserved exactly:

### Detection (from `prepare_dataset.py`):
- Binary mask processing (any non-black pixel is foreground)
- Connected components for bounding box extraction
- Morphological opening to remove noise
- MIN_AREA_THRESHOLD filtering (default: 50 pixels)
- Single category: "polyp"

### Classification (from `clean_and_organize_classification_dataset.py` and `prepare_dataset_classification.py`):
- Class-specific color mapping:
  - Adenoma: Red (255, 0, 0)
  - Hyperplastic: Green (0, 255, 0)
  - Benign: Purple (157, 0, 255)
  - No Pathology: White (255, 255, 255)
- Binary to colored mask conversion
- Image renaming with class suffix
- Multi-class bounding box extraction from colored masks

### Download (from `dataset_versions_downloader.py`):
- API integration with dataset version IDs
- S3 multipart download with retries
- Parallel downloads with ThreadPoolExecutor
- Post-download verification
- Class-based directory organization
- Negative sample handling

---

## Status: All Bugs Fixed ✓

The pipeline is now ready for testing and production use.
