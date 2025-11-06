# Visual Workflow Diagrams

## Detection Pipeline Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    DETECTION PIPELINE                            │
└─────────────────────────────────────────────────────────────────┘

STEP 1: DOWNLOAD (from dataset_versions_downloader.py)
┌──────────┐
│   API    │
│ Server   │
└────┬─────┘
     │ dataset_version_ids: [1, 2, 3]
     ↓
┌────────────────────────────────────────────────────────┐
│  client_data/                                          │
│  ├── positive_samples/                                 │
│  │   ├── adenoma/                                      │
│  │   │   ├── images/  (video1_frame1.png, ...)        │
│  │   │   └── masks/   (video1_frame1.tif, ...)        │
│  │   ├── hyperplastic/                                 │
│  │   │   ├── images/                                   │
│  │   │   └── masks/                                    │
│  │   └── benign/                                       │
│  │       ├── images/                                   │
│  │       └── masks/                                    │
│  └── negative_samples/  (no_polyp_1.png, ...)         │
└────────────────────────────────────────────────────────┘
     │
     │ STEP 2: ORGANIZE (aggregate all classes)
     ↓
┌────────────────────────────────────────────────────────┐
│  detection_dataset/                                    │
│  ├── images/                                           │
│  │   ├── video1_frame1.png  ← from adenoma            │
│  │   ├── video1_frame2.png  ← from hyperplastic       │
│  │   ├── video2_frame1.png  ← from benign             │
│  │   └── ...                                           │
│  ├── masks/                                            │
│  │   ├── video1_frame1.tif  ← binary mask             │
│  │   ├── video1_frame2.tif  ← binary mask             │
│  │   ├── video2_frame1.tif  ← binary mask             │
│  │   └── ...                                           │
│  └── negative_samples/                                 │
│      └── no_polyp_1.png                                │
└────────────────────────────────────────────────────────┘
     │
     │ STEP 3: CLEAN (remove unmatched/corrupted)
     ↓
┌────────────────────────────────────────────────────────┐
│  detection_dataset/ (cleaned)                          │
│  ├── images/  (only matched pairs)                     │
│  ├── masks/   (only matched pairs)                     │
│  └── negative_samples/                                 │
└────────────────────────────────────────────────────────┘
     │
     │ STEP 4: PREPARE (COCO format)
     │ - Binary mask → Connected components → BBoxes
     │ - Single category: "polyp" (id=1)
     │ - 80/20 train/val split
     ↓
┌────────────────────────────────────────────────────────┐
│  coco/                                                 │
│  ├── train2017/                                        │
│  │   ├── video1_frame1.png                            │
│  │   ├── video1_frame2.png                            │
│  │   └── ...                                           │
│  ├── val2017/                                          │
│  │   ├── video2_frame1.png                            │
│  │   └── ...                                           │
│  └── annotations/                                      │
│      ├── instances_train2017.json                     │
│      │   {                                             │
│      │     "categories": [{"id": 1, "name": "polyp"}] │
│      │     "images": [...],                            │
│      │     "annotations": [                            │
│      │       {"bbox": [x,y,w,h], "category_id": 1}    │
│      │     ]                                           │
│      │   }                                             │
│      └── instances_val2017.json                       │
└────────────────────────────────────────────────────────┘
```

---

## Classification Pipeline Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                 CLASSIFICATION PIPELINE                          │
└─────────────────────────────────────────────────────────────────┘

STEP 1: DOWNLOAD (from dataset_versions_downloader.py)
┌──────────┐
│   API    │
│ Server   │
└────┬─────┘
     │ dataset_version_ids: [1, 2, 3]
     ↓
┌────────────────────────────────────────────────────────┐
│  client_data/                                          │
│  ├── positive_samples/                                 │
│  │   ├── adenoma/                                      │
│  │   │   ├── images/  (video1_frame1.png, ...)        │
│  │   │   └── masks/   (video1_frame1.tif, ...)        │
│  │   ├── hyperplastic/                                 │
│  │   ├── benign/                                       │
│  │   └── no_pathology/                                 │
│  └── negative_samples/                                 │
└────────────────────────────────────────────────────────┘
     │
     │ STEP 2: ORGANIZE (convert to colored masks)
     │ - Adenoma masks → Red (255, 0, 0)
     │ - Hyperplastic masks → Green (0, 255, 0)
     │ - Benign masks → Purple (157, 0, 255)
     │ - No Pathology masks → White (255, 255, 255)
     │ - Add class suffix to filenames
     ↓
┌────────────────────────────────────────────────────────┐
│  classification_dataset/                               │
│  ├── images/                                           │
│  │   ├── video1_frame1_adenoma.jpg                    │
│  │   ├── video1_frame2_hyperplastic.jpg               │
│  │   ├── video2_frame1_benign.jpg                     │
│  │   ├── video2_frame2_no_pathology.jpg               │
│  │   └── ...                                           │
│  ├── masks/                                            │
│  │   ├── video1_frame1_adenoma.jpg      [RED MASK]    │
│  │   ├── video1_frame2_hyperplastic.jpg [GREEN MASK]  │
│  │   ├── video2_frame1_benign.jpg       [PURPLE MASK] │
│  │   ├── video2_frame2_no_pathology.jpg [WHITE MASK]  │
│  │   └── ...                                           │
│  └── negative_samples/                                 │
└────────────────────────────────────────────────────────┘
     │
     │ STEP 3: CLEAN (remove unmatched/corrupted)
     ↓
┌────────────────────────────────────────────────────────┐
│  classification_dataset/ (cleaned)                     │
│  ├── images/  (only matched pairs)                     │
│  ├── masks/   (only matched pairs)                     │
│  └── negative_samples/                                 │
└────────────────────────────────────────────────────────┘
     │
     │ STEP 4: PREPARE (COCO format)
     │ - Red channel → Adenoma (id=1)
     │ - Green channel → Hyperplastic (id=2)
     │ - Purple detection → Benign (id=3)
     │ - White detection → No Pathology (id=4)
     │ - Connected components per color → BBoxes
     │ - 80/20 train/val split
     ↓
┌────────────────────────────────────────────────────────┐
│  coco_classification/                                  │
│  ├── train2017/                                        │
│  │   ├── video1_frame1_adenoma.jpg                    │
│  │   ├── video1_frame2_hyperplastic.jpg               │
│  │   └── ...                                           │
│  ├── val2017/                                          │
│  │   ├── video2_frame1_benign.jpg                     │
│  │   └── ...                                           │
│  └── annotations/                                      │
│      ├── instances_train2017.json                     │
│      │   {                                             │
│      │     "categories": [                             │
│      │       {"id": 1, "name": "adenoma"},             │
│      │       {"id": 2, "name": "hyperplastic"},        │
│      │       {"id": 3, "name": "benign"},              │
│      │       {"id": 4, "name": "no_pathology"}         │
│      │     ],                                          │
│      │     "images": [...],                            │
│      │     "annotations": [                            │
│      │       {"bbox": [x,y,w,h], "category_id": 1},   │
│      │       {"bbox": [x,y,w,h], "category_id": 2}    │
│      │     ]                                           │
│      │   }                                             │
│      └── instances_val2017.json                       │
└────────────────────────────────────────────────────────┘
```

---

## Command Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│  USER COMMAND                                               │
└─────────────────────────────────────────────────────────────┘
                           │
    python -m data-pipelines.cli_detection \
        --base-dir ./project \
        --api-url http://... \
        --dataset-version-ids 1 2 3 \
        --step full
                           │
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  CLI PARSER (cli_detection.py)                              │
│  - Parses arguments                                         │
│  - Creates DetectionPipeline instance                       │
└─────────────────────────────────────────────────────────────┘
                           │
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  DETECTION PIPELINE (pipelines/detection_pipeline.py)       │
│  - Initializes all configs                                  │
│  - Coordinates all steps                                    │
└─────────────────────────────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┬──────────────┐
        │                  │                  │              │
        ↓                  ↓                  ↓              ↓
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ APIDownloader│  │  Detection   │  │   Dataset    │  │  Detection   │
│              │  │  Organizer   │  │   Cleaner    │  │  Preparer    │
│ (download)   │  │ (organize)   │  │  (clean)     │  │  (prepare)   │
└──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘
        │                  │                  │              │
        ↓                  ↓                  ↓              ↓
   client_data/    detection_dataset/  (cleaned)         coco/
```

---

## File Matching Logic

### Detection Organize Step

```
INPUT: client_data/positive_samples/
       ├── adenoma/
       │   ├── images/video1_frame1.png
       │   └── masks/video1_frame1.tif
       ├── hyperplastic/
       │   ├── images/video1_frame2.png
       │   └── masks/video1_frame2.tif
       └── benign/
           ├── images/video2_frame1.png
           └── masks/video2_frame1.tif

MATCHING LOGIC:
1. For each class folder:
   - Get all images from images/
   - For each image, find mask with same stem
   - Stem = filename without extension
   
   Example:
   - Image: video1_frame1.png → stem = "video1_frame1"
   - Mask:  video1_frame1.tif → stem = "video1_frame1"
   - MATCH! ✓

2. Copy matched pairs to output:
   - All images → detection_dataset/images/
   - All masks → detection_dataset/masks/

OUTPUT: detection_dataset/
        ├── images/
        │   ├── video1_frame1.png  (from adenoma)
        │   ├── video1_frame2.png  (from hyperplastic)
        │   └── video2_frame1.png  (from benign)
        └── masks/
            ├── video1_frame1.tif
            ├── video1_frame2.tif
            └── video2_frame1.tif
```

### Classification Organize Step

```
INPUT: client_data/positive_samples/
       ├── adenoma/
       │   ├── images/video1_frame1.png
       │   └── masks/video1_frame1.tif  (binary mask)
       └── hyperplastic/
           ├── images/video1_frame2.png
           └── masks/video1_frame2.tif  (binary mask)

CONVERSION LOGIC:
1. For each class folder:
   - Get class name (e.g., "adenoma")
   - Get class color (e.g., Red = (255, 0, 0))
   
2. For each image-mask pair:
   - Convert image to JPEG
   - Convert binary mask to colored JPEG:
     * Read binary mask (grayscale)
     * Create RGB image
     * Where mask > 0, apply class color
   - Add class suffix to filename
   
   Example (adenoma):
   - Input mask: video1_frame1.tif (binary: 0 or 255)
   - Output mask: video1_frame1_adenoma.jpg (RGB: black or red)
   - Where binary mask = 255 → RGB = (255, 0, 0) [RED]

OUTPUT: classification_dataset/
        ├── images/
        │   ├── video1_frame1_adenoma.jpg
        │   └── video1_frame2_hyperplastic.jpg
        └── masks/
            ├── video1_frame1_adenoma.jpg      [RED PIXELS]
            └── video1_frame2_hyperplastic.jpg [GREEN PIXELS]
```

---

## Color Channel Extraction (Classification COCO Prep)

```
INPUT: Colored mask (BGR format in OpenCV)

┌─────────────────────────────────────────────────────┐
│  Colored Mask (JPEG)                                │
│  - Adenoma regions: Red (255, 0, 0)                 │
│  - Hyperplastic regions: Green (0, 255, 0)          │
│  - Benign regions: Purple (157, 0, 255)             │
│  - No Pathology regions: White (255, 255, 255)      │
└─────────────────────────────────────────────────────┘
                    │
                    ↓
┌─────────────────────────────────────────────────────┐
│  CHANNEL EXTRACTION                                 │
└─────────────────────────────────────────────────────┘
        │
        ├─→ Red Channel (BGR[:,:,2])
        │   - Threshold > 127
        │   - Connected components
        │   - Extract bounding boxes
        │   - Category: Adenoma (id=1)
        │
        ├─→ Green Channel (BGR[:,:,1])
        │   - Threshold > 127
        │   - Connected components
        │   - Extract bounding boxes
        │   - Category: Hyperplastic (id=2)
        │
        ├─→ Purple Detection
        │   - Red > 127 AND Blue > 127 AND Green < 128
        │   - Connected components
        │   - Extract bounding boxes
        │   - Category: Benign (id=3)
        │
        └─→ White Detection
            - All channels > 200
            - Connected components
            - Extract bounding boxes
            - Category: No Pathology (id=4)
                    │
                    ↓
┌─────────────────────────────────────────────────────┐
│  COCO ANNOTATIONS                                   │
│  {                                                  │
│    "annotations": [                                 │
│      {"bbox": [10,20,50,60], "category_id": 1},    │
│      {"bbox": [100,150,30,40], "category_id": 2}   │
│    ]                                                │
│  }                                                  │
└─────────────────────────────────────────────────────┘
```

---

## Decision Tree: Which Pipeline to Use?

```
START
  │
  ├─ Do you need to detect polyps (single class)?
  │  └─ YES → Use DETECTION PIPELINE
  │           - Binary masks
  │           - Single category: "polyp"
  │           - All classes aggregated
  │
  └─ Do you need to classify polyp types (multiple classes)?
     └─ YES → Use CLASSIFICATION PIPELINE
              - Colored masks
              - Multiple categories: adenoma, hyperplastic, benign, no_pathology
              - Class-specific annotations
```

---

## Quick Reference: One-Line Commands

```bash
# Detection - Full pipeline
python -m data-pipelines.cli_detection --base-dir ./project --api-url http://... --dataset-version-ids 1 2 3 --step full

# Classification - Full pipeline
python -m data-pipelines.cli_classification --base-dir ./project --api-url http://... --dataset-version-ids 1 2 3 --step full

# Detection - Skip download
python -m data-pipelines.cli_detection --base-dir ./project --step full --skip-download

# Classification - Skip download
python -m data-pipelines.cli_classification --base-dir ./project --step full --skip-download

# Detection - Dry run
python -m data-pipelines.cli_detection --base-dir ./project --api-url http://... --dataset-version-ids 1 --step full --dry-run

# Classification - Dry run
python -m data-pipelines.cli_classification --base-dir ./project --api-url http://... --dataset-version-ids 1 --step full --dry-run
```
