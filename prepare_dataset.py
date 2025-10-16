import os
import json
import cv2
import numpy as np
import random
import shutil

# Constants
ADD_NEGATIVE_SAMPLES = True  # Set to True to include negative samples (images without polyps/masks, empty annotations)
MIN_AREA_THRESHOLD = 50  # Minimum area (in pixels) for a connected component to be considered a valid polyp; adjust as needed to filter noise


def prepare_data(dataset_root="dataset", output_dir="coco", train_split=0.8, seed=42):
    """
    Converts the dataset to COCO format for RT-DETR fine-tuning.
    - Matches images and masks by basename (without extension), supporting any names (e.g., integers, hashes).
    - Supports .png, .jpg, .jpeg for images and masks.
    - Bounding boxes are computed from binary masks using connected components.
    - Assumes masks where any non-black color is foreground (polyp).
    - Single category: 'polyp' with id=1.
    - If ADD_NEGATIVE_SAMPLES=True, includes images from 'negative_samples/' with empty annotations.
    - Blends negatives with positives, random split into train/val.
    - Filters small connected components (noise) using MIN_AREA_THRESHOLD.
    """
    random.seed(seed)

    images_dir = os.path.join(dataset_root, "images")
    masks_dir = os.path.join(dataset_root, "masks")
    negative_dir = (
        os.path.join(dataset_root, "negative_samples") if ADD_NEGATIVE_SAMPLES else None
    )

    # Supported extensions
    img_extensions = (".png", ".jpg", ".jpeg")
    mask_extensions = (".png", ".jpg", ".jpeg", ".tif")  # Added .tif if needed

    # Get all image files and their basenames (without extension)
    image_files = [
        f for f in os.listdir(images_dir) if f.lower().endswith(img_extensions)
    ]
    image_basenames = set(os.path.splitext(f)[0] for f in image_files)

    # Get all mask files and their basenames
    mask_files = [
        f for f in os.listdir(masks_dir) if f.lower().endswith(mask_extensions)
    ]
    mask_basenames = set(os.path.splitext(f)[0] for f in mask_files)

    # Find positive samples: basenames with both image and mask
    positive_basenames = sorted(image_basenames & mask_basenames)
    print(f"Found {len(positive_basenames)} positive image-mask pairs.")

    # Negative samples if enabled
    negative_basenames = []
    if ADD_NEGATIVE_SAMPLES and os.path.exists(negative_dir):
        negative_files = [
            f for f in os.listdir(negative_dir) if f.lower().endswith(img_extensions)
        ]
        negative_basenames = sorted(os.path.splitext(f)[0] for f in negative_files)
        print(f"Found {len(negative_basenames)} negative samples.")
    else:
        print("No negative samples added (directory not found or disabled).")

    # Combine positives and negatives for splitting
    all_basenames = positive_basenames + negative_basenames
    random.shuffle(all_basenames)
    train_size = int(train_split * len(all_basenames))
    train_basenames = all_basenames[:train_size]
    val_basenames = all_basenames[train_size:]

    # Create output directories
    os.makedirs(os.path.join(output_dir, "train2017"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val2017"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "annotations"), exist_ok=True)

    # Common COCO fields
    info = {
        "year": 2025,
        "version": "1",
        "description": "Polyp Detection Dataset",
        "contributor": "",
        "url": "",
        "date_created": "2025/10/16",
    }
    licenses = [{"id": 1, "name": "Unknown", "url": ""}]
    categories = [{"id": 1, "name": "polyp", "supercategory": "none"}]

    # Process each split
    for split, split_basenames, folder in [
        ("train", train_basenames, "train2017"),
        ("val", val_basenames, "val2017"),
    ]:
        images = []
        annotations = []
        ann_id = 1

        for image_id, basename in enumerate(split_basenames, start=1):
            # Determine if positive or negative
            is_positive = basename in positive_basenames
            source_dir = images_dir if is_positive else negative_dir

            # Find actual image file (support multiple extensions)
            img_file_candidates = [
                f
                for f in os.listdir(source_dir)
                if os.path.splitext(f)[0] == basename
                and f.lower().endswith(img_extensions)
            ]
            if not img_file_candidates:
                print(
                    f"Warning: No image file found for basename {basename} in {source_dir}"
                )
                continue
            img_file = img_file_candidates[0]  # Take the first match
            img_path = os.path.join(source_dir, img_file)

            # Read image to get dimensions
            img = cv2.imread(img_path)
            if img is None:
                print(f"Error: Could not read image {img_path}")
                continue
            height, width = img.shape[:2]

            # Copy image to output (keep original filename)
            shutil.copy(img_path, os.path.join(output_dir, folder, img_file))

            # Add image info
            images.append(
                {
                    "id": image_id,
                    "width": width,
                    "height": height,
                    "file_name": img_file,
                    "license": 1,
                    "flickr_url": "",
                    "coco_url": "",
                    "date_captured": "",
                }
            )

            # Process annotations only for positives
            if is_positive:
                # Find actual mask file
                mask_file_candidates = [
                    f
                    for f in os.listdir(masks_dir)
                    if os.path.splitext(f)[0] == basename
                    and f.lower().endswith(mask_extensions)
                ]
                if not mask_file_candidates:
                    print(f"Warning: No mask file found for basename {basename}")
                    continue
                mask_file = mask_file_candidates[0]
                mask_path = os.path.join(masks_dir, mask_file)

                # Read mask as color
                mask = cv2.imread(mask_path)
                if mask is None:
                    print(f"Error: Could not read mask {mask_path}")
                    continue

                # Create binary mask: any non-black pixel (any channel >0) is foreground
                binary_mask = np.any(mask > 0, axis=2).astype(np.uint8) * 255

                # Optional: Apply morphological opening to remove small noise
                kernel = np.ones((3, 3), np.uint8)
                binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

                # Find connected components
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                    binary_mask, connectivity=8
                )

                # Add annotations for each component (polyp), filtering small areas
                for label in range(1, num_labels):
                    x = stats[label, cv2.CC_STAT_LEFT]
                    y = stats[label, cv2.CC_STAT_TOP]
                    w = stats[label, cv2.CC_STAT_WIDTH]
                    h = stats[label, cv2.CC_STAT_HEIGHT]
                    area = stats[label, cv2.CC_STAT_AREA]

                    if area > MIN_AREA_THRESHOLD:
                        annotations.append(
                            {
                                "id": ann_id,
                                "image_id": image_id,
                                "category_id": 1,
                                "bbox": [float(x), float(y), float(w), float(h)],
                                "area": float(area),
                                "segmentation": [],
                                "iscrowd": 0,
                            }
                        )
                        ann_id += 1
            # For negatives: empty annotations (already handled by not adding any)

        # Create JSON
        json_data = {
            "info": info,
            "licenses": licenses,
            "categories": categories,
            "images": images,
            "annotations": annotations,
        }

        ann_file = (
            f"instances_{folder}.json"
            if folder == "train2017"
            else "instances_val2017.json"
        )
        with open(os.path.join(output_dir, "annotations", ann_file), "w") as f:
            json.dump(json_data, f, indent=4)

        print(
            f"{split.capitalize()} split: {len(images)} images, {len(annotations)} annotations"
        )


if __name__ == "__main__":
    # Customize paths if needed
    prepare_data(
        dataset_root="detection_dataset", output_dir="coco", train_split=0.8, seed=42
    )
