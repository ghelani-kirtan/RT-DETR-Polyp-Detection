import os
import json
import cv2
import numpy as np
import random
import shutil

# Constants
ADD_NEGATIVE_SAMPLES = True  # Set to True to include negative samples (images without polyps/masks, empty annotations)


def prepare_data(
    dataset_root="classification_dataset",
    output_dir="coco_classification",
    train_split=0.8,
    seed=42,
):
    """
    Converts the dataset to COCO format for RT-DETR fine-tuning.
    - Matches images and masks by basename (without extension), supporting any names (e.g., integers, hashes).
    - Supports .png, .jpg, .jpeg for images and masks.
    - Bounding boxes are computed from colored masks using connected components.
    - Assumes colored masks in JPEG where red spots (>127 in red channel) are Adenoma (class 1),
      green spots (>127 in green channel) are Hyperplastic (class 2).
    - Two categories: 'adenoma' with id=1, 'hyperplastic' with id=2.
    - If ADD_NEGATIVE_SAMPLES=True, includes images from 'negative_samples/' with empty annotations.
    - Blends negatives with positives, random split into train/val.
    """
    random.seed(seed)

    images_dir = os.path.join(dataset_root, "images")
    masks_dir = os.path.join(dataset_root, "masks")
    negative_dir = (
        os.path.join(dataset_root, "negative_samples") if ADD_NEGATIVE_SAMPLES else None
    )

    # Supported extensions
    img_extensions = (".png", ".jpg", ".jpeg")
    mask_extensions = (".png", ".jpg", ".jpeg")

    # Get all image files and their basenames (without extension)
    image_files = [
        f for f in os.listdir(images_dir) if f.lower().endswith(img_extensions)
    ]
    image_basenames = sorted(
        set(f.rsplit(".", 1)[0] for f in image_files)
    )  # unique hashes

    # Get all mask files and their basenames
    mask_files = [
        f for f in os.listdir(masks_dir) if f.lower().endswith(mask_extensions)
    ]
    mask_basenames = set(f.rsplit(".", 1)[0] for f in mask_files)

    # Find common hashes (present in both images and masks) - positives
    positive_basenames = sorted(set(image_basenames) & mask_basenames)
    print(
        f"Found {len(positive_basenames)} positive image-mask pairs out of {len(image_basenames)} images and {len(mask_basenames)} masks."
    )

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
        "description": "Polyp Detection Dataset with Classes",
        "contributor": "",
        "url": "",
        "date_created": "2025/08/18",
    }
    licenses = [{"id": 1, "name": "Unknown", "url": ""}]
    categories = [
        {"id": 1, "name": "adenoma", "supercategory": "none"},
        {"id": 2, "name": "hyperplastic", "supercategory": "none"},
    ]

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

            # Find actual image file (support .jpg and .jpeg)
            img_file_candidates = [
                f
                for f in os.listdir(source_dir)
                if f.rsplit(".", 1)[0] == basename
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

            # Copy image to output (keep original extension)
            file_name = img_file  # Use original filename
            shutil.copy(img_path, os.path.join(output_dir, folder, file_name))

            # Add image info
            images.append(
                {
                    "id": image_id,
                    "width": width,
                    "height": height,
                    "file_name": file_name,
                    "license": 1,
                    "flickr_url": "",
                    "coco_url": "",
                    "date_captured": "",
                }
            )

            # Process annotations only for positives
            if is_positive:
                # Find actual mask file (support .jpg and .jpeg)
                mask_file_candidates = [
                    f for f in mask_files if f.rsplit(".", 1)[0] == basename
                ]
                if not mask_file_candidates:
                    print(f"Warning: No mask file found for basename {basename}")
                    continue
                mask_file = mask_file_candidates[0]
                mask_path = os.path.join(masks_dir, mask_file)

                # Read and process mask (BGR format)
                mask_img = cv2.imread(mask_path)
                if mask_img is None:
                    print(f"Error: Could not read mask {mask_path}")
                    continue

                # Process for Adenoma (red channel)
                red_channel = mask_img[:, :, 2]
                _, binary_red = cv2.threshold(red_channel, 127, 255, cv2.THRESH_BINARY)
                num_labels_red, _, stats_red, _ = cv2.connectedComponentsWithStats(
                    binary_red, connectivity=8
                )

                for label in range(1, num_labels_red):
                    x = stats_red[label, cv2.CC_STAT_LEFT]
                    y = stats_red[label, cv2.CC_STAT_TOP]
                    w = stats_red[label, cv2.CC_STAT_WIDTH]
                    h = stats_red[label, cv2.CC_STAT_HEIGHT]
                    area = stats_red[label, cv2.CC_STAT_AREA]

                    annotations.append(
                        {
                            "id": ann_id,
                            "image_id": image_id,
                            "category_id": 1,  # adenoma
                            "bbox": [float(x), float(y), float(w), float(h)],
                            "area": float(area),
                            "segmentation": [],
                            "iscrowd": 0,
                        }
                    )
                    ann_id += 1

                # Process for Hyperplastic (green channel)
                green_channel = mask_img[:, :, 1]
                _, binary_green = cv2.threshold(
                    green_channel, 127, 255, cv2.THRESH_BINARY
                )
                num_labels_green, _, stats_green, _ = cv2.connectedComponentsWithStats(
                    binary_green, connectivity=8
                )

                for label in range(1, num_labels_green):
                    x = stats_green[label, cv2.CC_STAT_LEFT]
                    y = stats_green[label, cv2.CC_STAT_TOP]
                    w = stats_green[label, cv2.CC_STAT_WIDTH]
                    h = stats_green[label, cv2.CC_STAT_HEIGHT]
                    area = stats_green[label, cv2.CC_STAT_AREA]

                    annotations.append(
                        {
                            "id": ann_id,
                            "image_id": image_id,
                            "category_id": 2,  # hyperplastic
                            "bbox": [float(x), float(y), float(w), float(h)],
                            "area": float(area),
                            "segmentation": [],
                            "iscrowd": 0,
                        }
                    )
                    ann_id += 1

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
    # prepare_data(dataset_root='dataset_classification', output_dir='coco_classification', train_split=0.8, seed=42)
    prepare_data(
        dataset_root="classification_dataset",
        output_dir="coco_classification",
        train_split=0.8,
        seed=42,
    )
