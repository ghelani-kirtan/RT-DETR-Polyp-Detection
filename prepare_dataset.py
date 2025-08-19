import os
import json
import cv2
import numpy as np
import random
import shutil

def prepare_data(dataset_root='dataset', output_dir='coco', train_split=0.8, seed=42):
    """
    Converts the dataset to COCO format for RT-DETR fine-tuning.
    - Images are copied as .png (original format).
    - Bounding boxes are computed from binary masks using connected components.
    - Assumes binary masks where >127 is foreground (polyp).
    - Single category: 'polyp' with id=1.
    - Random split into train/val.
    """
    random.seed(seed)

    images_dir = os.path.join(dataset_root, 'images')
    masks_dir = os.path.join(dataset_root, 'masks')

    # Get list of image IDs (assuming filenames like 1.png, 2.png, etc.)
    ids = sorted([int(f[:-4]) for f in os.listdir(images_dir) if f.endswith('.png')])

    # Shuffle and split
    random.shuffle(ids)
    train_size = int(train_split * len(ids))
    train_ids = ids[:train_size]
    val_ids = ids[train_size:]

    # Create output directories
    os.makedirs(os.path.join(output_dir, 'train2017'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val2017'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'annotations'), exist_ok=True)

    # Common COCO fields
    info = {
        "year": 2025,
        "version": "1",
        "description": "Polyp Detection Dataset",
        "contributor": "",
        "url": "",
        "date_created": "2025/08/18"
    }
    licenses = [{"id": 1, "name": "Unknown", "url": ""}]
    categories = [{"id": 1, "name": "polyp", "supercategory": "none"}]

    # Process each split
    for split, split_ids, folder in [('train', train_ids, 'train2017'), ('val', val_ids, 'val2017')]:
        images = []
        annotations = []
        ann_id = 1

        for image_id, orig_id in enumerate(split_ids, start=1):
            img_path = os.path.join(images_dir, f'{orig_id}.png')
            mask_path = os.path.join(masks_dir, f'{orig_id}.tif')

            if not os.path.exists(img_path) or not os.path.exists(mask_path):
                print(f"Warning: Missing file for ID {orig_id}")
                continue

            # Read image to get dimensions
            img = cv2.imread(img_path)
            if img is None:
                print(f"Error: Could not read image {img_path}")
                continue
            height, width = img.shape[:2]

            # Copy image to output (keep as .png)
            file_name = f'{orig_id}.png'
            shutil.copy(img_path, os.path.join(output_dir, folder, file_name))

            # Add image info
            images.append({
                "id": image_id,
                "width": width,
                "height": height,
                "file_name": file_name,
                "license": 1,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": ""
            })

            # Read and process mask
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Error: Could not read mask {mask_path}")
                continue

            # Binarize mask
            _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

            # Find connected components
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

            # Add annotations for each component (polyp)
            for label in range(1, num_labels):
                x = stats[label, cv2.CC_STAT_LEFT]
                y = stats[label, cv2.CC_STAT_TOP]
                w = stats[label, cv2.CC_STAT_WIDTH]
                h = stats[label, cv2.CC_STAT_HEIGHT]
                area = stats[label, cv2.CC_STAT_AREA]

                annotations.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [float(x), float(y), float(w), float(h)],
                    "area": float(area),
                    "segmentation": [],
                    "iscrowd": 0
                })
                ann_id += 1

        # Create JSON
        json_data = {
            "info": info,
            "licenses": licenses,
            "categories": categories,
            "images": images,
            "annotations": annotations
        }

        ann_file = f'instances_{folder}.json' if folder == 'train2017' else 'instances_val2017.json'
        with open(os.path.join(output_dir, 'annotations', ann_file), 'w') as f:
            json.dump(json_data, f, indent=4)

        print(f"{split.capitalize()} split: {len(images)} images, {len(annotations)} annotations")

if __name__ == "__main__":
    # Customize paths if needed
    prepare_data(dataset_root='dataset', output_dir='coco', train_split=0.8, seed=42)