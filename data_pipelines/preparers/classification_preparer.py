"""Classification dataset COCO format preparer."""

import cv2
import numpy as np
import shutil
from pathlib import Path
from typing import Dict
from tqdm import tqdm
from .base_preparer import BasePreparer
from ..core import PreparerConfig, FileUtils


class ClassificationPreparer(BasePreparer):
    """Prepare classification dataset in COCO format."""
    
    def prepare(self) -> Dict[str, any]:
        """
        Prepare classification dataset in COCO format.
        
        Returns:
            Preparation statistics
        """
        images_dir = self.config.input_dir / "images"
        masks_dir = self.config.input_dir / "masks"
        negative_dir = self.config.input_dir / "negative_samples"
        
        if not images_dir.exists() or not masks_dir.exists():
            self.logger.error("Images or masks directory not found")
            return {}
        
        # Get positive and negative samples
        positive_stems, _ = FileUtils.get_matching_pairs(
            images_dir,
            masks_dir,
            self.config.image_extensions,
            self.config.mask_extensions
        )
        
        negative_stems = []
        if self.config.add_negative_samples and negative_dir.exists():
            negative_stems = list(
                FileUtils.get_file_stems(
                    negative_dir,
                    self.config.image_extensions
                )
            )
        
        self.logger.info(
            f"Found {len(positive_stems)} positive and "
            f"{len(negative_stems)} negative samples"
        )
        
        # Split dataset
        train_stems, val_stems = self.split_dataset(
            positive_stems,
            negative_stems
        )
        
        # Prepare output structure
        self.prepare_output_structure()
        
        # Process each split
        stats = {}
        for split, stems in [("train", train_stems), ("val", val_stems)]:
            stats[split] = self._process_split(
                split,
                stems,
                positive_stems,
                images_dir,
                masks_dir,
                negative_dir
            )
        
        return stats
    
    def _process_split(
        self,
        split: str,
        stems: list,
        positive_stems: list,
        images_dir: Path,
        masks_dir: Path,
        negative_dir: Path
    ) -> Dict:
        """Process a single split (train/val)."""
        folder = (
            self.config.train_folder if split == "train"
            else self.config.val_folder
        )
        output_folder = self.config.output_dir / folder
        
        coco_data = self.create_coco_base()
        ann_id = 1
        
        for image_id, stem in enumerate(
            tqdm(stems, desc=f"Processing {split}"),
            start=1
        ):
            is_positive = stem in positive_stems
            source_dir = images_dir if is_positive else negative_dir
            
            # Find image file
            img_file = FileUtils.find_matching_file(
                source_dir,
                stem,
                self.config.image_extensions
            )
            
            if not img_file:
                continue
            
            # Read image
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            
            height, width = img.shape[:2]
            
            # Copy image to output
            if not self.config.dry_run:
                shutil.copy(img_file, output_folder / img_file.name)
            
            # Add image info
            coco_data["images"].append({
                "id": image_id,
                "width": width,
                "height": height,
                "file_name": img_file.name,
                "license": 1,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": ""
            })
            
            # Process annotations for positive samples
            if is_positive:
                mask_file = FileUtils.find_matching_file(
                    masks_dir,
                    stem,
                    self.config.mask_extensions
                )
                
                if mask_file:
                    annotations = self._extract_annotations(
                        mask_file,
                        image_id,
                        ann_id
                    )
                    coco_data["annotations"].extend(annotations)
                    ann_id += len(annotations)
        
        # Save COCO JSON
        self.save_coco_json(coco_data, split)
        
        return {
            "images": len(coco_data["images"]),
            "annotations": len(coco_data["annotations"])
        }
    
    def _extract_annotations(
        self,
        mask_file: Path,
        image_id: int,
        start_ann_id: int
    ) -> list:
        """Extract annotations from colored mask."""
        annotations = []
        
        # Read mask (BGR format)
        mask_img = cv2.imread(str(mask_file))
        if mask_img is None:
            return annotations
        
        ann_id = start_ann_id
        
        # Color channel mapping for each class
        # Based on CLASS_COLORS from original script:
        # adenoma: (255, 0, 0) - Red
        # hyperplastic: (0, 255, 0) - Green
        # benign: (157, 0, 255) - Purple (high red + high blue)
        # no_pathology: (255, 255, 255) - White (all channels)
        
        # Process each category based on color channels
        for category in self.config.categories:
            cat_id = category["id"]
            cat_name = category["name"].lower()
            
            # Determine which channel/combination to use based on category
            if cat_name == "adenoma":
                # Red channel only (high red, low green, low blue)
                channel = mask_img[:, :, 2]  # Red channel (BGR format)
            elif cat_name == "hyperplastic":
                # Green channel only (low red, high green, low blue)
                channel = mask_img[:, :, 1]  # Green channel
            elif cat_name == "benign":
                # Purple: high red + high blue, low green
                # Detect where red and blue are high but green is low
                red_channel = mask_img[:, :, 2]
                blue_channel = mask_img[:, :, 0]
                green_channel = mask_img[:, :, 1]
                # Purple mask: red > 127 AND blue > 127 AND green < 128
                channel = np.where(
                    (red_channel > 127) & (blue_channel > 127) & (green_channel < 128),
                    255,
                    0
                ).astype(np.uint8)
            elif cat_name == "no_pathology" or cat_name == "no pathology":
                # White: all channels high
                channel = np.where(
                    (mask_img[:, :, 0] > 200) & 
                    (mask_img[:, :, 1] > 200) & 
                    (mask_img[:, :, 2] > 200),
                    255,
                    0
                ).astype(np.uint8)
            else:
                continue
            
            # For adenoma and hyperplastic, threshold the channel
            if cat_name in ["adenoma", "hyperplastic"]:
                _, binary_mask = cv2.threshold(
                    channel,
                    127,
                    255,
                    cv2.THRESH_BINARY
                )
            else:
                # For benign and no_pathology, channel is already binary
                binary_mask = channel
            
            # Find connected components
            num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
                binary_mask,
                connectivity=8
            )
            
            # Create annotations for each component
            for label in range(1, num_labels):
                x = stats[label, cv2.CC_STAT_LEFT]
                y = stats[label, cv2.CC_STAT_TOP]
                w = stats[label, cv2.CC_STAT_WIDTH]
                h = stats[label, cv2.CC_STAT_HEIGHT]
                area = stats[label, cv2.CC_STAT_AREA]
                
                if area > self.config.min_area_threshold:
                    annotations.append({
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": cat_id,
                        "bbox": [float(x), float(y), float(w), float(h)],
                        "area": float(area),
                        "segmentation": [],
                        "iscrowd": 0
                    })
                    ann_id += 1
        
        return annotations
