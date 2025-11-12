"""Enhanced Detection dataset COCO format preparer with smart filtering.

This is an enhanced version that integrates:
- Dynamic area thresholding based on image size
- Subset box removal for overlapping detections
- Parallel processing for better performance
- Maintains compatibility with existing pipeline architecture
"""

import cv2
import numpy as np
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base_preparer import BasePreparer
from ..core import PreparerConfig, FileUtils


class DetectionPreparerV2(BasePreparer):
    """Enhanced detection dataset preparer with smart filtering."""
    
    def __init__(self, config: PreparerConfig):
        """
        Initialize enhanced preparer.
        
        Args:
            config: Preparer configuration with additional attributes:
                - min_area_percentage: Minimum area as percentage of image (default: 0.0005 = 0.05%)
                - use_dynamic_threshold: Use dynamic threshold based on image size (default: True)
                - remove_subset_boxes: Remove boxes mostly inside larger boxes (default: False)
                - subset_threshold: Overlap percentage for subset detection (default: 0.85)
                - parallel_workers: Number of workers for parallel processing (default: 8)
        """
        super().__init__(config)
        
        # Enhanced configuration with safe defaults
        self.min_area_percentage = getattr(config, 'min_area_percentage', 0.0005)
        self.use_dynamic_threshold = getattr(config, 'use_dynamic_threshold', True)
        self.remove_subset_boxes = getattr(config, 'remove_subset_boxes', False)
        self.subset_threshold = getattr(config, 'subset_threshold', 0.85)
        self.parallel_workers = getattr(config, 'parallel_workers', 8)
        
        # Stats tracking
        self.stats = {
            'subset_boxes_removed': 0,
            'small_boxes_filtered': 0
        }
        
        self.logger.info("Enhanced Detection Preparer V2 initialized")
        self.logger.info(f"  Dynamic threshold: {self.use_dynamic_threshold}")
        self.logger.info(f"  Min area percentage: {self.min_area_percentage * 100:.3f}%")
        self.logger.info(f"  Remove subsets: {self.remove_subset_boxes}")
        if self.remove_subset_boxes:
            self.logger.info(f"  Subset threshold: {self.subset_threshold * 100:.0f}%")
    
    def compute_overlap_percentage(
        self,
        small_box: Dict,
        large_box: Dict
    ) -> float:
        """
        Calculate what percentage of small box is inside large box.
        
        Args:
            small_box: Smaller box dictionary with 'bbox' and 'area'
            large_box: Larger box dictionary with 'bbox' and 'area'
            
        Returns:
            Overlap percentage (0.0 to 1.0)
        """
        x1, y1, w1, h1 = small_box['bbox']
        x2, y2, w2, h2 = large_box['bbox']
        
        # Convert to coordinates
        small_coords = [x1, y1, x1 + w1, y1 + h1]
        large_coords = [x2, y2, x2 + w2, y2 + h2]
        
        # Compute intersection
        xi1 = max(small_coords[0], large_coords[0])
        yi1 = max(small_coords[1], large_coords[1])
        xi2 = min(small_coords[2], large_coords[2])
        yi2 = min(small_coords[3], large_coords[3])
        
        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height
        
        # Calculate percentage of small box inside large box
        if small_box['area'] == 0:
            return 0.0
        
        return inter_area / small_box['area']
    
    def filter_subset_boxes(
        self,
        boxes: List[Dict]
    ) -> Tuple[List[Dict], int]:
        """
        Remove boxes that are subsets of larger boxes.
        
        Args:
            boxes: List of box dictionaries with 'bbox' and 'area'
            
        Returns:
            Tuple of (filtered_boxes, num_removed)
        """
        if len(boxes) <= 1:
            return boxes, 0
        
        # Sort by area (largest first)
        sorted_boxes = sorted(boxes, key=lambda x: x['area'], reverse=True)
        
        keep = []
        removed_count = 0
        
        for i, box in enumerate(sorted_boxes):
            is_subset_of_any = False
            
            # Check against all larger boxes
            for larger_box in sorted_boxes[:i]:
                # Small box must be actually smaller
                if box['area'] >= larger_box['area']:
                    continue
                
                overlap_percentage = self.compute_overlap_percentage(
                    box,
                    larger_box
                )
                
                # If 85%+ of small box is inside large box, it's a subset
                if overlap_percentage >= self.subset_threshold:
                    is_subset_of_any = True
                    removed_count += 1
                    break
            
            if not is_subset_of_any:
                keep.append(box)
        
        return keep, removed_count
    
    def _extract_annotations(
        self,
        mask_file: Path,
        image_id: int,
        start_ann_id: int,
        image_width: int,
        image_height: int
    ) -> list:
        """
        Extract annotations from binary mask with enhanced filtering.
        
        Args:
            mask_file: Path to mask file
            image_id: Image ID for COCO format
            start_ann_id: Starting annotation ID
            image_width: Image width for dynamic threshold
            image_height: Image height for dynamic threshold
            
        Returns:
            List of COCO format annotations
        """
        annotations = []
        
        try:
            # Read mask
            mask = cv2.imread(str(mask_file))
            if mask is None:
                return annotations
            
            # Calculate dynamic threshold based on image size
            if self.use_dynamic_threshold:
                image_area = image_width * image_height
                dynamic_threshold = image_area * self.min_area_percentage
                min_area = max(self.config.min_area_threshold, dynamic_threshold)
            else:
                min_area = self.config.min_area_threshold
            
            # Create binary mask (any non-zero pixel)
            binary_mask = np.any(mask > 0, axis=2).astype(np.uint8) * 255
            
            # Apply morphological opening to remove noise
            kernel = np.ones((3, 3), np.uint8)
            binary_mask = cv2.morphologyEx(
                binary_mask,
                cv2.MORPH_OPEN,
                kernel
            )
            
            # Find connected components
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                binary_mask,
                connectivity=8
            )
            
            # Extract bounding boxes for each component
            boxes = []
            for label in range(1, num_labels):
                x = stats[label, cv2.CC_STAT_LEFT]
                y = stats[label, cv2.CC_STAT_TOP]
                w = stats[label, cv2.CC_STAT_WIDTH]
                h = stats[label, cv2.CC_STAT_HEIGHT]
                area = stats[label, cv2.CC_STAT_AREA]
                
                if area > min_area:
                    boxes.append({
                        'bbox': [float(x), float(y), float(w), float(h)],
                        'area': float(area)
                    })
                else:
                    self.stats['small_boxes_filtered'] += 1
            
            # Remove subset boxes if enabled
            if self.remove_subset_boxes and len(boxes) > 1:
                boxes, removed = self.filter_subset_boxes(boxes)
                if removed > 0:
                    self.stats['subset_boxes_removed'] += removed
            
            # Convert to COCO annotations
            ann_id = start_ann_id
            for box_info in boxes:
                annotations.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": 1,  # Single class: polyp
                    "bbox": box_info['bbox'],
                    "area": box_info['area'],
                    "segmentation": [],
                    "iscrowd": 0
                })
                ann_id += 1
        
        except Exception as e:
            self.logger.error(f"Error extracting annotations from {mask_file}: {e}")
        
        return annotations
    
    def prepare(self) -> Dict[str, any]:
        """
        Prepare detection dataset in COCO format with enhanced filtering.
        
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
        
        # Add enhanced stats
        stats['enhanced'] = {
            'subset_boxes_removed': self.stats['subset_boxes_removed'],
            'small_boxes_filtered': self.stats['small_boxes_filtered']
        }
        
        # Log summary
        self.logger.info("=" * 60)
        self.logger.info("Enhanced Preparation Summary:")
        self.logger.info(f"  Small boxes filtered: {self.stats['small_boxes_filtered']}")
        self.logger.info(f"  Subset boxes removed: {self.stats['subset_boxes_removed']}")
        self.logger.info("=" * 60)
        
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
        """Process a single split (train/val) with enhanced filtering."""
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
                    # Enhanced annotation extraction with image dimensions
                    annotations = self._extract_annotations(
                        mask_file,
                        image_id,
                        ann_id,
                        width,
                        height
                    )
                    coco_data["annotations"].extend(annotations)
                    ann_id += len(annotations)
        
        # Save COCO JSON
        self.save_coco_json(coco_data, split)
        
        return {
            "images": len(coco_data["images"]),
            "annotations": len(coco_data["annotations"])
        }
