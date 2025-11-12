"""
Bounding Box Processor for Detection Annotations.

Handles:
1. Merging overlapping boxes from same colored masks
2. Filtering noise/artifacts while keeping small polyps
3. Removing boxes that are subsets of larger boxes
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path


class BBoxProcessor:
    """Process and refine bounding boxes from masks."""
    
    def __init__(
        self,
        min_area_threshold: int = 100,
        min_aspect_ratio: float = 0.1,
        max_aspect_ratio: float = 10.0,
        iou_merge_threshold: float = 0.3,
        subset_iou_threshold: float = 0.8
    ):
        """
        Initialize bbox processor.
        
        Args:
            min_area_threshold: Minimum area for valid boxes
            min_aspect_ratio: Minimum width/height ratio (filters very thin boxes)
            max_aspect_ratio: Maximum width/height ratio (filters very thin boxes)
            iou_merge_threshold: IoU threshold for merging boxes
            subset_iou_threshold: IoU threshold for detecting subset boxes
        """
        self.min_area_threshold = min_area_threshold
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.iou_merge_threshold = iou_merge_threshold
        self.subset_iou_threshold = subset_iou_threshold
    
    def compute_iou(self, box1: List[float], box2: List[float]) -> float:
        """
        Compute IoU between two boxes.
        
        Args:
            box1: [x, y, w, h]
            box2: [x, y, w, h]
            
        Returns:
            IoU value
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Convert to [x1, y1, x2, y2]
        box1_coords = [x1, y1, x1 + w1, y1 + h1]
        box2_coords = [x2, y2, x2 + w2, y2 + h2]
        
        # Compute intersection
        xi1 = max(box1_coords[0], box2_coords[0])
        yi1 = max(box1_coords[1], box2_coords[1])
        xi2 = min(box1_coords[2], box2_coords[2])
        yi2 = min(box1_coords[3], box2_coords[3])
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Compute union
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    def is_subset(self, small_box: List[float], large_box: List[float]) -> bool:
        """
        Check if small_box is a subset of large_box.
        
        Args:
            small_box: [x, y, w, h]
            large_box: [x, y, w, h]
            
        Returns:
            True if small_box is mostly contained in large_box
        """
        iou = self.compute_iou(small_box, large_box)
        
        # If IoU is high, check if small box is actually smaller
        if iou > self.subset_iou_threshold:
            small_area = small_box[2] * small_box[3]
            large_area = large_box[2] * large_box[3]
            return small_area < large_area
        
        return False
    
    def merge_boxes(self, boxes: List[Dict]) -> List[Dict]:
        """
        Merge overlapping boxes.
        
        Args:
            boxes: List of box dictionaries with 'bbox' and 'area'
            
        Returns:
            List of merged boxes
        """
        if len(boxes) <= 1:
            return boxes
        
        # Sort by area (largest first)
        boxes = sorted(boxes, key=lambda x: x['area'], reverse=True)
        
        merged = []
        used = [False] * len(boxes)
        
        for i, box1 in enumerate(boxes):
            if used[i]:
                continue
            
            # Start with current box
            merge_group = [box1]
            used[i] = True
            
            # Find boxes to merge
            for j, box2 in enumerate(boxes[i+1:], start=i+1):
                if used[j]:
                    continue
                
                iou = self.compute_iou(box1['bbox'], box2['bbox'])
                if iou > self.iou_merge_threshold:
                    merge_group.append(box2)
                    used[j] = True
            
            # Merge the group
            if len(merge_group) > 1:
                merged_box = self._merge_box_group(merge_group)
                merged.append(merged_box)
            else:
                merged.append(box1)
        
        return merged
    
    def _merge_box_group(self, boxes: List[Dict]) -> Dict:
        """Merge a group of boxes into one."""
        # Find bounding box that contains all
        x_min = min(b['bbox'][0] for b in boxes)
        y_min = min(b['bbox'][1] for b in boxes)
        x_max = max(b['bbox'][0] + b['bbox'][2] for b in boxes)
        y_max = max(b['bbox'][1] + b['bbox'][3] for b in boxes)
        
        w = x_max - x_min
        h = y_max - y_min
        area = w * h
        
        return {
            'bbox': [float(x_min), float(y_min), float(w), float(h)],
            'area': float(area)
        }
    
    def remove_subsets(self, boxes: List[Dict]) -> List[Dict]:
        """
        Remove boxes that are subsets of larger boxes.
        
        Args:
            boxes: List of box dictionaries
            
        Returns:
            Filtered list without subsets
        """
        if len(boxes) <= 1:
            return boxes
        
        # Sort by area (largest first)
        boxes = sorted(boxes, key=lambda x: x['area'], reverse=True)
        
        keep = []
        
        for i, box in enumerate(boxes):
            is_subset_of_any = False
            
            # Check against all larger boxes
            for larger_box in boxes[:i]:
                if self.is_subset(box['bbox'], larger_box['bbox']):
                    is_subset_of_any = True
                    break
            
            if not is_subset_of_any:
                keep.append(box)
        
        return keep
    
    def filter_invalid_boxes(self, boxes: List[Dict]) -> List[Dict]:
        """
        Filter out invalid boxes (too small, bad aspect ratio).
        
        Args:
            boxes: List of box dictionaries
            
        Returns:
            Filtered list
        """
        valid = []
        
        for box in boxes:
            x, y, w, h = box['bbox']
            area = box['area']
            
            # Check area
            if area < self.min_area_threshold:
                continue
            
            # Check aspect ratio
            if w == 0 or h == 0:
                continue
            
            aspect_ratio = w / h
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                continue
            
            valid.append(box)
        
        return valid
    
    def extract_boxes_by_color(
        self,
        mask_path: Path,
        image_id: int,
        start_ann_id: int
    ) -> List[Dict]:
        """
        Extract bounding boxes from colored mask, grouping by color.
        
        Args:
            mask_path: Path to mask file
            image_id: Image ID
            start_ann_id: Starting annotation ID
            
        Returns:
            List of annotation dictionaries
        """
        try:
            # Read mask
            mask = cv2.imread(str(mask_path))
            if mask is None:
                return []
            
            # Check if mask is binary (white) or colored
            is_binary = self._is_binary_mask(mask)
            
            if is_binary:
                # Process as single binary mask
                return self._extract_from_binary_mask(mask, image_id, start_ann_id)
            else:
                # Process by color groups
                return self._extract_from_colored_mask(mask, image_id, start_ann_id)
        
        except Exception as e:
            print(f"Error extracting boxes from {mask_path}: {e}")
            return []
    
    def _is_binary_mask(self, mask: np.ndarray) -> bool:
        """Check if mask is binary (white/black only)."""
        # Check if all non-zero pixels are white (255, 255, 255)
        non_zero_mask = np.any(mask > 0, axis=2)
        if not np.any(non_zero_mask):
            return True
        
        non_zero_pixels = mask[non_zero_mask]
        unique_colors = np.unique(non_zero_pixels.reshape(-1, 3), axis=0)
        
        # Binary if only one color (white)
        return len(unique_colors) == 1
    
    def _extract_from_binary_mask(
        self,
        mask: np.ndarray,
        image_id: int,
        start_ann_id: int
    ) -> List[Dict]:
        """Extract boxes from binary mask."""
        # Create binary mask
        binary_mask = np.any(mask > 0, axis=2).astype(np.uint8) * 255
        
        # Apply morphological opening to remove noise
        kernel = np.ones((3, 3), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        
        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary_mask,
            connectivity=8
        )
        
        # Extract boxes
        boxes = []
        for label in range(1, num_labels):
            x = stats[label, cv2.CC_STAT_LEFT]
            y = stats[label, cv2.CC_STAT_TOP]
            w = stats[label, cv2.CC_STAT_WIDTH]
            h = stats[label, cv2.CC_STAT_HEIGHT]
            area = stats[label, cv2.CC_STAT_AREA]
            
            boxes.append({
                'bbox': [float(x), float(y), float(w), float(h)],
                'area': float(area)
            })
        
        # Process boxes
        boxes = self.filter_invalid_boxes(boxes)
        boxes = self.merge_boxes(boxes)
        boxes = self.remove_subsets(boxes)
        
        # Convert to annotations
        annotations = []
        for i, box in enumerate(boxes):
            annotations.append({
                'id': start_ann_id + i,
                'image_id': image_id,
                'category_id': 1,
                'bbox': box['bbox'],
                'area': box['area'],
                'segmentation': [],
                'iscrowd': 0
            })
        
        return annotations
    
    def _extract_from_colored_mask(
        self,
        mask: np.ndarray,
        image_id: int,
        start_ann_id: int
    ) -> List[Dict]:
        """Extract boxes from colored mask, processing each color separately."""
        # Get unique colors
        mask_2d = mask.reshape(-1, 3)
        unique_colors = np.unique(mask_2d, axis=0)
        
        # Remove black (background)
        unique_colors = unique_colors[np.any(unique_colors > 0, axis=1)]
        
        all_boxes = []
        
        # Process each color separately
        for color in unique_colors:
            # Create mask for this color
            color_mask = np.all(mask == color, axis=2).astype(np.uint8) * 255
            
            # Apply morphological opening
            kernel = np.ones((3, 3), np.uint8)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
            
            # Find connected components
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                color_mask,
                connectivity=8
            )
            
            # Extract boxes for this color
            color_boxes = []
            for label in range(1, num_labels):
                x = stats[label, cv2.CC_STAT_LEFT]
                y = stats[label, cv2.CC_STAT_TOP]
                w = stats[label, cv2.CC_STAT_WIDTH]
                h = stats[label, cv2.CC_STAT_HEIGHT]
                area = stats[label, cv2.CC_STAT_AREA]
                
                color_boxes.append({
                    'bbox': [float(x), float(y), float(w), float(h)],
                    'area': float(area)
                })
            
            # Process boxes for this color
            color_boxes = self.filter_invalid_boxes(color_boxes)
            color_boxes = self.merge_boxes(color_boxes)
            
            all_boxes.extend(color_boxes)
        
        # Final processing across all colors
        all_boxes = self.remove_subsets(all_boxes)
        
        # Convert to annotations
        annotations = []
        for i, box in enumerate(all_boxes):
            annotations.append({
                'id': start_ann_id + i,
                'image_id': image_id,
                'category_id': 1,
                'bbox': box['bbox'],
                'area': box['area'],
                'segmentation': [],
                'iscrowd': 0
            })
        
        return annotations
