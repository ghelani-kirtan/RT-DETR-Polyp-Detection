#!/usr/bin/env python3
"""
Test script to visualize detection annotations before preparing COCO dataset.

This script:
1. Reads images and masks from detection_dataset
2. Extracts bounding boxes from masks (same logic as preparer)
3. Draws overlays on images
4. Saves to temp folder for verification
5. Uses ThreadPoolExecutor for parallel processing
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse
import json


class DetectionAnnotationVisualizer:
    """Visualize detection annotations for verification."""
    
    def __init__(
        self,
        images_dir: Path,
        masks_dir: Path,
        output_dir: Path,
        min_area_threshold: int = 100,
        min_area_percentage: float = 0.0005,
        use_dynamic_threshold: bool = True,
        remove_subsets: bool = False,
        subset_threshold: float = 0.85,
        max_workers: int = 8
    ):
        """
        Initialize visualizer.
        
        Args:
            images_dir: Directory containing images
            masks_dir: Directory containing masks
            output_dir: Directory to save visualizations
            min_area_threshold: Minimum area for bounding boxes (static, fallback)
            min_area_percentage: Minimum area as percentage of image (dynamic)
            use_dynamic_threshold: Use dynamic threshold based on image size
            remove_subsets: Remove boxes that are mostly inside larger boxes
            subset_threshold: Percentage of overlap to consider as subset (0.85 = 85%)
            max_workers: Number of parallel workers
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.output_dir = Path(output_dir)
        self.min_area_threshold = min_area_threshold
        self.min_area_percentage = min_area_percentage
        self.use_dynamic_threshold = use_dynamic_threshold
        self.remove_subsets = remove_subsets
        self.subset_threshold = subset_threshold
        self.max_workers = max_workers
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Stats
        self.stats = {
            'total_images': 0,
            'images_with_annotations': 0,
            'total_annotations': 0,
            'images_without_annotations': 0,
            'subset_boxes_removed': 0,
            'errors': 0
        }
    
    def compute_iou(self, box1: List[float], box2: List[float]) -> float:
        """
        Compute IoU (Intersection over Union) between two boxes.
        
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
        
        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height
        
        # Compute union
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    def is_subset(self, small_box: Dict, large_box: Dict) -> bool:
        """
        Check if small_box is mostly contained within large_box.
        
        Args:
            small_box: Smaller box dictionary with 'bbox' and 'area'
            large_box: Larger box dictionary with 'bbox' and 'area'
            
        Returns:
            True if small_box is mostly inside large_box
        """
        # Small box must be actually smaller
        if small_box['area'] >= large_box['area']:
            return False
        
        # Calculate intersection
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
        
        # Calculate what percentage of small box is inside large box
        overlap_percentage = inter_area / small_box['area']
        
        # If 85%+ of small box is inside large box, it's a subset
        return overlap_percentage >= self.subset_threshold
    
    def remove_subset_boxes(self, boxes: List[Dict]) -> Tuple[List[Dict], int]:
        """
        Remove boxes that are subsets of larger boxes.
        
        Args:
            boxes: List of box dictionaries
            
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
            
            # Check against all larger boxes (those before current in sorted list)
            for larger_box in sorted_boxes[:i]:
                if self.is_subset(box, larger_box):
                    is_subset_of_any = True
                    removed_count += 1
                    break
            
            if not is_subset_of_any:
                keep.append(box)
        
        return keep, removed_count
    
    def extract_bboxes_from_mask(self, mask_path: Path, image_width: int, image_height: int) -> List[Dict]:
        """
        Extract bounding boxes from mask with dynamic thresholding.
        Simple and fast - treats all masks as binary.
        
        Note: For colored masks, subset removal only happens within connected components,
        not across different colors. This is handled by connected components algorithm.
        
        Args:
            mask_path: Path to mask file
            image_width: Image width for dynamic threshold
            image_height: Image height for dynamic threshold
            
        Returns:
            List of bbox dictionaries with [x, y, w, h, area]
        """
        bboxes = []
        
        try:
            # Read mask
            mask = cv2.imread(str(mask_path))
            if mask is None:
                return bboxes
            
            # Calculate dynamic threshold based on image size
            if self.use_dynamic_threshold:
                image_area = image_width * image_height
                dynamic_threshold = image_area * self.min_area_percentage
                min_area = max(self.min_area_threshold, dynamic_threshold)
            else:
                min_area = self.min_area_threshold
            
            # Create binary mask (any non-zero pixel)
            binary_mask = np.any(mask > 0, axis=2).astype(np.uint8) * 255
            
            # Apply morphological opening to remove noise
            kernel = np.ones((3, 3), np.uint8)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
            
            # Find connected components
            # This naturally separates different colored regions if they're not touching
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                binary_mask,
                connectivity=8
            )
            
            # Extract bounding boxes for each component
            for label in range(1, num_labels):
                x = stats[label, cv2.CC_STAT_LEFT]
                y = stats[label, cv2.CC_STAT_TOP]
                w = stats[label, cv2.CC_STAT_WIDTH]
                h = stats[label, cv2.CC_STAT_HEIGHT]
                area = stats[label, cv2.CC_STAT_AREA]
                
                if area > min_area:
                    bboxes.append({
                        'bbox': [int(x), int(y), int(w), int(h)],
                        'area': int(area),
                        'threshold_used': int(min_area),
                        'area_percentage': (area / (image_width * image_height)) * 100
                    })
            
            # Remove subset boxes if enabled
            # Note: This works well for binary masks and touching colored regions
            # For separate colored regions, connected components already separates them
            if self.remove_subsets and len(bboxes) > 1:
                bboxes, removed = self.remove_subset_boxes(bboxes)
                if removed > 0:
                    self.stats['subset_boxes_removed'] += removed
        
        except Exception as e:
            print(f"Error extracting bboxes from {mask_path}: {e}")
        
        return bboxes
    
    def draw_annotations(
        self,
        image: np.ndarray,
        bboxes: List[Dict],
        show_area: bool = True
    ) -> np.ndarray:
        """
        Draw bounding boxes on image.
        
        Args:
            image: Input image
            bboxes: List of bbox dictionaries
            show_area: Whether to show area text
            
        Returns:
            Image with drawn annotations
        """
        overlay = image.copy()
        
        for i, bbox_info in enumerate(bboxes, 1):
            x, y, w, h = bbox_info['bbox']
            area = bbox_info['area']
            area_pct = bbox_info.get('area_percentage', 0)
            
            # Draw bounding box
            color = (0, 255, 0)  # Green
            thickness = 2
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, thickness)
            
            # Draw label with area percentage
            label = f"#{i}"
            if show_area:
                label += f" ({area}px, {area_pct:.2f}%)"
            
            # Background for text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
            
            cv2.rectangle(
                overlay,
                (x, y - text_h - 5),
                (x + text_w, y),
                color,
                -1
            )
            
            # Draw text
            cv2.putText(
                overlay,
                label,
                (x, y - 5),
                font,
                font_scale,
                (0, 0, 0),
                font_thickness
            )
        
        return overlay
    
    def process_single_image(self, image_stem: str) -> Dict:
        """
        Process a single image and create visualization.
        
        Args:
            image_stem: Image file stem (without extension)
            
        Returns:
            Processing result dictionary
        """
        result = {
            'stem': image_stem,
            'success': False,
            'num_annotations': 0,
            'error': None
        }
        
        try:
            # Find image file
            image_path = None
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
                candidate = self.images_dir / f"{image_stem}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break
            
            if not image_path:
                result['error'] = "Image not found"
                return result
            
            # Find mask file
            mask_path = None
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
                candidate = self.masks_dir / f"{image_stem}{ext}"
                if candidate.exists():
                    mask_path = candidate
                    break
            
            if not mask_path:
                result['error'] = "Mask not found"
                return result
            
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                result['error'] = "Failed to read image"
                return result
            
            image_height, image_width = image.shape[:2]
            
            # Extract bounding boxes with dynamic threshold
            bboxes = self.extract_bboxes_from_mask(mask_path, image_width, image_height)
            result['num_annotations'] = len(bboxes)
            
            # Draw annotations
            annotated = self.draw_annotations(image, bboxes)
            
            # Add info text
            info_text = f"Image: {image_path.name} | Annotations: {len(bboxes)}"
            cv2.putText(
                annotated,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            cv2.putText(
                annotated,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                1
            )
            
            # Save visualization
            output_path = self.output_dir / f"{image_stem}_annotated.jpg"
            cv2.imwrite(str(output_path), annotated)
            
            result['success'] = True
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def visualize_all(self) -> Dict:
        """
        Visualize all images with annotations using parallel processing.
        
        Returns:
            Statistics dictionary
        """
        print(f"🔍 Scanning images in {self.images_dir}")
        
        # Get all image stems
        image_stems = set()
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
            for img_path in self.images_dir.glob(f"*{ext}"):
                image_stems.add(img_path.stem)
        
        image_stems = sorted(image_stems)
        self.stats['total_images'] = len(image_stems)
        
        print(f"📊 Found {len(image_stems)} images")
        print(f"🚀 Processing with {self.max_workers} workers...")
        
        # Process in parallel
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_stem = {
                executor.submit(self.process_single_image, stem): stem
                for stem in image_stems
            }
            
            # Process results with progress bar
            with tqdm(total=len(image_stems), desc="Visualizing") as pbar:
                for future in as_completed(future_to_stem):
                    result = future.result()
                    results.append(result)
                    
                    if result['success']:
                        if result['num_annotations'] > 0:
                            self.stats['images_with_annotations'] += 1
                            self.stats['total_annotations'] += result['num_annotations']
                        else:
                            self.stats['images_without_annotations'] += 1
                    else:
                        self.stats['errors'] += 1
                    
                    pbar.update(1)
        
        # Save detailed report
        self._save_report(results)
        
        return self.stats
    
    def _save_report(self, results: List[Dict]):
        """Save detailed report to JSON."""
        report = {
            'stats': self.stats,
            'images': []
        }
        
        for result in sorted(results, key=lambda x: x['stem']):
            report['images'].append({
                'stem': result['stem'],
                'success': result['success'],
                'num_annotations': result['num_annotations'],
                'error': result['error']
            })
        
        report_path = self.output_dir / 'annotation_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Detailed report saved to {report_path}")
    
    def print_summary(self):
        """Print summary statistics."""
        print("\n" + "=" * 60)
        print("📊 ANNOTATION VISUALIZATION SUMMARY")
        print("=" * 60)
        print(f"Total images:                {self.stats['total_images']}")
        print(f"Images with annotations:     {self.stats['images_with_annotations']}")
        print(f"Images without annotations:  {self.stats['images_without_annotations']}")
        print(f"Total annotations:           {self.stats['total_annotations']}")
        
        if self.remove_subsets:
            print(f"Subset boxes removed:        {self.stats['subset_boxes_removed']}")
        
        print(f"Errors:                      {self.stats['errors']}")
        
        if self.stats['images_with_annotations'] > 0:
            avg_annotations = self.stats['total_annotations'] / self.stats['images_with_annotations']
            print(f"Avg annotations per image:   {avg_annotations:.2f}")
        
        print("=" * 60)
        print(f"\n✅ Visualizations saved to: {self.output_dir}")
        print(f"📄 Report saved to: {self.output_dir / 'annotation_report.json'}")


def main():
    """Main function."""
    import sys
    
    # Fix Windows console encoding
    if sys.platform == 'win32':
        import io
        if hasattr(sys.stdout, 'buffer'):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'buffer'):
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    
    parser = argparse.ArgumentParser(
        description="Visualize detection annotations before preparing COCO dataset"
    )
    
    parser.add_argument(
        "--images-dir",
        required=True,
        help="Directory containing images"
    )
    parser.add_argument(
        "--masks-dir",
        required=True,
        help="Directory containing masks"
    )
    parser.add_argument(
        "--output-dir",
        default="temp_annotation_visualization",
        help="Output directory for visualizations (default: temp_annotation_visualization)"
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=100,
        help="Minimum area threshold in pixels (static fallback, default: 100)"
    )
    parser.add_argument(
        "--min-area-pct",
        type=float,
        default=0.0005,
        help="Minimum area as percentage of image (dynamic, default: 0.0005 = 0.05%%)"
    )
    parser.add_argument(
        "--use-static",
        action="store_true",
        help="Use static threshold instead of dynamic (not recommended)"
    )
    parser.add_argument(
        "--remove-subsets",
        action="store_true",
        help="Remove boxes that are mostly inside larger boxes"
    )
    parser.add_argument(
        "--subset-threshold",
        type=float,
        default=0.85,
        help="Overlap threshold for subset detection (default: 0.85 = 85%%)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)"
    )
    
    args = parser.parse_args()
    
    # Validate directories
    images_dir = Path(args.images_dir)
    masks_dir = Path(args.masks_dir)
    
    if not images_dir.exists():
        print(f"❌ Error: Images directory not found: {images_dir}")
        return
    
    if not masks_dir.exists():
        print(f"❌ Error: Masks directory not found: {masks_dir}")
        return
    
    # Initialize visualizer
    print("🎨 Detection Annotation Visualizer")
    print("=" * 60)
    print(f"Images dir:       {images_dir}")
    print(f"Masks dir:        {masks_dir}")
    print(f"Output dir:       {args.output_dir}")
    print(f"Min area (px):    {args.min_area}")
    print(f"Min area (%):     {args.min_area_pct * 100:.3f}%")
    print(f"Threshold mode:   {'Static' if args.use_static else 'Dynamic'}")
    print(f"Remove subsets:   {'Yes' if args.remove_subsets else 'No'}")
    if args.remove_subsets:
        print(f"Subset threshold: {args.subset_threshold * 100:.0f}%")
    print(f"Workers:          {args.workers}")
    print("=" * 60)
    
    visualizer = DetectionAnnotationVisualizer(
        images_dir=images_dir,
        masks_dir=masks_dir,
        output_dir=Path(args.output_dir),
        min_area_threshold=args.min_area,
        min_area_percentage=args.min_area_pct,
        use_dynamic_threshold=not args.use_static,
        remove_subsets=args.remove_subsets,
        subset_threshold=args.subset_threshold,
        max_workers=args.workers
    )
    
    # Visualize all
    stats = visualizer.visualize_all()
    
    # Print summary
    visualizer.print_summary()
    
    # Warnings
    if stats['images_without_annotations'] > 0:
        print(f"\n⚠️  Warning: {stats['images_without_annotations']} images have no annotations")
        print("   These might be negative samples or have issues with masks")
    
    if stats['errors'] > 0:
        print(f"\n⚠️  Warning: {stats['errors']} images had errors")
        print("   Check annotation_report.json for details")
    
    print("\n💡 Next steps:")
    print("   1. Review visualizations in temp_annotation_visualization/")
    print("   2. Check annotation_report.json for details")
    print("   3. If annotations look good, proceed with COCO preparation")
    print("   4. If issues found, fix masks and re-run this script")


if __name__ == "__main__":
    main()
