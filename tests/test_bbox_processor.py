#!/usr/bin/env python3
"""
Test script to compare old vs new bbox extraction logic.

This script:
1. Processes masks with both old and new logic
2. Shows side-by-side comparison
3. Highlights differences (merged boxes, removed subsets, etc.)
4. Saves comparison visualizations
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse
import json

from data_pipelines.preparers.bbox_processor import BBoxProcessor


class BBoxComparisonTester:
    """Compare old vs new bbox extraction logic."""
    
    def __init__(
        self,
        images_dir: Path,
        masks_dir: Path,
        output_dir: Path,
        min_area: int = 100,
        max_workers: int = 8
    ):
        """Initialize tester."""
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.output_dir = Path(output_dir)
        self.min_area = min_area
        self.max_workers = max_workers
        
        # Create output directories
        (self.output_dir / "old_logic").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "new_logic").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "comparison").mkdir(parents=True, exist_ok=True)
        
        # Initialize new processor
        self.processor = BBoxProcessor(
            min_area_threshold=min_area,
            min_aspect_ratio=0.1,
            max_aspect_ratio=10.0,
            iou_merge_threshold=0.3,
            subset_iou_threshold=0.8
        )
        
        self.stats = {
            'total_images': 0,
            'old_total_boxes': 0,
            'new_total_boxes': 0,
            'boxes_merged': 0,
            'boxes_removed': 0,
            'boxes_kept': 0
        }
    
    def extract_boxes_old_logic(self, mask_path: Path) -> List[Dict]:
        """Extract boxes using old logic (from detection_preparer.py)."""
        try:
            mask = cv2.imread(str(mask_path))
            if mask is None:
                return []
            
            # Create binary mask
            binary_mask = np.any(mask > 0, axis=2).astype(np.uint8) * 255
            
            # Apply morphological opening
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
                
                if area > self.min_area:
                    boxes.append({
                        'bbox': [float(x), float(y), float(w), float(h)],
                        'area': float(area)
                    })
            
            return boxes
        
        except Exception as e:
            print(f"Error with old logic on {mask_path}: {e}")
            return []
    
    def draw_boxes(self, image: np.ndarray, boxes: List[Dict], color: tuple) -> np.ndarray:
        """Draw boxes on image."""
        overlay = image.copy()
        
        for i, box in enumerate(boxes, 1):
            x, y, w, h = [int(v) for v in box['bbox']]
            area = int(box['area'])
            
            # Draw box
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label = f"#{i} ({area})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
            
            cv2.rectangle(overlay, (x, y - text_h - 5), (x + text_w, y), color, -1)
            cv2.putText(overlay, label, (x, y - 5), font, font_scale, (255, 255, 255), thickness)
        
        return overlay
    
    def create_comparison(
        self,
        image: np.ndarray,
        old_boxes: List[Dict],
        new_boxes: List[Dict]
    ) -> np.ndarray:
        """Create side-by-side comparison."""
        h, w = image.shape[:2]
        
        # Create canvas
        canvas = np.zeros((h, w * 2 + 20, 3), dtype=np.uint8)
        
        # Draw old logic (left, red)
        old_vis = self.draw_boxes(image, old_boxes, (0, 0, 255))
        canvas[:, :w] = old_vis
        
        # Draw new logic (right, green)
        new_vis = self.draw_boxes(image, new_boxes, (0, 255, 0))
        canvas[:, w+20:] = new_vis
        
        # Add labels
        cv2.putText(canvas, f"OLD: {len(old_boxes)} boxes", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(canvas, f"NEW: {len(new_boxes)} boxes", (w + 30, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Add difference
        diff = len(old_boxes) - len(new_boxes)
        diff_text = f"Difference: {diff:+d} boxes"
        diff_color = (0, 255, 255) if diff > 0 else (255, 255, 255)
        cv2.putText(canvas, diff_text, (w // 2 - 100, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, diff_color, 2)
        
        return canvas
    
    def process_single_image(self, image_stem: str) -> Dict:
        """Process single image with both logics."""
        result = {
            'stem': image_stem,
            'success': False,
            'old_boxes': 0,
            'new_boxes': 0,
            'difference': 0
        }
        
        try:
            # Find image
            image_path = None
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
                candidate = self.images_dir / f"{image_stem}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break
            
            if not image_path:
                return result
            
            # Find mask
            mask_path = None
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
                candidate = self.masks_dir / f"{image_stem}{ext}"
                if candidate.exists():
                    mask_path = candidate
                    break
            
            if not mask_path:
                return result
            
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                return result
            
            # Extract with old logic
            old_boxes = self.extract_boxes_old_logic(mask_path)
            
            # Extract with new logic
            new_annotations = self.processor.extract_boxes_by_color(mask_path, 1, 1)
            new_boxes = [{'bbox': ann['bbox'], 'area': ann['area']} for ann in new_annotations]
            
            result['old_boxes'] = len(old_boxes)
            result['new_boxes'] = len(new_boxes)
            result['difference'] = len(old_boxes) - len(new_boxes)
            result['success'] = True
            
            # Save visualizations only if there's a difference
            if result['difference'] != 0:
                # Old logic visualization
                old_vis = self.draw_boxes(image, old_boxes, (0, 0, 255))
                cv2.imwrite(
                    str(self.output_dir / "old_logic" / f"{image_stem}.jpg"),
                    old_vis
                )
                
                # New logic visualization
                new_vis = self.draw_boxes(image, new_boxes, (0, 255, 0))
                cv2.imwrite(
                    str(self.output_dir / "new_logic" / f"{image_stem}.jpg"),
                    new_vis
                )
                
                # Comparison
                comparison = self.create_comparison(image, old_boxes, new_boxes)
                cv2.imwrite(
                    str(self.output_dir / "comparison" / f"{image_stem}_comparison.jpg"),
                    comparison
                )
        
        except Exception as e:
            print(f"Error processing {image_stem}: {e}")
        
        return result
    
    def test_all(self) -> Dict:
        """Test all images."""
        print(f"🔍 Scanning images in {self.images_dir}")
        
        # Get image stems
        image_stems = set()
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
            for img_path in self.images_dir.glob(f"*{ext}"):
                image_stems.add(img_path.stem)
        
        image_stems = sorted(image_stems)
        self.stats['total_images'] = len(image_stems)
        
        print(f"📊 Found {len(image_stems)} images")
        print(f"🚀 Processing with {self.max_workers} workers...")
        
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_stem = {
                executor.submit(self.process_single_image, stem): stem
                for stem in image_stems
            }
            
            with tqdm(total=len(image_stems), desc="Comparing") as pbar:
                for future in as_completed(future_to_stem):
                    result = future.result()
                    results.append(result)
                    
                    if result['success']:
                        self.stats['old_total_boxes'] += result['old_boxes']
                        self.stats['new_total_boxes'] += result['new_boxes']
                        
                        if result['difference'] > 0:
                            self.stats['boxes_removed'] += result['difference']
                        elif result['difference'] < 0:
                            self.stats['boxes_kept'] += abs(result['difference'])
                    
                    pbar.update(1)
        
        # Calculate merged
        self.stats['boxes_merged'] = self.stats['old_total_boxes'] - self.stats['new_total_boxes']
        
        # Save report
        self._save_report(results)
        
        return self.stats
    
    def _save_report(self, results: List[Dict]):
        """Save comparison report."""
        report = {
            'stats': self.stats,
            'images': []
        }
        
        for result in sorted(results, key=lambda x: abs(x.get('difference', 0)), reverse=True):
            if result['success'] and result['difference'] != 0:
                report['images'].append({
                    'stem': result['stem'],
                    'old_boxes': result['old_boxes'],
                    'new_boxes': result['new_boxes'],
                    'difference': result['difference']
                })
        
        report_path = self.output_dir / 'comparison_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Report saved to {report_path}")
    
    def print_summary(self):
        """Print summary."""
        print("\n" + "=" * 60)
        print("📊 BBOX LOGIC COMPARISON SUMMARY")
        print("=" * 60)
        print(f"Total images:           {self.stats['total_images']}")
        print(f"Old logic total boxes:  {self.stats['old_total_boxes']}")
        print(f"New logic total boxes:  {self.stats['new_total_boxes']}")
        print(f"Boxes removed/merged:   {self.stats['boxes_merged']}")
        
        if self.stats['old_total_boxes'] > 0:
            reduction = (self.stats['boxes_merged'] / self.stats['old_total_boxes']) * 100
            print(f"Reduction:              {reduction:.1f}%")
        
        print("=" * 60)
        print(f"\n✅ Comparisons saved to: {self.output_dir}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Compare old vs new bbox extraction logic"
    )
    
    parser.add_argument("--images-dir", required=True, help="Images directory")
    parser.add_argument("--masks-dir", required=True, help="Masks directory")
    parser.add_argument("--output-dir", default="bbox_comparison", help="Output directory")
    parser.add_argument("--min-area", type=int, default=100, help="Min area threshold")
    parser.add_argument("--workers", type=int, default=8, help="Number of workers")
    
    args = parser.parse_args()
    
    print("🔬 BBox Logic Comparison Tester")
    print("=" * 60)
    print(f"Images dir:  {args.images_dir}")
    print(f"Masks dir:   {args.masks_dir}")
    print(f"Output dir:  {args.output_dir}")
    print(f"Min area:    {args.min_area}")
    print(f"Workers:     {args.workers}")
    print("=" * 60)
    
    tester = BBoxComparisonTester(
        images_dir=Path(args.images_dir),
        masks_dir=Path(args.masks_dir),
        output_dir=Path(args.output_dir),
        min_area=args.min_area,
        max_workers=args.workers
    )
    
    stats = tester.test_all()
    tester.print_summary()
    
    print("\n💡 Next steps:")
    print("   1. Review comparison images in bbox_comparison/comparison/")
    print("   2. Check comparison_report.json for detailed differences")
    print("   3. If new logic looks good, integrate into detection_preparer.py")


if __name__ == "__main__":
    main()
