"""
Duplicate Image Remover with Quality-based Selection.

This module identifies and removes duplicate images while keeping the best quality version.
Selection criteria:
1. Non-blurred images preferred over blurred
2. Images with proper masks preferred over those without
3. Higher resolution preferred
"""

import hashlib
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from collections import defaultdict


class DuplicateRemover:
    """Remove duplicate images intelligently based on quality metrics."""
    
    def __init__(
        self,
        images_dir: Path,
        masks_dir: Optional[Path] = None,
        dry_run: bool = True,
        blur_threshold: float = 100.0
    ):
        """
        Initialize duplicate remover.
        
        Args:
            images_dir: Directory containing images
            masks_dir: Directory containing masks (optional)
            dry_run: If True, don't actually delete files
            blur_threshold: Laplacian variance threshold for blur detection (lower = more blurred)
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir) if masks_dir else None
        self.dry_run = dry_run
        self.blur_threshold = blur_threshold
        
        self.duplicates = defaultdict(list)
        self.stats = {
            'total_images': 0,
            'unique_images': 0,
            'duplicate_groups': 0,
            'images_to_remove': 0,
            'images_kept': 0
        }
    
    def compute_image_hash(self, image_path: Path) -> str:
        """
        Compute MD5 hash of image file.
        
        Args:
            image_path: Path to image
            
        Returns:
            MD5 hash string
        """
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def compute_perceptual_hash(self, image_path: Path, hash_size: int = 8) -> str:
        """
        Compute perceptual hash (pHash) for near-duplicate detection.
        
        Args:
            image_path: Path to image
            hash_size: Size of hash (default 8x8)
            
        Returns:
            Perceptual hash string
        """
        try:
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                return ""
            
            # Resize to hash_size x hash_size
            img = cv2.resize(img, (hash_size, hash_size))
            
            # Compute DCT
            dct = cv2.dct(np.float32(img))
            
            # Keep only top-left 8x8
            dct_low = dct[:hash_size, :hash_size]
            
            # Compute median
            median = np.median(dct_low)
            
            # Create hash
            hash_str = ''.join(['1' if x > median else '0' for x in dct_low.flatten()])
            return hash_str
        except Exception as e:
            print(f"Error computing perceptual hash for {image_path}: {e}")
            return ""
    
    def detect_blur(self, image_path: Path) -> float:
        """
        Detect blur using Laplacian variance.
        
        Args:
            image_path: Path to image
            
        Returns:
            Blur score (higher = sharper, lower = more blurred)
        """
        try:
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                return 0.0
            
            # Compute Laplacian variance
            laplacian = cv2.Laplacian(img, cv2.CV_64F)
            variance = laplacian.var()
            return variance
        except Exception as e:
            print(f"Error detecting blur for {image_path}: {e}")
            return 0.0
    
    def check_mask_quality(self, image_path: Path) -> Tuple[bool, float]:
        """
        Check if image has a proper mask and its quality.
        
        Args:
            image_path: Path to image
            
        Returns:
            Tuple of (has_mask, mask_coverage_percentage)
        """
        if not self.masks_dir:
            return False, 0.0
        
        # Find corresponding mask
        mask_path = self.masks_dir / image_path.name
        
        if not mask_path.exists():
            return False, 0.0
        
        try:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                return False, 0.0
            
            # Calculate mask coverage (percentage of non-zero pixels)
            total_pixels = mask.shape[0] * mask.shape[1]
            non_zero_pixels = np.count_nonzero(mask)
            coverage = (non_zero_pixels / total_pixels) * 100
            
            # Consider mask valid if it has at least 0.1% coverage
            has_valid_mask = coverage > 0.1
            
            return has_valid_mask, coverage
        except Exception as e:
            print(f"Error checking mask for {image_path}: {e}")
            return False, 0.0
    
    def get_image_resolution(self, image_path: Path) -> int:
        """
        Get image resolution (width * height).
        
        Args:
            image_path: Path to image
            
        Returns:
            Total pixels (width * height)
        """
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return 0
            return img.shape[0] * img.shape[1]
        except Exception as e:
            print(f"Error getting resolution for {image_path}: {e}")
            return 0
    
    def score_image_quality(self, image_path: Path) -> Dict:
        """
        Score image quality based on multiple factors.
        
        Args:
            image_path: Path to image
            
        Returns:
            Dictionary with quality metrics
        """
        blur_score = self.detect_blur(image_path)
        has_mask, mask_coverage = self.check_mask_quality(image_path)
        resolution = self.get_image_resolution(image_path)
        
        # Calculate overall quality score
        quality_score = 0.0
        
        # Blur score (40% weight)
        if blur_score >= self.blur_threshold:
            quality_score += 40.0
        else:
            quality_score += (blur_score / self.blur_threshold) * 40.0
        
        # Mask quality (40% weight)
        if has_mask:
            quality_score += 40.0
        
        # Resolution (20% weight)
        # Normalize to 0-20 range (assuming max 4K resolution)
        max_resolution = 3840 * 2160
        quality_score += min((resolution / max_resolution) * 20.0, 20.0)
        
        return {
            'path': image_path,
            'blur_score': blur_score,
            'is_sharp': blur_score >= self.blur_threshold,
            'has_mask': has_mask,
            'mask_coverage': mask_coverage,
            'resolution': resolution,
            'quality_score': quality_score
        }
    
    def find_duplicates(self, use_perceptual: bool = False) -> Dict:
        """
        Find duplicate images in the directory.
        
        Args:
            use_perceptual: If True, use perceptual hashing for near-duplicates
            
        Returns:
            Dictionary mapping hash to list of image paths
        """
        print(f"🔍 Scanning for duplicates in {self.images_dir}")
        
        hash_to_images = defaultdict(list)
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [
            f for f in self.images_dir.rglob('*')
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        
        self.stats['total_images'] = len(image_files)
        print(f"📊 Found {len(image_files)} images")
        
        # Compute hashes
        for i, img_path in enumerate(image_files, 1):
            if i % 100 == 0:
                print(f"   Processing: {i}/{len(image_files)}")
            
            if use_perceptual:
                img_hash = self.compute_perceptual_hash(img_path)
            else:
                img_hash = self.compute_image_hash(img_path)
            
            if img_hash:
                hash_to_images[img_hash].append(img_path)
        
        # Filter to only duplicates
        self.duplicates = {
            h: paths for h, paths in hash_to_images.items()
            if len(paths) > 1
        }
        
        self.stats['duplicate_groups'] = len(self.duplicates)
        self.stats['unique_images'] = len(hash_to_images) - self.stats['duplicate_groups']
        
        print(f"✓ Found {self.stats['duplicate_groups']} groups of duplicates")
        
        return self.duplicates
    
    def select_best_image(self, duplicate_group: List[Path]) -> Tuple[Path, List[Path]]:
        """
        Select the best image from a group of duplicates.
        
        Args:
            duplicate_group: List of duplicate image paths
            
        Returns:
            Tuple of (best_image_path, images_to_remove)
        """
        # Score all images
        scored_images = []
        for img_path in duplicate_group:
            quality = self.score_image_quality(img_path)
            scored_images.append(quality)
        
        # Sort by quality score (highest first)
        scored_images.sort(key=lambda x: x['quality_score'], reverse=True)
        
        best_image = scored_images[0]['path']
        images_to_remove = [img['path'] for img in scored_images[1:]]
        
        return best_image, images_to_remove
    
    def generate_report(self, output_path: Path = None) -> Dict:
        """
        Generate detailed report of duplicates and decisions.
        
        Args:
            output_path: Path to save JSON report
            
        Returns:
            Report dictionary
        """
        report = {
            'stats': self.stats,
            'duplicate_groups': []
        }
        
        for img_hash, duplicate_group in self.duplicates.items():
            best_image, to_remove = self.select_best_image(duplicate_group)
            
            # Get quality metrics for all images
            group_info = {
                'hash': img_hash,
                'count': len(duplicate_group),
                'kept': str(best_image),
                'removed': [str(p) for p in to_remove],
                'details': []
            }
            
            for img_path in duplicate_group:
                quality = self.score_image_quality(img_path)
                group_info['details'].append({
                    'path': str(quality['path']),
                    'quality_score': round(quality['quality_score'], 2),
                    'blur_score': round(quality['blur_score'], 2),
                    'is_sharp': quality['is_sharp'],
                    'has_mask': quality['has_mask'],
                    'mask_coverage': round(quality['mask_coverage'], 2),
                    'resolution': quality['resolution'],
                    'action': 'KEEP' if quality['path'] == best_image else 'REMOVE'
                })
            
            report['duplicate_groups'].append(group_info)
        
        # Save report
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"📄 Report saved to {output_path}")
        
        return report
    
    def remove_duplicates(self) -> Dict:
        """
        Remove duplicate images, keeping the best quality version.
        
        Returns:
            Statistics dictionary
        """
        if not self.duplicates:
            print("⚠️  No duplicates found. Run find_duplicates() first.")
            return self.stats
        
        images_removed = 0
        images_kept = 0
        
        print(f"\n{'🔍 DRY RUN MODE' if self.dry_run else '🗑️  REMOVING DUPLICATES'}")
        print("=" * 60)
        
        for img_hash, duplicate_group in self.duplicates.items():
            best_image, to_remove = self.select_best_image(duplicate_group)
            
            print(f"\n📦 Duplicate group ({len(duplicate_group)} images):")
            print(f"   ✓ KEEPING: {best_image.name}")
            
            for img_path in to_remove:
                print(f"   ✗ REMOVING: {img_path.name}")
                
                if not self.dry_run:
                    try:
                        # Remove image
                        img_path.unlink()
                        
                        # Remove corresponding mask if exists
                        if self.masks_dir:
                            mask_path = self.masks_dir / img_path.name
                            if mask_path.exists():
                                mask_path.unlink()
                        
                        images_removed += 1
                    except Exception as e:
                        print(f"      ⚠️  Error removing {img_path}: {e}")
                else:
                    images_removed += 1
            
            images_kept += 1
        
        self.stats['images_removed'] = images_removed
        self.stats['images_kept'] = images_kept
        
        print("\n" + "=" * 60)
        print(f"📊 Summary:")
        print(f"   Total images: {self.stats['total_images']}")
        print(f"   Duplicate groups: {self.stats['duplicate_groups']}")
        print(f"   Images kept: {images_kept}")
        print(f"   Images removed: {images_removed}")
        
        if self.dry_run:
            print(f"\n⚠️  DRY RUN - No files were actually deleted")
        else:
            print(f"\n✅ Duplicates removed successfully")
        
        return self.stats


def main():
    """Example usage and testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Remove duplicate images intelligently")
    parser.add_argument("--images-dir", required=True, help="Directory containing images")
    parser.add_argument("--masks-dir", help="Directory containing masks (optional)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without deleting")
    parser.add_argument("--report", help="Path to save JSON report")
    parser.add_argument("--blur-threshold", type=float, default=100.0, 
                       help="Blur detection threshold (default: 100.0)")
    parser.add_argument("--perceptual", action="store_true",
                       help="Use perceptual hashing for near-duplicates")
    
    args = parser.parse_args()
    
    # Initialize remover
    remover = DuplicateRemover(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        dry_run=args.dry_run,
        blur_threshold=args.blur_threshold
    )
    
    # Find duplicates
    remover.find_duplicates(use_perceptual=args.perceptual)
    
    # Generate report
    if args.report:
        remover.generate_report(output_path=args.report)
    
    # Remove duplicates
    if remover.duplicates:
        remover.remove_duplicates()
    else:
        print("✓ No duplicates found!")


if __name__ == "__main__":
    main()
