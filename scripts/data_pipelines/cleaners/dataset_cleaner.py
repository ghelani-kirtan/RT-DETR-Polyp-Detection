"""Generic dataset cleaner."""

from pathlib import Path
from typing import Dict
from tqdm import tqdm
from .base_cleaner import BaseCleaner
from ..core import CleanerConfig, FileUtils


class DatasetCleaner(BaseCleaner):
    """Clean dataset by removing unmatched and corrupted files."""
    
    def clean(self) -> Dict[str, any]:
        """
        Clean dataset.
        
        Returns:
            Cleaning statistics
        """
        stats = {
            "removed_masks": 0,
            "removed_images": 0,
            "removed_corrupted": 0
        }
        
        images_dir = self.config.input_dir / "images"
        masks_dir = self.config.input_dir / "masks"
        
        if not images_dir.exists() or not masks_dir.exists():
            self.logger.error("Images or masks directory not found")
            return stats
        
        # Remove unmatched masks
        if self.config.remove_unmatched:
            stats["removed_masks"] = self._remove_unmatched_masks(
                images_dir,
                masks_dir
            )
        
        # Remove corrupted files
        if self.config.check_corrupted:
            stats["removed_corrupted"] = self._remove_corrupted_files(
                images_dir,
                masks_dir
            )
        
        return stats
    
    def analyze(self) -> Dict[str, any]:
        """
        Analyze dataset without cleaning.
        
        Returns:
            Analysis results
        """
        images_dir = self.config.input_dir / "images"
        masks_dir = self.config.input_dir / "masks"
        
        if not images_dir.exists() or not masks_dir.exists():
            return {"error": "Directories not found"}
        
        image_stems = FileUtils.get_file_stems(
            images_dir,
            self.config.image_extensions
        )
        mask_stems = FileUtils.get_file_stems(
            masks_dir,
            self.config.mask_extensions
        )
        
        matched = image_stems & mask_stems
        unmatched_images = image_stems - mask_stems
        unmatched_masks = mask_stems - image_stems
        
        # Count by class if filenames have class suffixes
        class_counts = self._count_by_class(images_dir)
        
        return {
            "total_images": len(image_stems),
            "total_masks": len(mask_stems),
            "matched_pairs": len(matched),
            "unmatched_images": len(unmatched_images),
            "unmatched_masks": len(unmatched_masks),
            "class_counts": class_counts
        }
    
    def _remove_unmatched_masks(
        self,
        images_dir: Path,
        masks_dir: Path
    ) -> int:
        """Remove masks without corresponding images."""
        image_stems = FileUtils.get_file_stems(
            images_dir,
            self.config.image_extensions
        )
        
        removed = 0
        mask_files = [
            f for f in masks_dir.iterdir()
            if f.is_file() and f.suffix.lower() in self.config.mask_extensions
        ]
        
        for mask_file in tqdm(mask_files, desc="Checking masks"):
            if mask_file.stem not in image_stems:
                if not self.config.dry_run:
                    mask_file.unlink()
                self.logger.info(f"Removed unmatched mask: {mask_file.name}")
                removed += 1
        
        return removed
    
    def _remove_corrupted_files(
        self,
        images_dir: Path,
        masks_dir: Path
    ) -> int:
        """Remove corrupted image and mask files."""
        removed = 0
        
        # Check images
        for img_file in tqdm(
            list(images_dir.glob('*')),
            desc="Checking images"
        ):
            if not img_file.is_file():
                continue
            
            if not FileUtils.is_image_valid(img_file):
                if not self.config.dry_run:
                    img_file.unlink()
                self.logger.warning(f"Removed corrupted image: {img_file.name}")
                removed += 1
        
        # Check masks
        for mask_file in tqdm(
            list(masks_dir.glob('*')),
            desc="Checking masks"
        ):
            if not mask_file.is_file():
                continue
            
            if not FileUtils.is_image_valid(mask_file):
                if not self.config.dry_run:
                    mask_file.unlink()
                self.logger.warning(f"Removed corrupted mask: {mask_file.name}")
                removed += 1
        
        return removed
    
    def _count_by_class(self, images_dir: Path) -> Dict[str, int]:
        """Count images by class based on filename suffixes."""
        class_counts = {}
        
        for img_file in images_dir.glob('*'):
            if not img_file.is_file():
                continue
            
            stem = img_file.stem
            # Try to extract class from filename (e.g., "image_adenoma")
            if '_' in stem:
                class_name = stem.split('_')[-1]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            else:
                class_counts['other'] = class_counts.get('other', 0) + 1
        
        return class_counts
