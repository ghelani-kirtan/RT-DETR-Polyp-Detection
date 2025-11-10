"""Detection dataset organizer."""

import shutil
from pathlib import Path
from typing import Dict
from tqdm import tqdm
from .base_organizer import BaseOrganizer
from ..core import OrganizerConfig, FileUtils


class DetectionOrganizer(BaseOrganizer):
    """Organize detection dataset with binary masks."""
    
    def organize(self) -> Dict[str, any]:
        """
        Organize detection dataset.
        
        Returns:
            Organization statistics
        """
        stats = {"images": 0, "masks": 0, "negatives": 0}
        
        # ROOT FIX: Simple, direct path - downloader now puts files here
        client_data_path = self.config.input_dir / "client_data"
        
        if not client_data_path.exists():
            self.logger.error(f"client_data directory not found at {client_data_path}")
            return stats
            
        positive_path = client_data_path / self.config.positive_subdir
        negative_path = client_data_path / self.config.negative_subdir
        
        output_images = self.config.output_dir / self.config.images_subdir
        output_masks = self.config.output_dir / self.config.masks_subdir
        output_negatives = self.config.output_dir / self.config.negative_subdir
        
        if not self.config.dry_run:
            FileUtils.ensure_dir(output_images)
            FileUtils.ensure_dir(output_masks)
        
        # Process positive samples (aggregate all classes)
        if positive_path.exists():
            stats.update(self._process_positive_samples(
                positive_path,
                output_images,
                output_masks
            ))
        
        # Process negative samples
        if negative_path.exists():
            stats["negatives"] = self._process_negative_samples(
                negative_path,
                output_negatives
            )
        
        return stats
    
    def _process_positive_samples(
        self,
        positive_path: Path,
        output_images: Path,
        output_masks: Path
    ) -> Dict[str, int]:
        """Process positive samples from all classes."""
        stats = {"images": 0, "masks": 0}
        
        for pathology_dir in positive_path.iterdir():
            if not pathology_dir.is_dir():
                continue
            
            images_src = pathology_dir / "images"
            masks_src = pathology_dir / "masks"
            
            if not images_src.exists() or not masks_src.exists():
                continue
            
            image_files = list(images_src.glob('*'))
            
            for img_file in tqdm(
                image_files,
                desc=f"Processing {pathology_dir.name}"
            ):
                if img_file.suffix.lower() not in self.config.image_extensions:
                    continue
                
                stem = img_file.stem
                mask_file = FileUtils.find_matching_file(
                    masks_src,
                    stem,
                    self.config.mask_extensions
                )
                
                if not mask_file:
                    self.logger.warning(
                        f"No mask for {img_file.name}"
                    )
                    continue
                
                # Copy to output (keep original names)
                img_out = output_images / img_file.name
                mask_out = output_masks / mask_file.name
                
                if not self.config.dry_run:
                    if not img_out.exists():
                        shutil.copy(img_file, img_out)
                        stats["images"] += 1
                    if not mask_out.exists():
                        shutil.copy(mask_file, mask_out)
                        stats["masks"] += 1
        
        return stats
    
    def _process_negative_samples(
        self,
        negative_path: Path,
        output_negatives: Path
    ) -> int:
        """Process negative samples."""
        if self.config.dry_run:
            return 0
        
        if output_negatives.exists():
            shutil.rmtree(output_negatives)
        
        shutil.copytree(negative_path, output_negatives)
        return sum(1 for _ in output_negatives.glob('*') if _.is_file())
    
    def validate(self) -> Dict[str, any]:
        """
        Validate organized dataset.
        
        Returns:
            Validation results
        """
        output_images = self.config.output_dir / self.config.images_subdir
        output_masks = self.config.output_dir / self.config.masks_subdir
        
        if not output_images.exists() or not output_masks.exists():
            return {"valid": False, "error": "Output directories not found"}
        
        image_stems = FileUtils.get_file_stems(
            output_images,
            self.config.image_extensions
        )
        mask_stems = FileUtils.get_file_stems(
            output_masks,
            self.config.mask_extensions
        )
        
        matched = len(image_stems & mask_stems)
        unmatched_images = len(image_stems - mask_stems)
        unmatched_masks = len(mask_stems - image_stems)
        
        return {
            "valid": True,
            "matched_pairs": matched,
            "unmatched_images": unmatched_images,
            "unmatched_masks": unmatched_masks
        }
