"""Classification dataset organizer."""

import shutil
from pathlib import Path
from typing import Dict
from tqdm import tqdm
from .base_organizer import BaseOrganizer
from ..core import OrganizerConfig, FileUtils


class ClassificationOrganizer(BaseOrganizer):
    """Organize classification dataset with colored masks."""
    
    def organize(self) -> Dict[str, any]:
        """
        Organize classification dataset.
        
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
        
        # Process positive samples
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
        """Process positive samples with class-specific colors."""
        stats = {"images": 0, "masks": 0}
        
        for pathology_dir in positive_path.iterdir():
            if not pathology_dir.is_dir():
                continue
            
            pathology = pathology_dir.name.lower()
            if pathology not in self.config.class_colors:
                self.logger.warning(
                    f"Skipping unknown class '{pathology}'"
                )
                continue
            
            images_src = pathology_dir / "images"
            masks_src = pathology_dir / "masks"
            
            if not images_src.exists() or not masks_src.exists():
                continue
            
            color = self.config.class_colors[pathology]
            image_files = list(images_src.glob('*'))
            
            for img_file in tqdm(
                image_files,
                desc=f"Processing {pathology}"
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
                        f"No mask for {img_file.name} in {pathology}"
                    )
                    continue
                
                # Output with class suffix
                img_out = output_images / f"{stem}_{pathology}.jpg"
                mask_out = output_masks / f"{stem}_{pathology}.jpg"
                
                if not self.config.dry_run:
                    if FileUtils.convert_image_format(img_file, img_out):
                        stats["images"] += 1
                    if FileUtils.convert_mask_to_color(mask_file, mask_out, color):
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
