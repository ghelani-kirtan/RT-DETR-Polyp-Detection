"""File utility functions for data pipelines."""

import os
import shutil
from pathlib import Path
from typing import List, Set, Tuple
from PIL import Image
import numpy as np


class FileUtils:
    """Utility class for file operations."""
    
    @staticmethod
    def get_file_stems(directory: Path, extensions: tuple = None) -> Set[str]:
        """
        Get file stems (names without extensions) from directory.
        
        Args:
            directory: Directory to scan
            extensions: Optional tuple of extensions to filter
            
        Returns:
            Set of file stems
        """
        if not directory.exists():
            return set()
        
        stems = set()
        for file in directory.iterdir():
            if file.is_file():
                if extensions is None or file.suffix.lower() in extensions:
                    stems.add(file.stem)
        return stems
    
    @staticmethod
    def find_matching_file(directory: Path, stem: str, extensions: tuple) -> Path:
        """
        Find file with given stem and any of the extensions.
        
        Args:
            directory: Directory to search
            stem: File stem to match
            extensions: Tuple of valid extensions
            
        Returns:
            Path to matching file or None
        """
        for ext in extensions:
            file_path = directory / f"{stem}{ext}"
            if file_path.exists():
                return file_path
        return None
    
    @staticmethod
    def ensure_dir(path: Path, clean: bool = False):
        """
        Ensure directory exists, optionally cleaning it first.
        
        Args:
            path: Directory path
            clean: If True, remove existing directory first
        """
        if clean and path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def convert_image_format(
        input_path: Path, 
        output_path: Path, 
        output_format: str = 'JPEG'
    ) -> bool:
        """
        Convert image to specified format.
        
        Args:
            input_path: Input image path
            output_path: Output image path
            output_format: Target format (JPEG, PNG, etc.)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if input_path.suffix.lower() == output_path.suffix.lower():
                shutil.copy(input_path, output_path)
            else:
                image = Image.open(input_path).convert('RGB')
                image.save(output_path, output_format)
            return True
        except Exception as e:
            return False
    
    @staticmethod
    def convert_mask_to_color(
        input_path: Path, 
        output_path: Path, 
        color: tuple
    ) -> bool:
        """
        Convert binary mask to colored RGB mask.
        
        Args:
            input_path: Input binary mask path
            output_path: Output colored mask path
            color: RGB color tuple
            
        Returns:
            True if successful, False otherwise
        """
        try:
            mask = Image.open(input_path).convert('L')
            mask_array = np.array(mask)
            
            colored_mask = np.zeros(
                (mask_array.shape[0], mask_array.shape[1], 3), 
                dtype=np.uint8
            )
            
            foreground = mask_array > 0
            colored_mask[foreground] = color
            
            colored_image = Image.fromarray(colored_mask, 'RGB')
            colored_image.save(output_path, 'JPEG')
            return True
        except Exception:
            return False
    
    @staticmethod
    def is_image_valid(file_path: Path) -> bool:
        """
        Check if image file is valid and not corrupted.
        
        Args:
            file_path: Path to image file
            
        Returns:
            True if valid, False otherwise
        """
        try:
            if not file_path.exists() or file_path.stat().st_size == 0:
                return False
            img = Image.open(file_path)
            img.verify()
            return True
        except Exception:
            return False
    
    @staticmethod
    def get_matching_pairs(
        images_dir: Path,
        masks_dir: Path,
        image_extensions: tuple,
        mask_extensions: tuple
    ) -> Tuple[List[str], List[str]]:
        """
        Find matching image-mask pairs by stem.
        
        Args:
            images_dir: Images directory
            masks_dir: Masks directory
            image_extensions: Valid image extensions
            mask_extensions: Valid mask extensions
            
        Returns:
            Tuple of (positive_stems, negative_stems)
        """
        image_stems = FileUtils.get_file_stems(images_dir, image_extensions)
        mask_stems = FileUtils.get_file_stems(masks_dir, mask_extensions)
        
        positive_stems = sorted(image_stems & mask_stems)
        negative_stems = sorted(image_stems - mask_stems)
        
        return positive_stems, negative_stems
