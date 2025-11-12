"""Base preparer class for COCO format conversion."""

from abc import ABC, abstractmethod
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List
from ..core import PreparerConfig, setup_logger, FileUtils


class BasePreparer(ABC):
    """Abstract base class for COCO format preparers."""
    
    def __init__(self, config: PreparerConfig):
        """
        Initialize preparer.
        
        Args:
            config: Preparer configuration
        """
        self.config = config
        self.logger = setup_logger(
            self.__class__.__name__,
            config.log_file
        )
        random.seed(config.seed)
    
    @abstractmethod
    def prepare(self) -> Dict[str, any]:
        """
        Prepare dataset in COCO format.
        
        Returns:
            Dictionary with preparation statistics
        """
        pass
    
    def prepare_output_structure(self):
        """Create output directory structure."""
        if not self.config.dry_run:
            FileUtils.ensure_dir(
                self.config.output_dir / self.config.train_folder
            )
            FileUtils.ensure_dir(
                self.config.output_dir / self.config.val_folder
            )
            FileUtils.ensure_dir(
                self.config.output_dir / self.config.annotations_folder
            )
    
    def split_dataset(
        self,
        positive_stems: List[str],
        negative_stems: List[str]
    ) -> tuple:
        """
        Split dataset into train and validation sets.
        
        Args:
            positive_stems: List of positive sample stems
            negative_stems: List of negative sample stems
            
        Returns:
            Tuple of (train_stems, val_stems)
        """
        all_stems = positive_stems + negative_stems
        random.shuffle(all_stems)
        
        train_size = int(self.config.train_split * len(all_stems))
        train_stems = all_stems[:train_size]
        val_stems = all_stems[train_size:]
        
        return train_stems, val_stems
    
    def create_coco_base(self) -> Dict:
        """Create base COCO format structure."""
        return {
            "info": {
                "year": 2025,
                "version": "1",
                "description": "Dataset for object detection",
                "contributor": "",
                "url": "",
                "date_created": "2025/11/06"
            },
            "licenses": [{"id": 1, "name": "Unknown", "url": ""}],
            "categories": self.config.categories,
            "images": [],
            "annotations": []
        }
    
    def save_coco_json(
        self,
        data: Dict,
        split: str
    ):
        """
        Save COCO format JSON file.
        
        Args:
            data: COCO format dictionary
            split: Split name (train/val)
        """
        if self.config.dry_run:
            return
        
        folder = (
            self.config.train_folder if split == "train"
            else self.config.val_folder
        )
        filename = f"instances_{folder}.json"
        
        output_path = (
            self.config.output_dir /
            self.config.annotations_folder /
            filename
        )
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=4)
