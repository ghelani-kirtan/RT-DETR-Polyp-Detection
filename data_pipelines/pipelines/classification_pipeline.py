"""End-to-end classification dataset pipeline."""

from pathlib import Path
from typing import Dict
from ..core import (
    DownloaderConfig,
    OrganizerConfig,
    CleanerConfig,
    PreparerConfig,
    setup_logger
)
from ..downloaders import APIDownloader
from ..organizers import ClassificationOrganizer
from ..cleaners import DatasetCleaner
from ..preparers import ClassificationPreparer


class ClassificationPipeline:
    """Complete pipeline for classification dataset preparation."""
    
    def __init__(
        self,
        base_dir: Path,
        dataset_version_ids: list = None,
        api_url: str = None,
        dry_run: bool = False
    ):
        """
        Initialize classification pipeline.
        
        Args:
            base_dir: Base directory for all operations
            dataset_version_ids: List of dataset version IDs to download
            api_url: API URL for downloading
            dry_run: If True, simulate operations
        """
        self.base_dir = Path(base_dir)
        self.dry_run = dry_run
        self.logger = setup_logger("ClassificationPipeline")
        
        # Setup paths
        self.client_data_dir = self.base_dir / "client_data"
        self.classification_dataset_dir = self.base_dir / "classification_dataset"
        self.coco_dir = self.base_dir / "coco_classification"
        
        # Class mappings and colors
        class_mappings = {
            "adenoma": "adenoma",
            "hyperplastic": "hyperplastic",
            "benign": "benign",
            "no pathology": "no_pathology",
            "multiple pathologies": "multiple_pathologies"
        }
        
        class_colors = {
            "adenoma": (255, 0, 0),         # Red
            "hyperplastic": (0, 255, 0),    # Green
            "benign": (157, 0, 255),        # Purple
            "no_pathology": (255, 255, 255) # White
        }
        
        # Initialize configs
        self.downloader_config = DownloaderConfig(
            api_url=api_url or "",
            dataset_version_ids=dataset_version_ids or [],
            output_dir=self.base_dir,
            dry_run=dry_run,
            class_mappings=class_mappings
        )
        
        self.organizer_config = OrganizerConfig(
            input_dir=self.base_dir,
            output_dir=self.classification_dataset_dir,
            dry_run=dry_run,
            class_mappings=class_mappings,
            class_colors=class_colors
        )
        
        self.cleaner_config = CleanerConfig(
            input_dir=self.classification_dataset_dir,
            dry_run=dry_run
        )
        
        self.preparer_config = PreparerConfig(
            input_dir=self.classification_dataset_dir,
            output_dir=self.coco_dir,
            dry_run=dry_run,
            categories=[
                {"id": 1, "name": "adenoma", "supercategory": "none"},
                {"id": 2, "name": "hyperplastic", "supercategory": "none"},
                {"id": 3, "name": "benign", "supercategory": "none"},
                {"id": 4, "name": "no_pathology", "supercategory": "none"}
            ]
        )
    
    def run_download(self) -> Dict:
        """Run download step."""
        self.logger.info("Step 1: Downloading data from API")
        downloader = APIDownloader(self.downloader_config)
        stats = downloader.download()
        self.logger.info(f"Download complete: {stats}")
        return stats
    
    def run_organize(self) -> Dict:
        """Run organization step."""
        self.logger.info("Step 2: Organizing dataset with colored masks")
        organizer = ClassificationOrganizer(self.organizer_config)
        stats = organizer.organize()
        validation = organizer.validate()
        self.logger.info(f"Organization complete: {stats}")
        self.logger.info(f"Validation: {validation}")
        return {"stats": stats, "validation": validation}
    
    def run_clean(self) -> Dict:
        """Run cleaning step."""
        self.logger.info("Step 3: Cleaning dataset")
        cleaner = DatasetCleaner(self.cleaner_config)
        analysis = cleaner.analyze()
        self.logger.info(f"Analysis: {analysis}")
        stats = cleaner.clean()
        self.logger.info(f"Cleaning complete: {stats}")
        return {"analysis": analysis, "stats": stats}
    
    def run_prepare(self) -> Dict:
        """Run COCO preparation step."""
        self.logger.info("Step 4: Preparing COCO format")
        preparer = ClassificationPreparer(self.preparer_config)
        stats = preparer.prepare()
        self.logger.info(f"COCO preparation complete: {stats}")
        return stats
    
    def run_full_pipeline(self, skip_download: bool = False) -> Dict:
        """
        Run complete pipeline.
        
        Args:
            skip_download: If True, skip download step
            
        Returns:
            Dictionary with all step results
        """
        results = {}
        
        if not skip_download:
            results["download"] = self.run_download()
        
        results["organize"] = self.run_organize()
        results["clean"] = self.run_clean()
        results["prepare"] = self.run_prepare()
        
        self.logger.info("Pipeline complete!")
        return results
