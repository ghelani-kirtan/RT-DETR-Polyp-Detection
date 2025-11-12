"""End-to-end detection dataset pipeline."""

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
from ..organizers import DetectionOrganizer
from ..cleaners import DatasetCleaner
from ..preparers import DetectionPreparer


class DetectionPipeline:
    """Complete pipeline for detection dataset preparation."""
    
    def __init__(
        self,
        base_dir: Path,
        dataset_version_ids: list = None,
        api_url: str = None,
        dry_run: bool = False
    ):
        """
        Initialize detection pipeline.
        
        Args:
            base_dir: Base directory for all operations
            dataset_version_ids: List of dataset version IDs to download
            api_url: API URL for downloading
            dry_run: If True, simulate operations
        """
        self.base_dir = Path(base_dir)
        self.dry_run = dry_run
        self.logger = setup_logger("DetectionPipeline")
        
        # Setup paths
        self.client_data_dir = self.base_dir / "client_data"
        self.detection_dataset_dir = self.base_dir / "detection_dataset"
        self.coco_dir = self.base_dir / "coco"
        
        # Initialize configs
        self.downloader_config = DownloaderConfig(
            api_url=api_url or "",
            dataset_version_ids=dataset_version_ids or [],
            output_dir=self.base_dir,
            dry_run=dry_run,
            class_mappings={
                "adenoma": "adenoma",
                "hyperplastic": "hyperplastic",
                "benign": "benign",
                "no pathology": "no_pathology",
                "multiple pathologies": "multiple_pathologies"
            }
        )
        
        self.organizer_config = OrganizerConfig(
            input_dir=self.base_dir,
            output_dir=self.detection_dataset_dir,
            dry_run=dry_run
        )
        
        self.cleaner_config = CleanerConfig(
            input_dir=self.detection_dataset_dir,
            dry_run=dry_run
        )
        
        self.preparer_config = PreparerConfig(
            input_dir=self.detection_dataset_dir,
            output_dir=self.coco_dir,
            dry_run=dry_run,
            categories=[{"id": 1, "name": "polyp", "supercategory": "none"}]
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
        self.logger.info("Step 2: Organizing dataset")
        organizer = DetectionOrganizer(self.organizer_config)
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
        preparer = DetectionPreparer(self.preparer_config)
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
