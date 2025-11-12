"""Base downloader class for data acquisition."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List
from ..core import DownloaderConfig, setup_logger


class BaseDownloader(ABC):
    """Abstract base class for data downloaders."""
    
    def __init__(self, config: DownloaderConfig):
        """
        Initialize downloader.
        
        Args:
            config: Downloader configuration
        """
        self.config = config
        self.logger = setup_logger(
            self.__class__.__name__,
            config.log_file
        )
    
    @abstractmethod
    def download(self) -> Dict[str, any]:
        """
        Download data from source.
        
        Returns:
            Dictionary with download statistics
        """
        pass
    
    @abstractmethod
    def verify_downloads(self) -> Dict[str, any]:
        """
        Verify downloaded files.
        
        Returns:
            Dictionary with verification results
        """
        pass
    
    def prepare_output_structure(self, base_path: Path):
        """
        Prepare output directory structure.
        
        Args:
            base_path: Base output path
        """
        if not self.config.dry_run:
            base_path.mkdir(parents=True, exist_ok=True)
