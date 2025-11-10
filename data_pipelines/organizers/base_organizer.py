"""Base organizer class for dataset organization."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict
from ..core import OrganizerConfig, setup_logger


class BaseOrganizer(ABC):
    """Abstract base class for dataset organizers."""
    
    def __init__(self, config: OrganizerConfig):
        """
        Initialize organizer.
        
        Args:
            config: Organizer configuration
        """
        self.config = config
        self.logger = setup_logger(
            self.__class__.__name__,
            config.log_file
        )
    
    @abstractmethod
    def organize(self) -> Dict[str, any]:
        """
        Organize dataset into target structure.
        
        Returns:
            Dictionary with organization statistics
        """
        pass
    
    @abstractmethod
    def validate(self) -> Dict[str, any]:
        """
        Validate organized dataset.
        
        Returns:
            Dictionary with validation results
        """
        pass
    
    def prepare_output_structure(self):
        """Prepare output directory structure."""
        if not self.config.dry_run:
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
