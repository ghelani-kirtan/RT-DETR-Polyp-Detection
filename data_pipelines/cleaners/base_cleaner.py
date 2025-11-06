"""Base cleaner class for dataset cleaning."""

from abc import ABC, abstractmethod
from typing import Dict
from ..core import CleanerConfig, setup_logger


class BaseCleaner(ABC):
    """Abstract base class for dataset cleaners."""
    
    def __init__(self, config: CleanerConfig):
        """
        Initialize cleaner.
        
        Args:
            config: Cleaner configuration
        """
        self.config = config
        self.logger = setup_logger(
            self.__class__.__name__,
            config.log_file
        )
    
    @abstractmethod
    def clean(self) -> Dict[str, any]:
        """
        Clean dataset.
        
        Returns:
            Dictionary with cleaning statistics
        """
        pass
    
    @abstractmethod
    def analyze(self) -> Dict[str, any]:
        """
        Analyze dataset without cleaning.
        
        Returns:
            Dictionary with analysis results
        """
        pass
