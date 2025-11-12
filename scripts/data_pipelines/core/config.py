"""Configuration management for data pipelines."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class PipelineConfig:
    """Base configuration for data pipelines."""
    
    # Input/Output paths
    input_dir: Path = Path(".")
    output_dir: Path = Path(".")
    
    # Processing options
    dry_run: bool = False
    max_workers: int = 20
    seed: int = 42
    
    # Split ratios
    train_split: float = 0.8
    
    # File extensions
    image_extensions: tuple = (".png", ".jpg", ".jpeg")
    mask_extensions: tuple = (".png", ".jpg", ".jpeg", ".tif")
    
    # Logging
    log_file: Optional[str] = None
    verbose: bool = True
    
    def __post_init__(self):
        """Convert string paths to Path objects."""
        self.input_dir = Path(self.input_dir)
        self.output_dir = Path(self.output_dir)


@dataclass
class DownloaderConfig(PipelineConfig):
    """Configuration for data downloaders."""
    
    # API settings
    api_url: str = ""
    dataset_version_ids: List[int] = field(default_factory=list)
    
    # S3 settings
    bucket_name: str = ""
    s3_prefix: str = ""
    
    # Download settings
    multipart_threshold_mb: int = 25
    multipart_chunksize_mb: int = 25
    max_concurrency: int = 10
    max_pool_connections: int = 200
    retry_attempts: int = 15
    retry_delay: int = 5
    
    # Class mappings for organizing downloads
    class_mappings: Dict[str, str] = field(default_factory=dict)


@dataclass
class OrganizerConfig(PipelineConfig):
    """Configuration for dataset organizers."""
    
    # Directory structure
    positive_subdir: str = "positive_samples"
    negative_subdir: str = "negative_samples"
    images_subdir: str = "images"
    masks_subdir: str = "masks"
    
    # Class mappings
    class_mappings: Dict[str, str] = field(default_factory=dict)
    class_colors: Dict[str, tuple] = field(default_factory=dict)


@dataclass
class CleanerConfig(PipelineConfig):
    """Configuration for dataset cleaners."""
    
    # Cleaning options
    remove_unmatched: bool = True
    min_file_size: int = 0
    check_corrupted: bool = True


@dataclass
class PreparerConfig(PipelineConfig):
    """Configuration for COCO format preparers."""
    
    # COCO settings
    add_negative_samples: bool = True
    min_area_threshold: int = 50
    
    # Categories
    categories: List[Dict[str, any]] = field(default_factory=list)
    
    # Output structure
    train_folder: str = "train2017"
    val_folder: str = "val2017"
    annotations_folder: str = "annotations"
