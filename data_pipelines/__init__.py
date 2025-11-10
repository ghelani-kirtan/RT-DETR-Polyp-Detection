"""
Data Pipelines Package

Generic, reusable pipelines for detection and classification datasets.

Quick Start:
    from data_pipelines.pipelines import DetectionPipeline, ClassificationPipeline
    
    # Detection
    pipeline = DetectionPipeline(base_dir="./project", dataset_version_ids=[1,2])
    pipeline.run_full_pipeline()
    
    # Classification
    pipeline = ClassificationPipeline(base_dir="./project", dataset_version_ids=[1,2])
    pipeline.run_full_pipeline()
"""

from .pipelines import DetectionPipeline, ClassificationPipeline
from .core import (
    PipelineConfig,
    DownloaderConfig,
    OrganizerConfig,
    CleanerConfig,
    PreparerConfig
)

__version__ = "1.0.0"
__all__ = [
    'DetectionPipeline',
    'ClassificationPipeline',
    'PipelineConfig',
    'DownloaderConfig',
    'OrganizerConfig',
    'CleanerConfig',
    'PreparerConfig'
]
