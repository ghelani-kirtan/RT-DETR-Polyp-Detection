"""End-to-end data pipelines."""

from .detection_pipeline import DetectionPipeline
from .classification_pipeline import ClassificationPipeline

__all__ = ['DetectionPipeline', 'ClassificationPipeline']
