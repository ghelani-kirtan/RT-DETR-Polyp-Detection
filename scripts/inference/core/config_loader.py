"""Configuration loader for inference pipeline."""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file (can be absolute or relative)
        
    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)
    
    # If not absolute, try multiple locations
    if not config_file.is_absolute():
        # Try as-is first (for when running from scripts/)
        if config_file.exists():
            pass
        # Try relative to scripts directory (for when running from project root)
        elif (Path(__file__).parent.parent / config_path).exists():
            config_file = Path(__file__).parent.parent / config_path
        # Try in scripts/ subdirectory (for when running from project root with scripts/config.yaml)
        elif Path(config_path).exists():
            config_file = Path(config_path)
        else:
            raise FileNotFoundError(
                f"Config file not found: {config_path}\n"
                f"Tried:\n"
                f"  - {Path(config_path).absolute()}\n"
                f"  - {(Path(__file__).parent.parent / config_path).absolute()}"
            )
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required fields
    required_fields = ['model', 'video', 'tracker', 'visualization']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required config section: {field}")
    
    return config


def get_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract model configuration."""
    return config['model']


def get_video_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract video configuration."""
    return config['video']


def get_tracker_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract tracker configuration."""
    return config['tracker']


def get_smoothing_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract smoothing configuration."""
    return config.get('smoothing', {'enabled': False})


def get_visualization_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract visualization configuration."""
    return config['visualization']


def get_performance_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract performance configuration."""
    return config.get('performance', {
        'providers': ['CUDAExecutionProvider', 'CPUExecutionProvider']
    })
