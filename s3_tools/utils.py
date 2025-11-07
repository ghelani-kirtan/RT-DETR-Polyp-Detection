"""Utility functions for S3 tools."""

import yaml
from pathlib import Path
from typing import Dict, Any
from datetime import datetime


def load_config(config_path: str = "s3_tools/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def format_timestamp(timestamp: datetime) -> str:
    """
    Format timestamp for display.
    
    Args:
        timestamp: Datetime object
        
    Returns:
        Formatted timestamp string
    """
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")


def format_size(size_bytes: int) -> str:
    """
    Format size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def confirm_action(message: str, default: bool = False) -> bool:
    """
    Ask user for confirmation.
    
    Args:
        message: Confirmation message
        default: Default response if user just presses Enter
        
    Returns:
        True if confirmed, False otherwise
    """
    suffix = " [Y/n]" if default else " [y/N]"
    response = input(f"{message}{suffix}: ").strip().lower()
    
    if not response:
        return default
    
    return response in ['y', 'yes', 'true', '1']


def build_s3_prefix(config: Dict, model_type: str, model_name: str) -> str:
    """
    Build S3 prefix for a model.
    
    Args:
        config: Configuration dictionary
        model_type: Model type ('classification', 'detection', etc.)
        model_name: Model directory name
        
    Returns:
        Full S3 prefix
    """
    base_prefix = config['s3']['base_prefix']
    type_prefix = config['s3']['prefixes'].get(model_type, model_type)
    
    return f"{base_prefix}/{type_prefix}/{model_name}"


def print_model_info(model: Dict):
    """
    Print formatted model information.
    
    Args:
        model: Model info dictionary
    """
    size_str = format_size(model['size'])
    modified_str = datetime.fromtimestamp(model['modified']).strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"  📁 {model['name']}")
    print(f"     Type: {model['type']}")
    print(f"     Size: {size_str} ({model['file_count']} files)")
    print(f"     Modified: {modified_str}")


def print_s3_objects(objects: list, prefix: str):
    """
    Print formatted S3 object list.
    
    Args:
        objects: List of S3 object dictionaries
        prefix: S3 prefix being listed
    """
    if not objects:
        print(f"  No objects found at {prefix}")
        return
    
    total_size = sum(obj['size'] for obj in objects)
    size_str = format_size(total_size)
    
    print(f"  📦 {len(objects)} files, {size_str} total")
    
    # Show first few files
    for obj in objects[:5]:
        file_size = format_size(obj['size'])
        modified = format_timestamp(obj['last_modified'])
        filename = obj['key'].split('/')[-1]
        print(f"     {filename} ({file_size}, {modified})")
    
    if len(objects) > 5:
        print(f"     ... and {len(objects) - 5} more files")


def validate_model_name(name: str) -> bool:
    """
    Validate model name format.
    
    Args:
        name: Model name to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not name:
        return False
    
    # Check for invalid characters
    invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    if any(char in name for char in invalid_chars):
        return False
    
    return True


def get_model_summary(models: list) -> str:
    """
    Get summary of models.
    
    Args:
        models: List of model dictionaries
        
    Returns:
        Summary string
    """
    if not models:
        return "No models found"
    
    types = {}
    total_size = 0
    
    for model in models:
        model_type = model['type']
        types[model_type] = types.get(model_type, 0) + 1
        total_size += model['size']
    
    size_str = format_size(total_size)
    
    type_summary = ", ".join([f"{count} {type}" for type, count in types.items()])
    
    return f"{len(models)} models ({type_summary}), {size_str} total"
