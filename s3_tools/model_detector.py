"""Auto-detect model directories and types."""

import os
from pathlib import Path
from typing import List, Dict, Optional
from fnmatch import fnmatch


class ModelDetector:
    """Detects and analyzes model directories."""
    
    def __init__(self, config: dict):
        """
        Initialize model detector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.output_dir = Path(config['local']['output_dir'])
        self.model_patterns = config['detection']['model_patterns']
        self.required_files = config['detection']['required_files']
    
    def find_models(self) -> List[Dict]:
        """
        Find all model directories in output directory.
        
        Returns:
            List of model info dictionaries
        """
        if not self.output_dir.exists():
            return []
        
        models = []
        for item in self.output_dir.iterdir():
            if item.is_dir() and self._is_model_directory(item):
                model_info = self._analyze_model(item)
                if model_info:
                    models.append(model_info)
        
        return sorted(models, key=lambda x: x['modified'], reverse=True)
    
    def _is_model_directory(self, path: Path) -> bool:
        """Check if directory matches model patterns."""
        return any(fnmatch(path.name, pattern) for pattern in self.model_patterns)
    
    def _analyze_model(self, path: Path) -> Optional[Dict]:
        """
        Analyze a model directory.
        
        Args:
            path: Path to model directory
            
        Returns:
            Model info dictionary or None
        """
        # Check for required files
        has_required = False
        for pattern in self.required_files:
            if list(path.glob(pattern)):
                has_required = True
                break
        
        if not has_required:
            return None
        
        # Detect model type
        model_type = self._detect_model_type(path.name)
        
        # Get directory stats
        total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
        file_count = sum(1 for _ in path.rglob('*') if _.is_file())
        modified = max(
            (f.stat().st_mtime for f in path.rglob('*') if f.is_file()),
            default=0
        )
        
        return {
            'name': path.name,
            'path': path,
            'type': model_type,
            'size': total_size,
            'file_count': file_count,
            'modified': modified
        }
    
    def _detect_model_type(self, name: str) -> str:
        """
        Detect model type from directory name.
        
        Args:
            name: Directory name
            
        Returns:
            Model type ('classification', 'detection', or 'unknown')
        """
        name_lower = name.lower()
        
        if 'classification' in name_lower or 'classifier' in name_lower:
            return 'classification'
        elif 'detection' in name_lower or 'detector' in name_lower:
            return 'detection'
        else:
            return 'unknown'
    
    def get_latest_model(self, model_type: Optional[str] = None) -> Optional[Dict]:
        """
        Get the most recently modified model.
        
        Args:
            model_type: Filter by type ('classification', 'detection', or None for any)
            
        Returns:
            Model info dictionary or None
        """
        models = self.find_models()
        
        if model_type:
            models = [m for m in models if m['type'] == model_type]
        
        return models[0] if models else None
    
    def format_size(self, size_bytes: int) -> str:
        """Format size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
