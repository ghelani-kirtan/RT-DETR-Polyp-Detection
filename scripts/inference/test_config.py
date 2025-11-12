#!/usr/bin/env python3
"""Test configuration loading."""

from core import load_config

def test_config(config_file):
    """Test loading a config file."""
    try:
        config = load_config(config_file)
        print(f"✓ Config loaded: {config_file}")
        print(f"  Model: {config['model']['path']}")
        print(f"  Task: {config['model']['task']}")
        print(f"  Classes: {config['model']['classes']['names']}")
        print(f"  Tracker: {'enabled' if config['tracker']['enabled'] else 'disabled'}")
        print(f"  Smoothing: {'enabled' if config.get('smoothing', {}).get('enabled', False) else 'disabled'}")
        print(f"  High-res: {'yes' if config['video'].get('high_resolution', False) else 'no'}")
        return True
    except Exception as e:
        print(f"✗ Error loading {config_file}: {e}")
        return False

if __name__ == "__main__":
    configs = [
        'config.yaml',
        'config_detection.yaml',
        'config_highres.yaml',
        'config_no_tracker.yaml'
    ]
    
    print("Testing all configurations...\n")
    for cfg in configs:
        test_config(cfg)
        print()
