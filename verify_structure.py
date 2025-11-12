#!/usr/bin/env python3
"""
Verification script to ensure project restructure is complete and correct.
Run this after restructuring to verify all files are in the right place.
"""

import os
import sys
from pathlib import Path

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'

def check_file_exists(path, description):
    """Check if a file exists and print result."""
    if os.path.exists(path):
        print(f"{GREEN}✓{RESET} {description}: {path}")
        return True
    else:
        print(f"{RED}✗{RESET} {description}: {path} (MISSING)")
        return False

def check_file_not_exists(path, description):
    """Check if a file does NOT exist (should be removed)."""
    if not os.path.exists(path):
        print(f"{GREEN}✓{RESET} {description}: {path} (correctly removed)")
        return True
    else:
        print(f"{RED}✗{RESET} {description}: {path} (SHOULD BE REMOVED)")
        return False

def check_import(module_path, import_statement, description):
    """Check if an import statement works correctly."""
    try:
        exec(import_statement)
        print(f"{GREEN}✓{RESET} {description}")
        return True
    except ImportError as e:
        print(f"{YELLOW}⚠{RESET} {description} (dependencies not installed: {e})")
        return True  # Don't fail on missing dependencies
    except Exception as e:
        print(f"{RED}✗{RESET} {description}: {e}")
        return False

def main():
    print("=" * 70)
    print("Project Structure Verification")
    print("=" * 70)
    print()
    
    all_checks_passed = True
    
    # Check new structure exists
    print("Checking new structure...")
    print("-" * 70)
    
    new_files = [
        ("scripts/inference/classification_vid_ann_saver.py", "Classification inference script"),
        ("scripts/inference/run_inference.py", "Main inference script"),
        ("scripts/inference/core/video_processor.py", "Video processor"),
        ("scripts/inference/deploy/rtdetrv2_onnxruntime.py", "ONNX deployment script"),
        ("scripts/cloud/s3_sync_models.py", "S3 model sync"),
        ("scripts/cloud/s3_sync_datasets.py", "S3 dataset sync"),
        ("scripts/cloud/s3_tools/cli.py", "S3 tools CLI"),
        ("scripts/data/cli_classification.py", "Classification data CLI"),
        ("scripts/data/cli_detection.py", "Detection data CLI"),
        ("scripts/data/pipelines/classification_pipeline.py", "Classification pipeline"),
        ("scripts/data/pipelines/detection_pipeline.py", "Detection pipeline"),
        ("scripts/training_commands.txt", "Training commands"),
        ("src/tracker/byte_tracker.py", "Byte tracker"),
        ("src/tracker/basetrack.py", "Base tracker"),
        ("requirements/requirements.txt", "Base requirements"),
        ("requirements/inference-requirements.txt", "Inference requirements"),
        ("requirements/cloud-requirements.txt", "Cloud requirements"),
        ("PROJECT_RESTRUCTURE_GUIDE.md", "Restructure guide"),
        ("QUICK_START.md", "Quick start guide"),
    ]
    
    for path, desc in new_files:
        if not check_file_exists(path, desc):
            all_checks_passed = False
    
    print()
    
    # Check old structure removed
    print("Checking old structure removed...")
    print("-" * 70)
    
    old_paths = [
        ("data_pipelines", "Old data_pipelines folder"),
        ("s3_tools", "Old s3_tools folder"),
        ("tracker", "Old tracker folder"),
        ("references", "Old references folder"),
        ("rtdetr_polyp", "Old rtdetr_polyp folder"),
        ("classification_vid_ann_saver.py", "Old root classification script"),
        ("s3_sync.py", "Old root s3_sync script"),
        ("s3_sync_dataset.py", "Old root s3_sync_dataset script"),
        ("cmds.txt", "Old root cmds.txt"),
    ]
    
    for path, desc in old_paths:
        if not check_file_not_exists(path, desc):
            all_checks_passed = False
    
    print()
    
    # Check Python syntax
    print("Checking Python syntax...")
    print("-" * 70)
    
    import py_compile
    
    python_files = [
        "scripts/inference/classification_vid_ann_saver.py",
        "scripts/cloud/s3_sync_models.py",
        "scripts/cloud/s3_sync_datasets.py",
        "scripts/data/cli_classification.py",
        "src/tracker/byte_tracker.py",
    ]
    
    for file in python_files:
        try:
            py_compile.compile(file, doraise=True)
            print(f"{GREEN}✓{RESET} Syntax valid: {file}")
        except py_compile.PyCompileError as e:
            print(f"{RED}✗{RESET} Syntax error in {file}: {e}")
            all_checks_passed = False
    
    print()
    print("=" * 70)
    
    if all_checks_passed:
        print(f"{GREEN}✓ All checks passed! Project structure is correct.{RESET}")
        return 0
    else:
        print(f"{RED}✗ Some checks failed. Please review the output above.{RESET}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
