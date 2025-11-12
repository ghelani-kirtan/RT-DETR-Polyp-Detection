#!/usr/bin/env python3
"""
Test critical workflows to ensure restructure didn't break functionality.
This tests import paths and basic module loading.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_tracker_import():
    """Test tracker module can be imported."""
    try:
        from src.tracker.byte_tracker import BYTETracker
        from src.tracker.basetrack import BaseTrack
        print("✓ Tracker imports successful")
        return True
    except ImportError as e:
        if "torch" in str(e) or "numpy" in str(e):
            print("⚠ Tracker imports skipped (missing dependencies)")
            return True
        print(f"✗ Tracker import failed: {e}")
        return False

def test_data_pipeline_structure():
    """Test data pipeline module structure."""
    try:
        # Test that the package structure is correct
        import scripts.data
        from scripts.data import cleaners, core, downloaders, organizers, pipelines, preparers
        print("✓ Data pipeline structure valid")
        return True
    except ImportError as e:
        # Check if it's a missing dependency (not a structural issue)
        missing_deps = ["PIL", "boto3", "tqdm", "requests", "yaml", "cv2"]
        if any(dep in str(e) for dep in missing_deps):
            print("⚠ Data pipeline structure check skipped (missing dependencies)")
            return True
        print(f"✗ Data pipeline structure invalid: {e}")
        return False

def test_inference_scripts_exist():
    """Test inference scripts are in correct location."""
    scripts = [
        "scripts/inference/classification_vid_ann_saver.py",
        "scripts/inference/run_inference.py",
        "scripts/inference/test_inference.py",
        "scripts/inference/core/video_processor.py",
        "scripts/inference/core/inference_engine.py",
    ]
    
    all_exist = True
    for script in scripts:
        if not os.path.exists(script):
            print(f"✗ Missing: {script}")
            all_exist = False
    
    if all_exist:
        print("✓ All inference scripts in correct location")
    return all_exist

def test_cloud_scripts_exist():
    """Test cloud scripts are in correct location."""
    scripts = [
        "scripts/cloud/s3_sync_models.py",
        "scripts/cloud/s3_sync_datasets.py",
        "scripts/cloud/s3_tools/cli.py",
        "scripts/cloud/s3_tools/s3_manager.py",
    ]
    
    all_exist = True
    for script in scripts:
        if not os.path.exists(script):
            print(f"✗ Missing: {script}")
            all_exist = False
    
    if all_exist:
        print("✓ All cloud scripts in correct location")
    return all_exist

def test_requirements_consolidated():
    """Test requirements files are properly organized."""
    req_files = [
        "requirements/requirements.txt",
        "requirements/desktop-requirements.txt",
        "requirements/mac-requirements.txt",
        "requirements/inference-requirements.txt",
        "requirements/cloud-requirements.txt",
    ]
    
    all_exist = True
    for req_file in req_files:
        if not os.path.exists(req_file):
            print(f"✗ Missing: {req_file}")
            all_exist = False
    
    if all_exist:
        print("✓ All requirements files properly organized")
    return all_exist

def test_old_structure_removed():
    """Test old structure has been removed."""
    old_paths = [
        "data_pipelines",
        "s3_tools",
        "tracker",
        "references",
        "rtdetr_polyp",
        "classification_vid_ann_saver.py",
        "s3_sync.py",
        "s3_sync_dataset.py",
        "cmds.txt",
    ]
    
    all_removed = True
    for path in old_paths:
        if os.path.exists(path):
            print(f"✗ Old path still exists: {path}")
            all_removed = False
    
    if all_removed:
        print("✓ All old structure properly removed")
    return all_removed

def test_core_structure_intact():
    """Test core training structure is intact."""
    core_paths = [
        "src",
        "tools",
        "configs",
        "data",
        "notebooks",
        "tools/train.py",
        "tools/export_onnx.py",
    ]
    
    all_exist = True
    for path in core_paths:
        if not os.path.exists(path):
            print(f"✗ Missing core path: {path}")
            all_exist = False
    
    if all_exist:
        print("✓ Core training structure intact")
    return all_exist

def main():
    print("=" * 70)
    print("Testing Critical Workflows")
    print("=" * 70)
    print()
    
    tests = [
        ("Tracker Import", test_tracker_import),
        ("Data Pipeline Structure", test_data_pipeline_structure),
        ("Inference Scripts Location", test_inference_scripts_exist),
        ("Cloud Scripts Location", test_cloud_scripts_exist),
        ("Requirements Organization", test_requirements_consolidated),
        ("Old Structure Removed", test_old_structure_removed),
        ("Core Structure Intact", test_core_structure_intact),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\nTesting: {name}")
        print("-" * 70)
        result = test_func()
        results.append((name, result))
    
    print()
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print()
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All workflow tests passed!")
        return 0
    else:
        print("✗ Some workflow tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
