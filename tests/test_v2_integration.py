#!/usr/bin/env python3
"""Quick test to verify DetectionPreparerV2 integration."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test that all imports work."""
    print("Testing imports...")
    
    try:
        from data_pipelines.preparers import DetectionPreparerV2
        print("  [OK] DetectionPreparerV2 import successful")
    except ImportError as e:
        print(f"  [FAIL] Failed to import DetectionPreparerV2: {e}")
        return False
    
    try:
        from data_pipelines.pipelines import DetectionPipeline
        print("  [OK] DetectionPipeline import successful")
    except ImportError as e:
        print(f"  [FAIL] Failed to import DetectionPipeline: {e}")
        return False
    
    try:
        from data_pipelines.core import PreparerConfig
        print("  [OK] PreparerConfig import successful")
    except ImportError as e:
        print(f"  [FAIL] Failed to import PreparerConfig: {e}")
        return False
    
    return True


def test_preparer_initialization():
    """Test that V2 preparer can be initialized."""
    print("\nTesting V2 preparer initialization...")
    
    try:
        from data_pipelines.preparers import DetectionPreparerV2
        from data_pipelines.core import PreparerConfig
        
        config = PreparerConfig(
            input_dir=Path("test_input"),
            output_dir=Path("test_output"),
            categories=[{"id": 1, "name": "polyp", "supercategory": "none"}]
        )
        
        config.min_area_percentage = 0.0005
        config.use_dynamic_threshold = True
        config.remove_subset_boxes = True
        config.subset_threshold = 0.85
        config.parallel_workers = 8
        
        preparer = DetectionPreparerV2(config)
        print("  [OK] V2 preparer initialized successfully")
        
        assert preparer.min_area_percentage == 0.0005
        assert preparer.use_dynamic_threshold == True
        assert preparer.remove_subset_boxes == True
        assert preparer.subset_threshold == 0.85
        assert preparer.parallel_workers == 8
        print("  [OK] V2 configuration attributes set correctly")
        
        return True
    except Exception as e:
        print(f"  [FAIL] Failed to initialize V2 preparer: {e}")
        return False


def test_pipeline_integration():
    """Test that pipeline can use V2."""
    print("\nTesting pipeline integration...")
    
    try:
        from data_pipelines.pipelines import DetectionPipeline
        
        pipeline_v1 = DetectionPipeline(
            base_dir=Path("test_dir"),
            use_enhanced_preparer=False
        )
        assert pipeline_v1.use_enhanced_preparer == False
        print("  [OK] Pipeline V1 mode works")
        
        pipeline_v2 = DetectionPipeline(
            base_dir=Path("test_dir"),
            use_enhanced_preparer=True,
            min_area_percentage=0.0005,
            remove_subset_boxes=True,
            subset_threshold=0.85,
            parallel_workers=8
        )
        assert pipeline_v2.use_enhanced_preparer == True
        assert pipeline_v2.preparer_config.min_area_percentage == 0.0005
        assert pipeline_v2.preparer_config.remove_subset_boxes == True
        print("  [OK] Pipeline V2 mode works")
        
        return True
    except Exception as e:
        print(f"  [FAIL] Failed pipeline integration test: {e}")
        return False


def test_subset_removal_logic():
    """Test subset removal logic."""
    print("\nTesting subset removal logic...")
    
    try:
        from data_pipelines.preparers import DetectionPreparerV2
        from data_pipelines.core import PreparerConfig
        
        config = PreparerConfig(
            input_dir=Path("test_input"),
            output_dir=Path("test_output"),
            categories=[{"id": 1, "name": "polyp", "supercategory": "none"}]
        )
        config.remove_subset_boxes = True
        config.subset_threshold = 0.85
        
        preparer = DetectionPreparerV2(config)
        
        # Test case 1: Box inside another
        boxes = [
            {'bbox': [10, 10, 100, 100], 'area': 10000},
            {'bbox': [15, 15, 80, 80], 'area': 6400}
        ]
        
        filtered, removed = preparer.filter_subset_boxes(boxes)
        assert len(filtered) == 1
        assert removed == 1
        print("  [OK] Subset removal works correctly")
        
        # Test case 2: Non-overlapping boxes
        boxes = [
            {'bbox': [10, 10, 50, 50], 'area': 2500},
            {'bbox': [100, 100, 50, 50], 'area': 2500}
        ]
        
        filtered, removed = preparer.filter_subset_boxes(boxes)
        assert len(filtered) == 2
        assert removed == 0
        print("  [OK] Non-overlapping boxes preserved")
        
        return True
    except Exception as e:
        print(f"  [FAIL] Failed subset removal test: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("DetectionPreparerV2 Integration Tests")
    print("=" * 70)
    
    tests = [
        ("Imports", test_imports),
        ("Preparer Initialization", test_preparer_initialization),
        ("Pipeline Integration", test_pipeline_integration),
        ("Subset Removal Logic", test_subset_removal_logic),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n[FAIL] Test '{name}' crashed: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status}: {name}")
    
    print("-" * 70)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\nAll tests passed! V2 integration is working correctly.")
        return 0
    else:
        print(f"\n{total - passed} test(s) failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
