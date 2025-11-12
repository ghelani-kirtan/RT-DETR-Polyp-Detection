#!/usr/bin/env python3
"""
Test inference system without GUI.
Tests all components individually.
"""

import sys
import numpy as np
import cv2
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import load_config, InferenceEngine, VideoProcessor, Visualizer
from core.inference_engine_highres import InferenceEngineHighRes


def test_config_loading():
    """Test 1: Configuration loading."""
    print("=" * 60)
    print("TEST 1: Configuration Loading")
    print("=" * 60)
    
    configs = [
        'config.yaml',
        'config_detection.yaml',
        'config_highres.yaml',
        'config_no_tracker.yaml'
    ]
    
    for cfg in configs:
        try:
            config = load_config(cfg)
            print(f"✓ {cfg}: OK")
        except Exception as e:
            print(f"✗ {cfg}: FAILED - {e}")
            return False
    
    print()
    return True


def test_inference_engine():
    """Test 2: Inference engine initialization."""
    print("=" * 60)
    print("TEST 2: Inference Engine")
    print("=" * 60)
    
    try:
        config = load_config('config.yaml')
        model_config = config['model']
        performance_config = config.get('performance', {})
        
        # Check if model exists
        model_path = Path(model_config['path'])
        if not model_path.exists():
            print(f"✗ Model not found: {model_path}")
            print(f"  Please update model path in config.yaml")
            return False
        
        print(f"✓ Model found: {model_path}")
        
        # Initialize engine
        engine = InferenceEngine(model_config, performance_config)
        print(f"✓ Inference engine initialized")
        
        # Test with dummy frame
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        outputs, pre_time, inf_time = engine.run_inference(dummy_frame)
        print(f"✓ Inference successful")
        print(f"  Pre-processing: {pre_time:.2f}ms")
        print(f"  Inference: {inf_time:.2f}ms")
        
        engine.cleanup()
        print(f"✓ Cleanup successful")
        
    except Exception as e:
        print(f"✗ Inference engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    return True


def test_high_res_engine():
    """Test 3: High-resolution inference engine."""
    print("=" * 60)
    print("TEST 3: High-Resolution Engine")
    print("=" * 60)
    
    try:
        config = load_config('config_highres.yaml')
        model_config = config['model']
        performance_config = config.get('performance', {})
        
        # Check if model exists
        model_path = Path(model_config['path'])
        if not model_path.exists():
            print(f"✗ Model not found: {model_path}")
            return False
        
        # Initialize high-res engine
        engine = InferenceEngineHighRes(model_config, performance_config)
        print(f"✓ High-res engine initialized")
        
        # Test with dummy frame
        dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        outputs, pre_time, inf_time, scale, pads = engine.run_inference(dummy_frame)
        print(f"✓ High-res inference successful")
        print(f"  Pre-processing: {pre_time:.2f}ms")
        print(f"  Inference: {inf_time:.2f}ms")
        print(f"  Scale: {scale:.3f}, Pads: {pads}")
        
        engine.cleanup()
        print(f"✓ Cleanup successful")
        
    except Exception as e:
        print(f"✗ High-res engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    return True


def test_video_processor():
    """Test 4: Video processor."""
    print("=" * 60)
    print("TEST 4: Video Processor")
    print("=" * 60)
    
    try:
        config = load_config('config.yaml')
        processor = VideoProcessor(config)
        print(f"✓ Video processor initialized")
        print(f"  Tracker: {'enabled' if processor.tracker_enabled else 'disabled'}")
        
        # Test with dummy outputs
        dummy_outputs = [
            np.array([[0, 1]]),  # labels
            np.array([[[100, 100, 200, 200], [300, 300, 400, 400]]]),  # boxes
            np.array([[0.9, 0.8]])  # scores
        ]
        
        tracked_objects, post_time = processor.process_outputs(dummy_outputs)
        print(f"✓ Processing successful")
        print(f"  Post-processing: {post_time:.2f}ms")
        print(f"  Tracked objects: {tracked_objects.shape[0]}")
        
    except Exception as e:
        print(f"✗ Video processor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    return True


def test_visualizer():
    """Test 5: Visualizer."""
    print("=" * 60)
    print("TEST 5: Visualizer")
    print("=" * 60)
    
    try:
        config = load_config('config.yaml')
        visualizer = Visualizer(config)
        print(f"✓ Visualizer initialized")
        print(f"  Classes: {visualizer.class_names}")
        print(f"  Smoothing: {'enabled' if visualizer.smoothing_enabled else 'disabled'}")
        print(f"  Trails: {'enabled' if visualizer.show_trails else 'disabled'}")
        
        # Test with dummy frame and detections
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_tracks = np.array([
            [100, 100, 200, 200, 1, 0.9, 0],  # x1, y1, x2, y2, track_id, score, class_id
            [300, 300, 400, 400, 2, 0.85, 1]
        ])
        
        result_frame = visualizer.draw_detections(dummy_frame, dummy_tracks)
        print(f"✓ Visualization successful")
        print(f"  Output shape: {result_frame.shape}")
        
        # Test latency overlay
        result_frame = visualizer.add_latency_overlay(result_frame, 10.0, 20.0, 5.0)
        print(f"✓ Latency overlay successful")
        
    except Exception as e:
        print(f"✗ Visualizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    return True


def test_video_source():
    """Test 6: Video source access."""
    print("=" * 60)
    print("TEST 6: Video Source")
    print("=" * 60)
    
    try:
        config = load_config('config.yaml')
        video_source = config['video']['source']
        
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"✗ Cannot open video source: {video_source}")
            print(f"  Try changing 'source' in config.yaml")
            print(f"  Options: 0 (webcam), 1, 2 (USB cameras), or video file path")
            return False
        
        ret, frame = cap.read()
        if not ret:
            print(f"✗ Cannot read frame from source: {video_source}")
            cap.release()
            return False
        
        print(f"✓ Video source accessible: {video_source}")
        print(f"  Resolution: {frame.shape[1]}x{frame.shape[0]}")
        print(f"  FPS: {cap.get(cv2.CAP_PROP_FPS)}")
        
        cap.release()
        
    except Exception as e:
        print(f"✗ Video source test failed: {e}")
        return False
    
    print()
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("INFERENCE SYSTEM TEST SUITE")
    print("=" * 60 + "\n")
    
    tests = [
        ("Configuration Loading", test_config_loading),
        ("Inference Engine", test_inference_engine),
        ("High-Res Engine", test_high_res_engine),
        ("Video Processor", test_video_processor),
        ("Visualizer", test_visualizer),
        ("Video Source", test_video_source),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ {name} crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Ready to run inference.")
        print("\nNext step:")
        print("  python run_inference.py --config config.yaml")
    else:
        print("\n⚠️  Some tests failed. Please fix issues before running inference.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
