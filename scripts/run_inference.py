#!/usr/bin/env python3
"""
Main inference script for polyp detection/classification.

Usage (from project root):
    python scripts/run_inference.py --config scripts/config.yaml
    python scripts/run_inference.py --config scripts/config_detection.yaml
    python scripts/run_inference.py --config scripts/config_highres.yaml

Usage (from scripts/ directory):
    python run_inference.py --config config.yaml
    python run_inference.py --config config_detection.yaml
    python run_inference.py --config config_highres.yaml
"""

import sys
import argparse
import cv2
import numpy as np
from pathlib import Path
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QHBoxLayout, QWidget
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QImage, QPixmap

# Add scripts directory to path (works from both project root and scripts/)
script_dir = Path(__file__).parent
if script_dir.name == 'scripts':
    # Running from scripts/ directory
    sys.path.insert(0, str(script_dir))
else:
    # Running from elsewhere, assume scripts/ is in current directory
    sys.path.insert(0, 'scripts')

# Also add parent directory for tracker module
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import load_config, InferenceEngine, VideoProcessor, Visualizer
from core.inference_engine_highres import InferenceEngineHighRes


class VideoThread(QThread):
    """Thread for video capture and inference."""
    
    original_frame_signal = pyqtSignal(np.ndarray)
    detected_frame_signal = pyqtSignal(np.ndarray)
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.running = True
        
        # Initialize components
        video_config = config['video']
        model_config = config['model']
        performance_config = config.get('performance', {})
        
        # Choose inference engine based on high_resolution flag
        self.high_res_mode = video_config.get('high_resolution', False)
        if self.high_res_mode:
            self.inference_engine = InferenceEngineHighRes(
                model_config,
                performance_config
            )
            print("✓ Using high-resolution mode")
        else:
            self.inference_engine = InferenceEngine(
                model_config,
                performance_config
            )
            print("✓ Using standard mode")
        
        self.video_processor = VideoProcessor(config)
        self.visualizer = Visualizer(config)
        
        # Video source
        self.cap_device = video_config['source']
        self.show_original = config['visualization'].get('show_original', False)
    
    def run(self):
        """Main video processing loop."""
        cap = cv2.VideoCapture(self.cap_device)
        if not cap.isOpened():
            print(f"✗ Error: Could not open video source: {self.cap_device}")
            self.running = False
            return
        
        print(f"✓ Video source opened: {self.cap_device}")
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("✗ End of video stream or cannot read frame")
                break
            
            # Emit original frame if needed
            if self.show_original:
                self.original_frame_signal.emit(frame.copy())
            
            # Run inference
            if self.high_res_mode:
                outputs, pre_time, inf_time, scale, pads = \
                    self.inference_engine.run_inference(frame)
                
                # Process with high-res scaling
                tracked_objects, post_time = self.video_processor.process_outputs(
                    outputs,
                    orig_frame_shape=frame.shape[:2],
                    scale=scale,
                    pads=pads
                )
            else:
                outputs, pre_time, inf_time = \
                    self.inference_engine.run_inference(frame)
                
                # Process normally
                tracked_objects, post_time = self.video_processor.process_outputs(
                    outputs
                )
            
            # Visualize
            detected_frame = self.visualizer.draw_detections(
                frame.copy(),
                tracked_objects
            )
            detected_frame = self.visualizer.add_latency_overlay(
                detected_frame,
                pre_time,
                inf_time,
                post_time
            )
            
            self.detected_frame_signal.emit(detected_frame)
        
        cap.release()
        print("✓ Video thread stopped")
    
    def stop(self):
        """Stop the video thread."""
        self.running = False
        self.wait()
        self.inference_engine.cleanup()


class MainWindow(QMainWindow):
    """Main UI window."""
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        
        # Window configuration
        viz_config = config['visualization']
        window_config = viz_config.get('window', {})
        self.setWindowTitle(window_config.get('title', 'Polyp Detection'))
        
        # Get video resolution for window size
        video_config = config['video']
        cap = cv2.VideoCapture(video_config['source'])
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Adjust window size based on show_original flag
            if viz_config.get('show_original', False):
                self.setGeometry(100, 100, width * 2, height)
            else:
                self.setGeometry(100, 100, width, height)
            cap.release()
        else:
            # Fallback size
            w = window_config.get('width', 800)
            h = window_config.get('height', 600)
            self.setGeometry(100, 100, w, h)
        
        # Setup UI
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)
        
        # Original feed (optional)
        if viz_config.get('show_original', False):
            self.original_label = QLabel("<h2>Original Feed</h2>")
            self.original_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(self.original_label)
        else:
            self.original_label = None
        
        # Detected feed
        self.detected_label = QLabel("<h2>Processed Feed</h2>")
        self.detected_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.detected_label)
        
        # Start video thread
        self.video_thread = VideoThread(config)
        if self.original_label:
            self.video_thread.original_frame_signal.connect(self.update_original_frame)
        self.video_thread.detected_frame_signal.connect(self.update_detected_frame)
        self.video_thread.start()
    
    def update_frame(self, label: QLabel, frame: np.ndarray):
        """Update a label with a frame."""
        try:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(
                rgb_image.data, w, h, bytes_per_line,
                QImage.Format.Format_RGB888
            )
            pixmap = QPixmap.fromImage(qt_image).scaled(
                label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            label.setPixmap(pixmap)
        except Exception as e:
            print(f"✗ UI update error: {e}")
    
    def update_original_frame(self, frame: np.ndarray):
        """Update original frame display."""
        if self.original_label:
            self.update_frame(self.original_label, frame)
    
    def update_detected_frame(self, frame: np.ndarray):
        """Update detected frame display."""
        self.update_frame(self.detected_label, frame)
    
    def closeEvent(self, event):
        """Handle window close event."""
        print("✓ Closing application...")
        self.video_thread.stop()
        event.accept()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run polyp detection/classification inference"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
        print(f"✓ Configuration loaded from: {args.config}")
    except Exception as e:
        print(f"✗ Error loading configuration: {e}")
        sys.exit(1)
    
    # Start application
    app = QApplication(sys.argv)
    window = MainWindow(config)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
