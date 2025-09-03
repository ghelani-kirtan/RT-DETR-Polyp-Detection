# CLASSIFICATION - POLYP CLASSIFICATION [RT-DETR-S] Testing script:
import sys
import time
import numpy as np
import onnxruntime as ort
import cv2
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QHBoxLayout, QWidget
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QImage, QPixmap
from PIL import Image
import torchvision.transforms as T
import torch
import gc

#! --- CONFIGS/CONSTANTS ---
MODEL_PATH = 'output/rtdetr_r18vd_6x_classification/polyp_classifier_e100_v1.onnx'  
INPUT_SIZE = (640, 640)  # Model input size [height, width]

# 
CLASS_NAMES = ['adenoma', 'hyperplastic']  # Classes: index 0=adenoma (id1), 1=hyperplastic (id2)
COLOR_ADENOMA = (0, 0, 255)
COLOR_HYPERPLASTIC = (0, 255, 0)
# 

SCORE_THRESHOLD = 0.75   # Confidence threshold for displaying detections
CAP_DEVICE = 2           # Capture device ID or path to a video file.

#* Toggle Buttons For overlays [prediction frame]
SHOW_PREPROCESSING_TIME = True   # Toggle to show preprocessing time overlay on the detected feed
SHOW_INFERENCE_TIME = True      # Toggle to show inference time overlay on the detected feed
SHOW_POSTPROCESSING_TIME = True # Toggle to show postprocessing time overlay on the detected feed
SHOW_OVERALL_LATENCY = True     # Toggle to show overall latency overlay on the detected feed
#! ---------------------------------------------------------------------------


#* New: Added GPU Cleanup Function:
class InferenceEngine:
    """
    Handles loading the ONNX model, image preprocessing, and running inference.
    This class abstracts the AI/ML-specific logic for easier integration.
    """
    def __init__(self, model_path):
        try:
            # Automatically select CUDA if available, fallback to CPU
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(model_path, providers=providers)
            print(f"ONNX Runtime is using: {self.session.get_providers()[0]}")
        except Exception as e:
            print(f"Error initializing ONNX session: {e}")
            sys.exit(1)
            
        # Image transformations for model input
        self.transforms = T.Compose([
            T.Resize(INPUT_SIZE),
            T.ToTensor(),
        ])

    def run_inference(self, frame: np.ndarray):
        """
        Preprocesses the input frame and runs model inference.
        Returns: outputs (list of labels, boxes, scores), pre_time (ms), inf_time (ms)
        """
        try:
            # Preprocessing time
            pre_start = time.time()
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            orig_w, orig_h = pil_image.size
            orig_size = np.array([[orig_w, orig_h]], dtype=np.int64)
            img_data = self.transforms(pil_image)[None].numpy()
            pre_time = (time.time() - pre_start) * 1000  # ms
            
            # Inference time
            inf_start = time.time()
            outputs = self.session.run(
                None, 
                {'images': img_data, 'orig_target_sizes': orig_size}
            )
            inf_time = (time.time() - inf_start) * 1000  # ms
            
            return outputs, pre_time, inf_time
        except Exception as e:
            print(f"Inference error: {e}")
            # Return empty outputs and zero times on error
            empty_array = np.array([[]])
            return [empty_array, empty_array, empty_array], 0.0, 0.0

    def cleanup(self):
        """
        Cleans up the ONNX session and flushes GPU resources if applicable.
        """
        self.session = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def draw_detections(frame, labels, boxes, scores):
    """
    Draws bounding boxes, labels, and scores on the frame for detections above threshold.
    - Adenoma (class 0): Red boxes
    - Hyperplastic (class 1): Green boxes
    """
    if len(scores) == 0:
        return frame  # No detections
    
    scores_i = scores[0]
    labels_i = labels[0]
    boxes_i = boxes[0]

    # Filter by confidence threshold
    mask = scores_i > SCORE_THRESHOLD
    filtered_labels = labels_i[mask]
    filtered_boxes = boxes_i[mask]
    filtered_scores = scores_i[mask]

    for label, box, score in zip(filtered_labels, filtered_boxes, filtered_scores):
        x1, y1, x2, y2 = map(int, box)
        class_idx = int(label)
        class_name = CLASS_NAMES[class_idx] if class_idx < len(CLASS_NAMES) else 'unknown'
        
        #! RED for adenoma (0), 
        #! GREEN for hyperplastic (1)
        color = COLOR_ADENOMA if class_idx == 0 else COLOR_HYPERPLASTIC  # BGR format
        
        #! Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background and text with score
        label_text = f"{class_name}: {score:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x1, y1 - text_height - baseline), (x1 + text_width, y1), color, -1)
        cv2.putText(frame, label_text, (x1, y1 - baseline + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
    return frame


class VideoThread(QThread):
    """
    Thread for video capture, inference, and signal emission.
    Runs in background to keep UI responsive.
    """
    original_frame_signal = pyqtSignal(np.ndarray)
    detected_frame_signal = pyqtSignal(np.ndarray)

    def __init__(self, inference_engine):
        super().__init__()
        self.cap = cv2.VideoCapture(CAP_DEVICE)
        self.running = True  # Added to fix AttributeError
        if not self.cap.isOpened():
            print(f"Error: Could not open video source: {CAP_DEVICE}")
            self.running = False
        
        self.inference_engine = inference_engine

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # Emit original frame
                self.original_frame_signal.emit(frame.copy())
                
                # Run inference
                outputs, pre_time, inf_time = self.inference_engine.run_inference(frame)
                
                # Post-processing time (drawing detections)
                post_start = time.time()
                detected_frame = draw_detections(frame, *outputs)
                post_time = (time.time() - post_start) * 1000  # ms
                
                # Calculate overall latency
                overall_latency = pre_time + inf_time + post_time
                
                # Log times to terminal
                print(f"Pre-process time: {pre_time:.2f} ms, Post-process time: {post_time:.2f} ms, Overall latency: {overall_latency:.2f} ms")
                
                # Add time overlays based on individual flags
                #* OVERLAY ADDITION BASED ON TOGGLES::::::
                y_pos = 30
                if SHOW_PREPROCESSING_TIME:
                    cv2.putText(detected_frame, f"Preprocessing: {pre_time:.2f} ms", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    y_pos += 30
                if SHOW_INFERENCE_TIME:
                    cv2.putText(detected_frame, f"Inference: {inf_time:.2f} ms", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    y_pos += 30
                if SHOW_POSTPROCESSING_TIME:
                    cv2.putText(detected_frame, f"Postprocessing: {post_time:.2f} ms", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    y_pos += 30
                if SHOW_OVERALL_LATENCY:
                    cv2.putText(detected_frame, f"Latency: {overall_latency:.2f} ms", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    y_pos += 30
                
                # Emit detected frame
                self.detected_frame_signal.emit(detected_frame)
            else:
                print("End of video stream or cannot read frame.")
                self.stop()

    def stop(self):
        self.running = False
        self.wait()
        if self.cap.isOpened():
            self.cap.release()
        print("Video thread stopped.")


class MainWindow(QMainWindow):
    """
    Main UI window displaying original and detected video feeds side-by-side.
    """
    def __init__(self, inference_engine):
        super().__init__()
        self.setWindowTitle("Real-Time Polyp Classification")
        self.setGeometry(100, 100, 1600, 800)

        self.inference_engine = inference_engine

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)

        self.original_label = QLabel("<h2>Original Feed</h2>")
        self.original_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.original_label)

        self.detected_label = QLabel("<h2>Detected Feed</h2>")
        self.detected_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.detected_label)

        self.thread = VideoThread(inference_engine)
        self.thread.original_frame_signal.connect(self.update_original_frame)
        self.thread.detected_frame_signal.connect(self.update_detected_frame)
        self.thread.start()

    def update_original_frame(self, frame):
        self._update_label(self.original_label, frame)

    def update_detected_frame(self, frame):
        self._update_label(self.detected_label, frame)

    def _update_label(self, label, frame):
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            
            pixmap = QPixmap.fromImage(qt_img)
            scaled_pixmap = pixmap.scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            label.setPixmap(scaled_pixmap)
        except Exception as e:
            print(f"UI update error: {e}")

    def closeEvent(self, event):
        self.thread.stop()
        self.inference_engine.cleanup()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    inference_engine = InferenceEngine(MODEL_PATH)
    
    window = MainWindow(inference_engine)
    window.show()
    
    sys.exit(app.exec())