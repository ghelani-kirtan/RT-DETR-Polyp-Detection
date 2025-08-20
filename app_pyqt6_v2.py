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

# --- CONFIGURATIONS ---
# These can be adjusted based on the model and setup
MODEL_PATH = 'output/rtdetr_r18vd_6x_coco/polyp_detector.onnx'  
INPUT_SIZE = (640, 640)  # Model input size [height, width]
CLASS_NAMES = ['polyp']  # List of class names for the model
SCORE_THRESHOLD = 0.85   # Confidence threshold for displaying detections
CAP_DEVICE = 2           # Webcam device ID (0 for default) or path to a video file 'path/to/video.mp4'
SHOW_LATENCY = True      # Toggle to show latency overlay on the detected feed
# ---------------------------------------------------------------------------


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


def draw_detections(frame, labels, boxes, scores):
    """
    Draws bounding boxes, labels, and scores on the frame for detections above threshold.
    Supports multiple classes via CLASS_NAMES.
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
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label background and text
        label_text = f"{class_name}: {score:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x1, y1 - text_height - baseline), (x1 + text_width, y1), (0, 255, 0), -1)
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
        if not self.cap.isOpened():
            print(f"Error: Could not open video source: {CAP_DEVICE}")
            self.running = False
        else:
            self.running = True
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
                
                # Add latency overlay if enabled
                if SHOW_LATENCY:
                    cv2.putText(detected_frame, f"Latency: {overall_latency:.2f} ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
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
        self.setWindowTitle("Real-Time Object Detection")
        self.setGeometry(100, 100, 1600, 800)

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
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    inference_engine = InferenceEngine(MODEL_PATH)
    
    window = MainWindow(inference_engine)
    window.show()
    
    sys.exit(app.exec())