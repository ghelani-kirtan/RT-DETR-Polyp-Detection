import sys
import time
import numpy as np
import onnxruntime as ort
import cv2
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QHBoxLayout, QWidget, QPushButton, QFileDialog
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QImage, QPixmap
import torch

# Constants for Configuration (Edit these as needed)
MODEL_TYPE = 'onnx'  # 'pth' for PyTorch .pth file, 'onnx' for ONNX file
MODEL_PATH = 'polyp_detector_ep1.onnx' if MODEL_TYPE == 'onnx' else 'checkpoint.pth'  # Path to model file
INPUT_SIZE = 640
CLASS_NAMES = ['polyp']
SCORE_THRESHOLD = 0.1
CAP_DEVICE = 2  # Webcam device ID (0 for default)
USE_GPU = torch.cuda.is_available()  # Auto-detect GPU for PyTorch; for ONNX, handled in providers
MEAN = [0.485, 0.456, 0.406]  # For PyTorch normalization if needed
STD = [0.229, 0.224, 0.225]

# Modular Inference Functions
class ModelLoader:
    def __init__(self, model_type, model_path):
        self.model_type = model_type
        self.model = None
        self.session = None
        self.load_model(model_path)

    def load_model(self, model_path):
        if self.model_type == 'pth':
            # Load PyTorch model (assume RT-DETRv2 deploy mode)
            from src.zoo.rtdetr import RTDETR  # Assume repo imports; adjust path if needed
            self.model = RTDETR.from_pretrained(model_path, deploy=True)  # Pseudo-code; use actual loading
            if USE_GPU:
                self.model.cuda()
            self.model.eval()
        elif self.model_type == 'onnx':
            providers = ['CUDAExecutionProvider' if USE_GPU else 'CPUExecutionProvider']
            self.session = ort.InferenceSession(model_path, providers=providers)
        else:
            raise ValueError("Unsupported model type. Use 'pth' or 'onnx'.")

    def infer(self, input_img, orig_size):
        if self.model_type == 'pth':
            with torch.no_grad():
                outputs = self.model(input_img)  # Assume model returns (labels, boxes, scores)
            return outputs  # Adjust to unpack labels, boxes, scores
        elif self.model_type == 'onnx':
            outputs = self.session.run(None, {'images': input_img, 'orig_target_sizes': orig_size})
            return outputs  # labels, boxes, scores

def preprocess(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img.shape[:2]
    orig_size = np.array([[orig_h, orig_w]], dtype=np.int64)
    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)[None]  # [1, 3, H, W]
    return img, orig_size

def draw_detections(frame, labels, boxes, scores):
    h, w = frame.shape[:2]
    scr = scores[0]
    lab = labels[0][scr > SCORE_THRESHOLD]
    box = boxes[0][scr > SCORE_THRESHOLD]
    for l, b in zip(lab, box):
        class_name = CLASS_NAMES[int(l)] if int(l) < len(CLASS_NAMES) else 'unknown'
        x1, y1, x2, y2 = map(int, b)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

# Thread for Video Capture and Inference
class VideoThread(QThread):
    original_frame_signal = pyqtSignal(np.ndarray)
    detected_frame_signal = pyqtSignal(np.ndarray)

    def __init__(self, model_loader):
        super().__init__()
        self.cap = cv2.VideoCapture(CAP_DEVICE)
        self.model_loader = model_loader
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                original_frame = frame.copy()
                self.original_frame_signal.emit(original_frame)

                input_img, orig_size = preprocess(frame)
                outputs = self.model_loader.infer(input_img, orig_size)
                labels, boxes, scores = outputs
                detected_frame = draw_detections(frame, labels, boxes, scores)
                self.detected_frame_signal.emit(detected_frame)
            time.sleep(0.01)  # Small sleep for threading efficiency

    def stop(self):
        self.running = False
        self.cap.release()

# PyQt6 UI Class
class MainWindow(QMainWindow):
    def __init__(self, model_loader):
        super().__init__()
        self.setWindowTitle("Polyp Detection UI")
        self.setGeometry(100, 100, 1280, 720)

        # Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)

        # Left: Original Video
        self.original_label = QLabel()
        self.original_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.original_label)

        # Right: Detected Video
        self.detected_label = QLabel()
        self.detected_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.detected_label)

        # Thread
        self.thread = VideoThread(model_loader)
        self.thread.original_frame_signal.connect(self.update_original)
        self.thread.detected_frame_signal.connect(self.update_detected)
        self.thread.start()

    def update_original(self, frame):
        self.update_label(self.original_label, frame)

    def update_detected(self, frame):
        self.update_label(self.detected_label, frame)

    def update_label(self, label, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qt_img).scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def closeEvent(self, event):
        self.thread.stop()
        super().closeEvent(event)

if __name__ == "__main__":
    # Load model first
    model_loader = ModelLoader(MODEL_TYPE, MODEL_PATH)

    app = QApplication(sys.argv)
    window = MainWindow(model_loader)
    window.show()
    sys.exit(app.exec())