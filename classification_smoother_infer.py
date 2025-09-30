# CLASSIFICATION - POLYP CLASSIFICATION [RT-DETR-S] Testing script with Tracking and Smoothing
import sys
import time
import numpy as np
import onnxruntime as ort
import cv2
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QImage, QPixmap
from PIL import Image
import torchvision.transforms as T
import torch
import gc
from collections import defaultdict, deque
from typing import Dict, Deque, Tuple, List

# LOCAL TRACKER MODULE IMPORT:
from tracker.byte_tracker import BYTETracker

#! --- CONFIGS/CONSTANTS ---
MODEL_PATH = "./output/rtdetr_r18vd_6x_classification_v1_2_1/polyp_classifier.onnx"
INPUT_SIZE = (640, 640)  # Model input size [height, width]

# update: More generic classes selection (implement if required):
CLASS_NAMES = ["adenoma", "hyperplastic"]
CLASS_COLORS = [(0, 0, 255), (0, 255, 0)]  # BGR format
NUM_CLASSES = len(CLASS_NAMES)


#! DETECTION & TRACKING THRESHOLDS:::::
SCORE_THRESHOLD = 0.70  # FINAL TRAKCS DISPLAY: Confidence threshold
TRACKER_INPUT_SCORE_THRESHOLD = 0.1  # Low threshold for detections fed into the tracker (used for GATING so keep low)

# Video source (camera index or path to video file) [keep as it was previous]
CAP_DEVICE = 2

# Overlay Toggles
SHOW_LATENCY_DETAILS = False
SHOW_OVERALL_LATENCY = True

#! Tracker Configurations (ByteTrack):::::
TRACKER_ARGS = {
    "track_thresh": 0.45,  # Min. score to start a new track
    "track_buffer": 30,  # Frames to keep a lost track
    "match_thresh": 0.75,  # IoU threshold for matching
    "mot20": False,  # Keep it False :)
}

#!
VIDEO_FRAME_RATE = 60  # Adjust to source FPS [can add dynamic FPS captureing here]


#! Class Smoothing Configurations (EMA) - To REDUCE Flickering between the classes
EMA_ALPHA = 0.60  # Higher alpha gives more weight to the new prediction
LOW_PROB_THRESHOLD = 0.80  # If max prob is below this, hold the previous class


# Tracking Trail Visualization configs - NOT IMP.
SHOW_TRAILS = True  # Flag to enable/disable tracking trails
TRAIL_MAXLEN = 40
TRAIL_BASE_COLOR = (255, 0, 0)  # BGR
TRAIL_FADING = True

#! ---------------------------------------------------------------------------


class InferenceEngine:
    def __init__(self, model_path: str):
        try:
            providers = ort.get_available_providers()
            if "CUDAExecutionProvider" in providers:
                self.session = ort.InferenceSession(
                    model_path, providers=["CUDAExecutionProvider"]
                )

            else:
                self.session = ort.InferenceSession(
                    model_path, providers=["CPUExecutionProvider"]
                )
            print(f"ONNX Runtime is using: {self.session.get_providers()[0]}")
        except Exception as e:
            print(f"Fatal Error initializing ONNX session: {e}")
            sys.exit(1)

        self.transforms = T.Compose([T.Resize(INPUT_SIZE), T.ToTensor()])

    def run_inference(self, frame: np.ndarray) -> Tuple[List[np.ndarray], float, float]:
        pre_start = time.perf_counter()
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        orig_w, orig_h = pil_image.size
        orig_size = np.array([[orig_w, orig_h]], dtype=np.int64)
        img_data = self.transforms(pil_image)[None].numpy()
        pre_time = (time.perf_counter() - pre_start) * 1000

        inf_start = time.perf_counter()
        try:
            outputs = self.session.run(
                None, {"images": img_data, "orig_target_sizes": orig_size}
            )
        except Exception:
            try:
                outputs = self.session.run(
                    None, {"image": img_data, "orig_size": orig_size}
                )
            except Exception as e_fallback:
                print(f"Inference failed with all known input name sets: {e_fallback}")
                return (
                    [np.empty((1, 0)), np.empty((1, 0, 4)), np.empty((1, 0))],
                    pre_time,
                    0.0,
                )

        inf_time = (time.perf_counter() - inf_start) * 1000

        return outputs, pre_time, inf_time

    def cleanup(self):
        del self.session
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def draw_visualizations(frame, tracked_objects, class_history, track_history):
    if tracked_objects.shape[0] == 0:
        return frame

    for track in tracked_objects:
        x1, y1, x2, y2, track_id, score, class_id = track
        track_id = int(track_id)
        class_id = int(class_id)

        current_probs = np.zeros(NUM_CLASSES)
        current_probs[class_id] = score

        if track_id in class_history and len(class_history[track_id]) > 0:
            prev_smoothed = class_history[track_id][-1]
            smoothed_probs = EMA_ALPHA * current_probs + (1 - EMA_ALPHA) * prev_smoothed
        else:
            smoothed_probs = current_probs
        class_history[track_id].append(smoothed_probs)

        smoothed_class_id = np.argmax(smoothed_probs)
        max_prob = np.max(smoothed_probs)

        if max_prob < LOW_PROB_THRESHOLD and len(class_history[track_id]) > 1:
            smoothed_class_id = np.argmax(class_history[track_id][-2])

        label_text = f"{CLASS_NAMES[smoothed_class_id]}: {score:.2f}"
        color = CLASS_COLORS[smoothed_class_id]

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        (text_w, text_h), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(
            frame,
            (int(x1), int(y1) - text_h - baseline),
            (int(x1) + text_w, int(y1)),
            color,
            -1,
        )
        cv2.putText(
            frame,
            label_text,
            (int(x1), int(y1) - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )

        # No Need to include:
        if SHOW_TRAILS:
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            track_history[track_id].append(center)

            points = list(track_history[track_id])
            for i in range(1, len(points)):
                if points[i - 1] is None or points[i] is None:
                    continue
                thickness = (
                    int(np.sqrt(TRAIL_MAXLEN / float(i + 1)) * 2.5)
                    if TRAIL_FADING
                    else 2
                )
                cv2.line(frame, points[i - 1], points[i], TRAIL_BASE_COLOR, thickness)

    return frame


class VideoThread(QThread):
    detected_frame_signal = pyqtSignal(np.ndarray)

    def __init__(self, inference_engine):
        super().__init__()
        self.inference_engine = inference_engine

        #! TRACKER INITIALIZATION:::::
        self.tracker = BYTETracker(TRACKER_ARGS, frame_rate=VIDEO_FRAME_RATE)
        self.class_history = defaultdict(lambda: deque(maxlen=10))
        self.track_history = defaultdict(lambda: deque(maxlen=TRAIL_MAXLEN))
        #! ------------------------------------------------------

        self.running = True

    def run(self):
        cap = cv2.VideoCapture(CAP_DEVICE)
        if not cap.isOpened():
            print(f"Error: Could not open video source: {CAP_DEVICE}")
            self.running = False
            return

        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream or cannot read frame.")
                break

            outputs, pre_time, inf_time = self.inference_engine.run_inference(frame)

            post_start = time.perf_counter()

            #! ------------------------------------------------------------------------------------------------
            #!  NEW POSTPROCESSING LAYER:
            labels_batch, boxes_batch, scores_batch = outputs

            # Functionality of the postprocessing have been moved here [partially]
            if scores_batch.shape[1] > 0:
                scores = scores_batch[0]
                labels = labels_batch[0]
                boxes = boxes_batch[0]

                mask = scores > TRACKER_INPUT_SCORE_THRESHOLD
                dets_to_track = np.column_stack(
                    [boxes[mask], scores[mask], labels[mask]]
                )
            else:
                dets_to_track = np.empty((0, 6))

            tracked_objects = self.tracker.update(dets_to_track)

            if tracked_objects.shape[0] > 0:
                display_tracks = tracked_objects[
                    tracked_objects[:, 5] > SCORE_THRESHOLD
                ]
            else:
                display_tracks = np.empty((0, 7))

            #! NEW: draw_detections -> draw_visualizations along with the trakcer and smoothening integrations
            detected_frame = draw_visualizations(
                frame, display_tracks, self.class_history, self.track_history
            )

            post_time = (time.perf_counter() - post_start) * 1000
            overall_latency = pre_time + inf_time + post_time

            #! ------------------------------------------------------------------------------------------------
            self.add_latency_overlays(
                detected_frame, pre_time, inf_time, post_time, overall_latency
            )
            self.detected_frame_signal.emit(detected_frame)

        cap.release()
        print("Video thread stopped.")

    def add_latency_overlays(self, frame, pre, inf, post, total):
        y_pos = 30
        if SHOW_OVERALL_LATENCY:
            cv2.putText(
                frame,
                f"Latency: {total:.1f} ms",
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            y_pos += 30
        if SHOW_LATENCY_DETAILS:
            cv2.putText(
                frame,
                f"Pre: {pre:.1f}ms Inf: {inf:.1f}ms Post: {post:.1f}ms",
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

    def stop(self):
        self.running = False
        self.wait()


class MainWindow(QMainWindow):
    def __init__(self, inference_engine):
        super().__init__()
        self.setWindowTitle("Real-Time Polyp Tracking and Classification")
        self.setGeometry(100, 100, 800, 600)

        self.inference_engine = inference_engine

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.detected_label = QLabel("<h2>Processed Feed</h2>")
        self.detected_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.detected_label)

        self.video_thread = VideoThread(inference_engine)
        self.video_thread.detected_frame_signal.connect(self.update_detected_frame)
        self.video_thread.start()

    def update_frame(self, label, frame):
        try:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(
                rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
            )
            pixmap = QPixmap.fromImage(qt_image).scaled(
                label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            label.setPixmap(pixmap)
        except Exception as e:
            print(f"UI update error: {e}")

    def update_detected_frame(self, frame):
        self.update_frame(self.detected_label, frame)

    def closeEvent(self, event):
        print("Closing application...")
        self.video_thread.stop()
        self.inference_engine.cleanup()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    engine = InferenceEngine(MODEL_PATH)
    main_window = MainWindow(engine)
    main_window.show()
    sys.exit(app.exec())
