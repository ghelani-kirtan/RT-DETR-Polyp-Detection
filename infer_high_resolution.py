# CLASSIFICATION - POLYP CLASSIFICATION [RT-DETR-S] Testing script with Tracking and Smoothing
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
from collections import defaultdict, deque
from tracker.byte_tracker import BYTETracker  # Import from local tracker module
from typing import Dict, Deque, Tuple, List
from queue import Queue, Full, Empty
import threading

#! --- CONFIGS/CONSTANTS (REFINED for Modularity) ---
MODEL_PATH = (
    "./output/rtdetr_r18vd_6x_classification_unlicensed/polyp_classifier_100.onnx"
)

# MODEL_PATH = "./output/rtdetr_r18vd_6x_classification_v1_2_1/polyp_classifier.onnx"
INPUT_SIZE = (640, 640)  # Model input size [height, width]

#! REFINED: Class configurations are now in lists for easy expansion
CLASS_NAMES = ["adenoma", "hyperplastic"]
CLASS_COLORS = [(0, 0, 255), (0, 255, 0)]  # BGR format for OpenCV
NUM_CLASSES = len(CLASS_NAMES)

# Detection and Tracking Thresholds
SCORE_THRESHOLD = 0.80  # Confidence threshold for displaying final tracks
TRACKER_INPUT_SCORE_THRESHOLD = (
    0.05  # Low threshold for detections fed into the tracker
)

# Video source (camera index or path to video file)
CAP_DEVICE = 2

# Overlay Toggles
SHOW_LATENCY_DETAILS = False  # Set to True to see Pre/Inf/Post times
SHOW_OVERALL_LATENCY = True

# Tracker Configurations (ByteTrack)
TRACKER_ARGS = {
    "track_thresh": 0.45,  # Min score to start a new track [Lower This -> If model is not catching polyps]
    "track_buffer": 60,  # Frames to keep a lost track
    "match_thresh": 0.60,  # IoU threshold for matching
    "mot20": False,
}
VIDEO_FRAME_RATE = 60  # Adjust to your video's FPS

# Class Smoothing Configurations (EMA)
EMA_ALPHA = 0.30  # Higher alpha gives more weight to the new prediction [lower this value to reduce the flickering]
LOW_PROB_THRESHOLD = 0.80  # If max prob is below this, hold the previous class

# Trail Visualization Configurations
TRAIL_MAXLEN = 40
TRAIL_BASE_COLOR = (255, 0, 0)  # BGR
TRAIL_FADING = True
#! ---------------------------------------------------------------------------


class InferenceEngine:
    """Handles model loading, preprocessing, and inference."""

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

        # Updated transforms without Resize, since we handle letterbox manually
        self.transforms = T.Compose([T.ToTensor()])

    def cleanup(self):
        """Release resources."""
        del self.session
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize and pad the image while preserving aspect ratio."""
    shape = img.shape[:2]  # [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # (width, height)
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return img, r, (top, left)  # padded img, scale ratio, pads (pad_h, pad_w)


def draw_visualizations(frame, tracked_objects, class_history, track_history):
    """Draws bounding boxes, smoothed labels, and trails on the frame."""
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

        label_status = ""
        if max_prob < LOW_PROB_THRESHOLD and len(class_history[track_id]) > 1:
            smoothed_class_id = np.argmax(class_history[track_id][-2])
            label_status = " (Held)"

        label_text = (
            f"ID:{track_id} {CLASS_NAMES[smoothed_class_id]}{label_status}: {score:.2f}"
        )
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

        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        track_history[track_id].append(center)

        points = list(track_history[track_id])
        for i in range(1, len(points)):
            if points[i - 1] is None or points[i] is None:
                continue
            thickness = (
                int(np.sqrt(TRAIL_MAXLEN / float(i + 1)) * 2.5) if TRAIL_FADING else 2
            )
            cv2.line(frame, points[i - 1], points[i], TRAIL_BASE_COLOR, thickness)

    return frame


class VideoThread(QThread):
    """Handles video capture, inference, and tracking in a separate thread."""

    detected_frame_signal = pyqtSignal(np.ndarray)

    def __init__(self, inference_engine):
        super().__init__()
        self.inference_engine = inference_engine
        self.class_history = defaultdict(lambda: deque(maxlen=10))
        self.track_history = defaultdict(lambda: deque(maxlen=TRAIL_MAXLEN))
        self.stop_event = threading.Event()
        self.frame_queue = Queue(maxsize=1)
        self.result_queue = Queue(maxsize=1)
        self.consumer_thread = None
        self.running = True  # Retained for compatibility

    def consumer_func(self):
        tracker = BYTETracker(TRACKER_ARGS, frame_rate=VIDEO_FRAME_RATE)
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.1)
            except Empty:
                continue

            orig_h, orig_w = frame.shape[:2]

            # Letterbox resize (preserve aspect ratio with padding)
            input_img, scale, (pad_h, pad_w) = letterbox(frame)

            pre_start = time.perf_counter()
            pil_image = Image.fromarray(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))
            img_data = self.inference_engine.transforms(pil_image)[None].numpy()
            # Pass input size as orig_target_sizes since we scale manually
            orig_size = np.array(
                [[INPUT_SIZE[1], INPUT_SIZE[0]]], dtype=np.int64
            )  # [width, height]
            pre_time = (time.perf_counter() - pre_start) * 1000

            inf_start = time.perf_counter()
            try:
                outputs = self.inference_engine.session.run(
                    None, {"images": img_data, "orig_target_sizes": orig_size}
                )
            except Exception:
                try:
                    outputs = self.inference_engine.session.run(
                        None, {"image": img_data, "orig_size": orig_size}
                    )
                except Exception as e_fallback:
                    print(
                        f"Inference failed with all known input name sets: {e_fallback}"
                    )
                    outputs = [
                        np.empty((1, 0)),
                        np.empty((1, 0, 4)),
                        np.empty((1, 0)),
                    ]
            inf_time = (time.perf_counter() - inf_start) * 1000

            post_start = time.perf_counter()
            labels_batch, boxes_batch, scores_batch = outputs

            if scores_batch.shape[1] > 0:
                scores = scores_batch[0]
                labels = labels_batch[0]
                boxes = boxes_batch[0]

                # Scale boxes back to original resolution (remove padding and scale)
                boxes[:, [0, 2]] -= pad_w
                boxes[:, [1, 3]] -= pad_h
                boxes /= scale
                boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_w)
                boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_h)
                boxes[:, 2] = np.clip(boxes[:, 2], 0, orig_w)
                boxes[:, 3] = np.clip(boxes[:, 3], 0, orig_h)

                mask = scores > TRACKER_INPUT_SCORE_THRESHOLD
                dets_to_track = np.column_stack(
                    [boxes[mask], scores[mask], labels[mask]]
                )
            else:
                dets_to_track = np.empty((0, 6))

            tracked_objects = tracker.update(dets_to_track)

            if tracked_objects.shape[0] > 0:
                display_tracks = tracked_objects[
                    tracked_objects[:, 5] > SCORE_THRESHOLD
                ]
            else:
                display_tracks = np.empty((0, 7))

            post_time = (time.perf_counter() - post_start) * 1000
            overall_latency = pre_time + inf_time + post_time

            results = {
                "display_tracks": display_tracks,
                "pre_time": pre_time,
                "inf_time": inf_time,
                "post_time": post_time,
                "overall_latency": overall_latency,
            }
            self.result_queue.put(results)

    def run(self):
        self.consumer_thread = threading.Thread(target=self.consumer_func)
        self.consumer_thread.start()

        cap = cv2.VideoCapture(CAP_DEVICE)
        if not cap.isOpened():
            print(f"Error: Could not open video source: {CAP_DEVICE}")
            self.running = False
            return

        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("End of video stream or cannot read frame.")
                break

            try:
                self.frame_queue.put(frame.copy(), block=False)
            except Full:
                continue  # Skip if queue full (consumer busy)

            results = self.result_queue.get()  # Blocking wait for results

            detected_frame = draw_visualizations(
                frame, results["display_tracks"], self.class_history, self.track_history
            )

            self.add_latency_overlays(
                detected_frame,
                results["pre_time"],
                results["inf_time"],
                results["post_time"],
                results["overall_latency"],
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
        self.stop_event.set()
        # Unblock queues if waiting
        try:
            self.frame_queue.put(None, block=False)
        except Full:
            pass
        if self.consumer_thread:
            self.consumer_thread.join()
        self.wait()


class MainWindow(QMainWindow):
    """Main UI window to display video feeds."""

    def __init__(self, inference_engine):
        super().__init__()
        self.setWindowTitle("Real-Time Polyp Tracking and Classification")

        # Get video resolution
        cap = cv2.VideoCapture(CAP_DEVICE)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.setGeometry(100, 100, width, height)
            cap.release()
        else:
            self.setGeometry(100, 100, 800, 600)  # Fallback size

        self.inference_engine = inference_engine

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)

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
