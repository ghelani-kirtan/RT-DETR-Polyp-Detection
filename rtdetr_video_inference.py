"""Testing ONNX Script"""

import cv2
import torch
import torchvision.transforms as T
import numpy as np
import onnxruntime as ort
from PIL import Image
import time

# --- CONFIGURATION ---
ONNX_MODEL_PATH = 'output/rtdetr_r18vd_6x_coco/polyp_detector.onnx'  
# Use a string for the camera index, e.g., '0' or '1'
# Or provide a path to a video file, e.g., 'data/my_video.mp4'
VIDEO_SOURCE = '2' # <<< 
CONFIDENCE_THRESHOLD = 0.75  # Confidence threshold for filtering detections

# --- Model & Preprocessing Constants ---
MODEL_INPUT_SIZE = (640, 640)


class RTDETR_ONNX_GPU:
    """
    A class for performing real-time object detection with an RT-DETR ONNX model.
    This version implements the EXACT logic from the provided reference script.
    """

    def __init__(self, onnx_file, threshold=0.5):
        self.threshold = threshold
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        try:
            self.onnx_session = ort.InferenceSession(onnx_file, providers=providers)
            print(f"ONNX Runtime is using: {self.onnx_session.get_providers()}")
            if 'CUDAExecutionProvider' not in self.onnx_session.get_providers():
                print("Warning: CUDA is not available. Falling back to CPU.")
        except Exception as e:
            print(f"Error initializing ONNX session: {e}")
            exit()

        # Preprocessing transforms exactly as in the reference.
        self.transforms = T.Compose([
            T.Resize(MODEL_INPUT_SIZE),
            T.ToTensor(),
        ])

    def preprocess_and_infer(self, frame):
        """
        Preprocesses a frame and runs inference, exactly following the reference logic.
        The model's output boxes are already scaled to the original frame size.
        """
        # Convert frame to PIL Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        # Fetch original Image size.
        w, h = pil_image.size
        
        orig_size = torch.tensor([[w, h]])

        # Transformations
        img_data = self.transforms(pil_image)[None]

        # Inference:::::
        outputs = self.onnx_session.run(
            output_names=None,
            input_feed={'images': img_data.data.numpy(), "orig_target_sizes": orig_size.data.numpy()}
        )
        
        labels, boxes, scores = outputs
        return labels, boxes, scores

    def draw_detections(self, frame, labels, boxes, scores):
        """
        Draws the bounding boxes on the frame. The box coordinates from the model
        are used directly as they are already scaled to the original frame size.
        """
        # Squeeze the batch dimension
        scores_i = scores[0]
        labels_i = labels[0]
        boxes_i = boxes[0]

        # Filter detections based on the confidence threshold
        mask = scores_i > self.threshold
        filtered_labels = labels_i[mask]
        filtered_boxes = boxes_i[mask]
        filtered_scores = scores_i[mask]

        for label, box, score in zip(filtered_labels, filtered_boxes, filtered_scores):
            # Convert box coordinates to integers for drawing
            x1, y1, x2, y2 = map(int, box)
            
            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Prepare label text
            label_text = f"Polyp: {score:.2f}"
            
            # Put label text above the bounding box
            (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1 - text_height - baseline), (x1 + text_width, y1), (0, 255, 0), -1)
            cv2.putText(frame, label_text, (x1, y1 - baseline + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
        return frame

def main():
    """
    Main function to run the real-time inference.
    """
    detector = RTDETR_ONNX_GPU(ONNX_MODEL_PATH, threshold=CONFIDENCE_THRESHOLD)

    if isinstance(VIDEO_SOURCE, str) and VIDEO_SOURCE.isdigit():
        video_source = int(VIDEO_SOURCE)
    else:
        video_source = VIDEO_SOURCE
        
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"Error: Could not open video source: {VIDEO_SOURCE}")
        return

    prev_frame_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or cannot read frame.")
            break

        # Get detections
        labels, boxes, scores = detector.preprocess_and_infer(frame)

        # Draw detections on the frame
        frame_with_detections = detector.draw_detections(frame.copy(), labels, boxes, scores)
        
        # Calculate and display FPS
        new_frame_time = time.time()
        if prev_frame_time > 0:
            fps = 1 / (new_frame_time - prev_frame_time)
            cv2.putText(frame_with_detections, f"FPS: {int(fps)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        prev_frame_time = new_frame_time

        cv2.imshow('RT-DETR Polyp Detection (Corrected)', frame_with_detections)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()