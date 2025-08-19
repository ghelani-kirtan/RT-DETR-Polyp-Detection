import onnxruntime as ort
import cv2
import numpy as np
import time
import sys

# Constants
MODEL_PATH = 'polyp_detector_ep1.onnx'
INPUT_SIZE = 640
CLASS_NAMES = ['polyp']  # Assuming single class for polyps
SCORE_THRESHOLD = 0.1
CAP_DEVICE = 2  # Default to 0 for webcam; adjust as needed

def preprocess(frame):
    """Preprocess frame for model input (optimized like reference: direct resize, no letterbox or normalization)."""
    start_time = time.perf_counter()
    
    # Convert BGR to RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Get original size
    orig_h, orig_w = img.shape[:2]
    orig_size = np.array([[orig_h, orig_w]], dtype=np.int64)  # int64 as expected by model
    
    # Direct resize to 640x640 (as in reference)
    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    
    # Normalize to [0,1] (as in reference ToTensor)
    img = img.astype(np.float32) / 255.0
    
    # Transpose to CHW and add batch dimension
    img = img.transpose(2, 0, 1)[None]  # [1, 3, H, W]
    
    end_time = time.perf_counter()
    pre_time = (end_time - start_time) * 1000
    
    return img, orig_size, pre_time

def draw(frame, labels, boxes, scores, thrh=SCORE_THRESHOLD):
    """Draw bounding boxes and labels on the frame (adapted from reference: simple rectangle and label text)."""
    start_time = time.perf_counter()
    
    h, w = frame.shape[:2]
    scr = scores[0]  # Flatten for filtering
    lab = labels[0][scr > thrh]
    box = boxes[0][scr > thrh]
    
    for l, b in zip(lab, box):
        class_name = CLASS_NAMES[int(l)] if int(l) < len(CLASS_NAMES) else 'unknown'
        x1, y1, x2, y2 = map(int, b)
        # Clip boxes to image bounds
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    end_time = time.perf_counter()
    post_time = (end_time - start_time) * 1000
    
    return frame, post_time

def main():
    # Load ONNX model with preferred providers
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(MODEL_PATH, providers=providers)
    print(f"Using provider: {session.get_providers()[0]}")  # Show which provider is used
    
    # Open webcam
    cap = cv2.VideoCapture(CAP_DEVICE)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    prev_time = time.perf_counter()  # For FPS calculation
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # Preprocess
        input_img, orig_size, pre_time = preprocess(frame)
        
        # Inference
        start_time = time.perf_counter()
        outputs = session.run(None, {'images': input_img, 'orig_target_sizes': orig_size})
        end_time = time.perf_counter()
        inf_time = (end_time - start_time) * 1000
        
        labels, boxes, scores = outputs  # Outputs: labels (1,300), boxes (1,300,4) xyxy, scores (1,300)
        
        # Draw on frame
        frame, post_time = draw(frame, labels, boxes, scores)
        
        # Calculate FPS
        curr_time = time.perf_counter()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        # Print single line log with all timings
        sys.stdout.write(f"\rPre: {pre_time:.2f} ms | Inf: {inf_time:.2f} ms | Post: {post_time:.2f} ms | FPS: {fps:.2f}")
        sys.stdout.flush()
        
        # Display
        cv2.imshow('Polyp Detection', frame)
        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()