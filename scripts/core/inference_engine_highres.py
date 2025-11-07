"""High-resolution inference engine with letterbox preprocessing."""

import sys
import time
import numpy as np
import onnxruntime as ort
import cv2
from PIL import Image
import torchvision.transforms as T
import torch
import gc
from typing import Tuple, List
from pathlib import Path


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """
    Resize and pad image while preserving aspect ratio.
    
    Args:
        img: Input image (numpy array)
        new_shape: Target size (height, width)
        color: Padding color (BGR)
        
    Returns:
        Tuple of (padded_img, scale_ratio, (pad_h, pad_w))
    """
    shape = img.shape[:2]  # [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    
    # Calculate scale ratio
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # (width, height)
    
    # Calculate padding
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    
    # Resize if needed
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    # Add padding
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=color
    )
    
    return img, r, (top, left)


class InferenceEngineHighRes:
    """Inference engine with letterbox preprocessing for high-resolution frames."""
    
    def __init__(self, model_config: dict, performance_config: dict):
        """
        Initialize high-resolution inference engine.
        
        Args:
            model_config: Model configuration dictionary
            performance_config: Performance configuration dictionary
        """
        self.model_path = model_config['path']
        self.input_size = tuple(model_config['input_size'])
        self.task = model_config.get('task', 'detection')
        
        # Validate model path
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Initialize ONNX session
        try:
            providers = performance_config.get('providers', [
                'CUDAExecutionProvider',
                'CPUExecutionProvider'
            ])
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            print(f"✓ High-Res ONNX Runtime using: {self.session.get_providers()[0]}")
        except Exception as e:
            print(f"✗ Fatal error initializing ONNX session: {e}")
            sys.exit(1)
        
        # Setup transforms (without resize, handled by letterbox)
        self.transforms = T.Compose([T.ToTensor()])
    
    def preprocess(
        self,
        frame: np.ndarray
    ) -> Tuple[np.ndarray, float, Tuple[int, int], float]:
        """
        Preprocess frame with letterbox for high-resolution.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Tuple of (preprocessed_data, scale_ratio, (pad_h, pad_w), preprocessing_time_ms)
        """
        pre_start = time.perf_counter()
        
        # Apply letterbox
        input_img, scale, (pad_h, pad_w) = letterbox(frame, self.input_size)
        
        # Convert to PIL and apply transforms
        pil_image = Image.fromarray(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))
        img_data = self.transforms(pil_image)[None].numpy()
        
        # Use input size as orig_target_sizes since we scale manually
        orig_size = np.array(
            [[self.input_size[1], self.input_size[0]]],
            dtype=np.int64
        )  # [width, height]
        
        pre_time = (time.perf_counter() - pre_start) * 1000
        
        return img_data, orig_size, scale, (pad_h, pad_w), pre_time
    
    def run_inference(
        self,
        frame: np.ndarray
    ) -> Tuple[List[np.ndarray], float, float, float, Tuple[int, int]]:
        """
        Run inference on a frame with letterbox preprocessing.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Tuple of (outputs, preprocessing_time_ms, inference_time_ms, scale, pads)
        """
        # Preprocess with letterbox
        img_data, orig_size, scale, pads, pre_time = self.preprocess(frame)
        
        # Inference
        inf_start = time.perf_counter()
        try:
            outputs = self.session.run(
                None,
                {"images": img_data, "orig_target_sizes": orig_size}
            )
        except Exception:
            try:
                outputs = self.session.run(
                    None,
                    {"image": img_data, "orig_size": orig_size}
                )
            except Exception as e:
                print(f"✗ Inference failed: {e}")
                outputs = [
                    np.empty((1, 0)),
                    np.empty((1, 0, 4)),
                    np.empty((1, 0))
                ]
        
        inf_time = (time.perf_counter() - inf_start) * 1000
        
        return outputs, pre_time, inf_time, scale, pads
    
    def cleanup(self):
        """Release resources."""
        del self.session
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("✓ High-Res inference engine cleaned up")
