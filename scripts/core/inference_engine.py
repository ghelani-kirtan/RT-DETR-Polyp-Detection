"""Inference engine for ONNX model."""

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


class InferenceEngine:
    """Handles model loading, preprocessing, and inference."""
    
    def __init__(self, model_config: dict, performance_config: dict):
        """
        Initialize inference engine.
        
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
            print(f"✓ ONNX Runtime using: {self.session.get_providers()[0]}")
        except Exception as e:
            print(f"✗ Fatal error initializing ONNX session: {e}")
            sys.exit(1)
        
        # Setup transforms
        self.transforms = T.Compose([
            T.Resize(self.input_size),
            T.ToTensor()
        ])
    
    def preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Preprocess frame for inference.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Tuple of (preprocessed_data, original_size, preprocessing_time_ms)
        """
        pre_start = time.perf_counter()
        
        # Convert to PIL and get original size
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        orig_w, orig_h = pil_image.size
        orig_size = np.array([[orig_w, orig_h]], dtype=np.int64)
        
        # Apply transforms
        img_data = self.transforms(pil_image)[None].numpy()
        
        pre_time = (time.perf_counter() - pre_start) * 1000
        
        return img_data, orig_size, pre_time
    
    def run_inference(
        self,
        frame: np.ndarray
    ) -> Tuple[List[np.ndarray], float, float]:
        """
        Run inference on a frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Tuple of (outputs, preprocessing_time_ms, inference_time_ms)
        """
        # Preprocess
        img_data, orig_size, pre_time = self.preprocess(frame)
        
        # Inference
        inf_start = time.perf_counter()
        try:
            # Try common input names
            outputs = self.session.run(
                None,
                {"images": img_data, "orig_target_sizes": orig_size}
            )
        except Exception:
            try:
                # Fallback input names
                outputs = self.session.run(
                    None,
                    {"image": img_data, "orig_size": orig_size}
                )
            except Exception as e:
                print(f"✗ Inference failed: {e}")
                # Return empty outputs
                outputs = [
                    np.empty((1, 0)),
                    np.empty((1, 0, 4)),
                    np.empty((1, 0))
                ]
        
        inf_time = (time.perf_counter() - inf_start) * 1000
        
        return outputs, pre_time, inf_time
    
    def cleanup(self):
        """Release resources."""
        del self.session
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("✓ Inference engine cleaned up")
