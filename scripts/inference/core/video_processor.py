"""Video processing with tracking and postprocessing."""

import time
import numpy as np
import cv2
from typing import Dict, Tuple
import sys
from pathlib import Path

# Add parent directory to path for tracker import
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.tracker.byte_tracker import BYTETracker


class VideoProcessor:
    """Handles video processing, tracking, and postprocessing."""
    
    def __init__(self, config: dict):
        """
        Initialize video processor.
        
        Args:
            config: Complete configuration dictionary
        """
        self.model_config = config['model']
        self.tracker_config = config['tracker']
        self.video_config = config['video']
        
        # Initialize tracker if enabled
        self.tracker_enabled = self.tracker_config.get('enabled', True)
        if self.tracker_enabled:
            tracker_args = {
                'track_thresh': self.tracker_config.get('track_thresh', 0.45),
                'track_buffer': self.tracker_config.get('track_buffer', 30),
                'match_thresh': self.tracker_config.get('match_thresh', 0.75),
                'mot20': self.tracker_config.get('mot20', False)
            }
            self.tracker = BYTETracker(
                tracker_args,
                frame_rate=self.video_config.get('fps', 60)
            )
            print(f"✓ Tracker initialized: {tracker_args}")
        else:
            self.tracker = None
            print("✓ Tracker disabled")
        
        # Thresholds
        self.score_threshold = self.model_config.get('score_threshold', 0.70)
        self.tracker_input_threshold = self.model_config.get('tracker_input_threshold', 0.10)
    
    def process_outputs(
        self,
        outputs: list,
        orig_frame_shape: Tuple[int, int] = None,
        scale: float = None,
        pads: Tuple[int, int] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Process model outputs with optional tracking.
        
        Args:
            outputs: Model outputs [labels, boxes, scores]
            orig_frame_shape: Original frame shape (height, width) for high-res mode
            scale: Scale ratio from letterbox (for high-res mode)
            pads: Padding (pad_h, pad_w) from letterbox (for high-res mode)
            
        Returns:
            Tuple of (tracked_objects, postprocessing_time_ms)
        """
        post_start = time.perf_counter()
        
        labels_batch, boxes_batch, scores_batch = outputs
        
        # Extract detections from batch
        if scores_batch.shape[1] > 0:
            scores = scores_batch[0]
            labels = labels_batch[0]
            boxes = boxes_batch[0]
            
            # Scale boxes back to original resolution if high-res mode
            if orig_frame_shape is not None and scale is not None and pads is not None:
                boxes = self._scale_boxes_to_original(
                    boxes, orig_frame_shape, scale, pads
                )
            
            # Filter detections for tracker input
            mask = scores > self.tracker_input_threshold
            dets_to_track = np.column_stack([
                boxes[mask],
                scores[mask],
                labels[mask]
            ])
        else:
            dets_to_track = np.empty((0, 6))
        
        # Apply tracking if enabled
        if self.tracker_enabled and self.tracker is not None:
            tracked_objects = self.tracker.update(dets_to_track)
            
            # Filter by final score threshold
            if tracked_objects.shape[0] > 0:
                display_tracks = tracked_objects[
                    tracked_objects[:, 5] > self.score_threshold
                ]
            else:
                display_tracks = np.empty((0, 7))
        else:
            # No tracking - just filter by score threshold
            if dets_to_track.shape[0] > 0:
                mask = dets_to_track[:, 4] > self.score_threshold
                # Add dummy track IDs
                track_ids = np.arange(dets_to_track.shape[0]).reshape(-1, 1)
                display_tracks = np.column_stack([
                    dets_to_track[mask, :4],  # boxes
                    track_ids[mask],  # track_id
                    dets_to_track[mask, 4:6]  # score, class_id
                ])
            else:
                display_tracks = np.empty((0, 7))
        
        post_time = (time.perf_counter() - post_start) * 1000
        
        return display_tracks, post_time
    
    def _scale_boxes_to_original(
        self,
        boxes: np.ndarray,
        orig_shape: Tuple[int, int],
        scale: float,
        pads: Tuple[int, int]
    ) -> np.ndarray:
        """
        Scale boxes from letterbox coordinates back to original image coordinates.
        
        Args:
            boxes: Bounding boxes [x1, y1, x2, y2]
            orig_shape: Original frame shape (height, width)
            scale: Scale ratio from letterbox
            pads: Padding (pad_h, pad_w)
            
        Returns:
            Scaled boxes
        """
        pad_h, pad_w = pads
        orig_h, orig_w = orig_shape
        
        # Remove padding
        boxes[:, [0, 2]] -= pad_w
        boxes[:, [1, 3]] -= pad_h
        
        # Scale back
        boxes /= scale
        
        # Clip to original image bounds
        boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_w)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_h)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, orig_w)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, orig_h)
        
        return boxes
