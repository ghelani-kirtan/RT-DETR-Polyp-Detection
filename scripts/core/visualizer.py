"""Visualization utilities for inference results."""

import cv2
import numpy as np
from collections import defaultdict, deque
from typing import Dict, Deque, Tuple


class Visualizer:
    """Handles all visualization tasks including bounding boxes, labels, and trails."""
    
    def __init__(self, config: dict):
        """
        Initialize visualizer.
        
        Args:
            config: Complete configuration dictionary
        """
        self.model_config = config['model']
        self.viz_config = config['visualization']
        self.smoothing_config = config.get('smoothing', {'enabled': False})
        
        # Extract class information
        self.class_names = self.model_config['classes']['names']
        self.class_colors = [tuple(c) for c in self.model_config['classes']['colors']]
        self.num_classes = len(self.class_names)
        
        # Trail configuration
        self.show_trails = self.viz_config.get('show_trails', True)
        trail_config = self.viz_config.get('trail', {})
        self.trail_maxlen = trail_config.get('max_length', 40)
        self.trail_color = tuple(trail_config.get('color', [255, 0, 0]))
        self.trail_fading = trail_config.get('fading', True)
        
        # Smoothing configuration
        self.smoothing_enabled = self.smoothing_config.get('enabled', False)
        self.ema_alpha = self.smoothing_config.get('ema_alpha', 0.65)
        self.low_prob_threshold = self.smoothing_config.get('low_prob_threshold', 0.75)
        
        # History tracking
        self.class_history: Dict[int, Deque] = defaultdict(
            lambda: deque(maxlen=self.smoothing_config.get('history_length', 10))
        )
        self.track_history: Dict[int, Deque] = defaultdict(
            lambda: deque(maxlen=self.trail_maxlen)
        )
    
    def draw_detections(
        self,
        frame: np.ndarray,
        tracked_objects: np.ndarray
    ) -> np.ndarray:
        """
        Draw bounding boxes, labels, and trails on frame.
        
        Args:
            frame: Input frame
            tracked_objects: Array of tracked objects [x1, y1, x2, y2, track_id, score, class_id]
            
        Returns:
            Frame with visualizations
        """
        if tracked_objects.shape[0] == 0:
            return frame
        
        for track in tracked_objects:
            x1, y1, x2, y2, track_id, score, class_id = track
            track_id = int(track_id)
            class_id = int(class_id)
            
            # Apply smoothing if enabled
            if self.smoothing_enabled:
                smoothed_class_id, label_status = self._apply_smoothing(
                    track_id, class_id, score
                )
            else:
                smoothed_class_id = class_id
                label_status = ""
            
            # Get color and label
            color = self.class_colors[smoothed_class_id]
            label_text = f"ID:{track_id} {self.class_names[smoothed_class_id]}{label_status}: {score:.2f}"
            
            # Draw bounding box
            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color,
                2
            )
            
            # Draw label background and text
            (text_w, text_h), baseline = cv2.getTextSize(
                label_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2
            )
            cv2.rectangle(
                frame,
                (int(x1), int(y1) - text_h - baseline),
                (int(x1) + text_w, int(y1)),
                color,
                -1
            )
            cv2.putText(
                frame,
                label_text,
                (int(x1), int(y1) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )
            
            # Draw trail if enabled
            if self.show_trails:
                self._draw_trail(frame, track_id, x1, y1, x2, y2)
        
        return frame
    
    def _apply_smoothing(
        self,
        track_id: int,
        class_id: int,
        score: float
    ) -> Tuple[int, str]:
        """
        Apply EMA smoothing to class predictions.
        
        Args:
            track_id: Track ID
            class_id: Current class ID
            score: Current confidence score
            
        Returns:
            Tuple of (smoothed_class_id, label_status)
        """
        # Create probability vector
        current_probs = np.zeros(self.num_classes)
        current_probs[class_id] = score
        
        # Apply EMA
        if track_id in self.class_history and len(self.class_history[track_id]) > 0:
            prev_smoothed = self.class_history[track_id][-1]
            smoothed_probs = (
                self.ema_alpha * current_probs +
                (1 - self.ema_alpha) * prev_smoothed
            )
        else:
            smoothed_probs = current_probs
        
        self.class_history[track_id].append(smoothed_probs)
        
        # Get smoothed class
        smoothed_class_id = np.argmax(smoothed_probs)
        max_prob = np.max(smoothed_probs)
        
        # Hold previous class if confidence is low
        label_status = ""
        if max_prob < self.low_prob_threshold and len(self.class_history[track_id]) > 1:
            smoothed_class_id = np.argmax(self.class_history[track_id][-2])
            label_status = " (Held)"
        
        return smoothed_class_id, label_status
    
    def _draw_trail(
        self,
        frame: np.ndarray,
        track_id: int,
        x1: float,
        y1: float,
        x2: float,
        y2: float
    ):
        """Draw tracking trail for an object."""
        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        self.track_history[track_id].append(center)
        
        points = list(self.track_history[track_id])
        for i in range(1, len(points)):
            if points[i - 1] is None or points[i] is None:
                continue
            
            thickness = (
                int(np.sqrt(self.trail_maxlen / float(i + 1)) * 2.5)
                if self.trail_fading
                else 2
            )
            cv2.line(frame, points[i - 1], points[i], self.trail_color, thickness)
    
    def add_latency_overlay(
        self,
        frame: np.ndarray,
        pre_time: float,
        inf_time: float,
        post_time: float
    ) -> np.ndarray:
        """
        Add latency information overlay to frame.
        
        Args:
            frame: Input frame
            pre_time: Preprocessing time (ms)
            inf_time: Inference time (ms)
            post_time: Postprocessing time (ms)
            
        Returns:
            Frame with latency overlay
        """
        latency_config = self.viz_config.get('latency', {})
        show_overall = latency_config.get('show_overall', True)
        show_details = latency_config.get('show_details', False)
        
        y_pos = 30
        total_latency = pre_time + inf_time + post_time
        
        if show_overall:
            cv2.putText(
                frame,
                f"Latency: {total_latency:.1f} ms",
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            y_pos += 30
        
        if show_details:
            cv2.putText(
                frame,
                f"Pre: {pre_time:.1f}ms Inf: {inf_time:.1f}ms Post: {post_time:.1f}ms",
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )
        
        return frame
    
    def reset_history(self):
        """Reset all tracking history."""
        self.class_history.clear()
        self.track_history.clear()
