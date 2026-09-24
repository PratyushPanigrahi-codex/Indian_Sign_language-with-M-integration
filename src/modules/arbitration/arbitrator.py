"""
Member 1: Unified Static + Dynamic Pipeline Arbitrator
Combines predictions from Static (Dense NN) and Dynamic (LSTM) models.
Uses wrist velocity & motion variance to arbitrate between held and moving gestures.
"""

import numpy as np
from collections import deque

class GestureArbitrator:
    def __init__(self, motion_threshold=0.015, window_size=15):
        self.motion_threshold = motion_threshold
        self.wrist_history = deque(maxlen=window_size)

    def update_motion(self, landmarks):
        """
        Calculates motion variance based on wrist landmarks (indices 1 & 64).
        Returns True if hands are in motion, False if held still.
        """
        # Wrist Y positions
        left_wrist_y = landmarks[1] if np.any(landmarks[0:63] != 0) else None
        right_wrist_y = landmarks[64] if np.any(landmarks[63:126] != 0) else None
        
        current_y = [y for y in (left_wrist_y, right_wrist_y) if y is not None]
        if current_y:
            self.wrist_history.append(np.mean(current_y))
            
        if len(self.wrist_history) < self.wrist_history.maxlen:
            return False
            
        motion_variance = float(np.var(self.wrist_history))
        return motion_variance > self.motion_threshold

    def arbitrate(self, static_pred, static_conf, dynamic_pred, dynamic_conf, landmarks):
        """
        Decides which prediction to commit based on motion variance and confidence scores.
        """
        in_motion = self.update_motion(landmarks)
        
        # If in motion and dynamic model has confidence, prefer dynamic
        if in_motion and dynamic_pred and dynamic_conf >= 0.6:
            return dynamic_pred, dynamic_conf, "dynamic"
            
        # If held still or dynamic model is uncertain, prefer static
        if static_pred and static_conf >= 0.6:
            return static_pred, static_conf, "static"
            
        return None, 0.0, "none"
