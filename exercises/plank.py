# exercises/plank.py
import cv2
import time
from constants import Constants as K
from exercises.base import Exercise
from utils.math_utils import calculate_angle
import numpy as np

def valid_keypoints(confs, indices, threshold=0.3):
    return np.all(confs[indices] > threshold)

class PlankExercise(Exercise):
    """_summary_

    Args:
        Exercise (_type_): _description_
    """
    def __init__(self):
        self.start_time = None
        self.elapsed_time = 0
        self.latest_angle = None
        self.angle_pos = None

    def update(self, keypoints, confs: float):
        """_summary_

        Args:
            keypoints (_type_): _description_
            confs (float): _description_

        Returns:
            _type_: _description_
        """
        shoulder_idx = K.YOLO_POSE_KEYPOINTS['RIGHT_SHOULDER']
        hip_idx = K.YOLO_POSE_KEYPOINTS['RIGHT_HIP']
        knee_idx = K.YOLO_POSE_KEYPOINTS['RIGHT_KNEE']
        if valid_keypoints(confs, [shoulder_idx, hip_idx, knee_idx]):
            shoulder = keypoints[shoulder_idx]
            hip = keypoints[hip_idx]
            knee = keypoints[knee_idx]
            torso_angle = calculate_angle(shoulder, hip, knee)
            self.latest_angle = torso_angle
            self.angle_pos = tuple(map(int, hip))
            if torso_angle >= K.PLANK_THRESHOLD_ANGLE:
                if self.start_time is None:
                    self.start_time = time.time()
                self.elapsed_time = time.time() - self.start_time
            else:
                self.start_time = None
                self.elapsed_time = 0
            return torso_angle
        return None

    def draw(self, frame):
        """_summary_

        Args:
            frame (_type_): _description_
        """
        cv2.rectangle(frame, (0, 0), (300, 73), (245, 117, 16), -1)
        cv2.putText(frame, 'PLANK', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Time: {self.elapsed_time:.1f}s",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        if self.latest_angle is not None and self.angle_pos is not None:
            cv2.putText(frame, f"{self.latest_angle:.1f}", self.angle_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 5, cv2.LINE_AA)
            cv2.putText(frame, f"{self.latest_angle:.1f}", self.angle_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)