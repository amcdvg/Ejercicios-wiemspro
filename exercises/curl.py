# exercises/curl.py
import cv2
from constants import Constants as K
from exercises.base import Exercise
from utils._utils import calculate_angle, valid_keypoints, valid_full_pose
import numpy as np
import time

class CurlExercise(Exercise):
    """

    Args:
        Exercise (_type_): _description_
    """
    def __init__(self):
        super().__init__()
        self.counter = 0
        self.stage = None
        self.latest_angle = None
        self.angle_pos = None

    def update(self, keypoints, confs):
        """_summary_

        Args:
            keypoints (_type_): _description_
            confs (_type_): _description_

        Returns:
            _type_: _description_
        """
        if not valid_full_pose(confs, threshold=0.3):
            return None
        shoulder_idx = K.YOLO_POSE_KEYPOINTS['RIGHT_SHOULDER']
        elbow_idx = K.YOLO_POSE_KEYPOINTS['RIGHT_ELBOW']
        wrist_idx = K.YOLO_POSE_KEYPOINTS['RIGHT_WRIST']
        if valid_keypoints(confs, [shoulder_idx, elbow_idx, wrist_idx]):
            shoulder = keypoints[shoulder_idx]
            elbow = keypoints[elbow_idx]
            wrist = keypoints[wrist_idx]
            angle = calculate_angle(shoulder, elbow, wrist)
            self.latest_angle = angle
            self.angle_pos = tuple(map(int, elbow))
            if angle > K.CURL_MAX_ANGLE:
                if self.stage == "up" and self.up_start_time is not None:
                    self.up_time = time.time() - self.up_start_time
                self.stage = "down"
                self.down_start_time = time.time()
            elif angle < K.CURL_MIN_ANGLE and self.stage == "down":
                self.down_time = time.time() - self.down_start_time
                self.stage = "up"
                self.counter += 1
                self.up_start_time = time.time()

            return angle
        return None
    
    def draw(self, frame):
        """_summary_
        
        Args:
            frame (_type_): _description_
        """
        cv2.rectangle(frame, (0, 0), (300, 73), (245, 117, 16), -1)
        cv2.putText(frame, 'CURL', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(frame, str(self.counter),
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(frame, 'STAGE', (135, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        stage_text = self.stage if self.stage is not None else ""
        cv2.putText(frame, stage_text,
                    (90, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        if self.latest_angle is not None and self.angle_pos is not None:
            cv2.putText(frame, f"{self.latest_angle:.1f}", self.angle_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 5, cv2.LINE_AA)
            cv2.putText(frame, f"{self.latest_angle:.1f}", self.angle_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
