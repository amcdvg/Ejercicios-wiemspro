import cv2
from constants import Constants as K
from exercises.base import Exercise
from utils.math_utils import calculate_angle
import numpy as np

def valid_keypoints(confs, indices, threshold=0.3):
    return np.all(confs[indices] > threshold)

class SquatExercise(Exercise):
    """_summary_

    Args:
        Exercise (_type_): _description_
    """
    def __init__(self):
        self.counter = 0
        self.stage = None
        self.latest_leg_angle = None
        self.latest_torso_angle = None
        self.leg_angle_pos = None
        self.torso_angle_pos = None

    def update(self, keypoints, confs: float):
        """_summary_

        Args:
            keypoints (_type_): _description_
            confs (float): _description_

        Returns:
            _type_: _description_
        """
        r_shoulder_idx = K.YOLO_POSE_KEYPOINTS['RIGHT_SHOULDER']
        r_hip_idx = K.YOLO_POSE_KEYPOINTS['RIGHT_HIP']
        r_knee_idx = K.YOLO_POSE_KEYPOINTS['RIGHT_KNEE']
        r_ankle_idx = K.YOLO_POSE_KEYPOINTS['RIGHT_ANKLE']
        if valid_keypoints(confs, [r_hip_idx, r_knee_idx, r_ankle_idx]):
            r_shoulder = keypoints[r_shoulder_idx]
            r_hip = keypoints[r_hip_idx]
            r_knee = keypoints[r_knee_idx]
            r_ankle = keypoints[r_ankle_idx]
            leg_angle = calculate_angle(r_hip, r_knee, r_ankle)
            torso_angle = calculate_angle(r_shoulder, r_hip, r_knee)
            self.latest_leg_angle = leg_angle
            self.latest_torso_angle = torso_angle
            self.leg_angle_pos = tuple(map(int, r_knee))
            self.torso_angle_pos = tuple(map(int, r_hip))
            if leg_angle < K.SQUAT_MIN_ANGLE and torso_angle < K.SQUAT_TORSO_MIN_ANGLE and self.stage != "down":
                self.stage = "down"
            elif leg_angle > K.SQUAT_MAX_ANGLE and self.stage == "down":
                self.stage = "up"
                self.counter += 1
            return leg_angle, torso_angle
        return None, None

    def draw(self, frame):
        """_summary_

        Args:
            frame (_type_): _description_
        """
        cv2.rectangle(frame, (0, 0), (300, 73), (245, 117, 16), -1)
        cv2.putText(frame, 'SQUAT', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(frame, str(self.counter),
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(frame, 'STAGE', (135, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        stage_text = self.stage if self.stage is not None else ""
        cv2.putText(frame, stage_text,
                    (90, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        if self.latest_leg_angle is not None and self.leg_angle_pos is not None:
            cv2.putText(frame, f"Leg: {self.latest_leg_angle:.1f}", self.leg_angle_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 5, cv2.LINE_AA)
            cv2.putText(frame, f"Leg: {self.latest_leg_angle:.1f}", self.leg_angle_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
        if self.latest_torso_angle is not None and self.torso_angle_pos is not None:
            cv2.putText(frame, f"Torso: {self.latest_torso_angle:.1f}", self.torso_angle_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 5, cv2.LINE_AA)
            cv2.putText(frame, f"Torso: {self.latest_torso_angle:.1f}", self.torso_angle_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)