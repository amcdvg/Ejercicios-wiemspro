import cv2 
import numpy as np
from utils._utils import calculate_angle, valid_full_pose, valid_keypoints
import time
from constants import Constants as K
from exercises.base import Exercise

class ExtensionTricepsExercise(Exercise):
    """_summary_

    Args:
        Exercise (_type_): _description_
    """
    def __init__(self, side: str):
        super().__init__()
        self.side = side.lower()
        self.counter = 0
        self.stage = "down"
        self.rep_start_time = None
        self._latest_angle = None
        self.angle_pos = None
        self.angles_history = []
    
    @property
    def latest_angle(self):
        return self._latest_angle
    
    def update(self, keypoints, confs: float, current_time):
        if not valid_full_pose(confs, threshold=0.3):
            return None

        side_str = self.side.upper()
        shoulder_idx = K.YOLO_POSE_KEYPOINTS[f'{side_str}_SHOULDER']
        elbow_idx    = K.YOLO_POSE_KEYPOINTS[f'{side_str}_ELBOW']
        wrist_idx    = K.YOLO_POSE_KEYPOINTS[f'{side_str}_WRIST']

        if valid_keypoints(confs, [shoulder_idx, elbow_idx, wrist_idx]):
            shoulder = keypoints[shoulder_idx]
            elbow    = keypoints[elbow_idx]
            wrist    = keypoints[wrist_idx]
            raw_angle = calculate_angle(shoulder, elbow, wrist)

            self.angles_history.append(raw_angle)
            smoothed_angle = np.median(self.angles_history[-5:]) if len(self.angles_history) >= 5 else raw_angle

            self._latest_angle = smoothed_angle
            self.angle_pos = tuple(map(int, elbow))

            if hasattr(self, 'rep_finished') and self.rep_finished:
                # Aquí definimos que el bloqueo se libera cuando el ángulo baja por debajo de (TRICEPS_EXTENSION_MIN_ANGLE - 10)
                if smoothed_angle < K.TRICEPS_EXTENSION_MIN_ANGLE - 5:
                    self.rep_finished = False
                    print(f"[Extension Triceps][DEBUG] Cooldown finalizado: angle bajó a {smoothed_angle:.2f}")
                return smoothed_angle

            # Transición de "down" a "up": inicio de la repetición
            if self.stage == "down" and smoothed_angle > K.TRICEPS_EXTENSION_MAX_ANGLE+5:
                # Usamos current_time para que coincida con la escala del overlay
                self.rep_start_time = current_time
                self.rep_start_time_abs = time.time()
                self.stage = "up"
                print(f"[Extension Triceps] Transition to UP: angle {smoothed_angle:.2f}")

            # Transición de "up" a "down": final de la repetición
            elif self.stage == "up" and smoothed_angle < K.TRICEPS_EXTENSION_MIN_ANGLE-5:
                self.current_rep_time = current_time - self.rep_start_time
                self.rep_start_time = None  # Reiniciamos para la siguiente rep
                self.counter += 1
                self.stage = "down"
                self.rep_finished = True  # Activamos el cooldown para evitar contar inmediatamente otra rep
                print(f"[Extension Triceps] Transition to DOWN: angle {smoothed_angle:.2f}, rep count: {self.counter}, rep time: {self.current_rep_time:.2f}")
                self.angles_history.clear()

            return smoothed_angle
        return None


    def draw(self, frame):
            """Dibuja la interfaz del ejercicio en el frame."""
            cv2.rectangle(frame, (0, 0), (300, 73), (245, 117, 16), -1)
            cv2.putText(frame, 'TRICEPS EXTENSION', (15, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, str(self.counter), (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, 'STAGE', (135, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            stage_text = self.stage if self.stage is not None else ""
            cv2.putText(frame, stage_text, (90, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            if self._latest_angle is not None and self.angle_pos is not None:
                cv2.putText(frame, f"Angle: {self._latest_angle:.1f}", self.angle_pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 5, cv2.LINE_AA)
                cv2.putText(frame, f"Angle: {self._latest_angle:.1f}", self.angle_pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            return frame