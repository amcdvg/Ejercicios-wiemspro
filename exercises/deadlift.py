import cv2
from constants import Constants as K
from exercises.base import Exercise
from utils._utils import calculate_angle, valid_keypoints, valid_full_pose
import numpy as np
import time


class DeadliftExercise(Exercise):
    """_summary_

    Args:
        Exercise (_type_): _description_
    """
    def __init__(self, side: str):
        super().__init__()
        self.side = side.lower()
        self.counter = 0
        self.stage = "up"  # Inicia en posición de pie (up)
        self.rep_finished = False  # Para evitar conteos inmediatos tras finalizar una rep
        self.angles_history = []   # Para suavizar el ángulo
        self.rep_start_time = None # Tiempo relativo de inicio de la rep
        self._latest_angle = None  # Último ángulo (torso)
        self.angle_pos = None      # Posición para d1ibujar el ángulo

    @property
    def latest_angle(self):
        return self._latest_angle
    
    def update(self, keypoints, confs: float, current_time):
        """_summary_

        Args:
            keypoints (_type_): _description_
            confs (float): _description_

        Returns:
            _type_: _description_
        """
        if not valid_full_pose(confs, threshold=0.3):
            return None
        side_str = self.side.upper()
        shoulder_idx = K.YOLO_POSE_KEYPOINTS[f'{side_str}_SHOULDER']
        hip_idx = K.YOLO_POSE_KEYPOINTS[f'{side_str}_HIP']
        knee_idx = K.YOLO_POSE_KEYPOINTS[f'{side_str}_KNEE']

        if valid_keypoints(confs, [shoulder_idx, hip_idx, knee_idx]):
            shoulder = keypoints[shoulder_idx]
            hip = keypoints[hip_idx]
            knee = keypoints[knee_idx]

            raw_angle = calculate_angle(shoulder, hip, knee)
            self.angles_history.append(raw_angle)
            
            if len(self.angles_history) >= 5:
                smoothed_angle = np.median(self.angles_history[-5:])
            else:
                smoothed_angle = raw_angle
            self._latest_angle = smoothed_angle
            self.angle_pos = tuple(map(int, hip))

            if self.rep_finished:
                if smoothed_angle < K.DEADLIFT_MAX_TORSO_ANGLE - 10:
                    self.rep_finished = False
                    print(f"[Deadlift][DEBUG] Cooldown finalizado: ángulo = {smoothed_angle:.2f}")
                return smoothed_angle
            # Transición de "up" a "down": Se inicia la repetición cuando el torso se inclina demasiado.
            if self.stage == "up" and smoothed_angle < K.DEADLIFT_MIN_TORSO_ANGLE:
                self.rep_start_time = current_time  # Usamos current_time para la medición
                self.stage = "down"
                print(f"[Deadlift] Transition to DOWN: angle {smoothed_angle:.2f}")
            elif self.stage == "down" and smoothed_angle > K.DEADLIFT_MAX_TORSO_ANGLE:
                self.current_rep_time = current_time - self.rep_start_time
                self.rep_start_time = None  # Reiniciamos para la siguiente rep
                self.counter += 1
                self.stage = "up"
                self.rep_finished = True
                print(f"[Deadlift] Transition to UP: angle {smoothed_angle:.2f}, rep count: {self.counter}, rep time: {self.current_rep_time:.2f}")
                self.angles_history.clear()
            return smoothed_angle
        return None

    def draw(self, frame):
        """
        Dibuja los overlays para el ejercicio de Deadlift en el frame.
        Se muestra un recuadro con el nombre del ejercicio, el contador de repeticiones, 
        el estado actual y el ángulo del torso detectado.
        """
        cv2.rectangle(frame, (0, 0), (300, 73), (245, 117, 16), -1)
        cv2.putText(frame, 'DEADLIFT', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(frame, str(self.counter),
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(frame, 'STAGE', (135, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        stage_text = self.stage if self.stage is not None else ""
        cv2.putText(frame, stage_text,
                    (90, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        if self._latest_angle is not None and self.angle_pos is not None:
            cv2.putText(frame, f"Torso: {self._latest_angle:.1f}", self.angle_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 5, cv2.LINE_AA)
            cv2.putText(frame, f"Torso: {self._latest_angle:.1f}", self.angle_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
        return frame
