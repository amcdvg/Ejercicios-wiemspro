import cv2
from constants import Constants as K
from exercises.base import Exercise
from utils._utils import calculate_angle, valid_keypoints, valid_full_pose
import numpy as np
import time
class SquatExercise(Exercise):
    """Ejercicio de Squat (sentadilla) utilizando el patrón Template Method.
    Esta clase implementa la lógica específica para el ejercicio de sentadilla,
    donde se miden dos ángulos:
      - El ángulo de la pierna, calculado entre el hombro, la cadera y la rodilla.
      - El ángulo del torso, calculado entre el hombro, la cadera y la rodilla.
    El ejercicio se modela en dos fases:
    - Estado "down": cuando el usuario está en posición de sentadilla (ángulo de pierna bajo).
    - Estado "up": cuando el usuario se levanta (ángulo de pierna alto).
    Se cuenta una repetición cuando se detecta el cambio completo entre estas fases:
      - Se inicia en "down" (posición agachada).
      - Al levantarse y alcanzar un ángulo mayor a SQUAT_MAX_ANGLE, se cuenta una repetición y se cambia a "up".
      - Luego, al bajar de nuevo y alcanzar un ángulo menor a SQUAT_MIN_ANGLE, se vuelve a cambiar a "down".
    Atributos:
        counter (int): Número de repeticiones completadas.
        stage (str): Estado actual del ejercicio ("down" o "up").
        latest_leg_angle (float): Último ángulo medido en la pierna.
        latest_torso_angle (float): Último ángulo medido en el torso.
        leg_angle_pos (tuple): Coordenadas (x, y) para dibujar el ángulo de la pierna.
        torso_angle_pos (tuple): Coordenadas (x, y) para dibujar el ángulo del torso.
        down_start_time (float): Tiempo de inicio de la fase de bajada.
        up_start_time (float): Tiempo de inicio de la fase de subida.
    """
    def __init__(self):
        """_summary_
        """
        super().__init__()
        self.counter = 0
        self.stage = "down"
        self.latest_leg_angle = None
        self.latest_torso_angle = None
        self.leg_angle_pos = None
        self.torso_angle_pos = None
    @property
    def latest_angle(self):
        """Devuelve el último ángulo relevante para el ejercicio (ángulo de la pierna).
        Returns:
            float or None: El ángulo medido de la pierna, o None si aún no se ha calculado.
        """
        return self.latest_leg_angle
    
    def update(self, keypoints, confs: float):
        """Actualiza el estado del ejercicio de sentadilla.
        Se calcula el ángulo de la pierna (entre cadera, rodilla y tobillo) y el ángulo del torso (entre hombro, cadera y rodilla).
        Luego, según el estado actual y los umbrales definidos en las constantes (SQUAT_MAX_ANGLE y SQUAT_MIN_ANGLE),
        se detecta si el usuario se levanta o baja, actualizando el contador y los tiempos correspondientes.
        Args:
            keypoints (array-like): Puntos de referencia provenientes de la estimación de pose.
            confs (array-like): Valores de confianza de la detección.
        Returns:
            float or tuple: Devuelve el ángulo de la pierna si se actualiza correctamente;
            de lo contrario, retorna (None, None).
        """
        if not valid_full_pose(confs, threshold=0.3):
            return None
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
            if self.stage == "down" and leg_angle > K.SQUAT_MAX_ANGLE:
                # Se detecta que el usuario se levanta desde la posición de sentadilla
                self.down_time = time.time() - (self.down_start_time if self.down_start_time else time.time())
                self.counter += 1
                self.up_start_time = time.time()
                self.stage = "up"
            elif self.stage == "up" and leg_angle < K.SQUAT_MIN_ANGLE:
                # Se detecta q
                # ue el usuario baja desde la posición de pie
                self.up_time = time.time() - (self.up_start_time if self.up_start_time else time.time())
                self.down_start_time = time.time()
                self.stage = "down"
            return leg_angle
        return None, None
    
    def draw(self, frame):
        """Dibuja la interfaz del ejercicio de curl en el frame.
        Se dibujan un recuadro con el nombre del ejercicio, el contador de repeticiones, 
        el estado actual y el ángulo detectado, con contorno para mejorar la legibilidad.
        Args:
            frame (numpy.ndarray): Imagen actual del frame.
        Returns:
            numpy.ndarray: El frame actualizado con los overlays del ejercicio.
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