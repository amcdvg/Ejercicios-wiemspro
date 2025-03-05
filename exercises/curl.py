# exercises/curl.py
import cv2
from constants import Constants as K
from exercises.base import Exercise
from utils._utils import calculate_angle, valid_keypoints, valid_full_pose
import numpy as np
import time

class CurlExercise(Exercise):
    """Ejercicio de Curl para bíceps utilizando la plantilla del patrón Template Method.

    Esta clase implementa la lógica específica para el ejercicio de curl,
    donde se mide el ángulo formado por el hombro, codo y muñeca.
    Se utiliza para contar repeticiones a partir del cambio de fases:
    - Estado "up": brazo extendido (ángulo cercano a CURL_MAX_ANGLE).
    - Estado "down": brazo flexionado (ángulo menor a CURL_MIN_ANGLE).

    Atributos:
        counter (int): Contador de repeticiones completadas.
        stage (str): Estado actual del ejercicio, ya sea "up" o "down".
        latest_angle (float): Último ángulo calculado en la detección.
        angle_pos (tuple): Posición (x, y) para dibujar el ángulo en el frame.
        down_start_time (float): Momento en que se inicia la fase de descenso.
        up_start_time (float): Momento en que se inicia la fase de ascenso.
    """

    def __init__(self):
        super().__init__()
        self.counter = 0
        self.stage = "up"
        self.latest_angle = None
        self.angle_pos = None

    def update(self, keypoints, confs):
        """Actualiza el estado del ejercicio de curl.

        Se calcula el ángulo formado entre el hombro, codo y muñeca, y se actualiza el estado
        ("up" o "down") según los umbrales definidos en las constantes. Se cuenta la repetición cuando
        se completa el movimiento de flexión y extensión.

        Args:
            keypoints (array-like): Lista o array con las coordenadas de los keypoints detectados.
            confs (array-like): Lista o array con las confianzas asociadas a cada keypoint.

        Returns:
            float or None: El ángulo calculado si los keypoints son válidos, de lo contrario None.
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
            if self.stage == "up" and angle < K.CURL_MIN_ANGLE:
                # Se detecta que el brazo se flexiona (baja el ángulo)
                self.down_start_time = time.time()
                self.stage = "down"
            elif self.stage == "down" and angle > K.CURL_MAX_ANGLE:
                # Se detecta que el brazo se extiende nuevamente (ángulo alto)
                self.down_time = time.time() - (self.down_start_time if self.down_start_time else time.time())
                self.counter += 1
                self.up_start_time = time.time()
                self.stage = "up"

            return angle
        return None
    
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
