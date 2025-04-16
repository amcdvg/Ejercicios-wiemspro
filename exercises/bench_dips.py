import cv2
import numpy as np
import time

from constants import Constants as K
from exercises.base import Exercise
from utils._utils import calculate_angle, valid_keypoints, valid_full_pose


class BenchDipsExercise(Exercise):
    """Clase que implementa la lógica para detectar y analizar las repeticiones
    de fondos de tríceps en banco (Bench Dips) a partir de la estimación de la posición
    de keypoints y cálculo del ángulo formado por hombro, codo y muñeca.

    La clase define la lógica para detectar la transición entre las fases del
    ejercicio y contar las repeticiones. Además, utiliza un filtrado simple (mediana)
    sobre una historia de ángulos para suavizar la señal.

    Attributes:
        side (str): Lado que se utiliza ('left' o 'right').
        counter (int): Conteo de repeticiones completadas.
        stage (str): Fase actual del ejercicio ("up" o "down").
        rep_finished (bool): Indica si se terminó una repetición y se espera el reinicio.
        angles_history (list): Lista para almacenar los ángulos brutos calculados.
        rep_start_time (float): Tiempo de inicio de la repetición actual.
        _latest_angle (float): Último ángulo filtrado calculado.
        angle_pos (tuple): Coordenadas enteras (x, y) del keypoint a usar como referencia en la visualización.
    """

    def __init__(self, side: str):
        """Inicializa una instancia de BenchDipsExercise.

        Args:
            side (str): Lado a utilizar, por ejemplo, 'left' o 'right'.
        """
        super().__init__()
        self.side = side.lower()  # Puede ser 'left' o 'right'
        self.counter = 0
        self.stage = "up"
        self.rep_finished = False
        self.angles_history = []
        self.rep_start_time = None
        self._latest_angle = None
        self.angle_pos = None

    @property
    def latest_angle(self):
        """float: Devuelve el último ángulo (filtrado) calculado."""
        return self._latest_angle

    def update(self, keypoints, confs: float, current_time):
        """Actualiza el estado del ejercicio utilizando los keypoints detectados.

        Se valida que exista una pose completa (usando una confianza mínima) y se calcula
        el ángulo formado por el hombro, codo y muñeca según el lado especificado. Se almacena
        el ángulo en una historia y se utiliza la mediana de los últimos 5 valores para suavizar la señal.
        Además, se determina la transición entre fases ("up" a "down" y viceversa) y se cuenta la repetición.

        Args:
            keypoints (ndarray): Array con las coordenadas de los puntos clave detectados.
            confs (float): Valor de confianza de la detección.
            current_time (float): Tiempo actual (relative time) de la medición.

        Returns:
            float or None: El ángulo suavizado si la pose es válida, o None en caso contrario.
        """
        if not valid_full_pose(confs, threshold=0.3):
            return None

        side_str = self.side.upper()
        shoulder_idx = K.YOLO_POSE_KEYPOINTS[f'{side_str}_SHOULDER']
        elbow_idx = K.YOLO_POSE_KEYPOINTS[f'{side_str}_ELBOW']
        wrist_idx = K.YOLO_POSE_KEYPOINTS[f'{side_str}_WRIST']

        if valid_keypoints(confs, [shoulder_idx, elbow_idx, wrist_idx]):
            shoulder = keypoints[shoulder_idx]
            elbow = keypoints[elbow_idx]
            wrist = keypoints[wrist_idx]

            raw_angle = calculate_angle(shoulder, elbow, wrist)
            self.angles_history.append(raw_angle)
            
            if len(self.angles_history) >= 5:
                smoothed_angle = np.median(self.angles_history[-5:])
            else:
                smoothed_angle = raw_angle

            self._latest_angle = smoothed_angle
            self.angle_pos = tuple(map(int, elbow))

            if self.rep_finished:
                if smoothed_angle > K.TRICEPS_BANCO_MAX_ANGLE - 10:
                    self.rep_finished = False
                    print(f"[TricepsBanco][DEBUG] Cooldown finalizado: ángulo = {smoothed_angle:.2f}")
                return smoothed_angle

            if self.stage == "up" and smoothed_angle < K.TRICEPS_BANCO_MIN_ANGLE:
                self.rep_start_time = current_time
                self.stage = "down"
                print(f"[TricepsBanco] Transition to DOWN: angle {smoothed_angle:.2f}")
            elif self.stage == "down" and smoothed_angle > K.TRICEPS_BANCO_MAX_ANGLE:
                self.current_rep_time = current_time - self.rep_start_time
                self.rep_start_time = None
                self.counter += 1
                self.stage = "up"
                self.rep_finished = True
                print(f"[TricepsBanco] Transition to UP: angle {smoothed_angle:.2f}, rep count: {self.counter}, rep time: {self.current_rep_time:.2f}")
                self.angles_history.clear()
            return smoothed_angle
        return None

    def draw(self, frame):
        """Dibuja los overlays visuales sobre el frame de video.

        Se dibuja un rectángulo y textos indicando el nombre del ejercicio, el conteo de repeticiones,
        la fase actual, y se muestra el valor del ángulo filtrado en la posición del codo.

        Args:
            frame (ndarray): Imagen del frame sobre la cual se dibujarán los overlays.

        Returns:
            ndarray: El frame modificado con los overlays dibujados.
        """
        cv2.rectangle(frame, (0, 0), (300, 73), (245, 117, 16), -1)
        cv2.putText(frame, 'TRICEPS BANCO', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, str(self.counter), (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, 'STAGE', (135, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        stage_text = self.stage if self.stage is not None else ""
        cv2.putText(frame, stage_text, (90, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        if self._latest_angle is not None and self.angle_pos is not None:
            cv2.putText(frame, f"Arm: {self._latest_angle:.1f}", self.angle_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(frame, f"Arm: {self._latest_angle:.1f}", self.angle_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        return frame
