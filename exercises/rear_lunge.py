import cv2
import numpy as np
import time

from constants import Constants as K
from exercises.base import Exercise
from utils._utils import calculate_angle, valid_keypoints, valid_full_pose


class RearLungeExercise(Exercise):
    """Implementa el ejercicio de rear lunge (zancada trasera) a partir de keypoints detectados,
    utilizando el ángulo formado por cadera, rodilla y tobillo de la pierna trasera.

    La lógica para contar las repeticiones se basa en la detección de transiciones de fase:
    
      - Inicia la repetición (transición de "up" a "down") cuando el ángulo cae por debajo de
        `ZANCADA_MIN_ANGLE`.
      - Finaliza la repetición (transición de "down" a "up") cuando el ángulo sube por encima de
        `ZANCADA_MAX_ANGLE`.
    
    Se utiliza un historial de ángulos y se aplica la mediana de los últimos cinco valores para suavizar
    la señal y minimizar el impacto del ruido en la detección.
    
    Attributes:
        side (str): Lado utilizado para la medición ('left' o 'right').
        stage (str): Fase actual del ejercicio; inicialmente "up".
        rep_finished (bool): Flag que indica si la repetición terminó y se activa el cooldown.
        counter (int): Número de repeticiones contadas.
        angles_history (list): Historial de ángulos brutos usados para suavizar la señal.
        rep_start_time (float): Tiempo (en segundos) de inicio de la repetición.
        _latest_angle (float): Último ángulo suavizado calculado.
        angle_pos (tuple): Coordenadas (x, y) del keypoint (usualmente la rodilla) para visualización.
    """

    def __init__(self, side: str):
        """Inicializa una instancia del ejercicio de rear lunge.

        Args:
            side (str): Lado a utilizar ('left' o 'right').
        """
        super().__init__()
        self.side = side.lower()
        self.stage = "up"
        self.rep_finished = False
        self.counter = 0
        self.angles_history = []
        self.rep_start_time = None
        self._latest_angle = None
        self.angle_pos = None

    @property
    def latest_angle(self) -> float:
        """float: Devuelve el último ángulo suavizado calculado."""
        return self._latest_angle

    def update(self, keypoints, confs: float, current_time: float):
        """Actualiza el estado del ejercicio rear lunge utilizando keypoints detectados.

        Se valida la pose mediante la función `valid_full_pose` y se extraen los keypoints correspondientes
        a la cadera, rodilla y tobillo. A partir de ellos se calcula el ángulo (utilizando `calculate_angle`) que
        representa la flexión de la pierna trasera. Se almacena el ángulo en un historial y se suaviza mediante
        la mediana de las últimas 5 muestras para reducir el ruido. Además, se gestionan las transiciones de estado:
        
            - Si la fase actual es "up" y el ángulo cae por debajo de `ZANCADA_MIN_ANGLE`, se inicia la repetición.
            - Si la fase es "down" y el ángulo supera `ZANCADA_MAX_ANGLE`, se finaliza la repetición (se incrementa el contador)
              y se activa un cooldown para evitar contar repeticiones consecutivas sin el adecuado reset.

        Args:
            keypoints (ndarray): Array que contiene las coordenadas de los puntos clave detectados.
            confs (float): Valor de confianza de la detección.
            current_time (float): Tiempo actual (en segundos) para gestionar la temporización.

        Returns:
            float or None: El ángulo suavizado calculado o None en caso de que la pose no cumpla los requisitos mínimos.
        """
        if not valid_full_pose(confs, threshold=0.3):
            return None

        side_str = self.side.upper()
        hip_idx = K.YOLO_POSE_KEYPOINTS[f'{side_str}_HIP']
        knee_idx = K.YOLO_POSE_KEYPOINTS[f'{side_str}_KNEE']
        ankle_idx = K.YOLO_POSE_KEYPOINTS[f'{side_str}_ANKLE']

        if valid_keypoints(confs, [hip_idx, knee_idx, ankle_idx]):
            hip = keypoints[hip_idx]
            knee = keypoints[knee_idx]
            ankle = keypoints[ankle_idx]

            raw_angle = calculate_angle(hip, knee, ankle)
            self.angles_history.append(raw_angle)

            # Suavización: se utiliza la mediana de las últimas 5 muestras para reducir el ruido.
            if len(self.angles_history) >= 5:
                smoothed_angle = np.median(self.angles_history[-5:])
            else:
                smoothed_angle = raw_angle

            self._latest_angle = smoothed_angle
            self.angle_pos = tuple(map(int, knee))

            # Si se ha finalizado la repetición (cooldown activo), se libera cuando el ángulo sube
            # por encima del umbral (ZANCADA_MAX_ANGLE - 10).
            if self.rep_finished:
                if smoothed_angle > K.ZANCADA_MAX_ANGLE - 10:
                    self.rep_finished = False
                    print(f"[RearLunge][DEBUG] Cooldown finalizado: ángulo = {smoothed_angle:.2f}")
                return smoothed_angle

            # Transición "up" a "down": iniciar la repetición cuando el ángulo cae por debajo de ZANCADA_MIN_ANGLE.
            if self.stage == "up" and smoothed_angle < K.ZANCADA_MIN_ANGLE:
                self.rep_start_time = current_time
                self.stage = "down"
                print(f"[RearLunge] Transition to DOWN: angle {smoothed_angle:.2f}")
            # Transición "down" a "up": finalizar la repetición cuando el ángulo sube por encima de ZANCADA_MAX_ANGLE.
            elif self.stage == "down" and smoothed_angle > K.ZANCADA_MAX_ANGLE:
                self.current_rep_time = current_time - self.rep_start_time
                self.rep_start_time = None
                self.counter += 1
                self.stage = "up"
                self.rep_finished = True
                print(f"[RearLunge] Transition to UP: angle {smoothed_angle:.2f}, rep count: {self.counter}, rep time: {self.current_rep_time:.2f}")
                self.angles_history.clear()
            return smoothed_angle
        return None

    def draw(self, frame):
        """Dibuja los overlays para el ejercicio de rear lunge en el frame.

        Se dibuja un rectángulo de fondo que contiene el nombre del ejercicio, el contador de repeticiones,
        el estado actual, y se muestra el ángulo actual de la pierna trasera (en la posición del keypoint 'knee').

        Args:
            frame (ndarray): Frame de video sobre el cual se dibujarán los elementos.

        Returns:
            ndarray: Frame modificado con los overlays dibujados.
        """
        cv2.rectangle(frame, (0, 0), (300, 73), (245, 117, 16), -1)
        cv2.putText(frame, 'ZANCADA TRASERA', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, str(self.counter), (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, 'STAGE', (135, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        stage_text = self.stage if self.stage is not None else ""
        cv2.putText(frame, stage_text, (90, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        if self._latest_angle is not None and self.angle_pos is not None:
            cv2.putText(frame, f"Leg: {self._latest_angle:.1f}", self.angle_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(frame, f"Leg: {self._latest_angle:.1f}", self.angle_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        return frame
