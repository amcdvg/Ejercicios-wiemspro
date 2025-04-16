import cv2
import numpy as np
import time

from constants import Constants as K
from exercises.base import Exercise
from utils._utils import calculate_angle, valid_keypoints, valid_full_pose


class RenegadeRowExercise(Exercise):
    """Implementa el ejercicio de renegade row (remo renegado) utilizando keypoints para detectar repeticiones.

    La lógica del ejercicio se basa en calcular el ángulo formado por hombro, codo y muñeca (usando 
    los keypoints correspondientes) y en detectar la transición entre estados ("up" y "down") para 
    contabilizar repeticiones. Se utiliza un historial de ángulos (suavizados mediante la mediana de 
    las últimas 3 muestras) para minimizar el ruido en la medición.

    Attributes:
        side (str): Lado a utilizar ('left' o 'right').
        counter (int): Número de repeticiones completadas.
        stage (str): Fase actual del ejercicio ("up" o "down").  
        rep_finished (bool): Bandera que indica que la repetición ha finalizado y se activa un cooldown.
        angles_history (list): Historial de ángulos brutos para aplicar suavizado.
        rep_start_time (float): Tiempo de inicio de la repetición actual.
        _latest_angle (float): Último ángulo suavizado calculado.
        angle_pos (tuple): Coordenadas (x, y) del keypoint utilizado para la visualización (normalmente el codo).
    """

    def __init__(self, side: str):
        """Inicializa una instancia de RenegadeRowExercise.

        Args:
            side (str): Lado a utilizar ('left' o 'right').
        """
        super().__init__()
        self.side = side.lower()
        self.counter = 0
        self.stage = "up"
        self.rep_finished = False
        self.angles_history = []
        self.rep_start_time = None
        self._latest_angle = None
        self.angle_pos = None

    @property
    def latest_angle(self) -> float:
        """float: Devuelve el último ángulo suavizado calculado."""
        return self._latest_angle

    def update(self, keypoints, confs: float, current_time: float):
        """Actualiza el estado del ejercicio de renegade row a partir de los keypoints detectados.

        Se valida la pose con un umbral mínimo de confianza y se obtienen los índices correspondientes 
        a los keypoints del hombro, codo y muñeca. Se calcula el ángulo entre estos puntos mediante 
        `calculate_angle`. El ángulo se almacena en un historial y se suaviza aplicando la mediana de las
        últimas 3 muestras. Además, se gestiona la transición de estados ("up" a "down" y viceversa) para 
        contabilizar repeticiones, activando un cooldown (rep_finished) hasta que se libere la condición de reinicio.

        Args:
            keypoints (ndarray): Array con las coordenadas de los keypoints detectados.
            confs (float): Valor de confianza de la detección.
            current_time (float): Tiempo actual (en segundos) para la temporización de la repetición.
        
        Returns:
            float or None: El ángulo suavizado calculado o None si no se cumple el umbral de confianza.
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

            # Suavización: se utiliza la mediana de las últimas 3 muestras.
            if len(self.angles_history) >= 3:
                smoothed_angle = np.median(self.angles_history[-3:])
            else:
                smoothed_angle = raw_angle

            self._latest_angle = smoothed_angle
            self.angle_pos = tuple(map(int, elbow))

            # En estado de cooldown (rep_finished activado), se espera a que el ángulo sea mayor
            # que (RENEGADEROW_MAX_ANGLE - 10) para liberar el bloqueo.
            if self.rep_finished:
                if smoothed_angle > K.RENEGADEROW_MAX_ANGLE - 10:
                    self.rep_finished = False
                    print(f"[RenegadeRow][DEBUG] Cooldown finalizado: ángulo = {smoothed_angle:.2f}")
                return smoothed_angle

            # Transición "up" a "down": si el ángulo cae por debajo del umbral mínimo (RENEGADEROW_MIN_ANGLE),
            # se inicia la repetición.
            if self.stage == "up" and smoothed_angle < K.RENEGADEROW_MIN_ANGLE:
                self.rep_start_time = current_time
                self.stage = "down"
                print(f"[RenegadeRow] Transition to DOWN: angle {smoothed_angle:.2f}")
            # Transición "down" a "up": si el ángulo supera el umbral máximo (RENEGADEROW_MAX_ANGLE), se
            # finaliza la repetición; se incrementa el contador y se activa el cooldown.
            elif self.stage == "down" and smoothed_angle > K.RENEGADEROW_MAX_ANGLE:
                self.current_rep_time = current_time - self.rep_start_time
                self.rep_start_time = None
                self.counter += 1
                self.stage = "up"
                self.rep_finished = True
                print(f"[RenegadeRow] Transition to UP: angle {smoothed_angle:.2f}, rep count: {self.counter}, rep time: {self.current_rep_time:.2f}")
                self.angles_history.clear()
            return smoothed_angle
        return None

    def draw(self, frame):
        """Dibuja los overlays para el ejercicio de renegade row en el frame.

        Se dibuja un fondo rectangular y se muestran el nombre del ejercicio, el contador de repeticiones,
        el estado actual y el valor del ángulo suavizado en la posición del keypoint (normalmente el codo).

        Args:
            frame (ndarray): El frame del video sobre el cual se dibujarán los elementos.
        
        Returns:
            ndarray: El frame modificado con los overlays.
        """
        cv2.rectangle(frame, (0, 0), (300, 73), (245, 117, 16), -1)
        cv2.putText(frame, 'REN_ROW', (15, 12),
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
