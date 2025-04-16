import cv2
import numpy as np
import time

from constants import Constants as K
from exercises.base import Exercise
from utils._utils import calculate_angle, valid_full_pose, valid_keypoints


class ExtensionTricepsExercise(Exercise):
    """Implementa el ejercicio de extensión de tríceps en banco usando el patrón Template Method.

    En este ejercicio se calcula el ángulo formado por el hombro, codo y muñeca
    para determinar el progreso de la repetición. Se cuentan repeticiones basándose en
    la detección de transiciones entre estados ("down" a "up" y viceversa):
    
      - Se inicia la repetición (transición de "down" a "up") cuando el ángulo supera
        `TRICEPS_EXTENSION_MAX_ANGLE + 5`.
      - Se finaliza la repetición (transición de "up" a "down") cuando el ángulo cae por
        debajo de `TRICEPS_EXTENSION_MIN_ANGLE - 5`, y se verifica que el delta entre el ángulo
        final y el inicial sea mayor a 15°.
    
    Durante la repetición, el ángulo se suaviza utilizando la mediana de las últimas cinco muestras,
    lo que ayuda a mitigar el ruido de la detección de keypoints. Además, se implementa un mecanismo de
    cooldown (rep_finished) para evitar contar repeticiones de forma consecutiva hasta que el ángulo
    baje lo suficiente.

    Attributes:
        side (str): Lado usado para el ejercicio ('left' o 'right').
        counter (int): Número de repeticiones completadas.
        stage (str): Estado actual del ejercicio ("down" o "up").
        rep_start_time (float): Tiempo de inicio de la repetición actual.
        _latest_angle (float): Último ángulo suavizado calculado.
        angle_pos (tuple): Coordenadas (x, y) del keypoint empleado para la visualización (en este caso, el codo).
        angles_history (list): Historial de ángulos brutos para aplicar el filtrado (mediana).
        rep_finished (bool): Flag que indica que la repetición ha finalizado y se activa el cooldown.
    """

    def __init__(self, side: str):
        """Inicializa la clase ExtensionTricepsExercise.

        Args:
            side (str): Lado a utilizar ('left' o 'right').
        """
        super().__init__()
        self.side = side.lower()
        self.counter = 0
        self.stage = "down"  # Se inicia en "down" para detectar el inicio de la extensión
        self.rep_start_time = None
        self._latest_angle = None
        self.angle_pos = None
        self.angles_history = []
        self.rep_finished = False
        # Los tiempos absolutos de inicio de rep se registran para calcular la duración.
        self.rep_start_time_abs = None

    @property
    def latest_angle(self):
        """float: Devuelve el último ángulo suavizado calculado."""
        return self._latest_angle

    def update(self, keypoints, confs: float, current_time):
        """Actualiza el estado del ejercicio de extensión de tríceps a partir de la pose detectada.

        Se valida que la pose tenga suficiente confianza, se extraen las coordenadas del hombro,
        codo y muñeca, y se calcula el ángulo formado entre ellos. El ángulo se suaviza usando la mediana
        de las últimas 5 muestras para minimizar el ruido. Además, se controla la transición entre estados (de
        "down" a "up" y de "up" a "down") para iniciar y finalizar repeticiones.

        Args:
            keypoints (ndarray): Array de coordenadas de los keypoints detectados.
            confs (float): Valor de confianza de la detección.
            current_time (float): Tiempo actual relativo para gestionar la temporización de la repetición.

        Returns:
            float or None: El ángulo suavizado calculado si la pose es válida; de lo contrario, None.
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

            # Suavizar el ángulo usando la mediana de las últimas cinco muestras.
            if len(self.angles_history) >= 5:
                smoothed_angle = np.median(self.angles_history[-5:])
            else:
                smoothed_angle = raw_angle

            self._latest_angle = smoothed_angle
            self.angle_pos = tuple(map(int, elbow))

            # Si estamos en cooldown (rep_finished activado), se espera a que el ángulo baje
            # por debajo de (TRICEPS_EXTENSION_MIN_ANGLE - 5) para liberar el bloqueo.
            if self.rep_finished:
                if smoothed_angle < K.TRICEPS_EXTENSION_MIN_ANGLE - 5:
                    self.rep_finished = False
                    print(f"[Extension Triceps][DEBUG] Cooldown finalizado: angle bajó a {smoothed_angle:.2f}")
                return smoothed_angle

            # Transición de "down" a "up": se inicia la repetición cuando el ángulo sube por encima de (TRICEPS_EXTENSION_MAX_ANGLE + 5).
            if self.stage == "down" and smoothed_angle > K.TRICEPS_EXTENSION_MAX_ANGLE + 5:
                self.rep_start_time = current_time
                self.rep_start_time_abs = time.time()
                self.stage = "up"
                print(f"[Extension Triceps] Transition to UP: angle {smoothed_angle:.2f}")

            # Transición de "up" a "down": se finaliza la repetición cuando el ángulo cae por debajo de (TRICEPS_EXTENSION_MIN_ANGLE - 5).
            elif self.stage == "up" and smoothed_angle < K.TRICEPS_EXTENSION_MIN_ANGLE - 5:
                self.current_rep_time = current_time - self.rep_start_time
                self.rep_start_time = None
                self.counter += 1
                self.stage = "down"
                self.rep_finished = True
                print(f"[Extension Triceps] Transition to DOWN: angle {smoothed_angle:.2f}, rep count: {self.counter}, rep time: {self.current_rep_time:.2f}")
                self.angles_history.clear()

            return smoothed_angle
        return None

    def draw(self, frame):
        """Dibuja los overlays con la información del ejercicio en el frame.

        Se dibuja un rectángulo de fondo, el nombre del ejercicio, el contador de repeticiones y el
        estado actual. Además, se muestra el valor del ángulo suavizado en la posición del codo.

        Args:
            frame (ndarray): El frame actual donde se dibujará la interfaz.

        Returns:
            ndarray: El frame modificado con los overlays dibujados.
        """
        cv2.rectangle(frame, (0, 0), (300, 73), (245, 117, 16), -1)
        cv2.putText(frame, 'TRICEPS EXTENSION', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, str(self.counter), (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, 'STAGE', (135, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        stage_text = self.stage if self.stage is not None else ""
        cv2.putText(frame, stage_text, (90, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        if self._latest_angle is not None and self.angle_pos is not None:
            cv2.putText(frame, f"Angle: {self._latest_angle:.1f}", self.angle_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(frame, f"Angle: {self._latest_angle:.1f}", self.angle_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        return frame
