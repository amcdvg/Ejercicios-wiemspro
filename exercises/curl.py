import cv2
import numpy as np
import time

from constants import Constants as K
from exercises.base import Exercise
from utils._utils import calculate_angle, valid_keypoints, valid_full_pose


class CurlExercise(Exercise):
    """Ejercicio de Curl para bíceps utilizando el patrón Template Method.

    Se mide el ángulo formado por el hombro, codo y muñeca y se cuentan las repeticiones
    detectando la transición de estado. En concreto:
    
      - Se inicia la repetición (transición de "up" a "down") cuando el ángulo cae por debajo
        de `CURL_MIN_ANGLE`.
      - Se finaliza la repetición (transición de "down" a "up") cuando el ángulo supera 
        `CURL_MAX_ANGLE` y se verifica que la diferencia entre el ángulo inicial y el final
        (delta_angle) sea superior a 15°.
      
    La señal de ángulo se suaviza mediante la mediana de las últimas 5 muestras para evitar
    ruido puntual. Durante la fase de cooldown (cuando se ha finalizado la repetición) la
    actualización de la medición se congela hasta que el ángulo disminuya lo suficiente.
    
    Attributes:
        side (str): Lado a utilizar ('left' o 'right').
        counter (int): Número de repeticiones completadas.
        stage (str): Estado actual del ejercicio ("up" o "down").
        _latest_angle (float): Último ángulo suavizado calculado.
        angle_pos (tuple): Coordenadas (x, y) del keypoint usado para la visualización (en este caso, el codo).
        angles_history (list): Historial de ángulos brutos para aplicar suavizado.
        rep_lock (bool): Flag utilizado para evitar contar repeticiones múltiples (no se usa en este código).
        down_start_time (float): Tiempo de inicio de la fase "down" (no se usa en este fragmento).
        up_start_time (float): Tiempo de inicio de la fase "up" (no se usa en este fragmento).
        rep_finished (bool): Indica que la repetición ha finalizado (cooldown activo).
        rep_start_time (float): Tiempo de inicio de la repetición.
        rep_end_angle (float): Valor de ángulo final de la repetición (para calcular delta).
    """

    def __init__(self, side: str):
        """Inicializa una instancia del ejercicio de Curl.

        Args:
            side (str): Lado a utilizar ('left' o 'right').
        """
        super().__init__()
        self.side = side.lower()
        self.counter = 0
        self.stage = "up"
        self._latest_angle = 0  # Inicialmente 0
        self.angle_pos = None
        self.angles_history = []
        self.rep_lock = False  
        self.down_start_time = None
        self.up_start_time = None
        self.rep_finished = False
        self.rep_start_time = None

    @property
    def latest_angle(self):
        """float: Devuelve el último ángulo suavizado calculado."""
        return self._latest_angle

    def update(self, keypoints, confs: float, current_time):
        """Actualiza el estado del ejercicio de Curl a partir de los keypoints detectados.

        Se valida la pose, se calcula el ángulo formado por hombro, codo y muñeca, y se
        suaviza la señal utilizando la mediana de las últimas 5 muestras. Además, se gestiona
        la transición entre estados "up" y "down" para contabilizar repeticiones.

        Args:
            keypoints (ndarray): Array con las coordenadas de los keypoints detectados.
            confs (float): Valor de confianza asociado a la detección.
            current_time (float): Tiempo (en segundos) actual, utilizado para medir la duración
                de la repetición.

        Returns:
            float or None: El ángulo suavizado calculado si la pose es válida, o None si no se cumple
            el umbral de confianza.
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

            # Calcular el ángulo entre hombro, codo y muñeca
            raw_angle = calculate_angle(shoulder, elbow, wrist)
            self.angles_history.append(raw_angle)
            # Suavizar el ángulo usando la mediana de las últimas 5 muestras
            if len(self.angles_history) >= 5:
                smoothed_angle = np.median(self.angles_history[-5:])
            else:
                smoothed_angle = raw_angle

            self._latest_angle = smoothed_angle
            self.angle_pos = tuple(map(int, elbow))

            # Durante el cooldown no se actualiza la repetición hasta que el ángulo caiga 15° por debajo
            # del ángulo final de la repetición anterior.
            if self.rep_finished:
                if smoothed_angle < self.rep_end_angle - 15:
                    self.rep_finished = False
                    print(f"[Curl][DEBUG] Cooldown finalizado: ángulo bajó de {self.rep_end_angle:.2f} a {smoothed_angle:.2f}")
                else:
                    return smoothed_angle

            # Estado "up": iniciar la repetición si el ángulo cae por debajo del umbral mínimo
            if self.stage == "up" and smoothed_angle < K.CURL_MIN_ANGLE:
                self.rep_start_time = current_time  
                self.rep_start_angle = smoothed_angle
                self.stage = "down"
                print(f"[Curl] Transition to DOWN: angle {smoothed_angle:.2f}")
                print(f"[Curl][DEBUG] Rep iniciada con ángulo = {self.rep_start_angle:.2f}")

            # Estado "down": registrar la rep cuando el ángulo sube por encima del umbral máximo
            elif self.stage == "down":
                print(f"[Curl][DEBUG] En rep: ángulo actual = {smoothed_angle:.2f}")
                if smoothed_angle > K.CURL_MAX_ANGLE:
                    self.rep_end_angle = smoothed_angle
                    delta_angle = self.rep_end_angle - self.rep_start_angle
                    print(f"[Curl][DEBUG] Rep finaliza: ángulo final = {self.rep_end_angle:.2f} (inicio = {self.rep_start_angle:.2f}), delta = {delta_angle:.2f}")
                    if delta_angle > 15:
                        self.current_rep_time = time.time() - self.rep_start_time
                        self.counter += 1
                        self.last_rep_time = self.current_rep_time
                        print(f"[Curl] Transition to UP: angle {smoothed_angle:.2f}, rep count: {self.counter}, rep time: {self.current_rep_time:.2f}")
                    else:
                        print(f"[Curl][DEBUG] Rep descartada: delta angle {delta_angle:.2f} insuficiente")
                    self.stage = "up"
                    self.rep_finished = True
                    self.angles_history.clear()
            return smoothed_angle
        return None

    def draw(self, frame):
        """Dibuja la interfaz de usuario y la visualización del ejercicio en el frame.

        Se dibuja un rectángulo de fondo, el nombre del ejercicio, el conteo de repeticiones,
        el estado actual y se muestra el valor del ángulo suavizado en la posición del codo.

        Args:
            frame (ndarray): Imagen del frame donde se dibujarán los elementos.

        Returns:
            ndarray: El frame modificado con los overlays.
        """
        cv2.rectangle(frame, (0, 0), (300, 73), (245, 117, 16), -1)
        cv2.putText(frame, 'CURL', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, str(self.counter),
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, 'STAGE', (135, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        stage_text = self.stage if self.stage is not None else ""
        cv2.putText(frame, stage_text,
                    (90, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        if self._latest_angle is not None and self.angle_pos is not None:
            cv2.putText(frame, f"{self._latest_angle:.1f}", self.angle_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(frame, f"{self._latest_angle:.1f}", self.angle_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        return frame
