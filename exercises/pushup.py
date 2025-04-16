import cv2
import numpy as np
import time

from constants import Constants as K
from exercises.base import Exercise
from utils._utils import calculate_angle, valid_keypoints, valid_full_pose


class PushupExercise(Exercise):
    """Clase que implementa el ejercicio de pushup (flexiones de pecho) usando el patrón Template Method.

    Esta clase utiliza keypoints para calcular el ángulo formado por el hombro, codo y muñeca,
    y determina las transiciones entre estados ("up" y "down") para contar repeticiones. La señal
    de ángulo se suaviza usando la mediana de las últimas 5 muestras para reducir el ruido.
    
    Durante la ejecución:
      - Se inicia la repetición (transición de "up" a "down") cuando el ángulo cae por debajo de 
        un umbral (CURL_MIN_ANGLE para pushup, definido en el objeto Constants).
      - Se finaliza la repetición (transición de "down" a "up") cuando el ángulo sube por encima de 
        otro umbral (CURL_MAX_ANGLE para pushup).
      - Se utiliza un flag (rep_finished) para activar un cooldown, de modo que no se cuenten repeticiones
        consecutivas hasta que el ángulo baje lo suficiente.
    
    Attributes:
        side (str): Lado utilizado en el ejercicio ('left' o 'right').
        counter (int): Número de repeticiones completadas.
        stage (str): Estado actual del ejercicio; por defecto "up".
        rep_finished (bool): Indica si se encuentra en estado de cooldown tras finalizar una repetición.
        angles_history (list): Historial de ángulos brutos para aplicar suavizado (mediana).
        rep_start_time (float): Tiempo de inicio de la repetición actual.
        _latest_angle (float): Último ángulo suavizado calculado.
        angle_pos (tuple): Coordenadas (x, y) del keypoint usado para la visualización (normalmente el codo).
    """

    def __init__(self, side: str):
        """Inicializa la clase PushupExercise.

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
        """Actualiza el estado del ejercicio de pushup a partir de los keypoints detectados.

        Se calcula el ángulo formado entre el hombro, el codo y la muñeca, se suaviza utilizando la mediana 
        de las últimas 5 muestras, y se gestiona la transición de estados para contar repeticiones.

        Args:
            keypoints (ndarray): Array que contiene las coordenadas de los puntos clave detectados.
            confs (float): Valor de confianza asociado a la detección.
            current_time (float): Tiempo actual (en segundos) usado para la temporización de la repetición.
            
        Returns:
            float or None: El ángulo suavizado calculado si la pose es válida; de lo contrario, None.
        """
        side_str = self.side.upper()  # 'LEFT' o 'RIGHT'
        shoulder_idx = K.YOLO_POSE_KEYPOINTS[f'{side_str}_SHOULDER']
        elbow_idx = K.YOLO_POSE_KEYPOINTS[f'{side_str}_ELBOW']
        wrist_idx = K.YOLO_POSE_KEYPOINTS[f'{side_str}_WRIST']

        if valid_keypoints(confs, [shoulder_idx, elbow_idx, wrist_idx]):
            shoulder = keypoints[shoulder_idx]
            elbow = keypoints[elbow_idx]
            wrist = keypoints[wrist_idx]

            raw_angle = calculate_angle(shoulder, elbow, wrist)
            self.angles_history.append(raw_angle)
            
            # Suavizar la señal usando la mediana de las últimas cinco muestras
            if len(self.angles_history) >= 5:
                smoothed_angle = np.median(self.angles_history[-5:])
            else:
                smoothed_angle = raw_angle

            self._latest_angle = smoothed_angle
            self.angle_pos = tuple(map(int, elbow))

            # Si estamos en cooldown, no se actualiza la rep hasta que el ángulo baje adecuadamente
            if self.rep_finished:
                if smoothed_angle > K.PUSHUP_MAX_ANGLE - 10:
                    self.rep_finished = False
                    print(f"[Pushup][DEBUG] Cooldown finalizado: ángulo = {smoothed_angle:.2f}")
                return smoothed_angle

            # Transición "up" a "down": iniciar la repetición cuando el ángulo cae por debajo del mínimo
            if self.stage == "up" and smoothed_angle < K.PUSHUP_MIN_ANGLE:
                self.rep_start_time = current_time
                self.stage = "down"
                print(f"[Pushup] Transition to DOWN: angle {smoothed_angle:.2f}")

            # Transición "down" a "up": finalizar la repetición cuando el ángulo supera el máximo
            elif self.stage == "down" and smoothed_angle > K.PUSHUP_MAX_ANGLE:
                self.current_rep_time = current_time - self.rep_start_time
                self.rep_start_time = None
                self.counter += 1
                self.stage = "up"
                self.rep_finished = True
                print(f"[Pushup] Transition to UP: angle {smoothed_angle:.2f}, rep count: {self.counter}, rep time: {self.current_rep_time:.2f}")
                self.angles_history.clear()
            return smoothed_angle
        return None

    def draw(self, frame):
        """Dibuja la interfaz de usuario y la información del ejercicio en el frame.

        Se dibuja un rectángulo, se muestra el nombre del ejercicio, el contador de repeticiones, el estado actual
        y se indica el valor del ángulo en la posición del codo.

        Args:
            frame (ndarray): La imagen del frame sobre la cual se dibujarán los elementos.

        Returns:
            ndarray: El frame modificado con los overlays.
        """
        cv2.rectangle(frame, (0, 0), (300, 73), (245, 117, 16), -1)
        cv2.putText(frame, 'PUSHUP', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, str(self.counter),
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, 'STAGE', (135, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        stage_text = self.stage if self.stage is not None else ""
        cv2.putText(frame, stage_text,
                    (90, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        if self.latest_angle is not None and self.angle_pos is not None:
            cv2.putText(frame, f"{self.latest_angle:.1f}", self.angle_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(frame, f"{self.latest_angle:.1f}", self.angle_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        return frame
