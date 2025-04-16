import cv2
import numpy as np
import time

from constants import Constants as K
from exercises.base import Exercise
from utils._utils import calculate_angle, valid_keypoints, valid_full_pose


class ReverseFlyExercise(Exercise):
    """Implementa el ejercicio de Reverse Fly (aperturas inversas) utilizando keypoints 
    para detectar el movimiento y contar repeticiones.

    La medición se basa en calcular el ángulo formado entre un punto de referencia,
    el hombro y el codo. En este caso, el punto de referencia se define a partir del hombro,
    desplazado horizontalmente 100 píxeles (hacia la izquierda si se usa el lado izquierdo y
    hacia la derecha si se usa el lado derecho). Esta referencia intenta simular un punto fijo
    para evaluar la apertura inversa de los brazos. La lógica para detectar repeticiones se basa 
    en las siguientes transiciones:
      - En estado "up": si el ángulo supera un valor umbral máximo (`REVERSEFLY_MAX_ANGLE`), 
        se inicia la fase "down".
      - En estado "down": si el ángulo cae por debajo del umbral mínimo (`REVERSEFLY_MIN_ANGLE`), 
        se finaliza la repetición, se incrementa el contador y se activa un cooldown (rep_finished)
        para evitar contar múltiples repeticiones sin que la señal se estabilice.

    Attributes:
        side (str): Lado a utilizar ('left' o 'right').
        counter (int): Número de repeticiones completadas.
        stage (str): Estado actual del ejercicio ("up" o "down").
        rep_finished (bool): Indica si la repetición ha finalizado (cooldown activo).
        angles_history (list): Historial de ángulos brutos para suavizar la señal.
        rep_start_time (float): Tiempo de inicio de la repetición actual.
        _latest_angle (float): Último ángulo suavizado calculado.
        angle_pos (tuple): Posición (x, y) del keypoint de referencia para visualización.
    """

    def __init__(self, side: str):
        """Inicializa la clase ReverseFlyExercise.

        Args:
            side (str): Lado a utilizar, por ejemplo 'left' o 'right'.
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
        """Actualiza el estado del ejercicio Reverse Fly en base a los keypoints detectados.

        Se valida la pose utilizando una confianza mínima. Posteriormente, se extraen los keypoints 
        correspondientes al hombro y al codo. Para calcular el ángulo se establece un punto de referencia:
        si se utiliza el lado izquierdo, se define un punto situado 100 píxeles a la izquierda del hombro; 
        si se utiliza el lado derecho, se define a 100 píxeles a la derecha. Se calcula el ángulo formado 
        entre ese punto de referencia, el hombro y el codo usando `calculate_angle`. El ángulo se añade a 
        un historial, y se suaviza utilizando la mediana de los últimos 5 valores para reducir el ruido.
        
        Se gestionan también las transiciones de estado para contar las repeticiones:
          - Si se encuentra en cooldown (rep_finished), se espera hasta que el ángulo sea menor que 
            `REVERSEFLY_MIN_ANGLE + 5` para liberar el bloqueo.
          - Si se está en el estado "up" y el ángulo es mayor que `REVERSEFLY_MAX_ANGLE`, se inicia la fase 
            "down".
          - Si se está en la fase "down" y el ángulo cae por debajo de `REVERSEFLY_MIN_ANGLE`, se finaliza la 
            repetición, se incrementa el contador y se activa el cooldown.
        
        Args:
            keypoints (ndarray): Array con las coordenadas de los keypoints detectados.
            confs (float): Valor de confianza de la detección.
            current_time (float): Tiempo actual (en segundos) para medir la duración de la repetición.
        
        Returns:
            float or None: El ángulo suavizado calculado o None si la pose no es válida.
        """
        if not valid_full_pose(confs, threshold=0.3):
            return None

        side_str = self.side.upper()
        shoulder_idx = K.YOLO_POSE_KEYPOINTS[f'{side_str}_SHOULDER']
        elbow_idx = K.YOLO_POSE_KEYPOINTS[f'{side_str}_ELBOW']

        if valid_keypoints(confs, [shoulder_idx, elbow_idx]):
            shoulder = keypoints[shoulder_idx]
            elbow = keypoints[elbow_idx]

            # Definir un punto de referencia en función del lado
            if self.side == "left":
                ref_point = (shoulder[0] - 100, shoulder[1])
            else:
                ref_point = (shoulder[0] + 100, shoulder[1])

            raw_angle = calculate_angle(ref_point, shoulder, elbow)
            self.angles_history.append(raw_angle)

            # Suavización del ángulo usando la mediana de las últimas 5 muestras
            if len(self.angles_history) >= 5:
                smoothed_angle = np.median(self.angles_history[-5:])
            else:
                smoothed_angle = raw_angle

            self._latest_angle = smoothed_angle
            self.angle_pos = tuple(map(int, shoulder))

            if self.rep_finished:
                if smoothed_angle < K.REVERSEFLY_MIN_ANGLE + 5:
                    self.rep_finished = False
                    print(f"[PajaroBanco][DEBUG] Cooldown finalizado: ángulo = {smoothed_angle:.2f}")
                return smoothed_angle

            if self.stage == "up" and smoothed_angle > K.REVERSEFLY_MAX_ANGLE:
                self.rep_start_time = current_time
                self.stage = "down"
                print(f"[PajaroBanco] Transition to DOWN: angle {smoothed_angle:.2f}")
            elif self.stage == "down" and smoothed_angle < K.REVERSEFLY_MIN_ANGLE:
                self.current_rep_time = current_time - self.rep_start_time
                self.rep_start_time = None
                self.counter += 1
                self.stage = "up"
                self.rep_finished = True
                print(f"[PajaroBanco] Transition to UP: angle {smoothed_angle:.2f}, rep count: {self.counter}, rep time: {self.current_rep_time:.2f}")
                self.angles_history.clear()
            return smoothed_angle
        return None

    def draw(self, frame):
        """Dibuja los overlays para el ejercicio Reverse Fly en el frame.

        Se dibuja un rectángulo de fondo, el nombre del ejercicio, el contador de repeticiones, el estado actual
        y se muestra el ángulo calculado (prefijado con la etiqueta "Arm:") en la posición del hombro, que actúa
        como punto de referencia para la visualización.

        Args:
            frame (ndarray): Frame de video sobre el cual se dibujarán los overlays.
        
        Returns:
            ndarray: El frame modificado con los overlays.
        """
        cv2.rectangle(frame, (0, 0), (300, 73), (245, 117, 16), -1)
        cv2.putText(frame, 'PAJARO BANCO', (15, 12),
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
