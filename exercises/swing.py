import cv2
import numpy as np
import time

from constants import Constants as K
from exercises.base import Exercise
from utils._utils import calculate_angle, valid_keypoints, valid_full_pose


class SwingExercise(Exercise):
    """Implementa el ejercicio de Swing (movimiento de swing, por ejemplo, estilo kettlebell)
    utilizando keypoints para detectar las repeticiones a partir del ángulo de "torso".
    
    En este ejercicio se calcula el ángulo formado por la cadera, el hombro y la muñeca, 
    que se utiliza para determinar la posición del torso. Se suaviza la señal aplicando la mediana 
    de las últimas 5 muestras para mitigar el ruido. La transición de estados se controla de la siguiente manera:
    
      - Se inicia la repetición (transición de "up" a "down") cuando el ángulo supera 
        `SWING_MIN_TORSO_ANGLE`.
      - Se finaliza la repetición (transición de "down" a "up") cuando el ángulo baja por debajo 
        de `SWING_MAX_TORSO_ANGLE`.
      
    Se emplea un flag (`rep_finished`) que activa un período de cooldown para evitar contar repeticiones 
    múltiples sin que la señal se estabilice.
    
    Attributes:
        side (str): Lado que se utiliza para la medición ('left' o 'right').
        counter (int): Número de repeticiones completadas.
        stage (str): Estado actual del ejercicio ("up" o "down").
        rep_finished (bool): Indica si la repetición se ha finalizado y se activa el cooldown.
        angles_history (list): Historial de ángulos brutos para suavizar la señal.
        rep_start_time (float): Tiempo de inicio de la repetición actual.
        _latest_angle (float): Último ángulo suavizado calculado, representativo del torso.
        angle_pos (tuple): Coordenadas (x, y) para dibujar el valor del ángulo (en este caso, la posición del hombro).
    """

    def __init__(self, side: str):
        """Inicializa una instancia de SwingExercise.

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
        """float: Devuelve el último ángulo suavizado (representa el ángulo del torso)."""
        return self._latest_angle

    def update(self, keypoints, confs: float, current_time: float):
        """Actualiza el estado del ejercicio de Swing a partir de los keypoints detectados.
        
        Se valida que la pose detectada tenga la confianza requerida. Luego se extraen los keypoints 
        correspondientes al hombro, cadera y muñeca, los cuales se usan para calcular el ángulo formado 
        (con la función `calculate_angle`). Este ángulo se almacena en un historial y se suaviza utilizando 
        la mediana de las últimas 5 muestras. Con base en los umbrales definidos en `Constants` 
        (`SWING_MIN_TORSO_ANGLE` y `SWING_MAX_TORSO_ANGLE`), se detecta la transición de estados:
        
          - Si se está en estado "up" y el ángulo supera `SWING_MIN_TORSO_ANGLE`, se inicia la fase "down".
          - Si se está en estado "down" y el ángulo baja por debajo de `SWING_MAX_TORSO_ANGLE`,
            se finaliza la repetición, se incrementa el contador y se activa el cooldown (rep_finished).
          - Durante el cooldown se espera hasta que el ángulo se recupere (por encima de un umbral de liberación)
            para reiniciar la detección de una nueva repetición.
        
        Args:
            keypoints (ndarray): Array con las coordenadas de los keypoints detectados.
            confs (float): Valor de confianza de la detección.
            current_time (float): Tiempo actual en segundos (utilizado para la temporización de la repetición).
        
        Returns:
            float or None: El ángulo suavizado calculado o None si la detección de pose no es válida.
        """
        if not valid_full_pose(confs, threshold=0.3):
            return None
        
        side_str = self.side.upper()
        shoulder_idx = K.YOLO_POSE_KEYPOINTS[f'{side_str}_SHOULDER']
        hip_idx = K.YOLO_POSE_KEYPOINTS[f'{side_str}_HIP']
        wrist_idx = K.YOLO_POSE_KEYPOINTS[f'{side_str}_WRIST']

        if valid_keypoints(confs, [shoulder_idx, hip_idx, wrist_idx]):
            shoulder = keypoints[shoulder_idx]
            hip = keypoints[hip_idx]
            wrist = keypoints[wrist_idx]
            # Se calcula el ángulo tomando como referencia la cadera, el hombro y la muñeca.
            raw_angle = calculate_angle(hip, shoulder, wrist)
            self.angles_history.append(raw_angle)
            
            # Suaviza la señal usando la mediana de las últimas 5 muestras
            if len(self.angles_history) >= 5:
                smoothed_angle = np.median(self.angles_history[-5:])
            else:
                smoothed_angle = raw_angle

            self._latest_angle = smoothed_angle
            self.angle_pos = tuple(map(int, shoulder))

            # Si se encuentra en cooldown (rep_finished), se espera a que el ángulo baje hasta liberar el bloqueo.
            if self.rep_finished:
                if smoothed_angle < K.SWING_MIN_TORSO_ANGLE + 10:
                    self.rep_finished = False
                    print(f"[Swing][DEBUG] Cooldown finalizado: ángulo = {smoothed_angle:.2f}")
                return smoothed_angle

            # Transición de "up" a "down": iniciar repetición cuando el ángulo supera SWING_MIN_TORSO_ANGLE.
            if self.stage == "up" and smoothed_angle > K.SWING_MIN_TORSO_ANGLE:
                self.rep_start_time = current_time
                self.stage = "down"
                print(f"[Swing] Transition to DOWN: angle {smoothed_angle:.2f}")
            # Transición de "down" a "up": finalizar repetición cuando el ángulo baja por debajo de SWING_MAX_TORSO_ANGLE.
            elif self.stage == "down" and smoothed_angle < K.SWING_MAX_TORSO_ANGLE:
                self.current_rep_time = current_time - self.rep_start_time
                self.rep_start_time = None
                self.counter += 1
                self.stage = "up"
                self.rep_finished = True
                print(f"[Swing] Transition to UP: angle {smoothed_angle:.2f}, rep count: {self.counter}, rep time: {self.current_rep_time:.2f}")
                self.angles_history.clear()
            return smoothed_angle
        return None

    def draw(self, frame):
        """Dibuja los overlays visuales para el ejercicio de Swing en el frame.

        Se dibuja un rectángulo de fondo en la parte superior del frame, se muestra el nombre
        del ejercicio, el contador de repeticiones, el estado actual y el valor del ángulo
        (representando el ángulo del torso) en la posición del keypoint correspondiente.

        Args:
            frame (ndarray): Imagen del frame sobre la cual se dibujarán los overlays.

        Returns:
            ndarray: El frame modificado con los overlays.
        """
        cv2.rectangle(frame, (0, 0), (300, 73), (245, 117, 16), -1)
        cv2.putText(frame, 'SWING', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, str(self.counter), (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, 'STAGE', (135, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        stage_text = self.stage if self.stage is not None else ""
        cv2.putText(frame, stage_text, (90, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        if self._latest_angle is not None and self.angle_pos is not None:
            cv2.putText(frame, f"Torso: {self._latest_angle:.1f}", self.angle_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(frame, f"Torso: {self._latest_angle:.1f}", self.angle_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        return frame
