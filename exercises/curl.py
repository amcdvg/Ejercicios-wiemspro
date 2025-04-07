import cv2
from constants import Constants as K
from exercises.base import Exercise
from utils._utils import calculate_angle, valid_keypoints, valid_full_pose
import numpy as np
import time
class CurlExercise(Exercise):
    """Ejercicio de Curl para bíceps utilizando el patrón Template Method.

    Se mide el ángulo formado por el hombro, codo y muñeca. Se cuentan repeticiones
    al detectar la transición:
      - De "up" a "down": cuando el ángulo cae por debajo de CURL_MIN_ANGLE.
      - De "down" a "up": cuando el ángulo sube por encima de CURL_MAX_ANGLE.
    """
    def __init__(self, side: str):
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
        # Flag para indicar que la rep ya se completó y se debe congelar la medición.
        self.rep_finished = False
        self.rep_start_time = None

    @property
    def latest_angle(self):
        return self._latest_angle

    def update(self, keypoints, confs: float, current_time):
        """
        Actualiza el estado del ejercicio de curl.
        Se calcula el ángulo formado por hombro, codo y muñeca y se suaviza usando la mediana
        de las últimas muestras.
        La lógica es la siguiente:
        - En estado "up": si el ángulo cae por debajo de CURL_MIN_ANGLE y no se está en cooldown,
            se inicia la repetición (transición a "down") y se guarda el ángulo inicial.
        - En estado "down": si el ángulo sube por encima de CURL_MAX_ANGLE, se considera que
            la repetición terminó; se calcula el delta entre el ángulo final y el inicial y, si es mayor a 15°,
            se cuenta la repetición, se guarda el tiempo de la repetición en last_rep_time, se activa el cooldown
            (rep_finished) y se limpia el historial.
        - Mientras se esté en cooldown (rep_finished == True), no se actualizará el temporizador
            hasta que se detecte que el ángulo ha bajado al menos 15° por debajo del ángulo final de la rep anterior.
        
        Returns:
            float or None: El ángulo suavizado calculado o None si no se detecta una pose válida.
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
            # Suavizamos usando la mediana de las últimas 5 muestras
            if len(self.angles_history) >= 5:
                smoothed_angle = np.median(self.angles_history[-5:])
            else:
                smoothed_angle = raw_angle

            self._latest_angle = smoothed_angle
            self.angle_pos = tuple(map(int, elbow))

            # Si estamos en cooldown (rep_finished activado), no se inicia una nueva repetición
            # hasta que el ángulo baje de forma significativa respecto al rep_end_angle.
            if self.rep_finished:
                if smoothed_angle < self.rep_end_angle - 15:
                    # Se reinicia el cooldown y se permite iniciar una nueva rep.
                    self.rep_finished = False
                    print(f"[Curl][DEBUG] Cooldown finalizado: ángulo bajó de {self.rep_end_angle:.2f} a {smoothed_angle:.2f}")
                else:
                    # Mientras no se baje lo suficiente, no se actualiza el tiempo.
                    return smoothed_angle

            # Estado "up": se inicia la repetición si se cumple la condición
            if self.stage == "up" and smoothed_angle < K.CURL_MIN_ANGLE:
                self.rep_start_time = current_time  
                self.rep_start_angle = smoothed_angle
                self.stage = "down"
                print(f"[Curl] Transition to DOWN: angle {smoothed_angle:.2f}")
                print(f"[Curl][DEBUG] Rep iniciada con ángulo = {self.rep_start_angle:.2f}")

            # Estado "down": durante la repetición, se muestra el ángulo y se detecta el final
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
        """Dibuja la interfaz del ejercicio de curl en el frame."""
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
