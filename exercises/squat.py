import cv2
import numpy as np
import time
from constants import Constants as K
from exercises.base import Exercise
from utils._utils import calculate_angle, valid_keypoints, valid_full_pose, smooth_angles


class SquatExercise(Exercise):
    """Ejercicio de Squat (sentadilla) utilizando el patrón Template Method.

    Esta clase implementa la lógica específica para el ejercicio de sentadilla. Se calculan dos ángulos:
      - El ángulo de la pierna, medido entre cadera, rodilla y tobillo.
      - El ángulo del torso, medido entre hombro, cadera y rodilla.
    
    El ejercicio se organiza en dos fases:
      - Estado "down": cuando el usuario está en posición agachada (ángulo de la pierna bajo).
      - Estado "up": cuando el usuario se levanta (ángulo de la pierna alto).

    Se cuenta una repetición cuando se detecta el cambio completo:
      - Se inicia en "up": cuando el ángulo baja por debajo de `SQUAT_MIN_ANGLE`.
      - Se finaliza en "down": cuando el ángulo sube por encima de `SQUAT_MAX_ANGLE`.
      - Adicionalmente, se activa un cooldown (flag rep_finished) hasta que el ángulo descienda lo suficiente (por ejemplo, que sea menor a `SQUAT_MAX_ANGLE - 10`).

    Attributes:
        side (str): Lado a utilizar ('left' o 'right').
        counter (int): Número de repeticiones completadas.
        stage (str): Estado actual del ejercicio ("up" o "down").
        rep_finished (bool): Flag que indica que la repetición ha finalizado y se activa el cooldown.
        angles_history (list): Historial de ángulos brutos para aplicar el suavizado.
        rep_start_time (float): Tiempo de inicio de la repetición actual.
        _latest_angle (float): Último ángulo suave calculado (se usa para la visualización).
        angle_pos (tuple): Coordenadas (x, y) en donde se dibuja el ángulo (por ejemplo, cerca de la cadera o rodilla).
        latest_leg_angle (float): Valor del ángulo de la pierna (calculado entre cadera, rodilla y tobillo).
        latest_torso_angle (float): Valor del ángulo del torso (calculado entre hombro, cadera y rodilla).
        leg_angle_pos (tuple): Coordenadas para visualizar el ángulo de la pierna.
        torso_angle_pos (tuple): Coordenadas para visualizar el ángulo del torso.
    """

    def __init__(self, side):
        """Inicializa la clase SquatExercise.

        Args:
            side (str): Lado a utilizar ('left' o 'right').
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
        self.latest_leg_angle = None
        # Estos atributos se usan para la visualización de los ángulos del torso y de la pierna.
        self.leg_angle_pos = None
        self.torso_angle_pos = None

    @property
    def latest_angle(self):
        """float or None: Retorna el último ángulo relevante calculado (ángulo de la pierna)."""
        return self.latest_leg_angle

    def update(self, keypoints, confs: float, current_time):
        """Actualiza el estado del ejercicio de sentadilla a partir de los keypoints detectados.

        Se calcula el ángulo de la pierna (entre cadera, rodilla y tobillo) y el ángulo del torso 
        (entre hombro, cadera y rodilla) a partir de los keypoints de la pose. Se almacena el ángulo de la pierna
        en un historial y se suaviza utilizando la mediana de las últimas 5 muestras para mitigar el ruido.
        Con base en umbrales definidos en `Constants` (`SQUAT_MIN_ANGLE` y `SQUAT_MAX_ANGLE`), se detecta la
        transición entre los estados:
            - De "up" a "down": cuando el ángulo de la pierna cae por debajo de `SQUAT_MIN_ANGLE`.
            - De "down" a "up": cuando el ángulo de la pierna sube por encima de `SQUAT_MAX_ANGLE`, lo que 
              marca el final de la repetición, activa el cooldown (rep_finished) y se incrementa el contador.
        
        Durante el cooldown, el sistema espera hasta que el ángulo baje lo suficiente (por ejemplo, sea menor que 
        `SQUAT_MAX_ANGLE - 10`) antes de permitir iniciar una nueva repetición.

        Args:
            keypoints (ndarray): Array con las coordenadas de los keypoints detectados.
            confs (float): Valor de confianza de la detección.
            current_time (float): Tiempo actual en segundos para la temporización de la repetición.

        Returns:
            float or None: El ángulo suavizado (de la pierna) o None si no se detecta una pose válida.
        """
        if not valid_full_pose(confs, threshold=0.3):
            return None

        side_str = self.side.upper()  # 'LEFT' o 'RIGHT'
        r_shoulder_idx = K.YOLO_POSE_KEYPOINTS[f'{side_str}_SHOULDER']
        r_hip_idx = K.YOLO_POSE_KEYPOINTS[f'{side_str}_HIP']
        r_knee_idx = K.YOLO_POSE_KEYPOINTS[f'{side_str}_KNEE']
        r_ankle_idx = K.YOLO_POSE_KEYPOINTS[f'{side_str}_ANKLE']

        if valid_keypoints(confs, [r_hip_idx, r_knee_idx, r_ankle_idx]):
            # Se extraen los keypoints para calcular el ángulo de la pierna y del torso.
            r_shoulder = keypoints[r_shoulder_idx]
            r_hip = keypoints[r_hip_idx]
            r_knee = keypoints[r_knee_idx]
            r_ankle = keypoints[r_ankle_idx]

            # Calcula el ángulo de la pierna (entre cadera, rodilla y tobillo)
            leg_angle = calculate_angle(r_hip, r_knee, r_ankle)
            # Calcula el ángulo del torso (entre hombro, cadera y rodilla)
            torso_angle = calculate_angle(r_shoulder, r_hip, r_knee)
            self.angles_history.append(leg_angle)
            
            if len(self.angles_history) >= 5:
                smoothed_leg_angle = np.median(self.angles_history[-5:])
            else:
                smoothed_leg_angle = leg_angle

            self.latest_leg_angle = smoothed_leg_angle
            self.latest_torso_angle = torso_angle
            self.leg_angle_pos = tuple(map(int, r_knee))
            self.torso_angle_pos = tuple(map(int, r_hip))

            if self.rep_finished:
                # Se libera el cooldown cuando el ángulo de la pierna baja por debajo del umbral (SQUAT_MAX_ANGLE - 10)
                if smoothed_leg_angle < K.SQUAT_MAX_ANGLE - 10:
                    self.rep_finished = False
                    print(f"[Squat][DEBUG] Cooldown finalizado: angle bajó a {smoothed_leg_angle:.2f}")
                return smoothed_leg_angle

            # Transición de "up" a "down": Iniciar la repetición cuando el ángulo baja por debajo de SQUAT_MIN_ANGLE.
            if self.stage == "up" and smoothed_leg_angle < K.SQUAT_MIN_ANGLE:
                self.rep_start_time = current_time
                self.rep_start_time_abs = time.time()
                self.stage = "down"
                print(f"[Squat] Transition to DOWN: angle {smoothed_leg_angle:.2f}")
            
            # Transición de "down" a "up": Finalizar la repetición cuando el ángulo sube por encima de SQUAT_MAX_ANGLE.
            elif self.stage == "down" and smoothed_leg_angle > K.SQUAT_MAX_ANGLE:
                self.current_rep_time = current_time - self.rep_start_time
                self.rep_start_time = None
                self.counter += 1
                self.stage = "up"
                self.rep_finished = True
                print(f"[Squat] Transition to UP: angle {smoothed_leg_angle:.2f}, rep count: {self.counter}, rep time: {self.current_rep_time:.2f}")
                self.angles_history.clear()
            return smoothed_leg_angle
        return None

    def draw(self, frame):
        """Dibuja los overlays del ejercicio de sentadilla en el frame.

        Se dibuja un recuadro de fondo en la parte superior del frame que incluye el nombre del
        ejercicio, el contador de repeticiones y el estado actual. Además, se muestran los ángulos 
        de la pierna y del torso cerca de los keypoints correspondientes para brindar una retroalimentación visual.

        Args:
            frame (numpy.ndarray): Imagen actual del frame.

        Returns:
            numpy.ndarray: El frame modificado con los overlays dibujados.
        """
        cv2.rectangle(frame, (0, 0), (300, 73), (245, 117, 16), -1)
        cv2.putText(frame, 'SQUAT', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, str(self.counter),
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, 'STAGE', (135, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        stage_text = self.stage if self.stage is not None else ""
        cv2.putText(frame, stage_text,
                    (90, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        if self.latest_leg_angle is not None and self.leg_angle_pos is not None:
            cv2.putText(frame, f"Leg: {self.latest_leg_angle:.1f}", self.leg_angle_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(frame, f"Leg: {self.latest_leg_angle:.1f}", self.leg_angle_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        if self.latest_torso_angle is not None and self.torso_angle_pos is not None:
            cv2.putText(frame, f"Torso: {self.latest_torso_angle:.1f}", self.torso_angle_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(frame, f"Torso: {self.latest_torso_angle:.1f}", self.torso_angle_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        return frame
