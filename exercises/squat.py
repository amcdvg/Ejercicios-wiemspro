import cv2
import numpy as np
import time
from constants import Constants as K
from exercises.base import Exercise
from scipy.signal import savgol_filter, medfilt
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

    def __init__(self, side: str, user_height: float, user_location : int):
        """Inicializa la clase SquatExercise.

        Args:
            side (str): Lado a utilizar ('left' o 'right').
        """
        self.side = side
        self.user_height = user_height  # Altura real del usuario en metros
        
        # Variables de calibración
        self.pixel_scale = None
        self.dist_top = None
        self.dist_button = None
        self.dist_higth = None
        
        # Seguimiento de repeticiones
        self.stage = "up"
        self.counter = 0
        self.rep_finished = False
        self.rep_start_time = None
        self.current_rep_time = 0
        
        # Historial de desplazamientos
        self.displacement_history = []
        if user_location == 0:
            self.savgol_window = 16
            self.savgol_polyorder = 2
        else:
            self.savgol_window = 17
            self.savgol_polyorder = 1
        self.disp_threshold = None
        self.calib_disp = []
        
        # Nuevas variables para distancia vertical
        self.bbox_height_history = []
        self.vertical_distance = None
        self.shoulder_pos = None
        self.ankle_pos = None
        self._latest_displacement = 0

    @property
    def latest_angle(self):
        """float or None: Retorna el último ángulo relevante calculado (ángulo de la pierna)."""
        return self._latest_displacement
    
    def get_pixel_scale(self):
        """Devuelve el factor de conversión de píxeles a metros calculado.

        Returns:
            float: El pixel_scale actual.
        """
        return self.pixel_scale
    def update(self, keypoints, confs: float, current_time):
        """Actualiza el estado del ejercicio de Deadlift.

        Primero se valida la existencia de una pose completa mediante la función `valid_full_pose`.
        Si la escala no se ha calibrado, se calcula usando el bounding box (diferencia entre el mínimo
        y máximo de las coordenadas Y). Luego, durante los primeros frames se acumulan desplazamientos
        (diferencia en Y entre muñeca y tobillo) para calcular un umbral dinámico. Una vez calibrado, se
        calcula el desplazamiento actual en metros, se aplica filtrado y se detectan las transiciones:
        
          - Si se está en estado "up" y el desplazamiento cae por debajo del umbral, se inicia la repetición
            (transición a "down").
          - Si se está en estado "down" y el desplazamiento supera el umbral + 0.02 m, se finaliza la rep,
            aumentando el contador y reiniciando el historial de desplazamientos.

        Args:
            keypoints (ndarray): Array con las coordenadas de los keypoints detectados.
            confs (float): Valor de confianza de la detección.
            current_time (float): Tiempo actual (valor relativo) para la medición.

        Returns:
            float or None: El desplazamiento filtrado (último valor) si la pose es válida, de lo contrario None.
        """
        if not valid_full_pose(confs, threshold=0.3):
            return None

        side_str = self.side.upper()
        shoulder_idx = K.YOLO_POSE_KEYPOINTS[f'{side_str}_SHOULDER']
        ankle_idx = K.YOLO_POSE_KEYPOINTS[f'{side_str}_EAR']
        
        # Calibración inicial usando el bounding box
        if self.pixel_scale is None:
            
            y_coords = keypoints[:, 1]
            self.dist_top = np.min(y_coords)
            self.dist_button = np.max(y_coords)
            bbox_height_px = self.dist_button - self.dist_top
            
            if bbox_height_px > 1000:
                self.pixel_scale = self.user_height / bbox_height_px
                print(f"Calibrado: 1px = {self.pixel_scale:.5f} m")
            else:
                self.pixel_scale = self.user_height / (bbox_height_px * 1.522)
                print(f"Calibrado: 1px = {self.pixel_scale:.5f} m")
                
        # Calcular distancia vertical actual
        if self.pixel_scale is not None:
            current_bbox_height_px = (self.dist_button - self.dist_top) 
            self.vertical_distance = current_bbox_height_px * self.pixel_scale
            self.bbox_height_history.append(self.vertical_distance)
        
        # Resto de la lógica de seguimiento...
        if self.disp_threshold is None:
            if valid_keypoints(confs, [shoulder_idx, ankle_idx]):
                shoulder = keypoints[shoulder_idx]
                ankle = keypoints[ankle_idx]
                disp_px = abs(self.dist_button - (shoulder[1] )) #- abs(self.dist_top - (shoulder[1]))
                self.calib_disp.append(disp_px)
                
                if len(self.calib_disp) >= 5:
                    max_disp_px = max(self.calib_disp)
                    self.disp_threshold = ((max_disp_px  * self.pixel_scale)) - 0.02
            return None

        if valid_keypoints(confs, [shoulder_idx, ankle_idx]):
            shoulder = keypoints[shoulder_idx]
            ankle = keypoints[ankle_idx]
            
            displacement_px = abs(self.dist_button - (shoulder[1]))  #- abs(self.dist_top - (shoulder[1]))
            displacement_m = (displacement_px * self.pixel_scale) 
            
            # Filtrado
            self.displacement_history.append(displacement_m)
            if len(self.displacement_history) >= self.savgol_window:
                smoothed_displacement = savgol_filter(
                    self.displacement_history,
                    self.savgol_window,
                    self.savgol_polyorder
                )[-1]
            else:
                smoothed_displacement = displacement_m
            
            # Lógica de transición de estados
            if self.stage == "up" and smoothed_displacement < self.disp_threshold:
                self.stage = "down"
                self.rep_start_time = current_time
            elif self.stage == "down" and smoothed_displacement > self.disp_threshold + 0.02:
                self.stage = "up"
                self.counter += 1
                self.current_rep_time = current_time - self.rep_start_time
                self.displacement_history.clear()
            
            self._latest_displacement = smoothed_displacement
            self.shoulder_pos = tuple(map(int, shoulder))
            self.ankle_pos = tuple(map(int, ankle))
            
            return smoothed_displacement
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
                       """ 
        return frame
