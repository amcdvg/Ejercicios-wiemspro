import cv2
import numpy as np
import time
from scipy.signal import savgol_filter, medfilt

from constants import Constants as K
from exercises.base import Exercise
from utils._utils import valid_keypoints, valid_full_pose


class DeadliftExercise(Exercise):
    """Implementa el ejercicio de peso muerto utilizando el desplazamiento relativo
    entre la muñeca y el tobillo para detectar fases y contar repeticiones.

    La lógica del ejercicio se basa en:
      - Calibrar la escala de píxeles a metros mediante el bounding box del sujeto.  
      - Calcular un umbral dinámico (disp_threshold) acumulando desplazamientos (wrist-ankle) en los
        primeros frames y restándole una constante (0.04 m) al máximo acumulado.
      - Durante la ejecución, se calcula el desplazamiento actual (en metros) y se almacena en un historial.
      - Se aplica un filtrado (Savitzky–Golay seguido de un filtro mediana) a la historia de desplazamientos
        para obtener una señal suavizada.
      - Se detectan transiciones de fase:
          * De "up" a "down": cuando el desplazamiento (después del filtrado) cae por debajo del umbral.
          * De "down" a "up": cuando el desplazamiento sube por encima del umbral + 0.02 m, contabilizando la repetición.
    
    Atributos:
        side (str): Lado a utilizar, puede ser 'left' o 'right'.
        counter (int): Conteo de repeticiones completadas.
        stage (str): Estado actual del ejercicio ('up' o 'down').
        rep_finished (bool): Indica si se finalizó la repetición y se encuentra en fase de cooldown.
        rep_start_time (float): Tiempo de inicio de la repetición actual.
        _latest_displacement (float): Desplazamiento filtrado (último valor calculado) en metros.
        wrist_pos (tuple): Coordenadas (x, y) del keypoint de la muñeca, para fines gráficos.
        ankle_pos (tuple): Coordenadas (x, y) del keypoint del tobillo, para fines gráficos.
        pixel_scale (float): Factor de conversión de píxeles a metros, calculado en base al bounding box.
        calib_disp (list): Historial de desplazamientos (en píxeles) durante la fase de calibración.
        disp_threshold (float): Umbral dinámico en metros para detectar el inicio de la repetición.
        user_height (float): Altura real del sujeto en metros.
        displacement_history (list): Historial de desplazamientos filtrados (en metros) durante la fase concéntrica.
        savgol_window (int): Tamaño de la ventana usada en el filtro Savitzky–Golay.
        savgol_polyorder (int): Orden del polinomio en el filtro Savitzky–Golay.
    """

    def __init__(self, side: str, user_height: float):
        """Inicializa una instancia de DeadliftExercise.

        Args:
            side (str): Lado a utilizar (por ejemplo, "left" o "right").
            user_height (float): Altura del sujeto (en metros) para calibrar la escala.
        """
        super().__init__()
        self.side = side.lower()
        self.counter = 0
        self.stage = "up"         # Inicialmente, se asume la posición de pie
        self.rep_finished = False # Se evita el conteo inmediato de repeticiones consecutivas.
        self.rep_start_time = None
        self._latest_displacement = 0
        self.wrist_pos = None
        self.ankle_pos = None

        self.pixel_scale = None   # Se calculará usando la diferencia en Y del bounding box.
        self.calib_disp = []      # Historial de desplazamientos (en píxeles) durante la calibración.
        self.disp_threshold = None  # Umbral dinámico en metros.
        self.user_height = user_height

        # Historial para el desplazamiento actual (fase concéntrica)
        self.displacement_history = []

        # Parámetros para el filtro Savitzky–Golay y mediana.
        self.savgol_window = 3
        self.savgol_polyorder = 2

    @property
    def latest_angle(self):
        """float: Devuelve el último desplazamiento filtrado calculado (en metros)."""
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
        wrist_idx = K.YOLO_POSE_KEYPOINTS[f'{side_str}_WRIST']
        ankle_idx = K.YOLO_POSE_KEYPOINTS[f'{side_str}_ANKLE']

        # Calibración de pixel_scale usando el bounding box.
        if self.pixel_scale is None:
            y_coords = keypoints[:, 1]
            bbox_top = np.min(y_coords)
            bbox_bottom = np.max(y_coords)
            bbox_height_px = bbox_bottom - bbox_top
            if bbox_height_px > 0:
                self.pixel_scale = self.user_height / bbox_height_px
                print(f"Calibrated using bbox: 1px = {self.pixel_scale:.4f} m")

        # Calibración del umbral dinámico a partir de los desplazamientos (wrist-ankle)
        if self.disp_threshold is None:
            if valid_keypoints(confs, [wrist_idx, ankle_idx]):
                wrist = keypoints[wrist_idx]
                ankle = keypoints[ankle_idx]
                disp_px = abs(ankle[1] - wrist[1])
                self.calib_disp.append(disp_px)
                print(f"Calibration displacement: {disp_px:.2f} px (acumulated: {len(self.calib_disp)})")
                if len(self.calib_disp) >= 5:
                    max_disp_px = max(self.calib_disp)
                    self.disp_threshold = (max_disp_px * self.pixel_scale) - 0.04
                    print(f"Dynamic threshold calculated: {self.disp_threshold:.4f} m")
            return None

        # Si la pose es válida, se procede a calcular el desplazamiento actual:
        if valid_keypoints(confs, [wrist_idx, ankle_idx]):
            wrist = keypoints[wrist_idx]
            ankle = keypoints[ankle_idx]
            displacement_px = abs(ankle[1] - wrist[1])
            displacement_m = displacement_px * self.pixel_scale

            # Agregar el desplazamiento a la historia para filtrado
            self.displacement_history.append(displacement_m)

            # Aplicar filtro Savitzky–Golay (adaptado a la cantidad de puntos disponibles)
            if len(self.displacement_history) >= self.savgol_window:
                # Se asegura que la ventana sea la adecuada; si hay menos puntos, se usa la señal sin filtrado adicional
                smoothed_displacement = savgol_filter(self.displacement_history,
                                                      window_length=self.savgol_window,
                                                      polyorder=self.savgol_polyorder)[-1]
            else:
                smoothed_displacement = displacement_m

            self._latest_displacement = smoothed_displacement
            print(f"[Deadlift] SavGol filtered displacement {smoothed_displacement:.2f} m")
            self.wrist_pos = tuple(map(int, wrist))
            self.ankle_pos = tuple(map(int, ankle))

            # Reinicia el cooldown si el desplazamiento baja al 95% del umbral
            if self.stage == "up" and self.rep_finished:
                reset_threshold = self.disp_threshold * 0.95
                if smoothed_displacement < reset_threshold:
                    self.rep_finished = False

            # Transición: de "up" a "down" para iniciar la rep
            if self.stage == "up" and not self.rep_finished and smoothed_displacement < self.disp_threshold:
                self.rep_start_time = current_time
                self.stage = "down"
            # Transición: de "down" a "up" para finalizar la rep si se supera el umbral + 0.02 m
            elif self.stage == "down" and smoothed_displacement > self.disp_threshold + 0.02:
                self.current_rep_time = current_time - self.rep_start_time
                self.rep_start_time = None
                self.counter += 1
                self.stage = "up"
                self.rep_finished = True
                print(f"[Deadlift] Transition to UP: filtered displacement {smoothed_displacement:.2f}, rep count: {self.counter}, rep time: {self.current_rep_time:.2f}")
                # Reiniciar la historia para la siguiente repetición
                self.displacement_history.clear()

            return smoothed_displacement
        return None

    def draw(self, frame):
        """Dibuja los overlays visuales (opcional) para el ejercicio de Deadlift.

        Este método genera gráficos sobre el frame para visualizar la conexión entre la
        posición detectada (muñeca y tobillo) y el desplazamiento calculado.

        Args:
            frame (ndarray): Imagen del frame del video.
            
        Returns:
            ndarray: El frame con los overlays dibujados.
        """
        # Código de dibujo comentado: descomentar para visualizar.
        """
        cv2.rectangle(frame, (0, 0), (300, 73), (245, 117, 16), -1)
        cv2.putText(frame, 'DEADLIFT', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, str(self.counter),
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, 'STAGE', (135, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        stage_text = self.stage if self.stage is not None else ""
        cv2.putText(frame, stage_text,
                    (90, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        if self._latest_displacement is not None and self.wrist_pos is not None and self.ankle_pos is not None:
            cv2.line(frame, self.wrist_pos, self.ankle_pos, (0, 255, 0), 2)
            midpoint = ((self.wrist_pos[0] + self.ankle_pos[0]) // 2,
                        (self.wrist_pos[1] + self.ankle_pos[1]) // 2)
            cv2.putText(frame, f"Desp: {self._latest_displacement:.1f}", midpoint,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(frame, f"Desp: {self._latest_displacement:.1f}", midpoint,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        """
        return frame

