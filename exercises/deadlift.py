import cv2
from constants import Constants as K
from exercises.base import Exercise
from utils._utils import valid_keypoints, valid_full_pose
import numpy as np
import time
from scipy.signal import savgol_filter  # Asegúrate de tener scipy instalada

class DeadliftExercise(Exercise):
    """
    Implementa el ejercicio de peso muerto utilizando el desplazamiento relativo
    entre la muñeca y el tobillo para detectar fases y contar repeticiones.
    
    Calibración:
      - Se estima el pixel_scale usando el bounding box de la persona, calculado como la diferencia
        en Y entre el punto superior y el inferior de los keypoints.
      - Se acumulan mediciones de desplazamiento (wrist-ankle) en los primeros frames para calcular
        un umbral dinámico (disp_threshold) en metros, restándole 0.05 m al máximo acumulado.
    """
    def __init__(self, side: str, user_height: float):
        super().__init__()
        self.side = side.lower()
        self.counter = 0
        self.stage = "up"         # Inicialmente en "up" (posición de pie)
        self.rep_finished = False # Flag para evitar conteos dobles
        self.rep_start_time = None
        self._latest_displacement = 0
        self.wrist_pos = None
        self.ankle_pos = None

        self.pixel_scale = None   # Factor de conversión de píxeles a metros
        self.calib_disp = []      # Historial de desplazamientos (wrist-ankle) durante la calibración
        self.disp_threshold = None  # Umbral dinámico en metros

        self.user_height = user_height

        # Para almacenar la historia de desplazamientos (para aplicar Savitzky-Golay)
        self.displacement_history = []

        # Parámetros para el Savitzky-Golay:
        self.savgol_window = 3     # Debe ser un entero impar
        self.savgol_polyorder = 2

    @property
    def latest_angle(self):
        # Se mantiene por compatibilidad; ahora es el desplazamiento filtrado (suavizado)
        return self._latest_displacement

    def get_pixel_scale(self):
        return self.pixel_scale

    def update(self, keypoints, confs: float, current_time):
        """
        Actualiza el estado del ejercicio basado en el desplazamiento entre muñeca y tobillo,
        aplicando un filtro Savitzky-Golay para suavizar la señal, y administra la detección de repeticiones.
        """
        if not valid_full_pose(confs, threshold=0.3):
            return None

        side_str = self.side.upper()
        wrist_idx = K.YOLO_POSE_KEYPOINTS[f'{side_str}_WRIST']
        ankle_idx = K.YOLO_POSE_KEYPOINTS[f'{side_str}_ANKLE']

        # ----------------------------
        # Calibrar el pixel_scale usando el bounding box
        # ----------------------------
        if self.pixel_scale is None:
            y_coords = keypoints[:, 1]
            bbox_top = np.min(y_coords)
            bbox_bottom = np.max(y_coords)
            bbox_height_px = bbox_bottom - bbox_top
            if bbox_height_px > 0:
                self.pixel_scale = self.user_height / bbox_height_px
                print(f"Calibrated using bbox: 1px = {self.pixel_scale:.4f} m")

        # ----------------------------
        # Calibrar el umbral dinámico acumulando desplazamientos
        # ----------------------------
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
            return None  # Mientras no se tenga el threshold, se retorna None

        # ----------------------------
        # Medición actual: obtener desplazamiento en metros a partir de keypoints
        # ----------------------------
        if valid_keypoints(confs, [wrist_idx, ankle_idx]):
            wrist = keypoints[wrist_idx]
            ankle = keypoints[ankle_idx]
            displacement_px = abs(ankle[1] - wrist[1])
            displacement_m = displacement_px * self.pixel_scale

            # Acumular la medición en la historia para el filtrado
            self.displacement_history.append(displacement_m)

            # Aplicar Savitzky-Golay si tenemos suficientes datos en la historia:
            if len(self.displacement_history) >= self.savgol_window:
                # Asegurarse de que la ventana sea impar y <= a la longitud del vector
                window = self.savgol_window if self.savgol_window <= len(self.displacement_history) else len(self.displacement_history) | 1
                smoothed_displacement = savgol_filter(self.displacement_history, window_length=window, polyorder=self.savgol_polyorder)[-1]
            else:
                smoothed_displacement = displacement_m

            self._latest_displacement = smoothed_displacement
            print(f"[Deadlift] SavGol filtered displacement {smoothed_displacement:.2f} m")
            self.wrist_pos = tuple(map(int, wrist))
            self.ankle_pos = tuple(map(int, ankle))

            # ----------------------------
            # Lógica para detección de repeticiones
            # ----------------------------
            # Resetear rep_finished si en estado "up" y el desplazamiento baja al 95% del threshold
            if self.stage == "up" and self.rep_finished:
                reset_threshold = self.disp_threshold * 0.95
                if smoothed_displacement < reset_threshold:
                    self.rep_finished = False

            # Transición de "up" a "down": iniciar la repetición cuando el desplazamiento baja por debajo del threshold
            if self.stage == "up" and not self.rep_finished and smoothed_displacement < self.disp_threshold:
                self.rep_start_time = current_time
                self.stage = "down"
            # Transición de "down" a "up": finalizar la repetición cuando el desplazamiento sube por encima del threshold + 0.02 m
            elif self.stage == "down" and smoothed_displacement > self.disp_threshold + 0.02:
                self.current_rep_time = current_time - self.rep_start_time
                self.rep_start_time = None
                self.counter += 1
                self.stage = "up"
                self.rep_finished = True
                print(f"[Deadlift] Transition to UP: filtered displacement {smoothed_displacement:.2f}, rep count: {self.counter}, rep time: {self.current_rep_time:.2f}")
                # Reiniciamos el historial de desplazamientos para la siguiente repetición
                self.displacement_history.clear()

            return smoothed_displacement
        return None

    def draw(self, frame):
        """
        Dibuja los overlays para el ejercicio de Deadlift en el frame.
        (El bloque de dibujo está comentado; descoméntalo para visualizar).
        """
        """
        cv2.rectangle(frame, (0, 0), (300, 73), (245, 117, 16), -1)
        cv2.putText(frame, 'DEADLIFT', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(frame, str(self.counter),
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(frame, 'STAGE', (135, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        stage_text = self.stage if self.stage is not None else ""
        cv2.putText(frame, stage_text,
                    (90, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        if self._latest_displacement is not None and self.wrist_pos is not None and self.ankle_pos is not None:
            cv2.line(frame, self.wrist_pos, self.ankle_pos, (0, 255, 0), 2)
            midpoint = ((self.wrist_pos[0] + self.ankle_pos[0]) // 2, 
                        (self.wrist_pos[1] + self.ankle_pos[1]) // 2)
            cv2.putText(frame, f"Desp: {self._latest_displacement:.1f}", midpoint,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 5, cv2.LINE_AA)
            cv2.putText(frame, f"Desp: {self._latest_displacement:.1f}", midpoint,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
        """
        return frame
