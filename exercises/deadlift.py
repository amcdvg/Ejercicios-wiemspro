import cv2
from constants import Constants as K
from exercises.base import Exercise
from utils._utils import valid_keypoints, valid_full_pose
import numpy as np
import time


class DeadliftExercise(Exercise):
    """
    Clase que implementa el ejercicio de peso muerto utilizando el desplazamiento relativo
    entre la muñeca y el tobillo para detectar fases y repeticiones, en lugar de ángulos articulares.
    
    Args:
        Exercise: Clase base para ejercicios
    """
    def __init__(self, side: str):
        super().__init__()
        self.side = side.lower()
        self.counter = 0
        self.stage = "up"  # Inicia en posición de pie (up)
        self.rep_finished = False  # Para evitar conteos inmediatos tras finalizar una rep
        self.displacement_history = []  # Para suavizar el desplazamiento relativo
        self.rep_start_time = None  # Tiempo relativo de inicio de la rep
        self._latest_displacement = None  # Último desplazamiento relativo (muñeca - tobillo)
        self.wrist_pos = None  # Posición de la muñeca para dibujar
        self.ankle_pos = None  # Posición del tobillo para dibujar
        self.scale_factor = None
        self.calibration_frames = []
        self.calibrated = False

    @property
    def latest_angle(self):
        # Mantenemos esta propiedad para compatibilidad con el código existente
        # que espera un ángulo, pero ahora devuelve un valor de desplazamiento normalizado
        return self._latest_displacement
    
    def update(self, keypoints, confs: float, current_time):
        """
        Actualiza el estado del ejercicio basado en el desplazamiento relativo entre
        la muñeca y el tobillo.

        Args:
            keypoints: Puntos clave detectados del cuerpo
            confs (float): Valores de confianza para los puntos detectados
            current_time: Tiempo actual relativo

        Returns:
            float or None: Valor del desplazamiento relativo si se detectó correctamente
        """
        if not valid_full_pose(confs, threshold=0.3):
            return None
        
        side_str = self.side.upper()
        wrist_idx = K.YOLO_POSE_KEYPOINTS[f'{side_str}_WRIST']
        ankle_idx = K.YOLO_POSE_KEYPOINTS[f'{side_str}_ANKLE']

        if valid_keypoints(confs, [wrist_idx, ankle_idx]):
            wrist = keypoints[wrist_idx]
            ankle = keypoints[ankle_idx]
            
            # Calculamos el desplazamiento vertical relativo (coordenada Y)
            # Nota: en coordenadas de imagen, Y aumenta hacia abajo
            raw_displacement =   ankle[1] - wrist[1]
            
            # Guardamos el desplazamiento en el historial para suavizarlo
            self.displacement_history.append(raw_displacement)
            
            # Suavizamos el desplazamiento con una ventana deslizante (mediana)
            if len(self.displacement_history) >= 5:
                smoothed_displacement = np.median(self.displacement_history[-5:])
            else:
                smoothed_displacement = raw_displacement
                
            self._latest_displacement = smoothed_displacement
            print(f"[Deadlift] displacement {smoothed_displacement:.2f}")
            self.wrist_pos = tuple(map(int, wrist))
            self.ankle_pos = tuple(map(int, ankle))

            # Lógica para evitar contar repeticiones inmediatamente después de terminar una
            if self.rep_finished:
                # Usamos un umbral relativo para determinar cuando podemos iniciar una nueva repetición
                # Valores positivos significa que la muñeca está por debajo del tobillo (posición inicial)
                if smoothed_displacement > K.DEADLIFT_MIN_VERTICAL_DISPLACEMENT + 10:
                    self.rep_finished = False
                return smoothed_displacement
                
            # Transición de "up" a "down": Se inicia la repetición cuando la muñeca
            # desciende por debajo de cierto umbral relativo al tobillo
            if self.stage == "up" and smoothed_displacement < K.DEADLIFT_MAX_VERTICAL_DISPLACEMENT:
                self.rep_start_time = current_time
                self.stage = "down"
            
            # Transición de "down" a "up": Se completa la repetición cuando la muñeca
            # vuelve a ascender por encima de cierto umbral relativo al tobillo
            elif self.stage == "down" and smoothed_displacement > K.DEADLIFT_MIN_VERTICAL_DISPLACEMENT:
                self.current_rep_time = current_time - self.rep_start_time
                self.rep_start_time = None  # Reiniciamos para la siguiente rep
                self.counter += 1
                self.stage = "up"
                self.rep_finished = True
                print(f"[Deadlift] Transition to UP: displacement {smoothed_displacement:.2f}, rep count: {self.counter}, rep time: {self.current_rep_time:.2f}")
                self.displacement_history.clear()
                
            return smoothed_displacement
        return None

    def draw(self, frame):
        """
        Dibuja los overlays para el ejercicio de Deadlift en el frame.
        Muestra un recuadro con el nombre del ejercicio, el contador de repeticiones,
        el estado actual y el desplazamiento relativo detectado.
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
        
        # Dibujamos el desplazamiento relativo entre muñeca y tobillo
        if self._latest_displacement is not None and self.wrist_pos is not None and self.ankle_pos is not None:
            # Dibuja una línea entre la muñeca y el tobillo
            cv2.line(frame, self.wrist_pos, self.ankle_pos, (0, 255, 0), 2)
            
            # Muestra el valor del desplazamiento
            midpoint = ((self.wrist_pos[0] + self.ankle_pos[0]) // 2, 
                         (self.wrist_pos[1] + self.ankle_pos[1]) // 2)
            cv2.putText(frame, f"Desp: {self._latest_displacement:.1f}", midpoint,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 5, cv2.LINE_AA)
            cv2.putText(frame, f"Desp: {self._latest_displacement:.1f}", midpoint,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
        """
        return frame