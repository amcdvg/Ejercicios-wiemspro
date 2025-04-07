import cv2
from constants import Constants as K
from exercises.base import Exercise
from utils._utils import calculate_angle, valid_keypoints, valid_full_pose, smooth_angles
import numpy as np
import time

def logging_debug(f):
    def wrapper(*args, **kwards):
        print(f'Obtaining method: {f.__name__}')
        result = f(*args, **kwards)
        return result
    return wrapper
class SquatExercise(Exercise):
    """Ejercicio de Squat (sentadilla) utilizando el patrón Template Method.
    Esta clase implementa la lógica específica para el ejercicio de sentadilla,
    donde se miden dos ángulos:
      - El ángulo de la pierna, calculado entre el hombro, la cadera y la rodilla.
      - El ángulo del torso, calculado entre el hombro, la cadera y la rodilla.
    El ejercicio se modela en dos fases:
    - Estado "down": cuando el usuario está en posición de sentadilla (ángulo de pierna bajo).
    - Estado "up": cuando el usuario se levanta (ángulo de pierna alto).
    Se cuenta una repetición cuando se detecta el cambio completo entre estas fases:
      - Se inicia en "down" (posición agachada).
      - Al levantarse y alcanzar un ángulo mayor a SQUAT_MAX_ANGLE, se cuenta una repetición y se cambia a "up".
      - Luego, al bajar de nuevo y alcanzar un ángulo menor a SQUAT_MIN_ANGLE, se vuelve a cambiar a "down".
    Atributos:
        counter (int): Número de repeticiones completadas.
        stage (str): Estado actual del ejercicio ("down" o "up").
        latest_leg_angle (float): Último ángulo medido en la pierna.
        latest_torso_angle (float): Último ángulo medido en el torso.
        leg_angle_pos (tuple): Coordenadas (x, y) para dibujar el ángulo de la pierna.
        torso_angle_pos (tuple): Coordenadas (x, y) para dibujar el ángulo del torso.
        down_start_time (float): Tiempo de inicio de la fase de bajada.
        up_start_time (float): Tiempo de inicio de la fase de subida.
    """
    def __init__(self, side):
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
    @property
    def latest_angle(self):
        """Devuelve el último ángulo relevante para el ejercicio (ángulo de la pierna).
        Returns:
            float or None: El ángulo medido de la pierna, o None si aún no se ha calculado.
        """
        return self.latest_leg_angle
    
    def update(self, keypoints, confs: float, current_time):
        """
        Actualiza el estado del ejercicio de sentadilla.
        Se calcula el ángulo de la pierna (entre cadera, rodilla y tobillo) y el ángulo del torso (entre hombro, cadera y rodilla).
        Se aplica un suavizado (mediana de las últimas 5 muestras) y, según los umbrales definidos (SQUAT_MIN_ANGLE y SQUAT_MAX_ANGLE),
        se detecta la transición:
        - De "up" a "down": cuando el ángulo baja por debajo de SQUAT_MIN_ANGLE, se marca el inicio de la repetición y se asigna rep_start_time = current_time.
        - De "down" a "up": cuando el ángulo sube por encima de SQUAT_MAX_ANGLE, se calcula el tiempo de la repetición,
            se incrementa el contador, se activa el cooldown (rep_finished = True) y se limpia el historial.
        Durante el cooldown, no se actualiza el temporizador (el wheel muestra 0) hasta que se detecta que el ángulo ha
        caído lo suficiente (por ejemplo, que sea menor que SQUAT_MAX_ANGLE - 10).
        
        Returns:
            float or None: El ángulo suavizado (de la pierna) o None si no se detecta una pose válida.
        """
        from constants import Constants as K  # En caso de necesitar referencia interna

        if not valid_full_pose(confs, threshold=0.3):
            return None
        side_str = self.side.upper()  # 'LEFT' o 'RIGHT'

        r_shoulder_idx = K.YOLO_POSE_KEYPOINTS[f'{side_str}_SHOULDER']
        r_hip_idx = K.YOLO_POSE_KEYPOINTS[f'{side_str}_HIP']
        r_knee_idx = K.YOLO_POSE_KEYPOINTS[f'{side_str}_KNEE']
        r_ankle_idx = K.YOLO_POSE_KEYPOINTS[f'{side_str}_ANKLE']

        if valid_keypoints(confs, [r_hip_idx, r_knee_idx, r_ankle_idx]):
            r_shoulder = keypoints[r_shoulder_idx]
            r_hip = keypoints[r_hip_idx]
            r_knee = keypoints[r_knee_idx]
            r_ankle = keypoints[r_ankle_idx]

            leg_angle = calculate_angle(r_hip, r_knee, r_ankle)
            torso_angle = calculate_angle(r_shoulder, r_hip, r_knee)
            self.angles_history.append(leg_angle)
            # Suavizamos usando la mediana de las últimas 5 muestras
            if len(self.angles_history) >= 5:
                smoothed_leg_angle = np.median(self.angles_history[-5:])
            else:
                smoothed_leg_angle = leg_angle

            # Actualizamos atributos para visualización
            self.latest_leg_angle = smoothed_leg_angle
            self.latest_torso_angle = torso_angle
            self.leg_angle_pos = tuple(map(int, r_knee))
            self.torso_angle_pos = tuple(map(int, r_hip))

            # Si estamos en cooldown, esperamos a que el ángulo baje lo suficiente para liberar el bloqueo.
            if hasattr(self, 'rep_finished') and self.rep_finished:
                # Aquí definimos que el bloqueo se libera cuando el ángulo baja por debajo de (SQUAT_MAX_ANGLE - 10)
                if smoothed_leg_angle < K.SQUAT_MAX_ANGLE - 10:
                    self.rep_finished = False
                    print(f"[Squat][DEBUG] Cooldown finalizado: angle bajó a {smoothed_leg_angle:.2f}")
                return smoothed_leg_angle

            # Transición de "up" a "down": inicio de la repetición
            if self.stage == "up" and smoothed_leg_angle < K.SQUAT_MIN_ANGLE:
                # Usamos current_time para que coincida con la escala del overlay
                self.rep_start_time = current_time
                self.rep_start_time_abs = time.time()
                self.stage = "down"
                print(f"[Squat] Transition to DOWN: angle {smoothed_leg_angle:.2f}")
            
            # Transición de "down" a "up": final de la repetición
            elif self.stage == "down" and smoothed_leg_angle > K.SQUAT_MAX_ANGLE:
                self.current_rep_time = current_time - self.rep_start_time
                self.rep_start_time = None  # Reiniciamos para la siguiente rep
                self.counter += 1
                self.stage = "up"
                self.rep_finished = True  # Activamos el cooldown para evitar contar inmediatamente otra rep
                print(f"[Squat] Transition to UP: angle {smoothed_leg_angle:.2f}, rep count: {self.counter}, rep time: {self.current_rep_time:.2f}")
                self.angles_history.clear()
            return smoothed_leg_angle
        return None

    
    def draw(self, frame):
        """Dibuja la interfaz del ejercicio de curl en el frame.
        Se dibujan un recuadro con el nombre del ejercicio, el contador de repeticiones, 
        el estado actual y el ángulo detectado, con contorno para mejorar la legibilidad.
        Args:
            frame (numpy.ndarray): Imagen actual del frame.
        Returns:
            numpy.ndarray: El frame actualizado con los overlays del ejercicio.
        """
        cv2.rectangle(frame, (0, 0), (300, 73), (245, 117, 16), -1)
        cv2.putText(frame, 'SQUAT', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(frame, str(self.counter),
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(frame, 'STAGE', (135, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        stage_text = self.stage if self.stage is not None else ""
        cv2.putText(frame, stage_text,
                    (90, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        if self.latest_leg_angle is not None and self.leg_angle_pos is not None:
            cv2.putText(frame, f"Leg: {self.latest_leg_angle:.1f}", self.leg_angle_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 5, cv2.LINE_AA)
            cv2.putText(frame, f"Leg: {self.latest_leg_angle:.1f}", self.leg_angle_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
        if self.latest_torso_angle is not None and self.torso_angle_pos is not None:
            cv2.putText(frame, f"Torso: {self.latest_torso_angle:.1f}", self.torso_angle_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 5, cv2.LINE_AA)
            cv2.putText(frame, f"Torso: {self.latest_torso_angle:.1f}", self.torso_angle_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)