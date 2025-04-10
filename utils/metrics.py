from constants import Constants as K
import numpy as np
from scipy.signal import savgol_filter
from utils._utils import smooth_angles
from utils.anthropometry import Anthopometry
from datetime import datetime
import time

class Metrics:
    def __init__(self, exercise: str, height: float, gender: str, age: int):
        """
        Inicializa el objeto de métricas para análisis de ejercicios.
        
        Args:
            exercise (str): Tipo de ejercicio
            height (float): Altura del sujeto en metros
            gender (str): Género del sujeto
            age (int): Edad del sujeto
        """
        self.exercise = exercise.lower()
        self.height = height
        self.gender = gender.lower()
        self.age = age
        
        # Mantenemos las funciones de regresión antropométrica para la compatibilidad
        regression_functions = {
            'curl': Anthopometry.calc_forearm_length,
            'pushup': Anthopometry.calc_pushup_length,
            'squat': Anthopometry.calc_squat_length,
            'deadlift': Anthopometry.calc_deadlift_length,
            'reverse_fly': Anthopometry.calc_reverse_fly_length,
            'swing': Anthopometry.calc_swing_length,
            'renegade_row': Anthopometry.calc_renegade_row_length,
            'rear_lunge': Anthopometry.calc_rear_lunge_length,
            'bench_dips': Anthopometry.calc_bench_dips_length,
            'overhead_triceps': Anthopometry.calc_overhead_triceps_length,
            'plank': Anthopometry.calc_plank_length
        }
        
        calc_func = regression_functions.get(self.exercise, lambda h, a, g: 0.16 * h)
        self.segment_length = calc_func(self.height, self.age, self.gender)
        
        # Usamos las mismas variables para mantener compatibilidad, pero ahora
        # "angles" contendrá valores de desplazamiento relativo
        self.angles = []  # Ahora contiene valores de desplazamiento
        self.timestamps = []
        self.history = []
        self.start_time = None

    def update(self, displacement: float, timestamp: float):
        """
        Actualiza los datos con un nuevo valor de desplazamiento y timestamp.
        
        Args:
            displacement (float): Valor de desplazamiento relativo
            timestamp (float): Timestamp actual
        """
        if not self.timestamps:
            self.start_time = timestamp
        self.angles.append(displacement)  # Guardamos el desplazamiento en angles para mantener compatibilidad
        self.timestamps.append(timestamp)

    def calculate_rom(self) -> float:
        """
        Calcula el Rango de Movimiento (ROM) basado en los valores de desplazamiento.
        
        Returns:
            float: ROM en metros, multiplicado por 100 para obtener centímetros
        """
        if not self.angles:
            return 0.0
        
        # Suavizamos los valores de desplazamiento
        smoothed_displacements = smooth_angles(self.angles, window_length=5, polyorder=2)
        
        # Calculamos el rango de desplazamiento
        min_displacement = np.min(smoothed_displacements)
        max_displacement = np.max(smoothed_displacements)
        displacement_range = max_displacement - min_displacement
        
        # Normalizamos el rango de desplazamiento de píxeles a metros usando antropometría
        # Usamos la longitud del segmento como factor de escala
        segment_ratio = self.segment_length / self.height
        
        # Para deadlift, usamos la altura total como referencia y ajustamos por el segmento
        if self.exercise == 'deadlift':
            # El valor de desplazamiento representa una fracción de la altura corporal
            # El factor de escala lo ajustamos según la proporción típica del ejercicio
            rom_factor = K.ROM_BASE_FACTORS.get(self.exercise, 0.91)
            rom = displacement_range * segment_ratio * rom_factor
        else:
            # Para otros ejercicios mantenemos la lógica original
            delta_angle = max_displacement - min_displacement
            delta_rad = np.deg2rad(delta_angle)
            rom_factor = K.ROM_BASE_FACTORS.get(self.exercise, 0.91)
            rom = self._get_effective_length() * delta_rad * rom_factor
            
        return rom

    def _get_effective_length(self) -> float:
        """
        Obtiene la longitud efectiva del segmento para el ejercicio actual.
        
        Returns:
            float: Longitud efectiva en metros
        """
        if K.USE_REGRESSION_EFFECTIVE_LENGTH:
            return self.segment_length
        else:
            calc_func = K.EFFECTIVE_LENGTH_FUNCTIONS.get(self.exercise, lambda h: 0.15 * h)
            return calc_func(self.height)

    def calculate_vmed(self) -> float:
        """
        Calcula la velocidad media (VMED) basada en los valores de desplazamiento.
        
        Returns:
            float: Velocidad media en m/s
        """
        if len(self.timestamps) < 2:
            return 0.0
        
        # Para el caso específico de deadlift, usamos el enfoque de desplazamiento
        if self.exercise == 'deadlift':
            filtered_displacements = smooth_angles(self.angles, window_length=5, polyorder=2)
            delta_t = np.diff(self.timestamps)
            total_time = np.sum(delta_t)
            
            if total_time <= 0:
                return 0.0
                
            # Calculamos la distancia total recorrida en el espacio de desplazamiento
            total_displacement = sum(abs(filtered_displacements[i] - filtered_displacements[i - 1]) 
                                    for i in range(1, len(filtered_displacements)))
            
            # Convertimos el desplazamiento en píxeles a metros usando la altura como referencia
            # y ajustamos según la proporción del segmento
            segment_ratio = self.segment_length / self.height
            total_distance = total_displacement * segment_ratio
            
            raw_vmed = total_distance / total_time
            
            # Aplicamos los factores de corrección existentes
            exercise_factor = K.VMED_CORRECTION_FACTORS.get(self.exercise, 1.0)
            phase_factor = K.PHASE_CORRECTION.get('eccentric', 1.0)
            base_factor = K.VMED_BASE_FACTORS.get(self.exercise, 0.85)
            final_factor = K.VMED_FINAL_FACTORS.get(self.exercise, 0.62)
            
            corrected_vmed = raw_vmed * exercise_factor * phase_factor * base_factor * final_factor
            vmed_clipped = np.clip(corrected_vmed, 0.2, 2.6)
            
            return vmed_clipped
        else:
            # Para otros ejercicios, mantenemos la lógica original
            filtered_angles = smooth_angles(self.angles, window_length=5, polyorder=2)
            delta_t = np.diff(self.timestamps)
            total_time = np.sum(delta_t)
            
            if total_time <= 0:
                return 0.0
                
            total_deg = sum(abs(filtered_angles[i] - filtered_angles[i - 1]) 
                           for i in range(1, len(filtered_angles)))
            total_rad = np.deg2rad(total_deg)
            L = self._get_effective_length()
            total_distance = total_rad * L
            
            raw_vmed = total_distance / total_time
            
            exercise_factor = K.VMED_CORRECTION_FACTORS.get(self.exercise, 1.0)
            phase_factor = K.PHASE_CORRECTION.get('eccentric', 1.0)
            base_factor = K.VMED_BASE_FACTORS.get(self.exercise, 0.85)
            final_factor = K.VMED_FINAL_FACTORS.get(self.exercise, 0.62)
            
            corrected_vmed = raw_vmed * exercise_factor * phase_factor * base_factor * final_factor
            vmed_clipped = np.clip(corrected_vmed, 0.2, 2.6)
            
            return vmed_clipped

    def calculate_vmax(self) -> float:
        """
        Calcula la velocidad máxima (VMAX) basada en los valores de desplazamiento.
        
        Returns:
            float: Velocidad máxima en m/s
        """
        if len(self.angles) < 2:
            return 0.0
        
        # Para el caso específico de deadlift, usamos el enfoque de desplazamiento
        if self.exercise == 'deadlift':
            filtered_displacements = smooth_angles(self.angles, window_length=5, polyorder=2)
            delta_t = np.diff(self.timestamps)
            delta_disp = np.abs(np.diff(filtered_displacements))
            
            # Convertimos el desplazamiento en píxeles a metros usando la altura como referencia
            segment_ratio = self.segment_length / self.height
            
            # Calculamos velocidades instantáneas
            with np.errstate(divide='ignore', invalid='ignore'):
                v_instant = delta_disp * segment_ratio / delta_t
                
            valid_mask = (delta_t > 0.01) & np.isfinite(v_instant)
            v_clean = v_instant[valid_mask]
            
            if len(v_clean) == 0:
                return 0.0
                
            try:
                window_size = min(5, len(v_clean))
                window_size = window_size if window_size % 2 != 0 else window_size - 1
                if window_size < 3:
                    raise ValueError
                v_smooth = savgol_filter(v_clean, window_size, 3)
                vmax = np.max(v_smooth)
            except Exception:
                vmax = np.max(v_clean) if len(v_clean) > 0 else 0.0
                
            # Aplicamos los factores de corrección existentes
            factor = K.ADJUSTMENT_FACTORS_VMAX.get(self.exercise, 0.856)
            base_factor = K.VMAX_BASE_FACTORS.get(self.exercise, 0.48)
            
            corrected_vmax = np.clip(vmax * factor * base_factor, 0.0, 3.3)
            vmed = self.calculate_vmed()
            
            if corrected_vmax < vmed:
                corrected_vmax = vmed + 0.1
                
            return corrected_vmax
        else:
            # Para otros ejercicios, mantenemos la lógica original
            filtered_angles = smooth_angles(self.angles, window_length=5, polyorder=2)
            delta_t = np.diff(self.timestamps)
            delta_ang = np.abs(np.diff(filtered_angles))
            L = self._get_effective_length()
            
            with np.errstate(divide='ignore', invalid='ignore'):
                v_instant = L * np.deg2rad(delta_ang) / delta_t
                
            valid_mask = (delta_t > 0.01) & np.isfinite(v_instant)
            v_clean = v_instant[valid_mask]
            
            if len(v_clean) == 0:
                return 0.0
                
            try:
                window_size = min(5, len(v_clean))
                window_size = window_size if window_size % 2 != 0 else window_size - 1
                if window_size < 3:
                    raise ValueError
                v_smooth = savgol_filter(v_clean, window_size, 3)
                vmax = np.max(v_smooth)
            except Exception:
                vmax = np.max(v_clean) if len(v_clean) > 0 else 0.0
                
            factor = K.ADJUSTMENT_FACTORS_VMAX.get(self.exercise, 0.856)
            base_factor = K.VMAX_BASE_FACTORS.get(self.exercise, 0.48)
            
            corrected_vmax = np.clip(vmax * factor * base_factor, 0.0, 3.3)
            vmed = self.calculate_vmed()
            
            if corrected_vmax < vmed:
                corrected_vmax = vmed + 0.1
                
            return corrected_vmax

    def get_metrics(self, repetition: int = None) -> dict:
        """
        Obtiene un diccionario con todas las métricas calculadas.
        
        Args:
            repetition (int, optional): Número de repetición. Por defecto es None.
            
        Returns:
            dict: Diccionario con todas las métricas calculadas
        """
        if len(self.timestamps) < 2:
            rep_time = 0.0
        else:
            rep_time = self.timestamps[-1] - self.timestamps[0]
        
        start_datetime = datetime.fromtimestamp(time.time() - rep_time).isoformat()
        min_value = min(self.angles) if self.angles else None
        max_value = max(self.angles) if self.angles else None
        effective_length = self._get_effective_length()
        
        # Para deadlift, reportamos los valores de desplazamiento en lugar de ángulos
        if self.exercise == 'deadlift':
            metrics_dict = {
                "start_datetime": start_datetime,
                "age": self.age,
                "gender": self.gender,
                "height (m)": round(self.height, 4),
                "effective_length (m)": round(effective_length, 4),
                "min_displacement": round(min_value, 4) if min_value is not None else None,
                "max_displacement": round(max_value, 4) if max_value is not None else None,
                "ROM (cm)": round(self.calculate_rom()*100, 4),  # Convertimos a cm
                "VMED (m/s)": round(self.calculate_vmed(), 4),
                "VMAX (m/s)": round(self.calculate_vmax(), 4),
                "rep_time": round(rep_time, 4),
                "repetition": repetition if repetition is not None else 0,
                "exercise": self.exercise
            }
        else:
            # Para otros ejercicios, mantenemos el formato original
            metrics_dict = {
                "start_datetime": start_datetime,
                "age": self.age,
                "gender": self.gender,
                "height (m)": round(self.height, 4),
                "effective_length (m)": round(effective_length, 4),
                "min_angle (°)": round(min_value, 4) if min_value is not None else None,
                "max_angle (°)": round(max_value, 4) if max_value is not None else None,
                "ROM (cm)": round(self.calculate_rom()*100, 4),
                "VMED (m/s)": round(self.calculate_vmed(), 4),
                "VMAX (m/s)": round(self.calculate_vmax(), 4),
                "rep_time": round(rep_time, 4),
                "repetition": repetition if repetition is not None else 0,
                "exercise": self.exercise
            }
        
        return metrics_dict

    def reset(self):
        """
        Reinicia las listas de ángulos/desplazamientos y timestamps.
        """
        self.angles = []
        self.timestamps = []