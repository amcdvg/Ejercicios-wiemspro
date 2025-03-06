from constants import Constants as K
import numpy as np
from scipy.signal import savgol_filter
from utils._utils import filter_valid_angles, compute_total_angular_change, smooth_velocities
from utils.anthropometry import Anthopometry
from datetime import datetime
class Metrics:
    """
    Clase para calcular métricas (ROM, VMED, VMAX) a partir de la variación angular registrada 
    durante una repetición, utilizando modelos de regresión y proporciones antropométricas.
    
    Para "curl" y "pushup" se estima la longitud del antebrazo mediante un modelo de regresión, 
    mientras que para "squat" se utiliza la fórmula clásica de la tibia para estimar la longitud del 
    segmento (rodilla a tobillo). Además, se calcula una "longitud efectiva" según el ejercicio para 
    convertir cambios angulares en desplazamientos lineales.
    """
    def __init__(self, exercise: str, height: float, gender: str, age: int):
        """
        Inicializa la instancia de Metrics.

        Args:
            exercise (str): Tipo de ejercicio ('squat', 'curl', 'pushup', 'plank').
            height (float): Estatura del sujeto en metros.
            gender (str): 'male' o 'female'.
            age (int): Edad en años.
        """
        self.exercise = exercise.lower()
        self.height = height
        self.gender = gender.lower()
        self.age = age
        regression_functions = {
            'curl': lambda h, a, g: Anthopometry.calc_forearm_length(h, a, g, 'curl'),
            'pushup': lambda h, a, g: Anthopometry.calc_forearm_length(h, a, g, 'pushup'),
            'squat': Anthopometry.calc_squat_length,
            'plank': Anthopometry.calc_plank_length}
        calc_func = regression_functions.get(self.exercise, lambda h, a, g: 0.16 * h)
        self.segment_length = calc_func(self.height, self.age, self.gender)
        self.angles = []
        self.timestamps = []
        self.history = []

    def update(self, angle: float, timestamp: float):
        """
        Registra un nuevo ángulo y su correspondiente marca de tiempo.

        Args:
            angle (float): Ángulo medido (en grados) en el instante actual.
            timestamp (float): Marca de tiempo en segundos.
        """
        self.angles.append(angle)
        self.timestamps.append(timestamp)

    def calculate_rom(self) -> float:
        """
        Calcula el rango de movimiento (ROM) en metros para la repetición actual.

        Se utiliza la diferencia entre el ángulo máximo y mínimo (convertida a adianes)
        y se multiplica por la longitud del segmento.

        Returns:
            float: ROM en metros.
        """
        valid_angles = filter_valid_angles(self.angles)
        if not valid_angles:
            return 0.0
        min_angle = min(valid_angles)
        max_angle = max(valid_angles)
        delta_angle = max_angle - min_angle
        delta_rad = np.deg2rad(delta_angle)
        rom = self.segment_length * delta_rad
        return rom * 0.73
    """
    def calculate_rom(self) -> float:
        _summary_
        
        Calcula el rango de movimiento (ROM) en metros para la repetición actual.
 
        Se utiliza la diferencia entre el ángulo máximo y mínimo (convertida a radianes)
        y se multiplica por la longitud del segmento.
 
        Returns:
            float: ROM en metros.
        
        if not self.angles:
            return 0.0
        min_angle = min(self.angles)
        max_angle = max(self.angles)
        valid_angles = [a for a in self.angles if isinstance(a, (int, float))]
        if not valid_angles:
            return 0.0
        min_angle = min(valid_angles)
        max_angle = max(valid_angles)
        delta_angle = max_angle - min_angle
        delta_rad = np.deg2rad(delta_angle)
        rom = self.segment_length * delta_rad
        return rom * 0.73
    """
    def _get_effective_length(self) -> float:
        """
        Calcula la longitud efectiva del segmento basado en proporciones antropométricas.

        Para cada ejercicio se usa una proporción diferente:
        - squat: se utiliza una combinación de proporciones para muslo y pierna.
        - curl: se utiliza el 95% de la longitud del antebrazo.
        - pushup: se utiliza el 75% de la longitud combinada del brazo.
        - plank: se asume que el torso representa el 43% de la estatura.
        - Valor por defecto: 15% de la estatura.

        Returns:
            float: Longitud efectiva en metros.
        """
        # Obtiene la función según el ejercicio; si no existe, se usa un valor por defecto (15% de la estatura)
        calc_func = K.EFFECTIVE_LENGTH_FUNCTIONS.get(self.exercise, lambda h: 0.15 * h)
        return calc_func(self.height)

    def calculate_vmed(self) -> float:
        """
        Calcula la velocidad media (VMED) en m/s durante la repetición actual,
        asegurando que no sea más de 0.6 m/s menor que VMAX y aplicando límites fisiológicos.
        """
        if len(self.timestamps) < 2:
            return 0.0
        
        # 1. Calcular tiempo total ACTIVO (suma de intervalos entre mediciones válidas)
        delta_t = np.diff(self.timestamps)
        total_time = np.sum(delta_t)
        if total_time <= 0:
            return 0.0

        # 2. Calcular distancia angular total
        total_deg = sum(abs(self.angles[i] - self.angles[i-1]) for i in range(1, len(self.angles)))
        total_rad = np.deg2rad(total_deg)
        L = self._get_effective_length()
        total_distance = total_rad * L

        # 3. Cálculo base de VMED
        raw_vmed = total_distance / total_time
        
        # 4. Aplicar factores de corrección
        exercise_factor = K.VMED_CORRECTION_FACTORS.get(self.exercise, 1.0)
        phase_factor = K.PHASE_CORRECTION.get('eccentric', 1.0)
        corrected_vmed = raw_vmed * exercise_factor * phase_factor * 0.85
        
        # 5. Aplicar límites de validez (Sánchez-Moreno et al. 2020)
        vmed_clipped = np.clip(corrected_vmed * 0.63, 0.2, 2.3)
        
        # 6. Asegurar concordancia con VMAX (VMED >= VMAX - 0.6 m/s)
        vmax = self.calculate_vmax()
        lower_bound = max(vmax - 0.6, 0.2)  # No menos de 0.2 m/s
        upper_bound = min(vmax, 2.3)       # No más que VMAX
        return np.clip(vmed_clipped, lower_bound, upper_bound)

    def calculate_vmax(self) -> float:
        """
        Calcula la velocidad máxima (VMAX) en m/s,
        asegurando que no supere los 2.0 m/s con filtrado y ajustes.
        """
        if len(self.angles) < 2:
            return 0.0
        
        # 1. Cálculo de velocidades instantáneas
        delta_t = np.diff(self.timestamps)
        delta_ang = np.abs(np.diff(self.angles))
        L = self._get_effective_length()
        
        with np.errstate(divide='ignore', invalid='ignore'):
            v_instant = L * np.deg2rad(delta_ang) / delta_t
        
        # 2. Filtrar datos inválidos
        valid_mask = (delta_t > 0.01) & np.isfinite(v_instant)
        v_clean = v_instant[valid_mask]
        
        if len(v_clean) == 0:
            return 0.0
        
        # 3. Suavizado adaptativo
        try:
            window_size = min(5, len(v_clean))
            window_size = window_size if window_size % 2 != 0 else window_size - 1
            if window_size < 3:
                raise ValueError
            
            v_smooth = savgol_filter(v_clean, window_size, 3)
            vmax = np.max(v_smooth)
        except Exception:
            vmax = np.max(v_clean) if len(v_clean) > 0 else 0.0
        
        # 4. Aplicar factores y límite fisiológico
        factor = K.ADJUSTMENT_FACTORS_VMAX.get(self.exercise, 0.856)
        corrected_vmax = np.clip(vmax * factor * 0.5, 0.0, 2.0)
        
        return corrected_vmax
        

    def get_metrics(self, repetition: int = None) -> dict:
        """Retorna un diccionario con la información y las métricas calculadas para la repetición actual.

        Los campos incluyen información del sujeto, del ejercicio y las métricas de la repetición,
        organizados en un orden coherente.

        Args:
            repetition (int, optional): Número de repetición. Defaults a 0.

        Returns:
            dict: Diccionario con las métricas e información del ejercicio, que incluye:
                - start_datetime (str): Fecha y hora de inicio de la repetición (formato ISO).
                - age (int): Edad del sujeto.
                - gender (str): Género del sujeto.
                - height (m) (float): Altura en metros.
                - effective_length (m) (float): Longitud efectiva del segmento.
                - min_angle (°) (float): Ángulo mínimo registrado.
                - max_angle (°) (float): Ángulo máximo registrado.
                - ROM (m) (float): Rango de movimiento en metros.
                - VMED (m/s) (float): Velocidad media en m/s.
                - VMAX (m/s) (float): Velocidad máxima en m/s.
                - rep_time (float): Tiempo de repetición en segundos.
                - repetition (int): Número de repetición.
                - exercise (str): Tipo de ejercicio.
        """
        # Se determina la fecha y hora de inicio de la repetición
        start_datetime = datetime.now().isoformat()

        # Calcula el tiempo de repetición basado en la diferencia de timestamps, si existen.
        if len(self.timestamps) < 2:
            rep_time = 0.0
        else:
            delta_t = np.diff(self.timestamps)
            rep_time = float(np.sum(delta_t))

        # Obtener ángulos válidos (suponiendo que filter_valid_angles esté definida)
        valid_angles = filter_valid_angles(self.angles)
        min_angle = min(valid_angles) if valid_angles else None
        max_angle = max(valid_angles) if valid_angles else None

        # Obtener la longitud efectiva mediante la función correspondiente
        effective_length = self._get_effective_length()

        metrics_dict = {
            "start_datetime": start_datetime,
            "age": self.age,
            "gender": self.gender,
            "height (m)": self.height,
            "effective_length (m)": effective_length,
            "min_angle (°)": min_angle,
            "max_angle (°)": max_angle,
            "ROM (m)": self.calculate_rom(),
            "VMED (m/s)": self.calculate_vmed(),
            "VMAX (m/s)": self.calculate_vmax(),
            "rep_time": rep_time,
            "repetition": repetition if repetition is not None else 0,
            "exercise": self.exercise
        }
        return metrics_dict

    def reset(self):
        """Reinicia las listas de ángulos y tiempos para la siguiente repetición."""
        self.angles = []
        self.timestamps = []