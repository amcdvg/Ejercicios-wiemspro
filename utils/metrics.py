from constants import Constants as K
import numpy as np
from scipy.signal import savgol_filter
from utils.anthropometry import Anthopometry

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

        Se utiliza la diferencia entre el ángulo máximo y mínimo (convertida a radianes)
        y se multiplica por la longitud del segmento.

        Returns:
            float: ROM en metros.
        """
        if not self.angles:
            return 0.0
        valid_angles = [a for a in self.angles if isinstance(a, (int, float))]
        if not valid_angles:
            return 0.0
        min_angle = min(valid_angles)
        max_angle = max(valid_angles)
        delta_angle = max_angle - min_angle
        delta_rad = np.deg2rad(delta_angle)
        rom = self.segment_length * delta_rad
        return rom
    
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
        Calcula la velocidad media (VMED) en m/s durante la repetición actual.

        Se define como el desplazamiento total (derivado del cambio angular acumulado)
        dividido entre el tiempo total activo (sólo considerando intervalos con datos válidos),
        usando la longitud efectiva (calculada con _get_effective_length).

        Returns:
            float: VMED en m/s.
        """
        if len(self.timestamps) < 2:
            return 0.0
        total_time = self.timestamps[-1] - self.timestamps[0]
        if total_time <= 0:
            return 0.0
        # Sumar los cambios absolutos entre cada par de ángulos válidos
        total_deg = sum(abs(self.angles[i] - self.angles[i-1]) for i in range(1, len(self.angles)))
        total_rad = np.deg2rad(total_deg)
        L = self._get_effective_length()
        total_distance = L * total_rad
        return total_distance / total_time

    def calculate_vmax(self) -> float:
        """
        Calcula la velocidad máxima (VMAX) en m/s durante la repetición actual.

        Se calculan las velocidades instantáneas entre cada par de datos válidos, se aplica un 
        suavizado mediante Savitzky–Golay y se impone un límite fisiológico razonable.

        Returns:
            float: VMAX en m/s.
        """
        if len(self.angles) < 2:
            return 0.0
        delta_t = np.diff(self.timestamps)
        delta_ang = np.abs(np.diff(self.angles))
        L = self._get_effective_length()
        with np.errstate(divide='ignore', invalid='ignore'):
            v_instant = L * np.deg2rad(delta_ang) / delta_t
        
        v_clean = v_instant[np.isfinite(v_instant) & (delta_t > 0.01)]
        if len(v_clean) < 5:
            return np.max(v_clean) if len(v_clean) > 0 else 0.0
        
        window_size = min(5, len(v_clean))
        if window_size % 2 == 0:
            window_size -= 1
        try:
            v_smooth = savgol_filter(v_clean, window_length=window_size, polyorder=3)
            vmax = np.max(v_smooth)
        except Exception:
            vmax = np.max(v_clean)
        
        factor = K.ADJUSTMENT_FACTORS_VMAX.get(self.exercise, 0.856)
        return (vmax*0.5)*factor
        

    def get_metrics(self, repetition: int = None) -> dict:
        """
        Retorna un diccionario con la información y las métricas calculadas:
          - 'exercise': Tipo de ejercicio.
          - 'height (m)': Estatura del sujeto en metros.
          - 'gender': Género.
          - 'age': Edad en años.
          - 'leg_length (m)' o 'forearm_length (m)': Longitud estimada del segmento.
          - 'min_angle (°)': Ángulo mínimo registrado.
          - 'max_angle (°)': Ángulo máximo registrado.
          - 'ROM (m)': Rango de movimiento en metros.
          - 'VMED (m/s)': Velocidad media en m/s.
          - 'VMAX (m/s)': Velocidad máxima en m/s.
          - 'repetition': Número de repetición.
        
        Args:
            repetition (int, optional): Número de repetición. Defaults a 0.

        Returns:
            dict: Diccionario con las métricas y la información del ejercicio.
        """
        min_angle = min(self.angles) if self.angles else None
        max_angle = max(self.angles) if self.angles else None
        type_length = 'leg_length (m)' if self.exercise == 'squat' else 'forearm_length (m)'
        metrics_dict = {
            "exercise": self.exercise,
            "height (m)": self.height,
            "gender": self.gender,
            "age": self.age,
            type_length: self.segment_length,
            "min_angle (°)": min_angle,
            "max_angle (°)": max_angle,
            "ROM (m)": self.calculate_rom(),
            "VMED (m/s)": self.calculate_vmed(),
            "VMAX (m/s)": self.calculate_vmax(),
            "repetition": repetition if repetition is not None else 0
        }
        return metrics_dict

    def reset(self):
        """Reinicia las listas de ángulos y tiempos para la siguiente repetición."""
        self.angles = []
        self.timestamps = []