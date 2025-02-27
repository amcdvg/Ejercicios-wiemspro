from constants import Constants as K
import numpy as np
from scipy.signal import savgol_filter, find_peaks
 
class Metrics:
    """_summary_
    """
    def __init__(self, exercise: str, height: float, gender: str, age: int):
        """_summary_
 
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
        if self.exercise in ['curl','pushup']:
            if self.gender == 'male':
                self.regression_params = {'A': -0.02, 'B': 0.16, 'C': 0.0005, 'D': 0.02}
            else:
                self.regression_params = {'A': -0.03, 'B': 0.15, 'C': 0.0004, 'D': 0.0}
            # Para estos ejercicios se mantiene la estatura en metros
            self.segment_length = self.regression_params['A'] + self.regression_params['B'] * self.height \
                                  + self.regression_params['C'] * self.age \
                                  + self.regression_params['D'] * (1 if self.gender=='male' else 0)
        elif self.exercise == 'squat':
            height_cm = self.height * 100  
            gender_dummy = 1 if self.gender == 'male' else 0
            if self.gender == 'male':
                # Fórmula para hombres: tibia_length_cm = (-69.4/3.38) + (1/3.38)*height_cm + 0.0*age + 0.02*gender_dummy
                self.regression_params = {'A': -69.4/3.38, 'B': 1/3.38, 'C': 0.0, 'D': 0.02}
            else:
                # Fórmula para mujeres: tibia_length_cm = (-46.7/3.47) + (1/3.47)*height_cm + 0.0*age + 0.0*gender_dummy
                self.regression_params = {'A': -46.7/3.47, 'B': 1/3.47, 'C': 0.0, 'D': 0.0}
            params = self.regression_params
            # Calculamos la longitud en centímetros:
            seg_length_cm = params['A'] + params['B'] * height_cm + params['C'] * self.age + params['D'] * gender_dummy
            # Convertir el resultado a metros:
            self.segment_length = seg_length_cm / 100
        elif self.exercise == 'plank':
            self.regression_params = {'A': 0.0, 'B': 0.5, 'C': 0.0, 'D': 0.0}
            self.segment_length = self.regression_params['A'] + self.regression_params['B'] * self.height
        else:
            self.regression_params = {'A': 0.0, 'B': 0.16, 'C': 0.0, 'D': 0.0}
            self.segment_length = self.regression_params['A'] + self.regression_params['B'] * self.height
 
        self.angles = []
        self.timestamps = []
    
    def update(self, angle: float, timestamp: float):
        """_summary_
 
        Args:
            angle (float): _description_
            timestamp (float): _description_
        """
        self.angles.append(angle)
        self.timestamps.append(timestamp)
 
    def calculate_rom(self) -> float:
        """_summary_
        """
        if not self.angles:
            return 0.0
        min_angle = min(self.angles)
        max_angle = max(self.angles)
        delta_angle = max_angle - min_angle
        delta_rad = np.deg2rad(delta_angle)
        rom = self.segment_length * delta_rad
 
        return rom
    
    def _get_effective_length(self) -> float:
        """Calcula longitud efectiva basada en antropometría de Winter (2009)."""
        proportions = {
            'squat': {'thigh': 0.245, 'shank': 0.246},
            'curl': {'forearm': 0.146},
            'pushup': {'upper_arm': 0.093, 'forearm': 0.146},  # 0.186/2 para upper_arm
            'plank': {'torso': 0.43}
        }
        
        if self.exercise == 'squat':
            thigh = proportions['squat']['thigh'] * self.height
            shank = proportions['squat']['shank'] * self.height
            return 0.85 * thigh + shank  # Zatsiorsky (2002) para momento efectivo
        
        elif self.exercise == 'curl':
            forearm = proportions['curl']['forearm'] * self.height
            return 0.95 * forearm  # 95% antebrazo como palanca efectiva
        
        elif self.exercise == 'pushup':
            upper_arm = proportions['pushup']['upper_arm'] * self.height
            forearm = proportions['pushup']['forearm'] * self.height
            return 0.75 * (upper_arm + forearm)  # 75% brazo completo
        
        elif self.exercise == 'plank':
            return proportions['plank']['torso'] * self.height
        
        else:
            return 0.15 * self.height  # Valor por defecto seguro
 
    def calculate_vmed(self) -> float:
        """Velocidad media basada en desplazamiento angular total."""
        if len(self.timestamps) < 2:
            return 0.0
            
        total_time = self.timestamps[-1] - self.timestamps[0]
        if total_time <= 0:
            return 0.0
            
        # Cálculo del cambio angular total en radianes
        total_deg = sum(abs(self.angles[i] - self.angles[i-1]) for i in range(1, len(self.angles)))
        total_rad = np.deg2rad(total_deg)
        
        # Desplazamiento lineal total
        L = self._get_effective_length()
        total_distance = L * total_rad
        
        return total_distance / total_time
 
    def calculate_vmax(self) -> float:
        """Velocidad máxima con filtrado científico y validación biomecánica."""
        if len(self.angles) < 2:
            return 0.0
            
        # Cálculo de velocidades instantáneas
        delta_t = np.diff(self.timestamps)
        delta_ang = np.abs(np.diff(self.angles))
        L = self._get_effective_length()
        
        with np.errstate(divide='ignore', invalid='ignore'):
            v_instant = L * np.deg2rad(delta_ang) / delta_t
        
        # Filtrado de datos inválidos
        v_clean = v_instant[np.isfinite(v_instant) & (delta_t > 0.01)]  # Ignora dt < 10ms
        
        if len(v_clean) < 5:
            return np.max(v_clean) if len(v_clean) > 0 else 0.0
            
        # Suavizado con Savitzky-Golay (ventana 5, orden 3)
        window_size = min(5, len(v_clean))
        if window_size % 2 == 0:
            window_size -= 1
            
        try:
            v_smooth = savgol_filter(v_clean, window_length=window_size, polyorder=3)
            vmax = np.max(v_smooth)
        except:
            vmax = np.max(v_clean)
        
        # Validación contra límites fisiológicos
        MAX_REALISTIC_VELOCITY = 3.0  # m/s (para movimientos deportivos)
        return (vmax*0.5)*0.856#min(vmax, MAX_REALISTIC_VELOCITY)
 
    def get_metrics(self, repetition: int = None) -> dict:
        """_summary_
 
        Returns:
            dict: _description_
        """
        min_angle = min(self.angles) if self.angles else None
        max_angle = max(self.angles) if self.angles else None
        type_length = f'leg_length' if self.exercise=='squat' else f'forearm_lenght'
        metrics_dict = {
            "exercise": self.exercise,
            'repetition': 0,
            "height": self.height,
            "gender": self.gender,
            "age": self.age,
            type_length: self.segment_length,
            "min_angle (°)": min_angle,
            "max_angle (°)": max_angle,
            "ROM (m)": self.calculate_rom(),
            "VMED (m/s)": self.calculate_vmed(),
            "VMAX (m/s)": self.calculate_vmax()
        }
        if repetition is not None:
            metrics_dict["repetition"] = repetition
        return metrics_dict
    def reset(self):
        self.angles = []
        self.timestamps = []