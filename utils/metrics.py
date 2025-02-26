from constants import Constants as K
import numpy as np

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
    def calculate_vmed(self) -> float:
        """_summary_

        Returns:
            float: _description_
        """
        if len(self.timestamps)<2:
            return 0.0
        total_time = self.timestamps[-1] - self.timestamps[0]
        if total_time <= 0:
            return 0.0
        total_delta_angle = max(self.angles) - min(self.angles)
        total_delta_rad = np.deg2rad(total_delta_angle)
        total_displacement = self.segment_length * total_delta_rad

        return total_displacement / total_time
    
    def calculate_vmax(self) -> float:
        """_summary_

        Returns:
            float: _description_
        """
        velocities = []
        if len(self.angles) < 2:
            return 0.0
        for i in range(1, len(self.angles)):
            delta_time = self.timestamps[i] - self.timestamps[i-1]
            if delta_time <= 0:
                continue
            delta_angle = abs(self.angles[i] - self.angles[i-1])
            delta_rad = np.deg2rad(delta_angle)
            displacement = self.segment_length * delta_rad
            v = displacement / delta_time
            velocities.append(v)

        if not velocities:
            return 0.0
        # Aplicar filtro de mediana sobre ventanas de 3
        smoothed_velocities = []
        for i in range(len(velocities)):
            # Tomar ventana desde i-1 a i+1
            window = velocities[max(0, i-1) : min(len(velocities), i+2)]
            smoothed_velocities.append(np.median(window))
        
        vmax = max(smoothed_velocities)
        #vmax = min(vmax, 1.5)
        return vmax
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
