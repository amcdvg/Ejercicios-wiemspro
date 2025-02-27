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
        """
        Calcula el rango de movimiento (ROM) en metros para la repetición actual.

        Se utiliza la diferencia entre el ángulo máximo y el mínimo (después de filtrar los valores None),
        se convierte ese delta a radianes y se multiplica por la longitud del segmento estimado.

        Returns:
            float: ROM en metros.
        """
        # Filtrar los ángulos válidos (excluyendo None)
        valid_angles = [a for a in self.angles if isinstance(a, (int, float))]
        if not valid_angles:
            return 0.0
        min_angle = min(valid_angles)
        max_angle = max(valid_angles)
        delta_angle = max_angle - min_angle
        delta_rad = np.deg2rad(delta_angle)
        rom = self.segment_length * delta_rad
        return rom

    def calculate_vmed(self) -> float:
        """
        Calcula la velocidad media (VMED) en m/s durante la repetición actual.

        Se obtiene dividiendo el desplazamiento total (calculado a partir del cambio total de ángulo)
        entre el tiempo total activo (solo considerando segmentos donde se tiene trackeo válido).
        Se aplica un suavizado y se omiten los valores no numéricos.

        Returns:
            float: VMED en m/s.
        """
        # Filtrar datos válidos: ángulos que sean int o float
        valid_data = [(angle, ts) for angle, ts in zip(self.angles, self.timestamps)
                    if isinstance(angle, (int, float))]
        if len(valid_data) < 2:
            return 0.0
        valid_angles, valid_timestamps = zip(*valid_data)

        total_displacement = 0.0
        velocities = []
        for i in range(1, len(valid_angles)):
            delta_time = valid_timestamps[i] - valid_timestamps[i-1]
            if delta_time <= 0:
                continue
            delta_angle = abs(valid_angles[i] - valid_angles[i-1])
            delta_rad = np.deg2rad(delta_angle)
            displacement = self.segment_length * delta_rad
            total_displacement += displacement
            velocities.append(displacement / delta_time)

        if not velocities:
            return 0.0

        # Suavizado de las velocidades con ventana de 5
        window_size = 5
        half_window = window_size // 2
        smoothed_velocities = [
            np.median(velocities[max(0, i - half_window):min(len(velocities), i + half_window + 1)])
            for i in range(len(velocities))
        ]

        velocity_threshold = 0.05  # m/s, ajustable según necesidad
        # Calcular tiempo activo solo en intervalos donde la velocidad es mayor que el threshold
        active_time = sum(
            valid_timestamps[i] - valid_timestamps[i-1]
            for i in range(1, len(valid_timestamps))
            if smoothed_velocities[i-1] > velocity_threshold
        )
        return 1.5 * (total_displacement / active_time) if active_time > 0 else 0.0


    def calculate_vmax(self) -> float:
        """
        Calcula la velocidad máxima (VMAX) en m/s durante la repetición actual.

        Se evalúan las velocidades instantáneas entre cada par consecutivo de datos válidos,
        se suavizan usando un filtro de mediana en ventanas de 5 y se retorna el valor máximo.

        Returns:
            float: VMAX en m/s.
        """
        # Filtrar datos válidos
        valid_data = [(angle, ts) for angle, ts in zip(self.angles, self.timestamps)
                    if isinstance(angle, (int, float))]
        if len(valid_data) < 2:
            return 0.0
        valid_angles, valid_timestamps = zip(*valid_data)

        velocities = []
        for i in range(1, len(valid_angles)):
            delta_time = valid_timestamps[i] - valid_timestamps[i-1]
            if delta_time <= 0:
                continue
            delta_angle = abs(valid_angles[i] - valid_angles[i-1])
            delta_rad = np.deg2rad(delta_angle)
            displacement = self.segment_length * delta_rad
            velocities.append(displacement / delta_time)

        if not velocities:
            return 0.0

        # Suavizado con ventana de 5
        window_size = 5
        half_window = window_size // 2
        smoothed = [
            np.median(velocities[max(0, i - half_window):min(len(velocities), i + half_window + 1)])
            for i in range(len(velocities))
        ]

        vmax = max(smoothed)
        # Limitar a un máximo razonable, por ejemplo, 1.5 m/s
        vmax = min(vmax, 1.5)
        return vmax
    def get_metrics(self, repetition: int = None) -> dict:
        """
        Retorna un diccionario con la información y las métricas calculadas:
        - 'exercise': Tipo de ejercicio.
        - 'height (m)': Estatura del sujeto en metros.
        - 'gender': Género.
        - 'age': Edad (años).
        - 'leg_length (m)' o 'forearm_length (m)': Longitud estimada del segmento en metros.
        - 'min_angle (°)': Ángulo mínimo registrado.
        - 'max_angle (°)': Ángulo máximo registrado.
        - 'ROM (m)': Rango de movimiento en metros.
        - 'VMED (m/s)': Velocidad media en m/s.
        - 'VMAX (m/s)': Velocidad máxima en m/s.
        - 'repetition': Número de repetición.
        """
        # Filtrar los ángulos válidos (no None)
        valid_angles = [a for a in self.angles if isinstance(a, (int, float))]
        min_angle = min(valid_angles) if valid_angles else None
        max_angle = max(valid_angles) if valid_angles else None
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
        self.angles = []
        self.timestamps = []