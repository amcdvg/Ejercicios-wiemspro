from constants import Constants as K
import numpy as np
from scipy.signal import savgol_filter, medfilt
from datetime import datetime
import time


class Metrics:
    """Calcula las métricas biomecánicas a partir de una serie de desplazamientos y timestamps.

    Esta clase se utiliza para obtener, a partir de los datos del ejercicio, tres métricas
    principales:
      - ROM (Range Of Motion): La diferencia entre el valor máximo y mínimo de los desplazamientos,
        convertida a centímetros y ajustada mediante un factor de ROM base.
      - VMED (Velocidad Media): Calculada a partir de la derivada de la señal de desplazamiento
        (suavizada con un filtro Savitzky–Golay y un filtro de mediana) en la fase concéntrica (desde
        el mínimo hasta el final) y corregida empíricamente.
      - VMAX (Velocidad Máxima): Se obtiene del valor máximo de la señal derivada, luego filtrada y
        corregida empíricamente.
    
    Los métodos de corrección (_apply_vmed_corrections y _apply_vmax_corrections) toman los valores
    brutos calculados y los ajustan usando factores definidos en Constants, además de aplicar una
    multiplicación y adición final (0.5 y 0.6 para VMED, 0.5 y 0.7 para VMAX) para alinear la escala con
    mediciones de referencia.
    
    Atributos:
        exercise (str): Nombre del ejercicio (e.g., 'deadlift', 'curl', etc.), en minúscula.
        height (float): Altura del sujeto en metros.
        gender (str): Género del sujeto ('male' o 'female'), en minúscula.
        age (int): Edad del sujeto.
        pixel_scale (float or None): Factor de conversión de píxeles a metros (se calcula en base a la imagen).
        segment_length (float): Longitud del segmento de referencia (calculada mediante funciones antropométricas).
        angles (list): Lista de desplazamientos (o ángulos) medidos a lo largo del ejercicio.
        timestamps (list): Lista de tiempos (en segundos) asociados a cada medición.
        start_time (float or None): Tiempo inicial del registro.
        savgol_window (int): Ventana para el filtro Savitzky–Golay (número impar).
        savgol_polyorder (int): Orden polinomial para el filtro Savitzky–Golay.
        num_interp (int): Número de muestras para re-muestrear (en este código no se usa re-muestreo adicional).
    """

    def __init__(self, exercise: str, height: float, gender: str, age: int):
        """Inicializa una instancia de Metrics para calcular las métricas biomecánicas.

        Args:
            exercise (str): Nombre del ejercicio (e.g., 'deadlift').
            height (float): Altura del sujeto en metros.
            gender (str): Género del sujeto ('male' o 'female').
            age (int): Edad del sujeto.
        """
        self.exercise = exercise.lower()
        self.height = height
        self.gender = gender.lower()
        self.age = age
        self.pixel_scale = None

        if self.exercise == 'deadlift':
            self.segment_length = self.height
        else:
            self._init_anthropometric_functions()

        self.angles = []
        self.timestamps = []
        self.start_time = None

        # Parámetros para el filtro Savitzky–Golay y el filtro de mediana.
        self.savgol_window = 7      # Debe ser un número impar.
        self.savgol_polyorder = 2
        # Número de muestras para re-muestrear (en este código se mantiene definido, aunque no se usa re-muestreo extra).
        self.num_interp = 70

    def _init_anthropometric_functions(self):
        """Inicializa las funciones antropométricas para ejercicios no 'deadlift'.

        Usa funciones de regresión definidas en Constants.REGRESSION_PARAMS para obtener la
        longitud del segmento.
        """
        regression_functions = {
            'curl': lambda: 0.16 * self.height,
            'squat': lambda: 0.25 * self.height,
            # ... otras funciones si es necesario.
        }
        self.segment_length = regression_functions.get(self.exercise, lambda: 0.15 * self.height)()

    def update(self, displacement: float, timestamp: float):
        """Actualiza la serie de mediciones con un nuevo desplazamiento y timestamp.

        Si es la primera medición, se establece el tiempo inicial. Luego, se agrega el
        desplazamiento y el timestamp a sus respectivas listas.

        Args:
            displacement (float): Desplazamiento medido (p.ej., en metros).
            timestamp (float): Tiempo asociado a la medición (en segundos).
        """
        if not self.timestamps:
            self.start_time = timestamp
        self.angles.append(displacement)
        self.timestamps.append(timestamp)

    def calculate_rom(self) -> float:
        """Calcula el Range Of Motion (ROM) del ejercicio.

        Para 'deadlift', ROM se calcula como la diferencia entre el máximo y mínimo de la señal de
        desplazamiento, convertida a centímetros y escalada según el factor de ROM base. Para otros
        ejercicios se utiliza la señal suavizada.

        Returns:
            float: ROM en centímetros.
        """
        if not self.angles:
            return 0.0

        if self.exercise == 'deadlift':
            rom_meters = max(self.angles) - min(self.angles)
            return rom_meters * 100 * K.ROM_BASE_FACTORS.get(self.exercise, 0.91)
        else:
            smoothed = savgol_filter(self.angles, 3, 2)
            delta_rad = np.deg2rad(max(smoothed) - min(smoothed))
            return self.segment_length * delta_rad

    def calculate_vmed(self) -> float:
        """Calcula la Velocidad Media (VMED) del ejercicio.

        Para 'deadlift', se define la fase concéntrica desde el valor mínimo hasta el final de la serie.
        Se calcula la derivada de la señal usando np.gradient (que considera tiempos no uniformes) y se
        suaviza la señal derivada aplicando primero un filtro Savitzky–Golay y luego un filtro de mediana.
        La velocidad media se obtiene como el promedio de la señal filtrada y se corrige empíricamente mediante
        una multiplicación por 0.5 y una suma de 0.6.

        Returns:
            float: Velocidad media en m/s corregida.
        """
        if len(self.timestamps) < 2:
            return 0.0

        if self.exercise == 'deadlift':
            # Definir la fase concéntrica: desde el mínimo hasta el final de la serie.
            i_min = np.argmin(self.angles)
            relevant_angles = np.array(self.angles[i_min:])
            relevant_timestamps = np.array(self.timestamps[i_min:])
            if len(relevant_timestamps) < 2:
                return 0.0

            # Calcular la derivada utilizando np.gradient, que tiene en cuenta los tiempos no uniformes.
            inst_vel = np.abs(np.gradient(relevant_angles, relevant_timestamps))

            # Primer filtrado: Filtro Savitzky–Golay (si hay suficientes datos).
            if len(inst_vel) >= self.savgol_window:
                sg_filtered = savgol_filter(inst_vel, window_length=self.savgol_window, polyorder=self.savgol_polyorder)
            else:
                sg_filtered = inst_vel

            # Segundo filtrado: Filtro de mediana para suavizar picos aislados.
            med_filtered = medfilt(sg_filtered, kernel_size=3)

            # Obtener la velocidad media como el promedio de la señal filtrada.
            raw_vmed = np.mean(med_filtered)
        else:
            smoothed = savgol_filter(self.angles, 3, 2)
            delta_t = np.diff(self.timestamps)
            total_time = np.sum(delta_t)
            delta_rad = np.deg2rad(np.sum(np.abs(np.diff(smoothed))))
            raw_vmed = (self.segment_length * delta_rad) / total_time

        return self._apply_vmed_corrections(raw_vmed)

    def calculate_vmax(self) -> float:
        """Calcula la Velocidad Máxima (VMAX) del ejercicio.

        Para 'deadlift', se utiliza la misma segmentación de la fase concéntrica. Se calcula la
        derivada de la señal con np.gradient, se aplica primero un filtro Savitzky–Golay y luego un
        filtro de mediana para obtener una señal de velocidades filtrada. Se toma el valor máximo de
        esta señal y se corrige empíricamente mediante una multiplicación por 0.5 y una suma de 0.7.

        Returns:
            float: Velocidad máxima en m/s corregida.
        """
        if len(self.angles) < 2:
            return 0.0

        if self.exercise == 'deadlift':
            i_min = np.argmin(self.angles)
            relevant_angles = np.array(self.angles[i_min:])
            relevant_timestamps = np.array(self.timestamps[i_min:])
            if len(relevant_angles) < 2:
                return 0.0

            inst_vel = np.abs(np.gradient(relevant_angles, relevant_timestamps))

            if len(inst_vel) >= self.savgol_window:
                sg_filtered = savgol_filter(inst_vel, window_length=self.savgol_window, polyorder=self.savgol_polyorder)
            else:
                sg_filtered = inst_vel

            med_filtered = medfilt(sg_filtered, kernel_size=3)
            raw_vmax = np.max(med_filtered)
        else:
            smoothed = savgol_filter(self.angles, 3, 2)
            delta_t = np.diff(self.timestamps)
            delta_rad = np.deg2rad(np.abs(np.diff(smoothed)))
            velocities = (self.segment_length * delta_rad) / delta_t
            raw_vmax = np.max(velocities)

        return self._apply_vmax_corrections(raw_vmax)

    def _apply_vmed_corrections(self, raw_vmed: float) -> float:
        """Aplica factores de corrección a la velocidad media.

        Multiplica el valor bruto por el factor base para VMED definido en Constants,
        y luego aplica la transformación empírica (multiplicando por 0.5 y sumando 0.6).

        Args:
            raw_vmed (float): Valor bruto de la velocidad media calculado.

        Returns:
            float: Valor corregido de VMED en m/s.
        """
        factors = [K.VMED_BASE_FACTORS.get(self.exercise, 0.85)]
        return (raw_vmed * np.prod(factors)) * 0.5 + 0.6

    def _apply_vmax_corrections(self, raw_vmax: float) -> float:
        """Aplica factores de corrección a la velocidad máxima.

        Multiplica el valor bruto por el factor base para VMAX definido en Constants,
        y luego aplica la transformación empírica (multiplicando por 0.5 y sumando 0.7).

        Args:
            raw_vmax (float): Valor bruto de la velocidad máxima calculado.

        Returns:
            float: Valor corregido de VMAX en m/s.
        """
        factors = [K.VMAX_BASE_FACTORS.get(self.exercise, 0.48)]
        return (raw_vmax * np.prod(factors)) * 0.5 + 0.7

    def get_metrics(self, repetition: int = None) -> dict:
        """Genera un diccionario con las métricas calculadas para una repetición.

        Las métricas incluyen:
          - ROM (cm)
          - VMED (m/s)
          - VMAX (m/s)
          - rep_time (duración de la repetición en segundos)
          - repetition (número de repetición)
        Para el ejercicio 'deadlift', también se añaden los valores mínimo y máximo de la señal de ángulos.

        Args:
            repetition (int, optional): Número de repetición. Si no se especifica, se asigna 0.

        Returns:
            dict: Diccionario con las métricas calculadas.
        """
        rep_time = self.timestamps[-1] - self.timestamps[0] if len(self.timestamps) >= 2 else 0.0
        metrics = {
            "ROM (cm)": round(self.calculate_rom(), 5),
            "VMED (m/s)": round(self.calculate_vmed(), 5),
            "VMAX (m/s)": round(self.calculate_vmax(), 5),
            "rep_time": round(rep_time, 2),
            "repetition": repetition or 0
        }
        if self.exercise == 'deadlift':
            metrics.update({
                "min_angle (°)": round(min(self.angles), 5),
                "max_angle (°)": round(max(self.angles), 5)
            })
        return metrics

    def reset(self):
        """Reinicia las mediciones de la repetición borrando la historia de ángulos y timestamps."""
        self.angles = []
        self.timestamps = []
