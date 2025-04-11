from constants import Constants as K
import numpy as np
from scipy.signal import savgol_filter
from utils._utils import smooth_angles
from datetime import datetime
import time

class Metrics:
    def __init__(self, exercise: str, height: float, gender: str, age: int):
        self.exercise = exercise.lower()
        self.height = height
        self.gender = gender.lower()
        self.age = age
        self.pixel_scale = None  # Recibirá el factor de escala desde DeadliftExercise
        
        # Configuración específica para deadlift
        if self.exercise == 'deadlift':
            self.segment_length = self.height  # Usamos la altura completa como referencia
        else:
            # Mantenemos regresiones antropométricas para otros ejercicios
            self._init_anthropometric_functions()
        
        self.angles = []  # Almacenará desplazamientos en metros para deadlift
        self.timestamps = []
        self.start_time = None

    def _init_anthropometric_functions(self):
        """Inicializa funciones antropométricas para ejercicios que no sean deadlift."""
        regression_functions = {
            'curl': lambda: 0.16 * self.height,
            'squat': lambda: 0.25 * self.height,
            # ... otras funciones
        }
        self.segment_length = regression_functions.get(self.exercise, lambda: 0.15 * self.height)()

    def update(self, displacement: float, timestamp: float):
        if not self.timestamps:
            self.start_time = timestamp
        self.angles.append(displacement)
        self.timestamps.append(timestamp)

    def calculate_rom(self) -> float:
        if not self.angles:
            return 0.0
        
        smoothed = savgol_filter(self.angles, 3, 2)  # Filtro de suavizado
        
        if self.exercise == 'deadlift':
            # ROM directo en metros (ya escalado desde el ejercicio)
            rom_meters = max(self.angles) - min(self.angles)
            return rom_meters * 100  # Convertir a cm
        else:
            # Lógica original para otros ejercicios
            delta_rad = np.deg2rad(max(smoothed) - min(smoothed))
            return self.segment_length * delta_rad #* K.ROM_BASE_FACTORS.get(self.exercise, 0.91)

    def calculate_vmed(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        
        smoothed = savgol_filter(self.angles, 3, 2)
        delta_t = np.diff(self.timestamps)
        total_time = np.sum(delta_t)
        
        if self.exercise == 'deadlift':
            total_distance = np.sum(np.abs(np.diff(self.angles)))  # En metros
            raw_vmed = total_distance / total_time
        else:
            total_rad = np.deg2rad(np.sum(np.abs(np.diff(smoothed))))
            raw_vmed = (self.segment_length * total_rad) / total_time
        
        return self._apply_vmed_corrections(raw_vmed)
    def calculate_vmax(self) -> float:
        if len(self.angles) < 2:
            return 0.0
        
        smoothed = savgol_filter(self.angles, 3, 2)
        delta_t = np.diff(self.timestamps)
        
        if self.exercise == 'deadlift':
            velocities = np.abs(np.diff(self.angles)) / delta_t  # m/s
        else:
            delta_rad = np.deg2rad(np.abs(np.diff(smoothed)))
            velocities = (self.segment_length * delta_rad) / delta_t
        
        valid_velocities = velocities[np.isfinite(velocities) & (delta_t > 0.01)]
        return self._apply_vmax_corrections(np.max(valid_velocities)) if valid_velocities.size > 0 else 0.0

    def _apply_vmed_corrections(self, raw_vmed: float) -> float:
        factors = [
            K.VMED_CORRECTION_FACTORS.get(self.exercise, 1.0),
            K.VMED_BASE_FACTORS.get(self.exercise, 0.85),
            K.VMED_FINAL_FACTORS.get(self.exercise, 0.62)
        ]
        return raw_vmed #* np.prod(factors)

    def _apply_vmax_corrections(self, raw_vmax: float) -> float:
        factors = [
            K.ADJUSTMENT_FACTORS_VMAX.get(self.exercise, 0.856),
            K.VMAX_BASE_FACTORS.get(self.exercise, 0.48)
        ]
        return raw_vmax #* np.prod(factors)

    def get_metrics(self, repetition: int = None) -> dict:
        rep_time = self.timestamps[-1] - self.timestamps[0] if len(self.timestamps) >= 2 else 0.0
        # Renombrar las claves para mantener consistencia:
        metrics = {
            "ROM (cm)": round(self.calculate_rom(), 5),
            "VMED (m/s)": round(self.calculate_vmed(), 5),
            "VMAX (m/s)": round(self.calculate_vmax(), 5),
            "rep_time": round(rep_time, 2),
            "repetition": repetition or 0
        }
        if self.exercise == 'deadlift':
            # Renombramos usando los nombres de columna originales
            metrics.update({
                "min_angle (°)": round(min(self.angles), 5),
                "max_angle (°)": round(max(self.angles), 5)
            })
        return metrics
    def reset(self):
        self.angles = []
        self.timestamps = []