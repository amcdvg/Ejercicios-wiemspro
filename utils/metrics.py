from constants import Constants as K
import numpy as np
from scipy.signal import savgol_filter
from utils._utils import smooth_angles
from utils.anthropometry import Anthopometry
from datetime import datetime
import time
class Metrics:
    def __init__(self, exercise: str, height: float, gender: str, age: int):
        """_summary_

        Args:
            exercise (str): _description_
            height (float): _description_
            gender (str): _description_
            age (int): _description_
        """
        self.exercise = exercise.lower()
        self.height = height
        self.gender = gender.lower()
        self.age = age
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
        self.angles = []
        self.timestamps = []
        self.history = []
        self.start_time = None

    def update(self, angle: float, timestamp: float):
        """_summary_

        Args:
            angle (float): _description_
            timestamp (float): _description_
        """
        if not self.timestamps:
            self.start_time = timestamp
        self.angles.append(angle)
        self.timestamps.append(timestamp)

    def calculate_rom(self) -> float:
        """_summary_

        Returns:
            float: _description_
        """
        if not self.angles:
            return 0.0
        smoothed_angles = smooth_angles(self.angles, window_length=5, polyorder=2)
        min_angle = np.min(smoothed_angles)
        max_angle = np.max(smoothed_angles)
        delta_angle = max_angle - min_angle
        delta_rad = np.deg2rad(delta_angle)

        rom_factor = K.ROM_BASE_FACTORS.get(self.exercise, 0.91)
        rom = self._get_effective_length() * delta_rad * rom_factor
        return rom

    def _get_effective_length(self) -> float:
        """_summary_

        Returns:
            float: _description_
        """
        if K.USE_REGRESSION_EFFECTIVE_LENGTH:
            return self.segment_length
        else:
            calc_func = K.EFFECTIVE_LENGTH_FUNCTIONS.get(self.exercise, lambda h: 0.15 * h)
            return calc_func(self.height)

    def calculate_vmed(self) -> float:
        """_summary_

        Returns:
            float: _description_
        """
        if len(self.timestamps) < 2:
            return 0.0
        filtered_angles = smooth_angles(self.angles, window_length=5, polyorder=2)
        delta_t = np.diff(self.timestamps)
        total_time = np.sum(delta_t)
        if total_time <= 0:
            return 0.0
        total_deg = sum(abs(filtered_angles[i] - filtered_angles[i - 1]) for i in range(1, len(filtered_angles)))
        total_rad = np.deg2rad(total_deg)
        L = self._get_effective_length()
        total_distance = total_rad * L
        raw_vmed = total_distance / total_time

        exercise_factor = K.VMED_CORRECTION_FACTORS.get(self.exercise, 1.0)
        phase_factor = K.PHASE_CORRECTION.get('eccentric', 1.0)
        base_factor     = K.VMED_BASE_FACTORS.get(self.exercise, 0.85)
        final_factor    = K.VMED_FINAL_FACTORS.get(self.exercise, 0.62)

        corrected_vmed = raw_vmed * exercise_factor * phase_factor * base_factor * final_factor

        vmed_clipped = np.clip(corrected_vmed, 0.2, 2.6)
        return vmed_clipped

    def calculate_vmax(self) -> float:
        """_summary_

        Raises:
            ValueError: _description_

        Returns:
            float: _description_
        """
        if len(self.angles) < 2:
            return 0.0
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
        """_summary_

        Args:
            repetition (int, optional): _description_. Defaults to None.

        Returns:
            dict: _description_
        """
        if len(self.timestamps) < 2:
            rep_time = 0.0
        else:
            rep_time = self.timestamps[-1] - self.timestamps[0]
        start_datetime = datetime.fromtimestamp(time.time() - rep_time).isoformat()
        min_angle = min(self.angles) if self.angles else None
        max_angle = max(self.angles) if self.angles else None
        effective_length = self._get_effective_length()
        metrics_dict = {
            "start_datetime": start_datetime,
            "age": self.age,
            "gender": self.gender,
            "height (m)": round(self.height, 4),
            "effective_length (m)": round(effective_length, 4),
            "min_angle (°)": round(min_angle, 4),
            "max_angle (°)": round(max_angle, 4),
            "ROM (cm)": (round(self.calculate_rom()*100, 4))/2,
            "VMED (m/s)": round(self.calculate_vmed(), 4),
            "VMAX (m/s)": round(self.calculate_vmax(), 4),
            "rep_time": round(rep_time, 4),
            "repetition": repetition if repetition is not None else 0,
            "exercise": self.exercise
        }
        return metrics_dict

    def reset(self):
        """_summary_
        """
        self.angles = []
        self.timestamps = []