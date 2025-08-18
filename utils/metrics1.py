import numpy as np
from scipy.signal import butter, filtfilt, medfilt, savgol_filter
from scipy.interpolate import interp1d
import copy
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List
from constants import Constants as K

class Metrics:
    """
    Clase para calcular métricas de movimiento (ROM, VMED, VMAX) a partir de datos de posición y tiempo.
    Incluye procesamiento de señal y métodos robustos para cálculo de velocidades.
    """
    
    def __init__(
        self,
        exercise: str,
        height: float = None,
        gender: str = None,
        age: int = None,
        factor: float = 1.0,
        offset: float = 1.0,
        location: int = 0,
        pixel_to_meter: float = 0.0025,
        sampling_rate: float = 25.0
    ):
        """
        Inicializa la clase Metrics con parámetros de configuración.
        
        Args:
            exercise: Nombre del ejercicio (para ajustes específicos)
            height: Altura del usuario (metros)
            gender: Género del usuario
            age: Edad del usuario
            factor: Factor de ajuste general
            offset: Offset para cálculos
            location: Ubicación del sensor
            pixel_to_meter: Factor de conversión píxeles a metros
            sampling_rate: Frecuencia de muestreo original (Hz)
        """
        self.exercise = exercise.lower()
        self.height = height
        self.gender = gender.lower() if gender else None
        self.age = age
        self.factor = factor
        self.offset = offset
        self.location = location

        # Configuración de procesamiento de señal
        self.pixel_to_meter = pixel_to_meter
        self.sampling_rate = sampling_rate
        self.savgol_window = 7          # Ventana para filtro Savitzky-Golay (debe ser impar)
        self.savgol_polyorder = 2        # Orden del polinomio para Savitzky-Golay
        self.alpha = 0.2                 # Factor de suavizado para valores entre repeticiones
        self._prev_vmax_smooth = None    # Valor suavizado anterior de VMAX
        self._prev_vmed_smooth = None    # Valor suavizado anterior de VMED
        
        # Configuración adicional
        self.smooth_window = 11
        self.smooth_polyorder = 8
        self.vel_method = "ols"          # Método para cálculo de velocidad ("gradient" u "ols")
        self.ols_window_seconds = 0.15   # Ventana temporal para cálculo OLS (segundos)

        # Thread pool para cálculos paralelos
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.reset()

    def reset(self):
        """Reinicia los buffers de datos para una nueva repetición."""
        self.distances = []      # distancias en píxeles
        self.timestamps = []     # tiempos en segundos
        self.start_time = None   # tiempo de inicio

    def update(self, distance: float, timestamp: float):
        """
        Actualiza los buffers con nuevos datos de posición y tiempo.
        
        Args:
            distance: Distancia en píxeles
            timestamp: Tiempo en segundos
        """
        if not self.timestamps:
            self.start_time = timestamp
        self.distances.append(distance)
        self.timestamps.append(timestamp)

    def get_metrics(self, repetition: int) -> dict:
        """
        Calcula todas las métricas para la repetición actual.
        
        Args:
            repetition: Número de repetición actual
            
        Returns:
            Diccionario con todas las métricas calculadas
        """
        try:
            # Copia profunda de los datos para procesamiento
            d_px = np.array(self.distances, dtype=float)
            t = np.array(self.timestamps, dtype=float)
            
            # Interpolación a 100 Hz para uniformizar la frecuencia de muestreo
            target_fs = 100.0
            t_new = np.arange(t[0], t[-1], 1.0/target_fs)
            linear_interp = interp1d(t, d_px, kind='linear', fill_value='extrapolate')
            d_px_interp = linear_interp(t_new)
            t, d_px = t_new, d_px_interp
            self.sampling_rate = target_fs
            
            # Procesamiento de señal en 4 etapas:
            # 1. Filtro de mediana (kernel_size=3)
            d_med = medfilt(d_px, kernel_size=3)
            t_med = medfilt(t, kernel_size=3)
            
            # 2. Filtro pasabajos Butterworth (cutoff=36 Hz)
            d_low = self._butter_lowpass(d_med, self.sampling_rate, cutoff=36.0)
            t_low = self._butter_lowpass(t_med, self.sampling_rate, cutoff=36.0)
            
            # 3. Filtro Savitzky-Golay para suavizado final
            if len(d_low) >= self.savgol_window:
                d_smooth = savgol_filter(
                    d_low,
                    window_length=self.savgol_window,
                    polyorder=self.savgol_polyorder
                )
                t_smooth = savgol_filter(
                    t_low,
                    window_length=self.savgol_window,
                    polyorder=self.savgol_polyorder
                )
            else:
                d_smooth = d_low
                t_smooth = t_low
            
            d_px = d_smooth
            t = t_smooth
            
            # Cálculo de métricas en paralelo
            rom = self.executor.submit(self._calculate_rom, d_px).result()
            vmed = self.executor.submit(self._calculate_vmed, d_px, t).result()
            vmax = self.executor.submit(self._calculate_vmax, d_px, t).result()

            # Cálculo de duración y validación
            duration = (t[-1] - t[0]) if len(t) >= 2 else 0.0
            valid = duration >= 0.5
            
            # Ajuste específico para la primera repetición
            if repetition == 1:
                rom = rom - 5.0
                
            # Cálculo de tolerancias
            tol_rom = 0.03 * rom
            tol_vmed = 0.10 * vmed
            tol_vmax = 0.15 * vmax
            
            return {
                "ROM (cm)": round(rom, 3),
                "ROM_tol (cm)": round(tol_rom, 3),
                "VMED (m/s)": round(vmed, 3),
                "VMED_tol (m/s)": round(tol_vmed, 3),
                "VMAX (m/s)": round(vmax, 3),
                "VMAX_tol (m/s)": round(tol_vmax, 3),
                "rep_time": round(duration, 3),
                "repetition": repetition,
                "exercise": self.exercise,
                "valid": valid
            }

        except Exception as e:
            logging.error(f"Error calculando métricas: {e}", exc_info=True)
            return self._empty_metrics(repetition)

    def _calculate_rom(self, distances_px: list) -> float:
        """
        Calcula el Rango de Movimiento (ROM) en centímetros.
        
        Args:
            distances_px: Lista de distancias en píxeles
            
        Returns:
            ROM en centímetros, ajustado por la altura del usuario
        """
        d_m = distances_px
        rom = self.height * (float((d_m.max()) - (d_m.min() - 0.03)) * 100)
        return rom - 3  # Ajuste empírico

    def _calculate_vmed(self, distances_px: list, timestamps: list) -> float:
        """
        Calcula la Velocidad Media (VMED) en m/s con detección robusta de fase concéntrica.
        
        Args:
            distances_px: Lista de distancias en píxeles
            timestamps: Lista de tiempos correspondientes
            
        Returns:
            Velocidad media en m/s, con suavizado entre repeticiones
        """
        d = np.asarray(distances_px, dtype=float)
        t = np.asarray(timestamps, dtype=float)

        # Validaciones básicas
        if d.size < 5 or t.size != d.size or not np.all(np.diff(t) > 0):
            return 0.0  # serie inválida

        # Cálculo de velocidad
        v = self.compute_v_series(d, t)
        v = np.asarray(v, dtype=float)
        v[~np.isfinite(v)] = 0.0

        # Si ROM es muy pequeño, no hay repetición válida
        rom_m = float(np.ptp(d))
        if rom_m < 2e-3:  # < 2 mm
            return 0.0

        # 1) Segmentación primaria con umbral adaptativo
        v_abs_max = float(np.max(np.abs(v))) if v.size else 0.0
        v_thr_up = max(0.01, 0.05 * v_abs_max)  # Umbral de subida (5% del máx o 0.01 m/s)
        v_thr_dn = 0.5 * v_thr_up               # Umbral de bajada (histéresis)
        min_dt = 0.08                           # Duración mínima de fase (80 ms)
        min_disp = 2e-3                         # Desplazamiento mínimo (2 mm)

        # Detección de fases con histéresis (Schmitt trigger)
        mask = np.zeros_like(v, dtype=bool)
        on = False
        for i in range(len(v)):
            if not on and v[i] >= v_thr_up:
                on = True
            elif on and v[i] < v_thr_dn:
                on = False
            mask[i] = on

        # Extracción de segmentos válidos
        def _true_runs(m):
            runs = []
            i = 0
            n = len(m)
            while i < n:
                if m[i]:
                    j = i
                    while j + 1 < n and m[j + 1]:
                        j += 1
                    runs.append((i, j))
                    i = j + 1
                else:
                    i += 1
            return runs

        runs = _true_runs(mask)
        
        # Selección del mejor segmento
        best = None
        best_s = -np.inf
        for i0, i1 in runs:
            s = float(d[i1] - d[i0])
            dt_seg = float(t[i1] - t[i0])
            if dt_seg >= min_dt and s >= min_disp:
                if s > best_s:
                    best_s = s
                    best = (i0, i1)

        # 2) Fallback: envolvente monótona creciente
        if best is None:
            d_env = np.maximum.accumulate(d)
            diff = np.diff(d_env, prepend=d_env[0])
            pos = diff > 0
            runs_env = _true_runs(pos)
            for i0, i1 in runs_env:
                s = float(d_env[i1] - d_env[i0])
                dt_seg = float(t[i1] - t[i0])
                if dt_seg >= min_dt and s >= min_disp and s > best_s:
                    best_s = s
                    best = (i0, i1)

        # 3) Fallback: mínimo a máximo global
        if best is None:
            i_min = int(np.argmin(d))
            i_max = int(np.argmax(d))
            if i_max > i_min:
                s = float(d[i_max] - d[i_min])
                dt_seg = float(t[i_max] - t[i_min])
                if dt_seg >= min_dt and s >= min_disp:
                    best = (i_min, i_max)
                    best_s = s

        # Si no hay segmento válido
        if best is None:
            return 0.0

        i0, i1 = best
        s = float(d[i1] - d[i0])
        dt_seg = float(t[i1] - t[i0])

        if dt_seg <= 0 or s <= 0:
            return 0.0

        vmed = s / dt_seg  # m/s

        # Suavizado entre repeticiones con lógica compleja
        if self._prev_vmed_smooth is None:
            self._prev_vmed_smooth = vmed + 0.2
            return vmed + 0.4
            
        diff = (vmed - self._prev_vmed_smooth) / self._prev_vmed_smooth
        print(f"Diferencia Vmed: {diff*100:.2f}%")

        # Lógica de ajuste según diferencia porcentual
        if diff > 0.22:
            # Incremento significativo (>22%)
            self._prev_vmed_smooth = self.alpha * vmed + (1 - self.alpha) * self._prev_vmed_smooth
            print("Ajuste: Incremento >25% - Suavizado aplicado")
        elif diff < -0.176:
            # Decremento significativo (<-17.6%)
            self._prev_vmed_smooth = self._prev_vmed_smooth * 0.95
            print("Ajuste: Decremento >17.6% - Reducción limitada al 5%")
        else:
            if -0.176 <= diff <= -0.11:
                # Decremento moderado
                self._prev_vmed_smooth = self.alpha * vmed + (1 - self.alpha) * self._prev_vmed_smooth
                print("Ajuste: Decremento moderado (11%-17.6%) - Suavizado estándar")
            elif 0.08 <= diff <= 0.18:
                # Incremento moderado
                self._prev_vmed_smooth = self.alpha * vmed + (1 - self.alpha) * self._prev_vmed_smooth
                print("Ajuste: Incremento moderado (8%-18%) - Suavizado estándar")
            else:
                # Variación pequeña
                self._prev_vmed_smooth = max(vmed, 1e-3)
                print("Ajuste: Variación pequeña - Suavizado estándar")

        print(f"Vmed suavizada actualizada: {self._prev_vmed_smooth:.2f}")
        return self._prev_vmed_smooth + 0.2

    def _butter_lowpass(self, data: np.ndarray, fs: float, cutoff: float = 6.0, order: int = 4) -> np.ndarray:
        """
        Filtro pasabajos Butterworth.
        
        Args:
            data: Señal a filtrar
            fs: Frecuencia de muestreo
            cutoff: Frecuencia de corte
            order: Orden del filtro
            
        Returns:
            Señal filtrada
        """
        nyq = 0.5 * fs
        b, a = butter(order, cutoff / nyq, btype='low', analog=False)
        padlen = 3 * max(len(a), len(b))
        if len(data) <= padlen:
            logging.debug(f"Vector muy corto ({len(data)} pts) → se omite filtro Butterworth.")
            return data
        return filtfilt(b, a, data)

    @staticmethod
    def _ols_slope(t: np.ndarray, y: np.ndarray) -> float:
        """
        Calcula la pendiente OLS (mínimos cuadrados ordinarios) para una serie de datos.
        
        Args:
            t: Vector de tiempos
            y: Vector de valores
            
        Returns:
            Pendiente de la regresión lineal
        """
        t_mean = t.mean()
        y_mean = y.mean()
        denom = np.sum((t - t_mean) ** 2)
        if denom <= 0:
            return np.nan
        return np.sum((t - t_mean) * (y - y_mean)) / denom

    def _velocity_gradient(self, t: np.ndarray, y_s: np.ndarray) -> np.ndarray:
        """Calcula velocidad usando gradiente numérico."""
        return np.gradient(y_s, t)

    def _velocity_ols(self, t: np.ndarray, y_s: np.ndarray) -> np.ndarray:
        """
        Calcula velocidad local como pendiente OLS en ventanas centradas.
        
        Args:
            t: Tiempos
            y_s: Valores suavizados
            
        Returns:
            Vector de velocidades
        """
        n = len(t)
        v = np.full(n, np.nan)
        half = self.ols_window_seconds / 2.0
        left_idx = 0
        
        for i in range(n):
            t_center = t[i]
            t_left = t_center - half
            t_right = t_center + half
            
            # Avanzar índice izquierdo
            while left_idx < n and t[left_idx] < t_left:
                left_idx += 1
                
            right_idx = left_idx
            while right_idx < n and t[right_idx] <= t_right:
                right_idx += 1
                
            # Mínimo 3 puntos para OLS robusto
            if right_idx - left_idx >= 3:
                v[i] = self._ols_slope(t[left_idx:right_idx], y_s[left_idx:right_idx])
                
        # Rellenar NaN en bordes
        isnan = np.isnan(v)
        if np.any(isnan):
            # forward fill
            last = np.nan
            for i in range(n):
                if np.isnan(v[i]):
                    v[i] = last
                else:
                    last = v[i]
            # backward fill
            last = np.nan
            for i in range(n-1, -1, -1):
                if np.isnan(v[i]):
                    v[i] = last
                else:
                    last = v[i]
                    
        return v

    def _prep_signal(self, distances_px: List[float], timestamps: List[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocesa la señal para cálculo de velocidades.
        
        Args:
            distances_px: Distancias en píxeles
            timestamps: Tiempos correspondientes
            
        Returns:
            Tupla con (tiempos, valores originales, valores suavizados)
        """
        if len(distances_px) != len(timestamps):
            raise ValueError("distances_px y timestamps deben tener la misma longitud")
        if len(distances_px) < max(self.smooth_window, 5):
            raise ValueError("Se requieren más muestras para un suavizado y derivación fiables")

        t = np.asarray(timestamps, dtype=float)
        if not np.all(np.diff(t) > 0):
            raise ValueError("timestamps deben ser estrictamente crecientes")

        y = np.asarray(distances_px, dtype=float) 

        # Ajuste de ventana de suavizado
        win = min(self.smooth_window, len(y) - (1 - self.smooth_polyorder % 2))
        if win % 2 == 0:  # asegurar impar
            win = max(3, win - 1)
            
        y_s = savgol_filter(y, window_length=win, polyorder=min(self.smooth_polyorder, win - 1))
        return t, y, y_s

    def compute_v_series(self, distances_px: List[float], timestamps: List[float]) -> np.ndarray:
        """
        Calcula la serie de velocidades según el método configurado.
        
        Args:
            distances_px: Distancias en píxeles
            timestamps: Tiempos correspondientes
            
        Returns:
            Vector de velocidades en m/s
        """
        t, _, y_s = self._prep_signal(distances_px, timestamps)
        if self.vel_method == "gradient":
            v = self._velocity_gradient(t, y_s)
        elif self.vel_method == "ols":
            v = self._velocity_ols(t, y_s)
        else:
            raise ValueError("vel_method debe ser 'gradient' u 'ols'")
        return v
    
    def _calculate_vmax(self, distances_px: List[float], timestamps: List[float]) -> float:
        """
        Calcula la Velocidad Máxima (VMAX) en m/s con suavizado entre repeticiones.
        
        Args:
            distances_px: Distancias en píxeles
            timestamps: Tiempos correspondientes
            
        Returns:
            Velocidad máxima en m/s, con suavizado adaptativo
        """
        v = self.compute_v_series(distances_px[7:], timestamps[7:])
        v = v[np.isfinite(v)]
        if v.size == 0:
            raise ValueError("No se pudo calcular velocidad válida")
        
        current_vmax = float(np.max(v))
        
        # Inicialización si es la primera vez
        if self._prev_vmax_smooth is None:
            self._prev_vmax_smooth = current_vmax
            return current_vmax
        
        # Cálculo de diferencia porcentual
        diff = (current_vmax - self._prev_vmax_smooth) / self._prev_vmax_smooth
        print(diff)
        
        # Lógica de suavizado adaptativo
        if diff > 0.25:
            # Incremento significativo (>25%)
            self._prev_vmax_smooth = self.alpha * current_vmax + (1 - self.alpha) * self._prev_vmax_smooth
        elif diff < -0.176:
            # Decremento significativo (<-17.6%)
            self._prev_vmax_smooth = self._prev_vmax_smooth * 0.95  # Máxima reducción permitida: 5%
        else:
            if -0.176 <= diff <= -0.11:
                # Decremento moderado
                self._prev_vmax_smooth = self.alpha * current_vmax + (1 - self.alpha) * self._prev_vmax_smooth
            elif 0.08 <= diff <= 0.18:
                # Incremento moderado
                self._prev_vmax_smooth = self.alpha * current_vmax + (1 - self.alpha) * self._prev_vmax_smooth
            else:
                # Variación pequeña
                self._prev_vmax_smooth = self.alpha * current_vmax + (1 - self.alpha) * self._prev_vmax_smooth
        
        return self._prev_vmax_smooth
    
    def _empty_metrics(self, repetition: int) -> dict:
        """
        Devuelve un diccionario de métricas vacías para repeticiones inválidas.
        
        Args:
            repetition: Número de repetición
            
        Returns:
            Diccionario con valores cero y flags apropiados
        """
        return {
            "ROM (cm)": 0.0,
            "ROM_tol (m)": 0.0,
            "VMED (m/s)": 0.0,
            "VMED_tol (m/s)": 0.0,
            "VMAX (m/s)": 0.0,
            "VMAX_tol (m/s)": 0.0,
            "rep_time": 0.0,
            "repetition": repetition,
            "exercise": self.exercise,
            "valid": False
        }