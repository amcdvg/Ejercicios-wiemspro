from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.signal import savgol_filter, medfilt
import copy
import logging
from constants import Constants as K

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
        self.executor = ThreadPoolExecutor(max_workers=3)
        self._init_parameters()
        self.reset()

    def _init_parameters(self):
        """Inicializa parámetros dependientes del ejercicio.

        Configura factores de escala, longitudes de segmento y parámetros de filtrado
        según el tipo de ejercicio (ej: 'deadlift'). Utiliza valores predeterminados
        basados en constantes definidas en el módulo `K` (ajustables según el modelo).
        """
        # --- 1. Configuración específica para 'deadlift' ---
        if self.exercise == 'deadlift':
            # Longitud del segmento corporal (altura del sujeto)
            self.segment_length = self.height  # Ej: Altura usada para ROM (Rango de Movimiento)
            
            # Factor de ajuste para ROM (Rango de Movimiento)
            # - Obtiene el factor de K.ROM_BASE_FACTORS o usa 0.91 por defecto.
            # - Ej: 0.91 podría escalar el ROM según normas de deadlift estándar.
            self.rom_factor = K.ROM_BASE_FACTORS.get(self.exercise, 0.91)
        
        # --- 2. Configuración para otros ejercicios ---
        else:
            # Método que define parámetros antropométricos (ej: longitudes de brazo/pierna)
            # - Ej: Para 'sentadilla', podría usar proporciones de la altura.
            self._set_anthropometric_params()
        
        # --- 3. Factores de velocidad (VMED y VMAX) ---
        # - VMED (Velocidad Media): Factor de escala para cálculos de potencia.
        #   - Ej: 0.85 ajusta la velocidad según el ejercicio (mayor en deadlift).
        self.vmed_factor = K.VMED_BASE_FACTORS.get(self.exercise, 0.85)
        
        # - VMAX (Velocidad Máxima): Factor para estimar picos de velocidad.
        #   - Ej: 0.48 podría reducir el peso de VMAX en ejercicios explosivos.
        self.vmax_factor = K.VMAX_BASE_FACTORS.get(self.exercise, 0.48)
        
        # --- 4. Parámetros de filtrado para suavizado de datos ---
        # - Savitzky-Golay: Ventana y orden del polinomio para filtrar velocidad.
        #   - `savgol_window=7`: Tamaño de ventana impar para preservar características.
        #   - `savgol_polyorder=2`: Polinomio de 2do grado para suavizado suave.
        self.savgol_window = 7
        self.savgol_polyorder = 2

    def _set_anthropometric_params(self):
        """Configura parámetros antropométricos para otros ejercicios.

        Define la longitud de segmentos corporales (ej: brazo/pierna) como un porcentaje
        de la altura del sujeto, basado en proporciones biomecánicas estándar.
        Ajusta `self.segment_length` para ejercicios que no sean 'deadlift'.

        Valores clave:
        - 'curl': 16% de la altura (bíceps: antebrazo + parte del brazo).
        - 'squat': 25% de la altura (longitud de la pierna completa).
        - 'pushup': 18% de la altura (brazo + hombro).
        - 'reverse_fly': 15% de la altura (brazo en posición lateral).
        - 'swing': 22% de la altura (pierna + cadera en movimientos dinámicos).
        - Default: 15% (valor conservador para ejercicios no listados).
        """
        # Diccionario de factores antropométricos por ejercicio
        exercise_params = {
            'curl': 0.16,          # Ej: Bíceps curl (antebrazo + brazo)
            'squat': 0.25,         # Ej: Sentadilla (pierna completa)
            'pushup': 0.18,        # Ej: Flexiones (brazo + hombro)
            'reverse_fly': 0.15,   # Ej: Aperturas laterales (brazo en 'T')
            'swing': 0.22          # Ej: Swing de cadera (pierna + tronco)
        }
        
        # --- Cálculo de la longitud del segmento ---
        # - Factor antropométrico: Porcentaje de la altura del sujeto.
        # - Ej: Para 'squat', segment_length = altura * 0.25 (longitud de pierna).
        factor = exercise_params.get(self.exercise, 0.15)
        self.segment_length = self.height * factor

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

    def reset(self):
        """Reinicia las mediciones de la repetición borrando la historia de ángulos y timestamps."""
        self.angles = []
        self.timestamps = []
        self.start_time = None

    def get_metrics(self, repetition: int):
        
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
        try:
            # Crear copias locales para consistencia
            local_angles = copy.deepcopy(self.angles)
            local_timestamps = copy.deepcopy(self.timestamps)
            
            # Enviar cálculos al executor
            rom_future = self.executor.submit(
                self._calculate_rom,
                local_angles,
                local_timestamps
            )
            
            vmed_future = self.executor.submit(
                self._calculate_vmed,
                local_angles,
                local_timestamps
            )
            
            vmax_future = self.executor.submit(
                self._calculate_vmax,
                local_angles,
                local_timestamps
            )
            
            # Recoger y procesar resultados
            return self._build_metrics(
                rom_future.result(),
                vmed_future.result(),
                vmax_future.result(),
                local_angles,
                local_timestamps,
                repetition
            )
            
        except Exception as e:
            logging.error(f"Error en cálculo paralelo: {str(e)}")
            return self._empty_metrics(repetition)

    def _calculate_rom(self, angles, timestamps):
        
        """Calcula el Range Of Motion (ROM) del ejercicio.

        Para 'deadlift', ROM se calcula como la diferencia entre el máximo y mínimo de la señal de
        desplazamiento, convertida a centímetros y escalada según el factor de ROM base. Para otros
        ejercicios se utiliza la señal suavizada.

        Returns:
            float: ROM en centímetros.
        """
        if not angles:
            return 0.0
            
        if self.exercise == 'deadlift':
            rom_meters = max(angles) - min(angles)
            
            return rom_meters * 100 * self.rom_factor 
        else:
            return self._calculate_generic_rom(angles)

    def _calculate_generic_rom(self, angles):
        """ROM para ejercicios no deadlift"""
        try:
            smoothed = savgol_filter(angles, 3, 2)
            delta_rad = np.deg2rad(max(smoothed) - min(smoothed))
            return self.segment_length * delta_rad
        except:
            return 0.0

    def _calculate_vmed(self, angles, timestamps):
        """Calcula la Velocidad Media (VMED) del ejercicio.

        Para 'deadlift', se define la fase concéntrica desde el valor mínimo hasta el final de la serie.
        Se calcula la derivada de la señal usando np.gradient (que considera tiempos no uniformes) y se
        suaviza la señal derivada aplicando primero un filtro Savitzky–Golay y luego un filtro de mediana.
        La velocidad media se obtiene como el promedio de la señal filtrada y se corrige empíricamente mediante
        una multiplicación por 0.5 y una suma de 0.6.

        Returns:
            float: Velocidad media en m/s corregida.
        """
        if len(timestamps) < 2:
            return 0.0
            
        if self.exercise == 'deadlift':
            return self._calculate_deadlift_vmed(angles, timestamps)
        return self._calculate_generic_vmed(angles, timestamps)
    
    def _calculate_deadlift_vmed(self, distances, timestamps):
        """VMED para deadlift con factores de escala de Virtue."""
        try:
            i_min = np.argmin(distances)
            relevant_distances = distances[i_min:]
            relevant_times = timestamps[i_min:]
            
            if len(relevant_times) < 2:
                return 0.0

            # --- 1. Velocidad instantánea ---
            inst_vel = np.gradient(relevant_distances, relevant_times)
            
            # --- 2. Ajustar kernel_size para medfilt ---
            # Asegurar kernel impar y menor que la longitud de los datos
            kernel_size = min(7, len(inst_vel) // 2 * 2 + 1)
            if kernel_size < 3:  # Mínimo para medfilt
                kernel_size = 3
            
            # --- 3. Aplicar medfilt con el +0.1 (requerido por Virtue) ---
            filtered = medfilt(inst_vel + 0.1, kernel_size=kernel_size)
            
            # --- 4. Aplicar Savitzky-Golay solo si la ventana es válida ---
            if len(filtered) >= self.savgol_window:
                filtered = savgol_filter(
                    filtered, 
                    window_length=self.savgol_window,
                    polyorder=self.savgol_polyorder
                )
            
            # --- 5. Calcular VMED con factores de Virtue ---
            vmed = (np.mean(filtered) * self.vmed_factor * 0.377) + 0.69 #si el kernel_sisze es 9 entonces el factor de suma es 0.7
            
            # --- 6. Evitar valores negativos (si es necesario) ---
            return max(vmed, 0.0)
        
        except Exception as e:
            print(f"Error calculando VMED: {e}")
            return 0.0
        
    def _calculate_generic_vmed(self, angles, timestamps):
        """VMED para otros ejercicios.

        Calcula la velocidad media ajustada (VMED) para ejercicios basados en ángulos articulares,
        usando datos suavizados y factores de escala biomecánicos. Adecuado para movimientos
        rotacionales (ej: curls, sentadillas, aperturas).

        Parámetros clave:
        - `angles`: Lista de ángulos articulares (en grados) durante el ejercicio.
        - `timestamps`: Marca de tiempo asociada a cada ángulo (en segundos).

        Retorna:
        - VMED: Velocidad media ajustada (en m/s o unidades de potencia).
        """
        try:
            # --- 1. Suavizado de ángulos con Savitzky-Golay ---
            # - `window_length=3`: Ventana pequeña para preservar picos.
            # - `polyorder=2`: Polinomio cuadrático para reducir ruido.
            smoothed = savgol_filter(angles, 3, 2)

            # --- 2. Cálculo de tiempo total y cambio angular ---
            delta_t = np.diff(timestamps)  # Diferencias entre marcas de tiempo
            total_time = np.sum(delta_t)   # Tiempo total del movimiento
            
            # --- 3. Conversión a radianes y acumulación de desplazamiento angular ---
            # - `np.diff(smoothed)`: Diferencia entre ángulos consecutivos.
            # - `np.abs(...)`: Considera movimiento en ambas direcciones.
            # - `np.deg2rad(...)`: Convierte grados a radianes para cálculos físicos.
            delta_rad = np.deg2rad(np.sum(np.abs(np.diff(smoothed))))

            # --- 4. Cálculo de VMED con factores de escala ---
            # - `self.segment_length`: Longitud del segmento corporal (ej: brazo/pierna).
            # - `delta_rad / total_time`: Velocidad angular media (rad/s).
            # - `* self.vmed_factor * 0.5`: Ajuste específico del modelo (ej: conversión a m/s).
            # - `+ 0.6`: Offset para alinear con métricas de potencia.
            vmed = (
                self.segment_length 
                * delta_rad 
                / total_time 
                * self.vmed_factor 
                * 0.5 
            ) + 0.6

            return vmed

        except Exception as e:
            # Captura errores (ej: delta_t vacío, división por cero)
            print(f"Error en VMED genérico: {e}")
            return 0.0

    def _calculate_vmax(self, angles, timestamps):
        """Calcula la Velocidad Máxima (VMAX) del ejercicio.

        Para 'deadlift', se utiliza la misma segmentación de la fase concéntrica. Se calcula la
        derivada de la señal con np.gradient, se aplica primero un filtro Savitzky–Golay y luego un
        filtro de mediana para obtener una señal de velocidades filtrada. Se toma el valor máximo de
        esta señal y se corrige empíricamente mediante una multiplicación por 0.5 y una suma de 0.7.

        Returns:
            float: Velocidad máxima en m/s corregida.
        """
        if len(timestamps) < 2:
            return 0.0
            
        if self.exercise == 'deadlift':
            return self._calculate_deadlift_vmax(angles, timestamps)
        return self._calculate_generic_vmax(angles, timestamps)

    def _calculate_deadlift_vmax(self, distances, timestamps):
        """VMAX específico para deadlift (distancias)"""
        try:
            i_min = np.argmin(distances)  # Punto más bajo del movimiento
            relevant_distances = distances[i_min:]
            relevant_times = timestamps[i_min:]
            
            if len(relevant_distances) < 2:
                return 0.0

            # 1. Velocidad instantánea (magnitud absoluta)
            inst_vel = np.abs(np.gradient(relevant_distances, relevant_times))
            
            # 2. Aplicar Savitzky-Golay primero (si hay suficientes puntos)
            if len(inst_vel) >= self.savgol_window:
                filtered = savgol_filter(
                    inst_vel,
                    self.savgol_window,
                    self.savgol_polyorder
                )
            else:
                filtered = inst_vel  # Usar datos crudos si no hay suficientes puntos
                
            # 3. Filtro de mediana con kernel fijo = 3 (como en el ejemplo)
            filtered = medfilt(filtered, kernel_size=5)
            
            # 4. Calcular VMAX con factores de Virtue
            vmax = (np.max(filtered) * self.vmax_factor  * 0.557) + 0.62#* 0.475) + 0.75#* 0.5) + 0.7 #* 0.577) + 0.59
            
            return max(vmax, 0.0)  # Evitar valores negativos
        
        except Exception as e:
            print(f"Error calculando VMAX: {e}")
            return 0.0
    
    
    def _calculate_generic_vmax(self, angles, timestamps):
        """VMAX para otros ejercicios.

        Calcula la velocidad máxima ajustada (VMAX) para ejercicios basados en ángulos articulares,
        usando datos suavizados y factores de escala biomecánicos. Adecuado para movimientos
        rotacionales con picos de velocidad (ej: curls rápidos, swings).

        Parámetros clave:
        - `angles`: Lista de ángulos articulares (en grados) durante el ejercicio.
        - `timestamps`: Marca de tiempo asociada a cada ángulo (en segundos).

        Retorna:
        - VMAX: Velocidad máxima ajustada (en m/s o unidades de potencia).
        """
        try:
            # --- 1. Suavizado de ángulos con Savitzky-Golay ---
            # - Reduce ruido mientras preserva picos de velocidad.
            smoothed = savgol_filter(angles, 3, 2)

            # --- 2. Cálculo de diferencias temporales y angulares ---
            delta_t = np.diff(timestamps)  # Intervalos de tiempo entre muestras
            delta_rad = np.deg2rad(np.abs(np.diff(smoothed)))  # Diferencias angulares en radianes

            # --- 3. Velocidad lineal instantánea ---
            # - Fórmula: velocidad = (longitud_segmento * delta_ángulo) / delta_tiempo
            # - Ej: Para un brazo de 0.3m, un cambio de 0.5 rad en 0.1s → 1.5 m/s.
            velocities = (self.segment_length * delta_rad) / delta_t

            # --- 4. Cálculo de VMAX con factores de escala ---
            # - `np.max(velocities)`: Pico de velocidad durante el movimiento.
            # - `self.vmax_factor`: Ajuste específico del ejercicio (ej: 0.48 para curls).
            # - `* 0.5 + 0.7`: Factores de escala y offset del modelo de Virtue.
            vmax = (np.max(velocities) * self.vmax_factor * 0.5) + 0.7

            return vmax

        except Exception as e:
            # Captura errores (ej: delta_t=0, arrays vacíos)
            print(f"Error en VMAX genérico: {e}")
            return 0.0

    def _build_metrics(self, rom, vmed, vmax, angles, timestamps, repetition):
        """Construye el diccionario final de métricas.

        Agrupa métricas biomecánicas (ROM, VMED, VMAX), tiempo de repetición,
        y metadatos en un formato estandarizado. Incluye métricas adicionales
        específicas para 'deadlift' (ángulos mínimo/máximo).

        Parámetros clave:
        - `rom`: Rango de Movimiento (ROM) en centímetros.
        - `vmed`: Velocidad Media Ajustada (VMED) en m/s.
        - `vmax`: Velocidad Máxima Ajustada (VMAX) en m/s.
        - `angles`: Lista de ángulos articulares (para ejercicios como 'deadlift').
        - `timestamps`: Marca de tiempo de la repetición.
        - `repetition`: Número de la repetición (ej: 1, 2, 3).

        Retorna:
        - Diccionario con métricas listas para exportar/visualizar.
        """
        # --- 1. Cálculo del tiempo de repetición ---
        # - Duración total: último timestamp - primer timestamp.
        # - Asegura al menos 2 timestamps para evitar errores.
        rep_time = timestamps[-1] - timestamps[0] if len(timestamps) >= 2 else 0.0

        # --- 2. Diccionario base de métricas ---
        metrics = {
            "ROM (cm)": round(rom, 2),          # ROM en centímetros (redondeo a 2 decimales)
            "VMED (m/s)": round(vmed, 2),       # Velocidad media ajustada
            "VMAX (m/s)": round(vmax, 2),       # Velocidad máxima ajustada
            "rep_time": round(rep_time, 2),     # Tiempo total de la repetición (segundos)
            "repetition": repetition,           # Número de la repetición (ej: 1)
            "exercise": self.exercise           # Nombre del ejercicio (ej: 'deadlift')
        }

        # --- 3. Métricas específicas para 'deadlift' ---
        # - Incluye ángulos mínimo y máximo si existen datos.
        # - Útil para analizar la postura durante el movimiento.
        if self.exercise == 'deadlift' and angles:
            metrics.update({
                "min_angle": round(min(angles), 2),  # Ángulo más bajo (ej: posición inicial)
                "max_angle": round(max(angles), 2)   # Ángulo más alto (ej: posición de bloqueo)
            })

        return metrics

    def _empty_metrics(self, repetition):
        """Métricas vacías en caso de error.

        Retorna un diccionario con valores predeterminados (cero) para todas las métricas,
        manteniendo la estructura de datos consistente incluso cuando ocurre un error crítico
        durante el procesamiento de la repetición. Incluye metadatos esenciales.

        Parámetros clave:
        - `repetition`: Número de la repetición fallida (ej: 3).

        Retorna:
        - Diccionario con métricas en cero para evitar errores en sistemas externos.
        """
        return {
            "ROM (cm)": 0.0,          # Rango de Movimiento (valor nulo)
            "VMED (m/s)": 0.0,        # Velocidad Media Ajustada (sin datos)
            "VMAX (m/s)": 0.0,        # Velocidad Máxima Ajustada (sin datos)
            "rep_time": 0.0,          # Tiempo de repetición (0 segundos)
            "repetition": repetition, # Número de la repetición (ej: 2)
            "exercise": self.exercise # Nombre del ejercicio (ej: 'squat')
        }