import numpy as np
import logging
from scipy.signal import savgol_filter, medfilt
import cv2

_last_detected_percentage = None

def calculate_angle(a, b, c):
        """
        Calcula el ángulo formado por tres puntos.

        Este método toma tres puntos (a, b, c) y calcula el ángulo en el punto b,
        utilizando la función arctan2 para determinar la orientación de los segmentos (b→a) y (b→c).
        La diferencia de estos ángulos se convierte a grados y se ajusta para que el valor final esté en el rango [0, 180]°.

        Args:
            a (iterable): Coordenadas (x, y) del primer punto.
            b (iterable): Coordenadas (x, y) del punto vértice donde se mide el ángulo.
            c (iterable): Coordenadas (x, y) del tercer punto.

        Returns:
            float: El ángulo en grados formado en el punto b, dentro del rango [0, 180]°.
        """
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        return angle if angle <= 180.0 else 360 - angle


def valid_keypoints(confs, indices, threshold: float=0.3):
    """_summary_

    Args:
        confs (_type_): _description_
        indices (_type_): _description_
        threshold (float, optional): _description_. Defaults to 0.3.

    Returns:
        _type_: _description_
    """
    return np.all(confs[indices] > threshold)

def filter_valid_angles(angles):
    """Filters the input list and returns only numeric values.

    Args:
        angles (list): A list that may contain angles as numbers and/or None.

    Returns:
        list: A list containing only numeric (int or float) angle values.
    """
    return [a for a in angles if isinstance(a, (int, float))]

def compute_total_angular_change(angles):
    """Computes the total absolute angular change from a sequence of angles.

    Args:
        angles (list): A list of numeric angles in degrees.

    Returns:
        float: The total absolute change in degrees.
    """
    total_change = 0.0
    for i in range(1, len(angles)):
        total_change += abs(angles[i] - angles[i - 1])
    return total_change

def smooth_velocities(velocities, window_size=5, polyorder=3):
    """Smooths a sequence of velocities using the Savitzky–Golay filter.

    Args:
        velocities (np.ndarray): Array of velocity values.
        window_size (int, optional): The length of the filter window (must be odd). Defaults to 5.
        polyorder (int, optional): The order of the polynomial used in the filter. Defaults to 3.

    Returns:
        np.ndarray: The smoothed velocity array.
    """
    # Ensure window_size is odd
    if window_size % 2 == 0:
        window_size -= 1
    return savgol_filter(velocities, window_length=window_size, polyorder=polyorder)


def smooth_angles(angles, window_length=5, polyorder=2):
    """
    Suaviza la secuencia de ángulos usando el filtro de Savitzky-Golay.
    
    Args:
        angles (list[float]): Lista de ángulos en grados.
        window_length (int): Longitud de la ventana (debe ser impar y menor o igual al número de puntos).
        polyorder (int): Orden del polinomio para el filtro.
        
    Returns:
        numpy.ndarray: Arreglo de ángulos suavizados.
    """
    # Asegurarse de que la ventana sea impar y no mayor que la longitud de la lista
    if len(angles) < window_length:
        window_length = len(angles) if len(angles) % 2 != 0 else len(angles)-1
    if window_length < 3:
        # Si no tenemos suficientes puntos, devolvemos la señal original
        return np.array(angles)
    
    try:
        angles_array = np.array(angles, dtype=float)
        smoothed = savgol_filter(angles_array, window_length=window_length, polyorder=polyorder)
        return smoothed
    except Exception as e:
        # En caso de error, devolvemos la lista original
        return np.array(angles)

def valid_full_pose(confs, threshold=0.3, required_percentage = 0.7):
    """
    Verifica que se hayan detectado todos los 17 keypoints con una confianza mínima.

    Args:
        confs (array-like): Lista o array de valores de confianza para cada keypoint.
        threshold (float, optional): Valor mínimo de confianza requerido. Defaults to 0.3.

    Returns:
        bool: True si hay 17 keypoints y todos tienen una confianza >= threshold, False en caso contrario.
    """
    global _last_detected_percentage
    logger = logging.getLogger(__name__)
    
    # Convertir confs a un array de NumPy, si no lo es ya.
    if confs is None:
        confs_arr = np.array([])
    else:
        confs_arr = np.asarray(confs)
    
    total_points = confs_arr.size
    if total_points == 0:
        current_percentage = 0.0
    else:
        detected_points = np.sum(confs_arr >= threshold)
        current_percentage = detected_points / total_points

    # Solo loguear si el estado (por encima o por debajo de required_percentage) cambia
    current_state = current_percentage >= required_percentage
    if _last_detected_percentage is None or ((_last_detected_percentage >= required_percentage) != current_state):
        logger.info("Keypoints detected: %d of %d (%.1f%%)", 
                    int(current_percentage * total_points), 
                    total_points, 
                    current_percentage * 100)
    _last_detected_percentage = current_percentage

    return current_percentage >= required_percentage

def median_filter_angles(angles:list, kernel_size: int = 3):
    """
    Aplica un filtro de mediana a una lista de ángulos.
    
    Args:
        angles (list): Lista de ángulos en grados
        kernel_size (int, optional): Tamaño de la ventana para el filtro de mediana. Debe ser número impar. Por defecto es 3
    
    Returns:
        list: Lista de ángulos filtrada.
    """

    if len(angles) < kernel_size:
        return angles
    
    return medfilt(angles, kernel_size=kernel_size).tolist()

def resize_frame(frame, target_width=1280, target_height=720):
    # Obtener dimensiones originales
    h, w = frame.shape[:2]
    # Calcular la relación de aspecto
    aspect_ratio = w / h
    target_ratio = target_width / target_height

    if aspect_ratio > target_ratio:
        # La imagen es más ancha que la ventana objetivo: ajustar el ancho
        new_w = target_width
        new_h = int(target_width / aspect_ratio)
    else:
        # La imagen es más alta que la ventana objetivo: ajustar la altura
        new_h = target_height
        new_w = int(target_height * aspect_ratio)

    # Redimensionar el frame
    resized_frame = cv2.resize(frame, (new_w, new_h))
    
    # Crear un lienzo negro con la resolución objetivo
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    # Calcular posiciones para centrar el frame redimensionado
    start_y = (target_height - new_h) // 2
    start_x = (target_width - new_w) // 2
    # Pegar el frame redimensionado en el lienzo
    canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized_frame
    return canvas