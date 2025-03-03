import numpy as np
import logging
from scipy.signal import savgol_filter

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

def valid_full_pose(confs, threshold=0.3, required_percentage = 0.8):
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