import numpy as np
from ultralytics import YOLO
from constants import Constants as K


class PoseEstimator:
    """Clase para realizar la estimación de poses a partir de frames de video usando el modelo YOLO.

    Esta clase carga un modelo pre-entrenado de YOLO para la detección de poses y
    proporciona un método para estimar los keypoints (puntos clave) y sus correspondientes
    valores de confianza a partir de un frame de video. Dependiendo de la configuración,
    también se pueden aplicar overlays (anotaciones sobre el frame) si K.SHOW_POSE_OVERLAYS es True.
    """

    def __init__(self, model_path='models/yolo11n-pose.pt'):
        """Inicializa el estimador de poses cargando el modelo YOLO pre-entrenado.

        Args:
            model_path (str, optional): Ruta del modelo pre-entrenado YOLO para poses. 
                Por defecto, 'models/yolo11n-pose.pt'.
        """
        
        self.model = YOLO(model_path)

    def estimate(self, frame, relevant_indices=None):
        """Realiza la estimación de pose sobre un frame.

        Se ejecuta el modelo YOLO sobre el frame para obtener los keypoints y los valores de confianza.
        Si se activa la opción SHOW_POSE_OVERLAYS en Constants, se retorna el frame anotado; de lo 
        contrario, se retorna una copia del frame sin anotaciones.
        Si no se especifican índices relevantes, se usa la primera detección. Si se proporcionan, se 
        selecciona la detección que maximiza la media de las confianzas en los índices relevantes.

        Args:
            frame (numpy.ndarray): Imagen del frame en el que se realizará la estimación de la pose.
            relevant_indices (list[int], optional): Lista de índices de keypoints relevantes para seleccionar
                la detección óptima. Por defecto, None.

        Returns:
            tuple or None: Una tupla ((kpts, confs), annotated_frame), donde:
              - kpts (numpy.ndarray): Array de keypoints (coordenadas XY) para la detección seleccionada.
              - confs (numpy.ndarray): Array de valores de confianza para la detección seleccionada.
              - annotated_frame (numpy.ndarray): El frame original con overlays (si SHOW_POSE_OVERLAYS es True)
                o una copia del frame.
            Si no se detectan keypoints válidos, se retorna (None, annotated_frame).
        """
        results = self.model(frame, verbose=False)
        if K.SHOW_POSE_OVERLAYS:
            annotated_frame = results[0].plot()
        else:
            annotated_frame = frame.copy()

        # Verificar si se detectaron keypoints y si contienen datos
        if (
            results[0].keypoints is None or 
            not hasattr(results[0].keypoints, "xy") or 
            len(results[0].keypoints.xy) == 0 or
            results[0].keypoints.conf is None or 
            len(results[0].keypoints.conf) == 0
        ):
            return None, annotated_frame

        # Si no se especifican índices relevantes, usar la primera detección
        if relevant_indices is None:
            try:
                kpts = results[0].keypoints.xy[0].cpu().numpy()
                confs = results[0].keypoints.conf[0].cpu().numpy()
            except Exception:
                return None, annotated_frame
            return (kpts, confs), annotated_frame

        # Seleccionar la mejor detección en base a la media de confianzas en los índices relevantes.
        best_index = 0
        best_conf = -1
        for i in range(len(results[0].keypoints.xy)):
            try:
                current_confs = results[0].keypoints.conf[i].cpu().numpy()
            except Exception:
                continue
            if current_confs is None:
                continue
            mean_conf = np.mean(current_confs[relevant_indices])
            if mean_conf > best_conf:
                best_conf = mean_conf
                best_index = i

        try:
            kpts = results[0].keypoints.xy[best_index].cpu().numpy()
            confs = results[0].keypoints.conf[best_index].cpu().numpy()
        except Exception:
            return None, annotated_frame

        return (kpts, confs), annotated_frame
