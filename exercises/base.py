from abc import ABC, abstractmethod
import numpy as np
class Exercise(ABC):
    """Clase abstracta base que define el patrón de diseño Template Method.

    Esta clase define la plantilla para el procesamiento de un ejercicio, que incluye la actualziación del estado (para contar repeticiones, calculos de ánguos, etc) y el dibujo de la interfaz cada frame del video

    Métodos abstractos: 
        - update(keypoints, confs): Actualiza el estado del ejercicio a partir de los keypoints.
        - draw(frame): Dibuja en el frame la interfaz específica del ejercicio 
    Método Plantilla:
        - process(keypoints, confs, frame): Ejecuta el flujo completo del ejercicio llamado a update() y draw() en secuencia
    """
    def __init__(self):
        self.down_time = None 
        self.up_time = None  
        self.down_start_time = None  
        self.up_start_time = None
    def process(self, keypoints, confs, frame, current_time):
        """
        Ejecuta el flujo de procesamiento del ejercicio.

        Este método es la plantilla que define el algoritmo general:
          1. Llama a update(keypoints, confs) para actualizar el estado del ejercicio.
          2. Llama a draw(frame) para dibujar la información en el frame.

        Args:
            keypoints (array-like): Puntos clave detectados.
            confs (array-like): Confianzas asociadas a cada keypoint.
            frame (numpy.ndarray): Frame de video actual.

        Returns:
            numpy.ndarray: El frame actualizado con los overlays del ejercicio.
        """
        self.update(keypoints, confs, current_time)
        self.draw(frame)
        return frame

    @abstractmethod
    def update(self, keypoints, confs):
        """
        Actualiza el estado del ejercicio basado en los keypoints y sus confianzas.

        Debe ser implementado por cada subclase para definir la lógica específica

        Args:
            keypoints: Puntos clave detectados.
            confs: Valores de confianza asociados a los keypoints.
        """
        pass
    
    @abstractmethod
    def draw(self, frame: np.ndarray) -> np.ndarray:
        """
        Dibuja la interfaz del ejercicio en el frame.

        Debe ser implementado por cada subclase para definir cómo se debe visualizar
        la información (por ejemplo, contador de repeticiones, etapa del ejercicio).

        Args:
            frame (numpy.ndarray): Frame de video actual.
        """
        pass