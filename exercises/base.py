from abc import ABC, abstractmethod

class Exercise(ABC):
    """_summary_

    Args:
        ABC (_type_): _description_
    """
    def __init__(self):
        pass

    @abstractmethod
    def update(self, keypoints, confs):
        """
        Actualiza el estado del ejercicio según los keypoints y sus confianzas.
        Retorna información relevante (por ejemplo, ángulo, tiempo, etc.) si se requiere.
        """
        pass
    
    @abstractmethod
    def draw(self, frame):
        """
        Dibuja en el frame la información (reps, etapa, tiempo, etc.) del ejercicio.
        """
        pass