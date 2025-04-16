import time

class TimeSync:
    """Sincroniza el tiempo de ejecución en función de la fuente de video.

    Si se utiliza una cámara en vivo, se devuelve el tiempo transcurrido desde el inicio.
    Si se utiliza un video, se calcula el tiempo relativo a partir del conteo de frames y la 
    tasa de cuadros por segundo (fps).

    Attributes:
        is_camera (bool): Indica si la fuente es una cámara en vivo (True) o un archivo de video (False).
        fps (int): Cuadros por segundo del video, usado para calcular el tiempo relativo si no es una cámara.
        frame_count (int): Contador de frames procesados.
        start_time (float): Tiempo de inicio (en segundos) de la ejecución.
    """

    def __init__(self, is_camera=True, fps=30):
        """Inicializa TimeSync con la fuente de video y la tasa de cuadros.

        Args:
            is_camera (bool, optional): Verdadero si se está utilizando una cámara. Por defecto, True.
            fps (int, optional): Tasa de cuadros por segundo para video. Por defecto, 30.
        """
        self.is_camera = is_camera
        self.fps = fps
        self.frame_count = 0
        self.start_time = time.time()  # Guarda el tiempo de inicio

    def get_current_time(self):
        """Obtiene el tiempo transcurrido relativo según la fuente de video.

        Si se utiliza una cámara en vivo, retorna el tiempo real transcurrido desde el inicio.
        Si se utiliza un video, incrementa el contador de frames y calcula el tiempo relativo
        dividiendo el número de frames por la tasa de cuadros por segundo (fps).

        Returns:
            float: Tiempo transcurrido en segundos.
        """
        if self.is_camera:
            # Retorna el tiempo transcurrido desde el inicio en tiempo real.
            return time.time() - self.start_time
        else:
            self.frame_count += 1
            # Calcula el tiempo basado en el número de frames y la tasa de cuadros.
            return self.frame_count / self.fps
