import time

class TimeSync:
    def __init__(self, is_camera=True, fps=30):
        """_summary_

        Args:
            is_camera (bool, optional): _description_. Defaults to True.
            fps (int, optional): _description_. Defaults to 30.
        """
        self.is_camera = is_camera
        self.fps = fps
        self.frame_count = 0
        self.start_time = time.time()  # Guarda el tiempo de inicio

    def get_current_time(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        if self.is_camera:
            # Retorna el tiempo relativo (transcurrido desde el inicio)
            return time.time() - self.start_time
        else:
            self.frame_count += 1
            # Tiempo relativo calculado a partir de los frames
            return self.frame_count / self.fps