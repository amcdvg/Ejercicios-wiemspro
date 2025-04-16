import numpy as np

class KalmanFilter1D:
    """Implementa un filtro Kalman unidimensional para estimar la posición a partir de mediciones ruidosas.

    El modelo asume un sistema de segundo orden donde el estado es
    [posición, velocidad]. Se actualiza la estimación del estado y de la covarianza
    de error mediante la formulación estándar del filtro Kalman.

    Attributes:
        dt (float): Intervalo de tiempo entre mediciones (en segundos).
        x (np.ndarray): Vector de estado, de forma [posición, velocidad]^T.
        P (np.ndarray): Matriz de covarianza del estado.
        F (np.ndarray): Matriz de transición de estado.
        H (np.ndarray): Matriz de observación (se mide solo la posición).
        Q (np.ndarray): Matriz de covarianza del proceso (ruido del proceso).
        R (np.ndarray): Covarianza del ruido de medición.
    """

    def __init__(self, dt=0.033, process_var=1, measurement_var=1e-4):
        """Inicializa el filtro Kalman con los parámetros de tiempo y varianzas especificados.

        Args:
            dt (float, optional): Intervalo de tiempo entre mediciones. Por defecto 0.033 (aprox. 30 fps).
            process_var (float, optional): Varianza del proceso, que afecta la matriz Q. Por defecto 1.
            measurement_var (float, optional): Varianza del ruido de medición, que afecta la matriz R. Por defecto 1e-4.
        """
        self.dt = dt
        # Estado inicial: posición = 0, velocidad = 0.
        self.x = np.array([[0.0], [0.0]], dtype=np.float32)
        # Matriz de covarianza inicial: identidad.
        self.P = np.eye(2, dtype=np.float32)
        # Matriz de transición de estado, considerando un modelo constante de velocidad.
        self.F = np.array([[1, dt],
                           [0, 1]], dtype=np.float32)
        # Matriz de observación: se observa solo la posición.
        self.H = np.array([[1, 0]], dtype=np.float32)
        # Matriz de covarianza del proceso, derivada del modelo de movimiento.
        self.Q = process_var * np.array([[dt**4 / 4, dt**3 / 2],
                                         [dt**3 / 2, dt**2]], dtype=np.float32)
        # Covarianza del ruido de medición.
        self.R = np.array([[measurement_var]], dtype=np.float32)

    def predict(self):
        """Realiza el paso de predicción del filtro Kalman.

        Actualiza la estimación del estado y la matriz de covarianza del estado
        aplicando la matriz de transición F y sumando la covarianza del proceso Q.
        """
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z):
        """Actualiza la estimación del estado con una nueva medición.

        Se calcula la innovación (error entre la medición y la predicción),
        se actualiza la ganancia Kalman y se corrige la estimación del estado y 
        su covarianza.

        Args:
            z (np.ndarray): La medición (posiblemente un escalar) como un array de forma [[valor]].

        Returns:
            float: La posición estimada (primer elemento del vector de estado).
        """
        # Cálculo de la innovación (error de la medición).
        y = z - np.dot(self.H, self.x)
        # Cálculo del estadístico S.
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        # Cálculo de la ganancia Kalman.
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        # Actualización del estado con la innovación ponderada por la ganancia.
        self.x = self.x + np.dot(K, y)
        # Actualización de la covarianza.
        self.P = self.P - np.dot(K, np.dot(self.H, self.P))
        return self.x[0, 0]
