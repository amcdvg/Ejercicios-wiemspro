import numpy as np
class KalmanFilter1D:
    """_summary_
    """
    def __init__(self, dt=0.033, process_var=1, measurement_var=1e-4):
        self.dt = dt
        # Estado: [posición, velocidad]
        self.x = np.array([[0.0], [0.0]], dtype=np.float32)
        self.P = np.eye(2, dtype=np.float32)
        self.F = np.array([[1, dt],
                           [0, 1]], dtype=np.float32)  # Transición de estado
        self.H = np.array([[1, 0]], dtype=np.float32)  # Observación: medimos sólo posición
        # Matriz de covarianza del proceso
        self.Q = process_var * np.array([[dt**4/4, dt**3/2],
                                          [dt**3/2, dt**2]], dtype=np.float32)
        # Covarianza del ruido de medición
        self.R = np.array([[measurement_var]], dtype=np.float32)

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z):
        y = z - np.dot(self.H, self.x)  # innovación
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(K, np.dot(self.H, self.P))
        return self.x[0, 0]