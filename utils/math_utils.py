import numpy as np

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