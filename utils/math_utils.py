import numpy as np

def calculate_angle(a, b, c):
        """_summary_

        Args:
            a (_type_): _description_
            b (_type_): _description_
            c (_type_): _description_

        Returns:
            _type_: _description_
        """
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        return angle if angle <= 180.0 else 360 - angle