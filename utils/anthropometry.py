from constants import Constants as K


class Anthropometry:
    """Clase estática para realizar cálculos antropométricos que estiman longitudes de segmentos 
    corporales para distintos ejercicios usando regresiones específicas.

    Los métodos están definidos como estáticos y calculan, a partir de la altura (en metros), la edad 
    y el género, longitudes específicas en metros para diferentes ejercicios. Los valores se obtienen a partir 
    de parámetros de regresión definidos en el objeto Constants.
    """

    @staticmethod
    def calc_forearm_length(height, age, gender, exercise='curl'):
        """Calcula la longitud del antebrazo para el ejercicio especificado.

        Se utiliza una función de regresión definida en Constants.REGRESSION_PARAMS, que depende de la 
        altura (convertida a centímetros), la edad y el género.

        Args:
            height (float): Altura del sujeto en metros.
            age (int or float): Edad del sujeto.
            gender (str): Género del sujeto ('male' o 'female').
            exercise (str, optional): Ejercicio para el cual se calcula la longitud (por defecto 'curl').

        Returns:
            float: Longitud del antebrazo en metros.
        """
        params = K.REGRESSION_PARAMS[exercise][gender]
        length_cm = (
            params['A'] + 
            params['B'] * (height * 100) + 
            params['C'] * age + 
            params['D'] * (1 if gender == 'male' else 0)
        )
        return length_cm / 100

    @staticmethod
    def calc_squat_length(height, age, gender):
        """Calcula la longitud del segmento de sentadilla para el ejercicio de squat.

        Utiliza los parámetros de regresión correspondientes al ejercicio 'squat' definidos en Constants.

        Args:
            height (float): Altura del sujeto en metros.
            age (int or float): Edad del sujeto.
            gender (str): Género del sujeto ('male' o 'female').

        Returns:
            float: Longitud estimada en metros para el ejercicio de squat.
        """
        height_cm = height * 100
        params = K.REGRESSION_PARAMS['squat'][gender]
        seg_length_cm = (
            params['A'] + 
            params['B'] * height_cm + 
            params['C'] * age + 
            params['D'] * (1 if gender == 'male' else 0)
        )
        return seg_length_cm / 100

    @staticmethod
    def calc_pushup_length(height, age, gender):
        """Calcula la longitud efectiva para el ejercicio de pushup.

        Usa los parámetros de regresión definidos para 'pushup' en Constants.REGRESSION_PARAMS.

        Args:
            height (float): Altura del sujeto en metros.
            age (int or float): Edad del sujeto.
            gender (str): Género del sujeto ('male' o 'female').

        Returns:
            float: Longitud efectiva en metros para el ejercicio de pushup.
        """
        height_cm = height * 100
        params = K.REGRESSION_PARAMS['pushup'][gender]
        seg_length_cm = (
            params['A'] + 
            params['B'] * height_cm + 
            params['C'] * age + 
            params['D'] * (1 if gender == 'male' else 0)
        )
        return seg_length_cm / 100

    @staticmethod
    def calc_deadlift_length(height, age, gender):
        """Calcula la longitud del segmento para el ejercicio de deadlift (peso muerto).

        Se emplean los parámetros de regresión para 'deadlift' definidos en Constants.

        Args:
            height (float): Altura del sujeto en metros.
            age (int or float): Edad del sujeto.
            gender (str): Género del sujeto ('male' o 'female').

        Returns:
            float: Longitud en metros para el ejercicio de deadlift.
        """
        height_cm = height * 100
        params = K.REGRESSION_PARAMS['deadlift'][gender]
        seg_length_cm = (
            params['A'] + 
            params['B'] * height_cm + 
            params['C'] * age + 
            params['D'] * (1 if gender == 'male' else 0)
        )
        return seg_length_cm / 100

    @staticmethod
    def calc_reverse_fly_length(height, age, gender):
        """Calcula la longitud del segmento para el ejercicio de reverse fly.

        Usa los parámetros de regresión definidos en Constants para 'reverse_fly'.

        Args:
            height (float): Altura del sujeto en metros.
            age (int or float): Edad del sujeto.
            gender (str): Género del sujeto ('male' o 'female').

        Returns:
            float: Longitud del segmento en metros para reverse fly.
        """
        height_cm = height * 100
        params = K.REGRESSION_PARAMS['reverse_fly'][gender]
        seg_length_cm = (
            params['A'] + 
            params['B'] * height_cm + 
            params['C'] * age + 
            params['D'] * (1 if gender == 'male' else 0)
        )
        return seg_length_cm / 100

    @staticmethod
    def calc_swing_length(height, age, gender):
        """Calcula la longitud del segmento para el ejercicio de swing.

        Aplica los parámetros de regresión para 'swing' definidos en Constants.

        Args:
            height (float): Altura del sujeto en metros.
            age (int or float): Edad del sujeto.
            gender (str): Género del sujeto ('male' o 'female').

        Returns:
            float: Longitud del segmento en metros para el ejercicio de swing.
        """
        height_cm = height * 100
        params = K.REGRESSION_PARAMS['swing'][gender]
        seg_length_cm = (
            params['A'] + 
            params['B'] * height_cm + 
            params['C'] * age + 
            params['D'] * (1 if gender == 'male' else 0)
        )
        return seg_length_cm / 100

    @staticmethod
    def calc_renegade_row_length(height, age, gender):
        """Calcula la longitud del segmento para el ejercicio de renegade row.

        Utiliza los parámetros de regresión para 'renegade_row' de Constants.

        Args:
            height (float): Altura del sujeto en metros.
            age (int or float): Edad del sujeto.
            gender (str): Género del sujeto ('male' o 'female').

        Returns:
            float: Longitud del segmento en metros para renegade row.
        """
        height_cm = height * 100
        params = K.REGRESSION_PARAMS['renegade_row'][gender]
        seg_length_cm = (
            params['A'] + 
            params['B'] * height_cm + 
            params['C'] * age + 
            params['D'] * (1 if gender == 'male' else 0)
        )
        return seg_length_cm / 100

    @staticmethod
    def calc_rear_lunge_length(height, age, gender):
        """Calcula la longitud del segmento para el ejercicio de rear lunge.

        Se emplean los parámetros de regresión para 'rear_lunge' definidos en Constants.

        Args:
            height (float): Altura del sujeto en metros.
            age (int or float): Edad del sujeto.
            gender (str): Género del sujeto ('male' o 'female').

        Returns:
            float: Longitud del segmento en metros para rear lunge.
        """
        height_cm = height * 100
        params = K.REGRESSION_PARAMS['rear_lunge'][gender]
        seg_length_cm = (
            params['A'] +
            params['B'] * height_cm +
            params['C'] * age +
            params['D'] * (1 if gender == 'male' else 0)
        )
        return seg_length_cm / 100

    @staticmethod
    def calc_bench_dips_length(height, age, gender):
        """Calcula la longitud del segmento para el ejercicio de bench dips.

        Utiliza los parámetros de regresión para 'bench_dips' definidos en Constants.

        Args:
            height (float): Altura del sujeto en metros.
            age (int or float): Edad del sujeto.
            gender (str): Género del sujeto ('male' o 'female').

        Returns:
            float: Longitud del segmento en metros para bench dips.
        """
        height_cm = height * 100
        params = K.REGRESSION_PARAMS['bench_dips'][gender]
        seg_length_cm = (
            params['A'] +
            params['B'] * height_cm +
            params['C'] * age +
            params['D'] * (1 if gender == 'male' else 0)
        )
        return seg_length_cm / 100

    @staticmethod
    def calc_overhead_triceps_length(height, age, gender):
        """Calcula la longitud del segmento para el ejercicio de extensión de tríceps por encima de la cabeza.

        Usa los parámetros de regresión para 'overhead_triceps' definidos en Constants.

        Args:
            height (float): Altura del sujeto en metros.
            age (int or float): Edad del sujeto.
            gender (str): Género del sujeto ('male' o 'female').

        Returns:
            float: Longitud del segmento en metros para overhead triceps.
        """
        height_cm = height * 100
        params = K.REGRESSION_PARAMS['overhead_triceps'][gender]
        seg_length_cm = (
            params['A'] +
            params['B'] * height_cm +
            params['C'] * age +
            params['D'] * (1 if gender == 'male' else 0)
        )
        return seg_length_cm / 100

    @staticmethod
    def calc_plank_length(height, age, gender):
        """Calcula la longitud efectiva para el ejercicio de plank.

        En este ejercicio se utiliza únicamente la altura para calcular la longitud, 
        usando los parámetros definidos para 'plank' bajo la clave 'default' en Constants.

        Args:
            height (float): Altura del sujeto en metros.
            age (int or float): Edad del sujeto (no utilizada en este cálculo).
            gender (str): Género del sujeto (no utilizada en este cálculo).

        Returns:
            float: Longitud efectiva para plank (valor unitario).
        """
        params = K.REGRESSION_PARAMS['plank']['default']
        return params['A'] + params['B'] * height
