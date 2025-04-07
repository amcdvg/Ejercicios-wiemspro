from constants import Constants as K

class Anthopometry:
    @staticmethod
    def calc_forearm_length(height, age, gender, exercise='curl'):
        """_summary_

        Args:
            height (_type_): _description_
            age (_type_): _description_
            gender (_type_): _description_
            exercise (str, optional): _description_. Defaults to 'curl'.

        Returns:
            _type_: _description_
        """
        params = K.REGRESSION_PARAMS[exercise][gender]
        length_cm = params['A'] + params['B'] * (height * 100) + params['C'] * age + params['D'] * (1 if gender == 'male' else 0)
        return length_cm / 100

    @staticmethod
    def calc_squat_length(height, age, gender):
        """_summary_

        Args:
            height (_type_): _description_
            age (_type_): _description_
            gender (_type_): _description_

        Returns:
            _type_: _description_
        """
        height_cm = height * 100
        params = K.REGRESSION_PARAMS['squat'][gender]
        seg_length_cm = params['A'] + params['B'] * height_cm + params['C'] * age + params['D'] * (1 if gender == 'male' else 0)
        return seg_length_cm / 100

    @staticmethod
    def calc_pushup_length(height, age, gender):
        """_summary_

        Args:
            height (_type_): _description_
            age (_type_): _description_
            gender (_type_): _description_

        Returns:
            _type_: _description_
        """
        height_cm = height * 100
        params = K.REGRESSION_PARAMS['pushup'][gender]
        seg_length_cm = params['A'] + params['B'] * height_cm + params['C'] * age + params['D'] * (1 if gender == 'male' else 0)
        return seg_length_cm / 100

    @staticmethod
    def calc_deadlift_length(height, age, gender):
        """_summary_

        Args:
            height (_type_): _description_
            age (_type_): _description_
            gender (_type_): _description_

        Returns:
            _type_: _description_
        """
        height_cm = height * 100
        params = K.REGRESSION_PARAMS['deadlift'][gender]
        seg_length_cm = params['A'] + params['B'] * height_cm + params['C'] * age + params['D'] * (1 if gender == 'male' else 0)
        return seg_length_cm / 100

    @staticmethod
    def calc_reverse_fly_length(height, age, gender):
        """_summary_

        Args:
            height (_type_): _description_
            age (_type_): _description_
            gender (_type_): _description_

        Returns:
            _type_: _description_
        """
        height_cm = height * 100
        params = K.REGRESSION_PARAMS['reverse_fly'][gender]
        seg_length_cm = params['A'] + params['B'] * height_cm + params['C'] * age + params['D'] * (1 if gender == 'male' else 0)
        return seg_length_cm / 100

    @staticmethod
    def calc_swing_length(height, age, gender):
        """_summary_

        Args:
            height (_type_): _description_
            age (_type_): _description_
            gender (_type_): _description_

        Returns:
            _type_: _description_
        """
        height_cm = height * 100
        params = K.REGRESSION_PARAMS['swing'][gender]
        seg_length_cm = params['A'] + params['B'] * height_cm + params['C'] * age + params['D'] * (1 if gender == 'male' else 0)
        return seg_length_cm / 100

    @staticmethod
    def calc_renegade_row_length(height, age, gender):
        """_summary_

        Args:
            height (_type_): _description_
            age (_type_): _description_
            gender (_type_): _description_

        Returns:
            _type_: _description_
        """
        height_cm = height * 100
        params = K.REGRESSION_PARAMS['renegade_row'][gender]
        seg_length_cm = params['A'] + params['B'] * height_cm + params['C'] * age + params['D'] * (1 if gender == 'male' else 0)
        return seg_length_cm / 100

    @staticmethod
    def calc_rear_lunge_length(height, age, gender):
        """_summary_

        Args:
            height (_type_): _description_
            age (_type_): _description_
            gender (_type_): _description_

        Returns:
            _type_: _description_
        """
        height_cm = height * 100
        params = K.REGRESSION_PARAMS['rear_lunge'][gender]
        seg_length_cm = params['A'] + params['B'] * height_cm + params['C'] * age + params['D'] * (1 if gender == 'male' else 0)
        return seg_length_cm / 100

    @staticmethod
    def calc_bench_dips_length(height, age, gender):
        """_summary_

        Args:
            height (_type_): _description_
            age (_type_): _description_
            gender (_type_): _description_

        Returns:
            _type_: _description_
        """
        height_cm = height * 100
        params = K.REGRESSION_PARAMS['bench_dips'][gender]
        seg_length_cm = params['A'] + params['B'] * height_cm + params['C'] * age + params['D'] * (1 if gender == 'male' else 0)
        return seg_length_cm / 100

    @staticmethod
    def calc_overhead_triceps_length(height, age, gender):
        """_summary_

        Args:
            height (_type_): _description_
            age (_type_): _description_
            gender (_type_): _description_

        Returns:
            _type_: _description_
        """
        height_cm = height * 100
        params = K.REGRESSION_PARAMS['overhead_triceps'][gender]
        seg_length_cm = params['A'] + params['B'] * height_cm + params['C'] * age + params['D'] * (1 if gender == 'male' else 0)
        return seg_length_cm / 100

    @staticmethod
    def calc_plank_length(height, age, gender):
        """_summary_

        Args:
            height (_type_): _description_
            age (_type_): _description_
            gender (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Se usa solo altura
        params = K.REGRESSION_PARAMS['plank']['default']
        return params['A'] + params['B'] * height
