from constants import Constants

class Anthopometry:
    @staticmethod
    def calc_forearm_length(height, age, gender, exercise='curl'):
        params = Constants.REGRESSION_PARAMS[exercise][gender]
        return params['A'] + params['B'] * height + params['C'] * age + params['D'] * (1 if gender == 'male' else 0)

    @staticmethod
    def calc_squat_length(height, age, gender):
        height_cm = height * 100
        params = Constants.REGRESSION_PARAMS['squat'][gender]
        seg_length_cm = params['A'] + params['B'] * height_cm + params['C'] * age + params['D'] * (1 if gender == 'male' else 0)
        return seg_length_cm / 100

    @staticmethod
    def calc_plank_length(height, age, gender):
        params = Constants.REGRESSION_PARAMS['plank']['default']
        return params['A'] + params['B'] * height