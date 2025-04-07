class Constants:
    """_summary_
    """
    YOLO_POSE_KEYPOINTS = {'NOSE': 0,
        'LEFT_EYE': 1,
        'RIGHT_EYE': 2,
        'LEFT_EAR': 3,
        'RIGHT_EAR': 4,
        'LEFT_SHOULDER': 5,
        'RIGHT_SHOULDER': 6,
        'LEFT_ELBOW': 7,
        'RIGHT_ELBOW': 8,
        'LEFT_WRIST': 9,
        'RIGHT_WRIST': 10,
        'LEFT_HIP': 11,
        'RIGHT_HIP': 12,
        'LEFT_KNEE': 13,
        'RIGHT_KNEE': 14,
        'LEFT_ANKLE': 15,
        'RIGHT_ANKLE': 16}
    
    EXERCISE_MAPPING = {
    'curl': (
        'CurlExercise', 
        [
            '{side}_SHOULDER',
            '{side}_ELBOW',
            '{side}_WRIST'
        ]
    ),
    'squat': (
        'SquatExercise',
        [
            '{side}_HIP',
            '{side}_KNEE',
            '{side}_ANKLE'
        ]
    ),
    'pushup': (
        'PushupExercise',
        [
            '{side}_SHOULDER',
            '{side}_ELBOW',
            '{side}_WRIST'
        ]
    ),
    'deadlift': (
        'DeadliftExercise',
        [
            '{side}_SHOULDER',
            '{side}_HIP',
            '{side}_KNEE'
        ]
    ),
    "reverse_fly": (
        'ReverseFlyExercise',
        [
            '{side}_SHOULDER',
            '{side}_ELBOW'
        ]
    ),
    'swing': (
        'SwingExercise',
        [
            '{side}_SHOULDER',
            '{side}_HIP',
            '{side}_KNEE'
        ]
    ),
    'renegade_row': (
        'RenegadeRowExercise',
        [
            '{side}_SHOULDER',
            '{side}_ELBOW',
            '{side}_WRIST'
        ]
    ),
    'rear_lunge': (
        'RearLugeExercise',
        [
            '{side}_HIP',
            '{side}_KNEE',
            '{side}_ANKLE'
        ]
    ),
    'bench_dips': (
        'BenchDipsExercise',
        [
            '{side}_SHOULDER',
            '{side}_ELBOW',
            '{side}_WRIST'
        ]
    ),
    'overhead_triceps': (
        'ExtensionTricepsExercise',
        [
            '{side}_SHOULDER',
            '{side}_ELBOW',
            '{side}_WRIST'
        ]
    )
}

    
    REGRESSION_PARAMS = {
    'curl': {
        'male': {'A': 0.5, 'B': 0.15, 'C': -0.01, 'D': 0.0},
        'female': {'A': 0.45, 'B': 0.14, 'C': -0.015, 'D': 0.0},
    },
    'pushup': {
        'male': {'A': 0.6, 'B': 0.16, 'C': -0.005, 'D': 0.0},
        'female': {'A': 0.55, 'B': 0.15, 'C': -0.005, 'D': 0.0},
    },
    'squat': {
        'male': {'A': 30.0, 'B': 0.25, 'C': 0.1, 'D': 0.0},  # valores en cm; se convertirán a metros
        'female': {'A': 28.0, 'B': 0.24, 'C': 0.1, 'D': 0.0},
    },
    'deadlift': {
        'male': {'A': 32.0, 'B': 0.26, 'C': 0.08, 'D': 0.0},
        'female': {'A': 30.0, 'B': 0.25, 'C': 0.08, 'D': 0.0},
    },
    'reverse_fly': {
        'male': {'A': 15.0, 'B': 0.19, 'C': 0.0, 'D': 0.0},
        'female': {'A': 14.0, 'B': 0.18, 'C': 0.0, 'D': 0.0},
    },
    'swing': {
        'male': {'A': 25.0, 'B': 0.24, 'C': 0.07, 'D': 0.0},
        'female': {'A': 23.0, 'B': 0.23, 'C': 0.07, 'D': 0.0},
    },
    'renegade_row': {
        'male': {'A': 25.0, 'B': 0.24, 'C': 0.07, 'D': 0.0},
        'female': {'A': 23.0, 'B': 0.23, 'C': 0.07, 'D': 0.0},
    },
    'rear_lunge': {
        'male': {'A': 30.0, 'B': 0.25, 'C': 0.1, 'D': 0.0},
        'female': {'A': 28.0, 'B': 0.24, 'C': 0.1, 'D': 0.0},
    },
    'bench_dips': {
        'male': {'A': 14.0, 'B': 0.146, 'C': 0.0, 'D': 0.0},
        'female': {'A': 13.0, 'B': 0.146, 'C': 0.0, 'D': 0.0},
    },
    'overhead_triceps': {
        'male': {'A': 18.0, 'B': 0.186, 'C': 0.0, 'D': 0.0},
        'female': {'A': 17.0, 'B': 0.186, 'C': 0.0, 'D': 0.0},
    },
    'plank': {
        'default': {'A': 0.43, 'B': 0.0}
    }
}
    
    EFFECTIVE_LENGTH_FUNCTIONS = {
    'curl': lambda h: 0.95 * (0.146 * h),
    'squat': lambda h: 0.85 * ((0.245 * h) + (0.246 * h)),
    'pushup': lambda h: 0.75 * ((0.093 * h) + (0.146 * h)),
    'deadlift': lambda h: 0.90 * ((0.245 * h) + (0.246 * h)),
    'reverse_fly': lambda h: 0.95 * (0.186 * h),
    'swing': lambda h: 0.80 * ((0.245 * h) + (0.146 * h)),
    'renegade_row': lambda h: 0.80 * ((0.245 * h) + (0.146 * h)),
    'rear_lunge': lambda h: 0.90 * ((0.245 * h) + (0.246 * h)),
    'bench_dips': lambda h: 0.95 * (0.146 * h),
    'overhead_triceps': lambda h: 0.95 * ((0.186 * h) + (0.146 * h))
}
    USE_REGRESSION_EFFECTIVE_LENGTH = True
    ADJUSTMENT_FACTORS_VMAX = {
        'curl' : 0.858,
        'pushup' : 0.858,
        'squat' : 0.75}
    
    SHOW_POSE_OVERLAYS = True

    CURL_COUNTER = 0
    CURL_STAGE = None
    CURL_MAX_ANGLE = 145
    CURL_MIN_ANGLE = 130

    SQUAT_COUNTER = 0
    SQUAT_STAGE = None
    SQUAT_MAX_ANGLE = 170
    SQUAT_MIN_ANGLE = 160
    SQUAT_TORSO_MIN_ANGLE = 170
    
    PUSHUP_COUNTER = 0
    PUSHUP_STAGE = None
    PUSHUP_MAX_ANGLE = 152
    PUSHUP_MIN_ANGLE = 142

    DEADLIFT_MIN_TORSO_ANGLE = 150
    DEADLIFT_MAX_TORSO_ANGLE = 170
    
    REVERSEFLY_MIN_ANGLE = 120
    REVERSEFLY_MAX_ANGLE = 140

    SWING_MIN_TORSO_ANGLE = 35
    SWING_MAX_TORSO_ANGLE = 45

    RENEGADEROW_MIN_ANGLE = 118
    RENEGADEROW_MAX_ANGLE = 140

    ZANCADA_MIN_ANGLE = 140
    ZANCADA_MAX_ANGLE = 170

    TRICEPS_BANCO_MIN_ANGLE = 145
    TRICEPS_BANCO_MAX_ANGLE =  158

    TRICEPS_EXTENSION_MIN_ANGLE = 110
    TRICEPS_EXTENSION_MAX_ANGLE = 95

    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720

    EXERCISE_ANGLE_RANGES = {
    'curl': (40, 180),
    'pushup': (85, 170),
    'squat': (100, 180),
    'plank': (40, 180),
    'reverse_fly': (170, 90),
    'swing' : (120, 0),
    'renegade_row': (110, 170),
    'bench_dips': (90, 180),
    'overhead_triceps':(70, 180)
    }
      # Ya definidos:
    VMED_CORRECTION_FACTORS = {
        'curl': 1.05,
        'squat': 1.03,
        'pushup': 0.98,
        'deadlift': 1.00,
        'reverse_fly': 1.00,
        'swing': 1.00,
        'renegade_row': 1.00,
        'rear_lunge': 1.00,
        'bench_dips': 1.00,
        'overhead_triceps': 1.00,
        'plank': 1.00
    }

    VMED_BASE_FACTORS = {
        'curl': 0.85,
        'squat': 0.87,
        'pushup': 0.84,
        'deadlift': 0.90,
        'reverse_fly': 0.85,
        'swing': 0.85,
        'renegade_row': 0.85,
        'rear_lunge': 0.88,
        'bench_dips': 0.85,
        'overhead_triceps': 0.85,
        'plank': 0.85,     # si lo usas
    }

    ADJUSTMENT_FACTORS_VMAX = {
        'curl': 0.858,
        'pushup': 0.858,
        'squat': 0.75,
        'deadlift': 0.80,
        'reverse_fly': 0.80,
        'swing': 0.80,
        'renegade_row': 0.80,
        'rear_lunge': 0.80,
        'bench_dips': 0.80,
        'overhead_triceps': 0.80,
        'plank': 0.80
    }

    VMAX_BASE_FACTORS = {
        'curl': 1.4,
        'squat': 0.51,
        'pushup': 0.554,
        'deadlift': 0.35,
        'reverse_fly': 2.25,
        'swing': 1,
        'renegade_row': 0.38,
        'rear_lunge': 0.49,
        'bench_dips': 0.58,
        'overhead_triceps': 0.59,
        'plank': 0.48
    }

    # NUEVO: para sustituir el factor de ROM (antes 0.91)
    ROM_BASE_FACTORS = {
        'curl': 2.37,
        'squat': 0.93,
        'pushup': 1.797,
        'deadlift': 0.59,
        'reverse_fly': 10.146,
        'swing': 2.72,
        'renegade_row': 0.8,
        'rear_lunge': 0.48,
        'bench_dips': 1.01,
        'overhead_triceps': 1.31,
        'plank': 0.91
    }

    VMED_FINAL_FACTORS = {
        'curl': 1.95,
        'squat': 0.825,
        'pushup':0.756,
        'deadlift': 0.37,
        'reverse_fly': 3.32,
        'swing': 1.16,
        'renegade_row': 0.79,
        'rear_lunge': 0.23,
        'bench_dips': 0.71,
        'overhead_triceps': 2.51,
        'plank': 0.62
    }

    PHASE_CORRECTION = {
        'concentric': 0.97,
        'eccentric': 1.03
    }
    
    CONF_THRESHOLD = 0.25    # Umbral mínimo de confianza

    DEFAULT_COLUMNS_TO_EXPORT = [
    "start_datetime",    # Fecha y hora de inicio del ejercicio (formato ISO)
    "age",               # Edad del sujeto
    "gender",            # Género
    "height (m)",        # Estatura en metros
    "effective_length (m)",
    "min_angle (°)",     # Ángulo mínimo registrado
    "max_angle (°)",     # Ángulo máximo registrado
    "ROM (cm)",           # Rango de movimiento en metros
    "VMED (m/s)",        # Velocidad media en m/s
    "VMAX (m/s)",        # Velocidad máxima en m/s
    "rep_time",          # Tiempo de repetición en segundos
    "repetition",        # Número de repetición
]
    TORSO_LOWER_BOUND = 25.0  # Inclinación mínima recomendada (en grados)
    TORSO_UPPER_BOUND = 45.0  # Inclinación máxima recomendada (en grados)

    POSE_SKELETON = [
    (0, 1), (1, 3),  # Nariz -> Ojo/oreja Izq
    (0, 2), (2, 4),  # Nariz -> Ojo/oreja Der
    (5, 7), (7, 9),  # Brazo izq
    (6, 8), (8, 10), # Brazo der
    (5, 6),          # Hombros
    (5, 11), (6, 12),# Hombros -> Cadera
    (11, 13), (13, 15), # Pierna izq
    (12, 14), (14, 16)  # Pierna der
]