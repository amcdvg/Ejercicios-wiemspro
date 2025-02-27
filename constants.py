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
                YOLO_POSE_KEYPOINTS['RIGHT_SHOULDER'],
                YOLO_POSE_KEYPOINTS['RIGHT_ELBOW'],
                YOLO_POSE_KEYPOINTS['RIGHT_WRIST']
            ]
        ),
        'squat': (
            'SquatExercise',
            [
                YOLO_POSE_KEYPOINTS['RIGHT_HIP'],
                YOLO_POSE_KEYPOINTS['RIGHT_KNEE'],
                YOLO_POSE_KEYPOINTS['RIGHT_ANKLE']
            ]
        ),
        'pushup': (
            'PushupExercise',
            [
                YOLO_POSE_KEYPOINTS['RIGHT_SHOULDER'],
                YOLO_POSE_KEYPOINTS['RIGHT_ELBOW'],
                YOLO_POSE_KEYPOINTS['RIGHT_WRIST']
            ]
        ),
        'plank': (
            'PlankExercise',
            [
                YOLO_POSE_KEYPOINTS['RIGHT_SHOULDER'],
                YOLO_POSE_KEYPOINTS['RIGHT_HIP'],
                YOLO_POSE_KEYPOINTS['RIGHT_KNEE']
            ]
        )
    }
    REGRESSION_PARAMS = {
        'curl': {
            'male': {'A': -0.02, 'B': 0.16, 'C': 0.0005, 'D': 0.02},
            'female': {'A': -0.03, 'B': 0.15, 'C': 0.0004, 'D': 0.0}
        },
        'pushup': {
            'male': {'A': -0.02, 'B': 0.16, 'C': 0.0005, 'D': 0.02},
            'female': {'A': -0.03, 'B': 0.15, 'C': 0.0004, 'D': 0.0}
        },
        'plank': {
            'default': {'A': 0.0, 'B': 0.5, 'C': 0.0, 'D': 0.0}
        },
        'squat': {
            'male': {'A': -69.4/3.38, 'B': 1/3.38, 'C': 0.0, 'D': 0.02},
            'female': {'A': -46.7/3.47, 'B': 1/3.47, 'C': 0.0, 'D': 0.0}}}
    EFFECTIVE_LENGTH_FUNCTIONS = {
        'squat': lambda h: 0.85 * (0.245 * h) + (0.246 * h),
        'curl': lambda h: 0.95 * (0.146 * h),
        'pushup': lambda h: 0.75 * ((0.093 * h) + (0.146 * h)),
        'plank': lambda h: 0.43 * h}
    
    ADJUSTMENT_FACTORS_VMAX = {
        'curl' : 0.858,
        'pushup' : 0.858,
        'squat' : 0.75}
    
    SHOW_POSE_OVERLAYS = None

    CURL_COUNTER = 0
    CURL_STAGE = None
    CURL_MAX_ANGLE = 140
    CURL_MIN_ANGLE = 30

    SQUAT_COUNTER = 0
    SQUAT_STAGE = None
    SQUAT_MAX_ANGLE = 160
    SQUAT_MIN_ANGLE = 90
    SQUAT_TORSO_MIN_ANGLE = 170
    
    PUSHUP_COUNTER = 0
    PUSHUP_STAGE = None
    PUSHUP_MAX_ANGLE = 160
    PUSHUP_MIN_ANGLE = 90

    PLANK_START_TIME = None
    PLANK_THRESHOLD_ANGLE = 170