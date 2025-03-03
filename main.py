import cv2
import time
import logging
from loggin_config import setup_logging
from constants import Constants as K

from utils.metrics import Metrics
from pose_estimator import PoseEstimator

from exercises.curl import CurlExercise
from exercises.squat import SquatExercise
from exercises.pushup import PushupExercise
from exercises.plank import PlankExercise


# Diccionario de mapeo (como ya definimos en Constants o localmente)
exercises = ['curl', 'squat', 'pushup', 'plank']
exercise_mapping = K.EXERCISE_MAPPING
class_mapping = {
    'CurlExercise': CurlExercise,
    'SquatExercise': SquatExercise,
    'PushupExercise': PushupExercise,
    'PlankExercise': PlankExercise
}
def main():
    """Orquestador principal del proyecto."""
    setup_logging()
    logger = logging.getLogger('Main')
    # Seleccionar el ejercicio (por ejemplo, 'curl')
    try:
        exercise_type = exercises[0]
        class_name_str, relevant_indices = exercise_mapping.get(exercise_type, (None, None))
        if class_name_str is None:
            logger.error("Exercise not recognized.")
            return
        exercise_class = class_mapping.get(class_name_str, None)
        if exercise_class is None:
            logger.error("Exercise class not found.")
            return

        exercise = exercise_class()
        pose_estimator = PoseEstimator()
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Error opening camera.")
            return

        subject_height = 1.71  # in meters
        subject_gender = "male"
        subject_age = 22
        metrics_obj = Metrics(exercise_type, subject_height, subject_gender, subject_age)
        last_counter = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Frame not read correctly.")
                break
            try:
                data, annotated_frame = pose_estimator.estimate(frame, relevant_indices)
            except Exception as e:
                logger.exception("Pose estimation error: %s", e)
                annotated_frame = frame.copy()
                cv2.putText(annotated_frame, "Error in pose detection", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                data = None
            logger.debug("Procesando frame inicial...")
            if data is not None:
                try:
                    keypoints, confs = data
                    logger.debug("Keypoints detected: %s", len(keypoints))
                    # Usar process() si se implementa en la clase base; de lo contrario, update y draw
                    exercise.process(keypoints, confs, annotated_frame)
                    if exercise.latest_angle is not None:
                        metrics_obj.update(exercise.latest_angle, time.time())
                    if exercise.counter > last_counter:
                        rep_metrics = metrics_obj.get_metrics(exercise.counter)
                        logger.info("Repetition completed: %s", rep_metrics)
                        last_counter = exercise.counter
                        metrics_obj.reset()
                except Exception as e:
                    logger.exception("Error updating exercise or metrics: %s", e)
            else:
                cv2.putText(annotated_frame, "Incomplete pose detection", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            try:
                # Si no se usa process(), se llama a draw() aquí
                exercise.draw(annotated_frame)
            except Exception as e:
                logger.exception("Error drawing overlay: %s", e)
            cv2.imshow('Virtual GYM', annotated_frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    except Exception as e:
        logger.exception("Main loop error: %s", e)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Application terminated.")

if __name__ == '__main__':
    main()