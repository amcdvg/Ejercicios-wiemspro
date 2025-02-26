# main.py
import cv2
from constants import Constants as K
from pose_estimator import PoseEstimator

from exercises.curl import CurlExercise
from exercises.squat import SquatExercise
from exercises.pushup import PushupExercise
from exercises.plank import PlankExercise

from utils.metrics import Metrics
import time

exercises = ['curl', 'squat', 'pushup', 'plank']
class_mapping = {
            'CurlExercise': CurlExercise,
            'SquatExercise': SquatExercise,
            'PushupExercise': PushupExercise,
            'PlankExercise': PlankExercise}

def main():
    """orquestador de todo
    """
    exercise_type = exercises[1]
    class_name_str, relevant_indices = K.EXERCISE_MAPPING.get(exercise_type, (None, None))

    if class_name_str is None:
        print("Ejercicio no reconocido")
        return
    exercise_class = class_mapping.get(class_name_str, None)
    if exercise_class is None:
        print("La clase para el ejercicio no fue encontrada")
        return
    # Instanciar el ejercicio
    exercise = exercise_class()
    # instanciamos pose_estimator
    pose_estimator = PoseEstimator()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error al abrir la cámara")
        return
    subject_height = 1.71  # en metros
    subject_gender = "male"  # "male" o "female"
    subject_age = 22  # años
    # Instanciar la clase Metrics para el ejercicio actual
    metrics_obj = Metrics(exercise_type, subject_height, subject_gender, subject_age)
    last_counter = 0 
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        data, annotated_frame = pose_estimator.estimate(frame, relevant_indices)
        if data is not None:
            keypoints, confs = data
            angle_value = exercise.update(keypoints, confs)

            if angle_value is not None:
                current_time = time.time()
                metrics_obj.update(angle_value, current_time)

            if exercise.counter > last_counter:
                rep_metrics = metrics_obj.get_metrics()
                print("Repetición completada:", rep_metrics)
                last_counter = exercise.counter
                # Reiniciar las métricas para la próxima repetición
                metrics_obj.reset()

        exercise.draw(annotated_frame)
        cv2.imshow('Virtual GYM', annotated_frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
