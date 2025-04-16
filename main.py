import cv2
import time
import os
import argparse
import logging
from loggin_config import setup_logging
from constants import Constants as K

from utils.metrics import Metrics
from utils.visors import MotionAnalyzer
from utils.overlay import draw_overlays
from utils.csv_exporter import export_metrics
from utils._utils import resize_frame
from utils.time_sync import TimeSync
from pose_estimator import PoseEstimator

from exercises.curl import CurlExercise
from exercises.squat import SquatExercise
from exercises.pushup import PushupExercise
from exercises.deadlift import DeadliftExercise
from exercises.reverse_fly import ReverseFlyExercise
from exercises.swing import SwingExercise
from exercises.renegade_row import RenegadeRowExercise
from exercises.rear_lunge import RearLugeExercise
from exercises.bench_dips import BenchDipsExercise
from exercises.overhead_triceps import ExtensionTricepsExercise

# Lista de ejercicios disponibles y mapeo de ejercicio a sus clases
exercises = ['squat', 'curl', 'pushup', 'deadlift', 'reverse_fly', 'overhead_triceps', 'bench_dips', 'rear_lunge', 'renegade_row', 'swing']
exercise_mapping = K.EXERCISE_MAPPING

class_mapping = {
    'CurlExercise': CurlExercise,
    'SquatExercise': SquatExercise,
    'PushupExercise': PushupExercise,
    'DeadliftExercise': DeadliftExercise,
    'ReverseFlyExercise': ReverseFlyExercise,
    'SwingExercise': SwingExercise,
    'RenegadeRowExercise': RenegadeRowExercise,
    'RearLugeExercise': RearLugeExercise,
    'BenchDipsExercise': BenchDipsExercise,
    'ExtensionTricepsExercise': ExtensionTricepsExercise
}

# Configuración de parámetros del sujeto
subject_height = 1.84  # Altura en metros
subject_gender = "male"
subject_age = 21
# Selección del lado a analizar (puede ser 'left' o 'right')
selected_side = "right"


def main(input_source):
    """Ejecuta el analizador de ejercicios Virtual Gym.

    Este método configura el logger, instancia la clase de ejercicio correspondiente a partir
    de la cadena de ejercicios y su mapeo, inicializa el estimador de poses y la sincronización 
    del tiempo, y procesa cada frame del video o la cámara. Durante el procesamiento se actualizan
    las métricas mediante el objeto Metrics y se exportan los resultados a un archivo CSV. Además,
    se dibujan overlays sobre el video para visualizar el progreso del ejercicio, la repetición actual 
    y otros datos de interés.

    Args:
        input_source (str): Fuente de video. Puede ser un número (como '0') que indica la cámara o 
            la ruta a un archivo de video.

    Returns:
        None
    """
    setup_logging()
    logger = logging.getLogger('Main')

    try:
        # Seleccionar el ejercicio según el índice (en este caso, 'deadlift')
        exercise_type = exercises[3]
        logger.info("Exercise type: %s", exercise_type)

        class_name_str, relevant_indices = exercise_mapping.get(exercise_type, (None, None))
        if class_name_str is None:
            logger.error("Exercise not recognized.")
            return

        # Formatear los índices relevantes de keypoints en función del lado seleccionado
        relevant_indices = [K.YOLO_POSE_KEYPOINTS[tpl.format(side=selected_side.upper())] for tpl in relevant_indices]

        exercise_class = class_mapping.get(class_name_str, None)
        if exercise_class is None:
            logger.error("Exercise class not found.")
            return

        # Crear instancia del ejercicio (en este caso, DeadliftExercise)
        exercise = exercise_class(selected_side, user_height=subject_height)
        # Inicializar el estimador de pose con el modelo especificado.
        pose_estimator = PoseEstimator(model_path='models/yolo11n-pose.pt')

        try:
            # Intentar convertir la fuente de entrada a entero (para cámara) o dejarla como ruta.
            source = int(input_source)
        except ValueError:
            source = input_source

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logger.error("Error opening video source: %s", input_source)
            return

        is_camera = isinstance(source, int)
        if is_camera:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, K.CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, K.CAMERA_HEIGHT)
            time_sync = TimeSync(is_camera=True)
        else:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps < 1:
                fps = 30
            time_sync = TimeSync(is_camera=False, fps=fps)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            base, ext = os.path.splitext(input_source)
            output_file = base + "_processed" + ext
            out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
            logger.info("Output video will be saved to: %s", output_file)
        if is_camera:
            out = None

        # Instancia del objeto Metrics para calcular las métricas según el ejercicio.
        metrics_obj = Metrics(exercise_type, subject_height, subject_gender, subject_age)

        last_counter = 0
        smooth_progress = 0.0
        wheel_progress = 0.0
        rep_metrics = {}
        initialized_angles = False

        # Ciclo principal: Procesamiento de cada frame
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.info("video ended or read frame error")
                break

            frame_height, frame_width, _ = frame.shape
            # Obtener el tiempo relativo (sincronizado) del frame actual
            current_time = time_sync.get_current_time()

            try:
                # Estimar pose y obtener el frame anotado
                data, annotated_frame = pose_estimator.estimate(frame, relevant_indices)
            except Exception as e:
                logger.exception("Pose estimation error: %s", e)
                annotated_frame = frame.copy()
                cv2.putText(annotated_frame, "Error in pose detection", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            if data is not None:
                keypoints, confs = data
                # Se procesa la pose actualizada, propagando el current_time al método process().
                exercise.process(keypoints, confs, frame=annotated_frame, current_time=current_time)
                # Actualizar la escala (pixel_scale) en el objeto Metrics si se ha calculado en el ejercicio.
                if isinstance(exercise, DeadliftExercise) and exercise.get_pixel_scale() is not None:
                    if not hasattr(metrics_obj, 'pixel_scale'):
                        metrics_obj.pixel_scale = exercise.get_pixel_scale()  # Transferir la escala
                # Actualizar las métricas sólo si la repetición está activa (fase "down")
                if exercise.latest_angle is not None and hasattr(exercise, 'rep_start_time') and exercise.rep_start_time is not None:
                    metrics_obj.update(exercise.latest_angle, current_time)
                # Calcular el progreso basado en el ángulo del ejercicio
                angle_range = K.EXERCISE_ANGLE_RANGES.get(exercise_type, (40, 180))
                progress = MotionAnalyzer.normalize_value_bar(exercise.latest_angle, angle_range[0], angle_range[1])
                wheel_progress = 0.18 * progress + (1 - 0.18) * wheel_progress
                smooth_progress = 0.115 * progress + (1 - 0.115) * smooth_progress
            else:
                smooth_progress = 0.0

            # Verificar si se completó una repetición
            if exercise.counter > last_counter:
                rep_metrics = metrics_obj.get_metrics(exercise.counter)
                logger.info("Repetition completed: %s", rep_metrics)
                rep_metrics["exercise"] = exercise_type
                export_metrics(rep_metrics)
                last_counter = exercise.counter
                metrics_obj.reset()  # Reiniciar la recolección de métricas para la siguiente repetición

            try:
                # Dibujar overlays sobre el frame anotado
                annotated_frame = draw_overlays(annotated_frame, wheel_progress, smooth_progress, exercise,
                                                rep_metrics, frame_width, frame_height, exercise_type, current_time)
            except Exception as e:
                logger.exception("Error drawing overlay: %s", e)

            if out is not None:
                out.write(annotated_frame)

            # Redimensionar el frame para mostrarlo correctamente
            annotated_frame = resize_frame(annotated_frame, 1280, 720)
            cv2.imshow('Virtual GYM', annotated_frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    except Exception as e:
        logger.exception("Main loop error: %s", e)
    finally:
        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()
        logger.info("Application terminated.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Virtual Gym Exercise Analyzer")
    parser.add_argument("--source", type=str, default="0",
                        help="Fuente de video: un número (por ejemplo, '0') para cámara o una ruta para un archivo de video.")
    args = parser.parse_args()
    main(args.source)
