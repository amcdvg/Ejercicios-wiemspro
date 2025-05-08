import cv2
import time
import os
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from loggin_config import setup_logging
from constants import Constants as K

from utils.metrics1 import Metrics
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
from exercises.rear_lunge import RearLungeExercise
from exercises.bench_dips import BenchDipsExercise
from exercises.overhead_triceps import ExtensionTricepsExercise

# Configuración de ejercicios
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
    'RearLugeExercise': RearLungeExercise,
    'BenchDipsExercise': BenchDipsExercise,
    'ExtensionTricepsExercise': ExtensionTricepsExercise
}

# Parámetros del sujeto
subject_height = 1.84
subject_gender = "male"
subject_age = 21
selected_side = "right"

NUM_THREADS = 4  # Hilos paralelos
#BATCH_SIZE =  8  # Tamaño de lote para procesamiento

def process_frame(frame_data):
    """Procesa un frame individual con estimación de pose."""
    smooth_progress = 0.0
    wheel_progress = 0.0
    rep_metrics = {}
    try:
        frame, pose_estimator, relevant_indices, exercise, metrics_obj, frame_width, frame_height, exercise_type, current_time, smooth_progress,wheel_progress,rep_metrics = frame_data
        #annotated_frame = frame.copy()
        # Bloque de estimación de pose con manejo de errores
        try:
            data, annotated_frame = pose_estimator.estimate(frame, relevant_indices)
            
        except Exception as e:
            logger = logging.getLogger('Main')
            #logger.exception("Pose estimation error: %s", e)
            annotated_frame = frame.copy()
            #cv2.putText(annotated_frame, "Error in pose detection", (50, 50),
            #            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            data = None
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
            return data, annotated_frame,current_time, wheel_progress, smooth_progress, exercise, frame_width, frame_height
        else:
            #annotated_frame = frame.copy()
            smooth_progress = 0.0
            return data, annotated_frame,current_time, wheel_progress, smooth_progress, exercise, frame_width, frame_height
        
    except Exception as e:
        logging.exception("Error general procesando frame: %s", e)
        #error_frame = frame.copy()
        #cv2.putText(error_frame, "Processing error", (50, 50),
        #            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        #return error_frame, {}, 0, 0.0, 0.0

def main(input_source):
    """Ejecuta el analizador de ejercicios Virtual Gym."""
    setup_logging()
    logger = logging.getLogger('Main')

    try:
        # Configuración inicial del ejercicio
        exercise_type = exercises[3]
        logger.info("Exercise type: %s", exercise_type)

        class_name_str, relevant_indices = exercise_mapping.get(exercise_type, (None, None))
        if not class_name_str:
            logger.error("Ejercicio no reconocido")
            return

        relevant_indices = [K.YOLO_POSE_KEYPOINTS[tpl.format(side=selected_side.upper())] for tpl in relevant_indices]
        exercise_class = class_mapping.get(class_name_str)
        if not exercise_class:
            logger.error("Clase de ejercicio no encontrada")
            return

        exercise = exercise_class(selected_side, user_height=subject_height)
        pose_estimator = PoseEstimator(model_path='models/yolo11n-pose.pt')

        # Configuración de la fuente de video
        try:
            source = int(input_source)
        except ValueError:
            source = input_source

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logger.error("Error abriendo fuente de video: %s", input_source)
            return

        is_camera = isinstance(source, int)
        #width, height = 640, 360#1280, 720
        out = None
        fps = 30

        if is_camera:
            width1, height1 = 1280, 720
            fps = 30
            BATCH_SIZE =  1
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, K.CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, K.CAMERA_HEIGHT)
            time_sync = TimeSync(is_camera=True)
        else:
            BATCH_SIZE =  8 
            width1, height1 = 640, 360
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_file = os.path.splitext(input_source)[0] + "_processed.mp4"
            out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
            time_sync = TimeSync(is_camera=False, fps=fps)

        metrics_obj = Metrics(exercise_type, subject_height, subject_gender, subject_age)
        metrics_obj.last_counter = 0
        frame_count = 0

        with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            last_counter = 0
            smooth_progress = 0.0
            wheel_progress = 0.0
            rep_metrics = {}
            
            # Variables para calcular los frames por segundo
            frame_count = 0  # Contador de frames
            #start_time1 = time.time()  # Tiempo inicial
            #fps1 = 0
            while True:
                batch = []
                # Leer lote de frames
                for _ in range(BATCH_SIZE):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    batch.append((frame_count, frame))
                    frame_count += 1
                     
                if not batch:
                    break  # Fin del video
                
                ################################################################################################
                # Revisamos la cantidad de Frame por segundo
                # Calcular el tiempo transcurrido desde el inicio
                #elapsed_time1 = time.time() - start_time1

                # Si ha pasado más de un segundo, calcular y mostrar los FPS
                #if elapsed_time1 >= 1.0:
                #    fps1 = frame_count  # Los FPS son iguales al número de frames procesados en un segundo
                #    print(f"Frames procesados en el último segundo: {fps1}")
                #    frame_count = 0  # Reiniciar el contador
                #    start_time1 = time.time()  # Reiniciar el tiempo inicial
                ################################################################################################

                # Preparar datos para procesamiento paralelo
                processing_data = []
                for idx, frame in batch:
                    
                    current_time = time_sync.get_current_time() if is_camera else idx / fps
                    processing_data.append((
                        frame.copy(),
                        pose_estimator,
                        relevant_indices,
                        exercise,
                        metrics_obj,
                        width,
                        height,
                        exercise_type,
                        current_time,
                        smooth_progress,
                        wheel_progress,
                        rep_metrics
                    ))

                # Procesar lote actual
                futures = [executor.submit(process_frame, data) for data in processing_data]
                
                # Procesar resultados manteniendo el orden
                for future, (idx, original_frame) in zip(futures, batch):
                    
                    try:
                        data, annotated_frame,current_time, wheel_progress, smooth_progress, exercise, frame_width, frame_height = future.result()
                        
                        

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
                        # Escribir frame procesado
                        if out is not None:
                            out.write(annotated_frame)
                            
                        # Mostrar frame
                        resized_frame = resize_frame(annotated_frame, width1, height1)
                        cv2.imshow('Virtual GYM', resized_frame)
                        if cv2.waitKey(10) & 0xFF == ord('q'):
                            break

                    except Exception as e:
                        logger.error("Error procesando frame %d: %s", idx, str(e))
                        if out is not None:
                            out.write(original_frame)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

    except Exception as e:
        logger.exception("Error en loop principal: %s", e)
    finally:
        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()
        logger.info("Aplicación finalizada")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analizador de ejercicios Virtual Gym")
    parser.add_argument("--source", type=str, default="0", help="Fuente de video (0 para cámara o ruta de archivo)")
    args = parser.parse_args()
    main(args.source)