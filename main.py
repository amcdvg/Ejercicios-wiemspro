import cv2
import time
import logging
from loggin_config import setup_logging
from constants import Constants as K

from utils.metrics import Metrics
from utils.visors import MotionAnalyzer
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
        smooth_progress = 0.0
        min_angle = None
        max_angle = None
        initialized_angles = False 
        rep_metrics = {}
        while True:
            ret, frame = cap.read()
            frame_height, frame_width, _ = frame.shape
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
                        if not initialized_angles:
                            min_angle = exercise.latest_angle
                            max_angle = exercise.latest_angle
                            initialized_angles = True
                        else:
                            # Actualizar ángulos solo si tenemos valores válidos
                            if exercise.latest_angle < min_angle:
                                min_angle = exercise.latest_angle
                            if exercise.latest_angle > max_angle:
                                max_angle = exercise.latest_angle
                            # Actulización de los valores de progreso
                            print(min_angle)
                            print(max_angle)
                            progress = MotionAnalyzer.normalize_value_bar(exercise.latest_angle, 80.8, 180)
                            smooth_progress = 0.1 * progress + (1 - 0.1) * smooth_progress #
                    else:
                        
                        smooth_progress = 0.0
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
                #exercise.draw(annotated_frame)
                # Barra de progreso vertical
                analyzer = MotionAnalyzer()
                
                analyzer.draw_progress_bar(annotated_frame, smooth_progress, (frame_width - 110, 50), 20, 300, (128, 0, 128), 10)
                # Rueda de progreso temporal
                #center = (frame_width - 100, frame_height - frame_height // 3 - 50)
                #color_wheel = (0, 255, 0) if phase_type == 'concentric' else (0, 0, 255)
                #MotionAnalyzer.draw_progress_wheel(
                #    annotated_frame,
                #    1 - smooth_progress,  # Progreso invertido
                #    center,
                #    50,   # Radio
                #    color_wheel,
                #    10,   # Grosor
                #    elapsed_time_phase
                #)
                # Crear el texto de repeticiones
                if rep_metrics == {}:
                    rep_metrics['repetition'] = 0
                    rep_metrics['ROM (m)']=0.000
                    rep_metrics['VMED (m/s)']=0.000
                    rep_metrics['VMAX (m/s)']=0.000
                texts = [
                    f"REPS: {rep_metrics['repetition']}"
                ]
                # Ajustar el tamaño del recuadro con bordes redondeados
                top_left = (20, 40)  # Mantén el punto superior izquierdo
                bottom_right = (220, 180)  # Reduce el tamaño del rectángulo
                MotionAnalyzer.draw_rounded_rectangle_rep(annotated_frame, top_left, bottom_right, 20, (128, 0, 128), -1)
                # Dibujar texto formateado en la parte superior izquierda
                MotionAnalyzer.draw_formatted_text_rep(
                    image=annotated_frame,
                    texts=texts,
                    position=(40, 95),
                    font=cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale=3.0,
                    color=(255, 255, 255),
                    thickness=10,
                    line_spacing=10
                )

                # Ajustar el tamaño y la posición del recuadro en la parte inferior
                frame_height, frame_width, _ = frame.shape
                recuadro_top_left = (20, frame_height - 80)
                recuadro_bottom_right = (frame_width - 20, frame_height - 20)
                # Dibujar el recuadro en la parte inferior
                MotionAnalyzer.draw_rounded_rectangle_rep(annotated_frame, recuadro_top_left, recuadro_bottom_right, 20, (128, 0, 128), -1)

                # Crear el texto con las métricas solicitadas
                texts = [
                    f"ROM: {rep_metrics['ROM (m)']:.4f} m",
                    f"VMED: {rep_metrics['VMED (m/s)']:.4f} m/s",
                    f"VMAX: {rep_metrics['VMAX (m/s)']:.4f} m/s"
                ]

                # Posición inicial del texto dentro del recuadro
                text_position = (40, frame_height - 90)

                # Dibujar el texto formateado
                MotionAnalyzer.draw_formatted_text_metrics(annotated_frame, texts, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, 10)

                pass
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