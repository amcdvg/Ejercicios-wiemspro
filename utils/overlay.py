import cv2
import time
from utils.visors import MotionAnalyzer
from constants import Constants as K

def draw_overlays(annotated_frame, progress, smooth_progress, exercise, rep_metrics, frame_width, frame_height, exercise_type, current_time):
    """Dibuja overlays en el frame para visualizar métricas y progreso del ejercicio.

    Esta función agrega elementos gráficos sobre el frame (como barras y recuadros) para 
    mostrar información visual del ejercicio, incluyendo:
      - Una barra de progreso vertical que indica el avance basado en el ángulo actual.
      - Una rueda de progreso que muestra el tiempo transcurrido durante la fase activa (rep).
      - Un recuadro con el número de repeticiones y otro con las métricas calculadas (ROM, VMED, VMAX).

    Para la barra de progreso, si el ejercicio es 'overhead_triceps', se invierte el valor del 
    smooth_progress. Asimismo, si no se ha iniciado la repetición (no existe rep_start_time), se fuerza 
    el progreso a 1 para que la barra muestre un estado predeterminado.

    Para calcular el tiempo transcurrido (elapsed_phase) se usa el atributo rep_start_time; si no existe 
    o la repetición ha finalizado (rep_finished es True), elapsed_phase se establece en 0.

    Args:
        annotated_frame (numpy.ndarray): Imagen del frame en el que se dibujarán los overlays.
        progress (float): Valor de progreso actual (calculado generalmente a partir del ángulo).
        smooth_progress (float): Valor de progreso suavizado para la animación de la barra.
        exercise (object): Instancia del ejercicio, que contiene atributos como latest_angle y rep_start_time.
        rep_metrics (dict): Diccionario con las métricas de la repetición (por ejemplo, 'ROM (cm)', 'VMED (m/s)', 'VMAX (m/s)', 'repetition').
        frame_width (int): Ancho del frame.
        frame_height (int): Altura del frame.
        exercise_type (str): Tipo de ejercicio, usado para obtener el rango de ángulos específico de Constants.
        current_time (float): Tiempo actual (en segundos) utilizado para calcular el elapsed_phase (tiempo de la fase activa).

    Returns:
        numpy.ndarray: El frame anotado con los overlays gráficos (barras, recuadros y textos con métricas).
    """
    # Se importa de nuevo para asegurar consistencia si el archivo se ejecuta de forma independiente.
    from utils.visors import MotionAnalyzer
    from constants import Constants as K

    analyzer = MotionAnalyzer()

    # Obtener el rango de ángulos específico para el ejercicio.
    angle_range = K.EXERCISE_ANGLE_RANGES.get(exercise_type, (40, 180))

    # Calcular el progreso en función de latest_angle. Si no existe latest_angle, se asigna 0.
    if exercise.latest_angle is None:
        progress = 0
    else:
        progress = MotionAnalyzer.normalize_value_bar(exercise.latest_angle, angle_range[0], angle_range[1])

    # Se determina la barra de progreso: para 'overhead_triceps' se invierte el smooth_progress.
    effective_smooth_progress = smooth_progress
    if exercise_type == 'overhead_triceps':
        wheel_progress = progress
        effective_smooth_progress = 1 - smooth_progress
    else:
        wheel_progress = 1 - progress

    # Si no se ha iniciado la repetición, se fuerza el smooth_progress a 1.
    if not hasattr(exercise, 'rep_start_time') or getattr(exercise, 'rep_start_time', None) is None:
        effective_smooth_progress = 1

    # Dibujar la barra de progreso vertical en una posición fija.
    progress_bar_pos = (frame_width - 110, 50)
    fixed_color = (128, 0, 128)  # Color fijo para todos los ejercicios.
    analyzer.draw_progress_bar(annotated_frame, effective_smooth_progress, progress_bar_pos, 20, 300, fixed_color, 10)

    # Calcular el elapsed_phase a partir de rep_start_time (si es válido) y current_time.
    rep_start_time = getattr(exercise, 'rep_start_time', None)
    if rep_start_time is None or not isinstance(rep_start_time, (int, float)) or rep_start_time < 0 or getattr(exercise, 'rep_finished', False):
        elapsed_phase = 0.0
    else:
        elapsed_phase = current_time - rep_start_time

    # Dibujar la rueda de progreso, que utiliza elapsed_phase para mostrar el avance temporal.
    analyzer.draw_progress_wheel(annotated_frame, wheel_progress,
                                 (frame_width - 100, frame_height - frame_height // 3 - 50),
                                 50, fixed_color, 10, elapsed_phase)

    # Si rep_metrics no existe, se crea un diccionario con valores predeterminados.
    if not rep_metrics:
        rep_metrics = {'repetition': 0, 'ROM (m)': 0.000, 'VMED (m/s)': 0.000, 'VMAX (m/s)': 0.000}
    texts = [f"REPS: {rep_metrics['repetition']}"]
    top_left = (20, 40)
    bottom_right = (220, 180)
    analyzer.draw_rounded_rectangle_rep(annotated_frame, top_left, bottom_right, 20, fixed_color, -1)
    analyzer.draw_formatted_text_rep(annotated_frame, texts, (40, 95), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 255), 10, 10)

    # Dibujar un recuadro y los textos de las métricas en la parte inferior del frame.
    recuadro_top_left = (20, frame_height - 80)
    recuadro_bottom_right = (frame_width - 20, frame_height - 20)
    analyzer.draw_rounded_rectangle_rep(annotated_frame, recuadro_top_left, recuadro_bottom_right, 20, fixed_color, -1)
    texts = [
        f"ROM: {rep_metrics.get('ROM (cm)', 0.000):.4f} cm",
        f"VMED: {rep_metrics.get('VMED (m/s)', 0.000):.4f} m/s",
        f"VMAX: {rep_metrics.get('VMAX (m/s)', 0.000):.4f} m/s"
    ]
    text_position = (40, frame_height - 90)
    analyzer.draw_formatted_text_metrics(annotated_frame, texts, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, 10)

    return annotated_frame
