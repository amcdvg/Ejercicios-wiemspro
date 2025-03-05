import cv2
import time
from utils.visors import MotionAnalyzer
from constants import Constants as K

def draw_overlays(annotated_frame, smooth_progress, exercise, rep_metrics, frame_width, frame_height, exercise_type):
    """
    Dibuja los overlays en el frame, ajustando dinámicamente los rangos de ángulos
    según el ejercicio seleccionado. El color se mantiene constante.
    
    Args:
        annotated_frame (numpy.ndarray): Imagen donde se dibujarán los overlays.
        smooth_progress (float): Valor suavizado del progreso (0-1) para la barra.
        exercise: Instancia del ejercicio, con atributos como stage, latest_angle, etc.
        rep_metrics (dict): Métricas de la repetición.
        frame_width (int): Ancho del frame.
        frame_height (int): Alto del frame.
        exercise_type (str): Tipo de ejercicio ('curl', 'squat', etc.).
        
    Returns:
        numpy.ndarray: Imagen actualizada con los overlays.
    """
    analyzer = MotionAnalyzer()
    
    # Obtener el rango de ángulos específico para el ejercicio desde las constantes.
    angle_range = K.EXERCISE_ANGLE_RANGES.get(exercise_type, (40, 180))
    
    # Calcular el progreso basado en latest_angle (si es None, usamos 0 para evitar errores)
    if exercise.latest_angle is None:
        progress = 0
    else:
        progress = MotionAnalyzer.normalize_value_bar(exercise.latest_angle, angle_range[0], angle_range[1])
    
    # Para la barra de progreso, forzamos el valor a 0 si la repetición aún no ha iniciado.
    effective_smooth_progress = smooth_progress
    if not hasattr(exercise, 'rep_start_time') or exercise.rep_start_time is None:
        effective_smooth_progress = 1
    # Dibujar la barra de progreso vertical (posición fija)
    progress_bar_pos = (frame_width - 110, 50)
    fixed_color = (128, 0, 128)  # Color fijo para todos los ejercicios
    analyzer.draw_progress_bar(annotated_frame, effective_smooth_progress, progress_bar_pos, 20, 300, fixed_color, 10)
    
    # --- Manejo del rep_start_time para mantener el tiempo acumulado en la repetición actual ---
    # Inicializar rep_start_time y rep_start_count si no existen.
    if not hasattr(exercise, 'rep_start_time'):
        exercise.rep_start_time = None
    if not hasattr(exercise, 'rep_start_count'):
        exercise.rep_start_count = exercise.counter

    # Si el contador de repeticiones ha cambiado (es decir, inicia una nueva repetición),
    # reiniciamos rep_start_time y actualizamos rep_start_count.
    if exercise.counter != exercise.rep_start_count:
        exercise.rep_start_time = None
        exercise.rep_start_count = exercise.counter

     # --- Inicio de rep_start_time usando una comparación simple con el ángulo de partida esperado ---
    threshold = 5
    if exercise.counter == 0:
        threshold = 35
    if exercise.rep_start_time is None and exercise.latest_angle is not None:
        expected_angle = angle_range[1]
        # Sólo iniciamos el contador si la diferencia es mayor a un umbral 
        if abs(exercise.latest_angle - expected_angle) > threshold:
            exercise.rep_start_time = time.time()

    # Calcular el tiempo transcurrido desde el inicio de la repetición actual
    elapsed_phase = time.time() - exercise.rep_start_time if exercise.rep_start_time is not None else 0.0
    
    # Dibujar la rueda de progreso utilizando el elapsed_phase acumulado
    analyzer.draw_progress_wheel(annotated_frame, 1 - effective_smooth_progress,
                                 (frame_width - 100, frame_height - frame_height // 3 - 50),
                                 50, fixed_color, 10, elapsed_phase)
    
    # Recuadro de repeticiones y texto formateado
    if not rep_metrics:
        rep_metrics = {'repetition': 0, 'ROM (m)': 0.000, 'VMED (m/s)': 0.000, 'VMAX (m/s)': 0.000}
    texts = [f"REPS: {rep_metrics['repetition']}"]
    top_left = (20, 40)
    bottom_right = (220, 180)
    analyzer.draw_rounded_rectangle_rep(annotated_frame, top_left, bottom_right, 20, fixed_color, -1)
    analyzer.draw_formatted_text_rep(annotated_frame, texts, (40, 95), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 255), 10, 10)
    
    # Recuadro y textos de métricas en la parte inferior
    recuadro_top_left = (20, frame_height - 80)
    recuadro_bottom_right = (frame_width - 20, frame_height - 20)
    analyzer.draw_rounded_rectangle_rep(annotated_frame, recuadro_top_left, recuadro_bottom_right, 20, fixed_color, -1)
    texts = [
        f"ROM: {rep_metrics.get('ROM (m)', 0.000):.4f} m",
        f"VMED: {rep_metrics.get('VMED (m/s)', 0.000):.4f} m/s",
        f"VMAX: {rep_metrics.get('VMAX (m/s)', 0.000):.4f} m/s"]
    text_position = (40, frame_height - 90)
    analyzer.draw_formatted_text_metrics(annotated_frame, texts, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, 10)
    
    return annotated_frame
