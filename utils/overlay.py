import cv2
import time
from utils.visors import MotionAnalyzer
from constants import Constants as K

def draw_overlays(annotated_frame, progress, smooth_progress, exercise, rep_metrics, frame_width, frame_height, exercise_type, current_time):
    """
    Dibuja los overlays en el frame y ajusta el tiempo de la repetición mostrado en el wheel.
    Si el ejercicio está en cooldown (rep_finished == True), el elapsed_phase se establece en 0 para
    no mostrar tiempo muerto.
    """
    from utils.visors import MotionAnalyzer
    from constants import Constants as K

    analyzer = MotionAnalyzer()
    
    # Obtener el rango de ángulos específico para el ejercicio
    angle_range = K.EXERCISE_ANGLE_RANGES.get(exercise_type, (40, 180))
    
    # Calcular el progreso basado en latest_angle (si es None, usamos 0 para evitar errores)
    if exercise.latest_angle is None:
        progress = 0
    else:
        progress = MotionAnalyzer.normalize_value_bar(exercise.latest_angle, angle_range[0], angle_range[1])
    
    # Para la barra de progreso, si la repetición no ha iniciado, forzamos el valor a 1
    effective_smooth_progress = smooth_progress
    if exercise_type == 'overhead_triceps':
        wheel_progress = progress
        effective_smooth_progress = 1- smooth_progress
    else:
        wheel_progress = 1 - progress
    if not hasattr(exercise, 'rep_start_time') or getattr(exercise, 'rep_start_time', None) is None:
        effective_smooth_progress = 1

    

    # Dibujar la barra de progreso vertical (posición fija)
    progress_bar_pos = (frame_width - 110, 50)
    fixed_color = (128, 0, 128)  # Color fijo para todos los ejercicios
    analyzer.draw_progress_bar(annotated_frame, effective_smooth_progress, progress_bar_pos, 20, 300, fixed_color, 10)
    
    # Para calcular elapsed_phase, usamos getattr() para evitar error si no existe rep_start_time
    rep_start_time = getattr(exercise, 'rep_start_time', None)
    if rep_start_time is None or not isinstance(rep_start_time, (int, float)) or rep_start_time < 0 or getattr(exercise, 'rep_finished', False):
        elapsed_phase = 0.0
    else:
        elapsed_phase = current_time - rep_start_time

    # Dibujar la rueda de progreso usando elapsed_phase
    #wheel_progress = progress  # Esto puede ajustarse si se desea otra fórmula
    analyzer.draw_progress_wheel(annotated_frame, wheel_progress,
                                 (frame_width - 100, frame_height - frame_height // 3 - 50),
                                 50, fixed_color, 10, elapsed_phase)
    
    # Recuadro de repeticiones y métricas
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
        f"ROM: {rep_metrics.get('ROM (cm)', 0.000):.4f} cm",
        f"VMED: {rep_metrics.get('VMED (m/s)', 0.000):.4f} m/s",
        f"VMAX: {rep_metrics.get('VMAX (m/s)', 0.000):.4f} m/s"]
    text_position = (40, frame_height - 90)
    analyzer.draw_formatted_text_metrics(annotated_frame, texts, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, 10)
    
    return annotated_frame
