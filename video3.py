import cv2
import mediapipe as mp
import numpy as np
import time

# Inicializar MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Variables
min_angle = None
max_angle = None
start_time = None
elapsed_time = 0
smooth_progress = 0
smooth_factor = 0.1
stage = None  # Etapa del movimiento: "subida" o "bajada"

# Función para calcular el ángulo entre tres puntos
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians) * 180.0 / np.pi
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Dibujar una rueda de progreso circular
def draw_progress_wheel(image, progress, center, radius, color, thickness, elapsed_time):
    angle_start = -90  # Inicio en la parte superior (en grados)
    angle_end = int(angle_start + 360 * progress)  # Ángulo según el progreso
    # Dibujar la elipse en una capa separada
    overlay = image.copy()
    cv2.ellipse(overlay, center, (radius, radius), 0, 0, 360, color, thickness)

    # Combinar las capas con transparencia
    cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)

    cv2.ellipse(image, center, (radius, radius), 0, angle_start, angle_end, color, thickness)

    # Dibujar el tiempo transcurrido en el centro
    time_text = f"{elapsed_time:.1f}s"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    text_size = cv2.getTextSize(time_text, font, font_scale, 2)[0]
    text_x = center[0] - text_size[0] // 2
    text_y = center[1] + text_size[1] // 2
    cv2.putText(image, time_text, (text_x, text_y), font, font_scale, color, 2)

# Abrir la cámara
cap = cv2.VideoCapture(0)

# Configuración de MediaPipe Pose
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    prev_stage = None
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Recolorear la imagen a RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        # Procesar la imagen con MediaPipe
        results = pose.process(image_rgb)

        # Recolorear de vuelta a BGR
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        try:
            # Extraer landmarks
            landmarks = results.pose_landmarks.landmark

            # Obtener coordenadas de hombro, codo y muñeca
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * frame.shape[1], 
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * frame.shape[0]]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * frame.shape[1], 
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * frame.shape[0]]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * frame.shape[1], 
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * frame.shape[0]]

            # Calcular el ángulo entre el hombro, codo y muñeca
            angle = calculate_angle(shoulder, elbow, wrist)

            # Actualizar ángulos mínimos y máximos
            if min_angle is None or angle < min_angle:
                min_angle = angle
            if max_angle is None or angle > max_angle:
                max_angle = angle

            # Calcular progreso invertido
            progress = 1 - ((angle - min_angle) / (max_angle - min_angle)) if max_angle > min_angle else 0
            smooth_progress = smooth_factor * progress + (1 - smooth_factor) * smooth_progress

            # Detectar la etapa del movimiento
            if angle > 160:  # Brazo completamente extendido
                stage = "bajada"
            elif angle < 160:  # Brazo completamente doblado
                stage = "subida"

            # Reiniciar el tiempo al cambiar de etapa
            if stage != prev_stage:
                if stage == "subida":  # Iniciar temporizador en "subida"
                    start_time = time.time()
                elif stage == "bajada" :  # Reiniciar temporizador en "bajada"
                    elapsed_time = 0
                    start_time = None

            # Calcular tiempo transcurrido en la etapa "subida"
            if stage == "subida" and start_time is not None:
                elapsed_time = time.time() - start_time

            prev_stage = stage
            
            # Dibujar rueda de progreso
           
            center = (frame.shape[1] - 100, frame.shape[0] - frame.shape[0] // 3)
            radius = 50
            thickness = 10
            color = (128, 0, 128) if stage == "subida" else (128, 0, 128)  # Verde para "subida", rojo para "bajada"
            draw_progress_wheel(image_bgr, smooth_progress, center, radius, color, thickness, elapsed_time)

        except Exception as e:
            print(e)

        # Mostrar la imagen procesada
        cv2.imshow("Pose Estimation with Progress Wheel", image_bgr)

        # Salir si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()