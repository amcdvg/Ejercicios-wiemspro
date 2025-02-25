import cv2
import mediapipe as mp
import numpy as np

# Inicializar MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Variables para el contador de repeticiones
counter = 0
stage = None
min_angle = None
max_angle = None

# Variable para almacenar el valor suavizado de progreso
smooth_progress = 0
smooth_factor = 0.1  # Controla la cantidad de suavizado (0 es sin suavizado, 1 es muy suave)

# Función para calcular el ángulo entre tres puntos
def calculate_angle(a, b, c):
    a = np.array(a)  # Primer punto
    b = np.array(b)  # Segundo punto (vértice del ángulo)
    c = np.array(c)  # Tercer punto
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians) * 180.0 / np.pi
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Función para normalizar un valor en un rango dado
def normalize_value(value, min_value, max_value):
    if min_value is None or max_value is None:
        return 0  # Si no tenemos un rango válido, devolver 0 como valor de progreso por defecto
    if max_value == min_value:  # Para evitar división por cero
        return 0
    return (value - min_value) / (max_value - min_value)  # Normalizar en el rango [0, 1]

# Función para dibujar la barra de progreso vertical
def draw_rounded_rect(image, position, bar_width, bar_height, color, border_radius):
    x, y = position

    # Crear una copia de la imagen para dibujar el rectángulo
    overlay = image.copy()
    
    # Dibuja las esquinas redondeadas usando elipses
    cv2.ellipse(overlay, (x + border_radius, y + border_radius), (border_radius, border_radius), 180, 0, 90, color, -1)
    cv2.ellipse(overlay, (x + bar_width - border_radius, y + border_radius), (border_radius, border_radius), 270, 0, 90, color, -1)
    cv2.ellipse(overlay, (x + border_radius, y + bar_height - border_radius), (border_radius, border_radius), 90, 0, 90, color, -1)
    cv2.ellipse(overlay, (x + bar_width - border_radius, y + bar_height - border_radius), (border_radius, border_radius), 0, 0, 90, color, -1)
    
    # Dibuja los rectángulos centrales
    cv2.rectangle(overlay, (x + border_radius, y), (x + bar_width - border_radius, y + bar_height), color, -1)
    cv2.rectangle(overlay, (x, y + border_radius), (x + bar_width, y + bar_height - border_radius), color, -1)
    
    # Mezclar con la imagen original
    alpha = 0.4  # Transparencia
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

# Función para dibujar la barra de progreso
def draw_progress_bar(image, progress, position, bar_width, bar_height, color, border_radius):
    x, y = position
    progress = 1 - progress  # Invertir el progreso
    progress_height = int(bar_height * progress)  # Ajusta el progreso según el valor
    
    # Fondo (color que no cambia) - se dibuja desde abajo hacia arriba
    draw_rounded_rect(image, (x, y), bar_width, bar_height, (128, 0, 128), border_radius)
    
    # Barra de progreso invertida - se dibuja desde abajo hacia arriba
    draw_rounded_rect(image, (x, y + bar_height - progress_height), bar_width, progress_height, color, border_radius)
    
    # Mostrar el porcentaje dentro de un pequeño recuadro
    percentage_text = f"{int(progress * 100) + 1}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color_text = (255, 255, 255)  # Blanco
    thickness = 2  # Grosor aumentado para simular negrita
    text_size = cv2.getTextSize(percentage_text, font, font_scale, thickness)[0]
    
    # Posicionar el recuadro para el porcentaje
    box_width = text_size[0] + 10  # Un poco más grande que el texto
    box_height = text_size[1] + 10
    box_x = x + (bar_width - box_width) // 2
    box_y = int(y + bar_height - (progress * bar_height) - box_height - 5)  # Mover el recuadro hacia abajo del progreso
   
    # Dibujar el recuadro del porcentaje con bordes redondeados
    if (box_x + box_width <= image.shape[1]) and (box_y + box_height <= image.shape[0]):
        draw_rounded_rect_T(image, (box_x, box_y), box_width, box_height, (75, 0, 130), border_radius=5)
        cv2.putText(image, percentage_text, (box_x + 5, box_y + box_height - 5), font, font_scale, color_text, thickness)

def draw_rounded_rect_T(image, top_left, width, height, color, border_radius):
    """Dibuja un rectángulo con bordes redondeados."""
    x, y = top_left
    rect = (x, y, x + width, y + height)
    radius = border_radius

    # Dibuja el fondo redondeado
    cv2.rectangle(image, (x + radius, y), (x + width - radius, y + height), color, -1)
    cv2.rectangle(image, (x, y + radius), (x + width, y + height - radius), color, -1)
    cv2.circle(image, (x + radius, y + radius), radius, color, -1)
    cv2.circle(image, (x + width - radius, y + radius), radius, color, -1)
    cv2.circle(image, (x + radius, y + height - radius), radius, color, -1)
    cv2.circle(image, (x + width - radius, y + height - radius), radius, color, -1)
    

# Abrir la cámara
cap = cv2.VideoCapture(0)

# Configuración de MediaPipe Pose
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    prev_angle = None  # Para verificar la dirección del movimiento del ángulo
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

        # Extraer landmarks (puntos de referencia)
        try:
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

            # Actualizar el ángulo mínimo y máximo si es la primera vez que se detectan
            if min_angle is None or angle < min_angle:
                min_angle = angle
            if max_angle is None or angle > max_angle:
                max_angle = angle

            # Normalizar el ángulo para obtener el progreso (valor entre 0 y 1)
            progress = 0
            if min_angle is not None and max_angle is not None:
                progress = normalize_value(angle, min_angle, max_angle)
                progress = max(0, min(1, progress))  # Asegurarse de que el progreso esté entre 0 y 1

                # Aplicar suavizado al progreso
                smooth_progress = smooth_factor * progress + (1 - smooth_factor) * smooth_progress

                # Dibujar la barra de progreso
                # Dibujar la barra de progreso vertical
                frame_height, frame_width, _ = frame.shape
                draw_progress_bar(image_bgr, smooth_progress, (frame_width - 50, 50), 20, 300, (128, 0, 128), 10)

        except:
            pass

        # Mostrar el resultado
        cv2.imshow("Pose Estimation", image_bgr)

        # Salir si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()