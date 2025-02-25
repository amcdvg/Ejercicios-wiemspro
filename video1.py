import cv2
import mediapipe as mp
import numpy as np
import time

# Inicializar MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Variables para el contador de repeticiones
counter = 0
stage = None
rom = 0  # Rango de movimiento normalizado
velocities = []  # Lista de velocidades para calcular VMED y VMAX
prev_angle = None
prev_time = None
min_angle = None
max_angle = None
vmed = 0  # Velocidad media normalizada
vmax = 0  # Velocidad máxima normalizada
last_vmed = 0  # Última velocidad media mostrada
last_vmax = 0  # Última velocidad máxima mostrada
arm_length_meters = None  # Longitud estimada del brazo (en metros)

# Factor de escala: píxeles a metros (modificar según el caso)
pixels_to_meters = 0.002  # Suponiendo que 1 píxel = 0.002 metros
height = 1.74
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

# Función para calcular la distancia entre dos puntos
def calculate_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    distance = np.linalg.norm(a - b)
    return distance

# Función para normalizar un valor en un rango dado
def normalize_value(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value) * 2  # Normalizar en el rango [0, 2]

# Función para dibujar un recuadro con bordes redondeados
def draw_rounded_rectangle(image, top_left, bottom_right, radius, color, thickness):
    x1, y1 = top_left
    x2, y2 = bottom_right

    # Dibujar círculos en las esquinas
    cv2.circle(image, (x1 + radius, y1 + radius), radius, color, -1)
    cv2.circle(image, (x2 - radius, y1 + radius), radius, color, -1)
    cv2.circle(image, (x1 + radius, y2 - radius), radius, color, -1)
    cv2.circle(image, (x2 - radius, y2 - radius), radius, color, -1)

    # Dibujar rectángulos entre los círculos
    cv2.rectangle(image, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(image, (x1, y1 + radius), (x2, y2 - radius), color, -1)

# Función para dibujar texto con formato
def draw_formatted_text(image, texts, position, font, font_scale, color, thickness, line_spacing):
    x, y = position
    for text in texts:
        before_colon, after_colon = text.split(":")
        before_colon = before_colon.strip()
        after_colon = after_colon.strip()

        # Dibujar la parte antes de ":"
        italic_font = font | cv2.FONT_ITALIC
        cv2.putText(image, before_colon + ":", (x, y), italic_font, font_scale, color, thickness, cv2.LINE_AA)

        # Calcular el ancho del texto antes de ":"
        text_size, _ = cv2.getTextSize(before_colon + ":", italic_font, font_scale, thickness)
        text_width = text_size[0]

        # Dibujar la parte después de ":"
        bold_font = font
        cv2.putText(image, after_colon, (x + text_width + 5, y), bold_font, font_scale, color, thickness, cv2.LINE_AA)

        # Ajustar la posición vertical para la siguiente línea
        y += text_size[1] + line_spacing
import cv2

def draw_formatted_text_row(image, texts, position, font, font_scale, color, thickness, spacing):
    """
    Dibuja los textos formateados en una fila distribuidos uniformemente en la imagen.
    Cada texto tiene la parte antes de ":" en cursiva y la parte después de ":" en negrita.
    
    :param image: Imagen sobre la que dibujar.
    :param texts: Lista de textos formateados como "clave: valor".
    :param position: Coordenadas iniciales (x, y) para comenzar a dibujar.
    :param font: Fuente utilizada para el texto.
    :param font_scale: Escala de la fuente.
    :param color: Color del texto en BGR.
    :param thickness: Grosor del texto.
    :param spacing: Espaciado adicional entre textos en la fila.
    """
    x_start, y = position
    image_width = image.shape[1]  # Ancho de la imagen

    # Calcular el espacio total requerido para los textos y el espaciado entre ellos
    total_text_width = 0
    text_sizes = []

    for text in texts:
        before_colon, after_colon = text.split(":")
        before_colon = before_colon.strip()
        after_colon = after_colon.strip()

        # Obtener tamaños de texto
        italic_font = font | cv2.FONT_ITALIC
        bold_font = font

        before_colon_size, _ = cv2.getTextSize(before_colon + ":", italic_font, font_scale, thickness)
        after_colon_size, _ = cv2.getTextSize(after_colon, bold_font, font_scale, thickness)

        total_width = before_colon_size[0] + after_colon_size[0] + spacing
        text_sizes.append((before_colon_size, after_colon_size, total_width))
        total_text_width += total_width

    # Calcular el espacio restante para distribuir uniformemente
    remaining_space = image_width - total_text_width
    if remaining_space < 0:
        print("Advertencia: El texto no cabe en la imagen.")
        remaining_space = 0

    gap = remaining_space // (len(texts) + 1)

    # Dibujar los textos
    x = x_start + gap
    for i, text in enumerate(texts):
        before_colon, after_colon = text.split(":")
        before_colon = before_colon.strip()
        after_colon = after_colon.strip()

        # Dibujar la parte antes de ":"
        italic_font = font | cv2.FONT_ITALIC
        cv2.putText(image, before_colon + ":", (x, y+50), italic_font, font_scale, color, thickness, cv2.LINE_AA)

        # Calcular el ancho del texto antes de ":"
        before_colon_width = text_sizes[i][0][0]

        # Dibujar la parte después de ":"
        bold_font = font
        cv2.putText(image, after_colon, (x + before_colon_width + 5, y+50), bold_font, font_scale, color, thickness, cv2.LINE_AA)

        # Ajustar la posición horizontal para el siguiente texto
        x += text_sizes[i][2] + gap

def draw_formatted_text1(image, texts, position, font, font_scale, color, thickness, line_spacing):
    x, y = position
    for text in texts:
        before_colon, after_colon = text.split(":")
        before_colon = before_colon.strip()
        after_colon = after_colon.strip()

        # Dibujar la parte antes de ":"
        italic_font = font | cv2.FONT_ITALIC
        #cv2.putText(image, before_colon + ":", (x, y), italic_font, font_scale, color, thickness, cv2.LINE_AA)

        # Calcular el ancho del texto antes de ":"
        text_size, _ = cv2.getTextSize(before_colon + ":", italic_font, font_scale, thickness)
        text_width = text_size[0]

        # Dibujar la parte después de ":"
        bold_font = font
        n = int(after_colon)
        
        if n < 10:
            cv2.putText(image, after_colon, (x +50, y+45), bold_font, font_scale, color, thickness, cv2.LINE_AA)
        else:
            cv2.putText(image, after_colon, (x +20, y+45), bold_font, font_scale, color, thickness, cv2.LINE_AA)

        # Ajustar la posición vertical para la siguiente línea
        y += text_size[1] + line_spacing
# Abrir la cámara
cap = cv2.VideoCapture(0)

# Configuración de MediaPipe Pose
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
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

            # Calcular la longitud del brazo (en píxeles) si no está definida
            if arm_length_meters is None:
                arm_length_pixels = calculate_distance(shoulder, wrist)
                arm_length_meters = height*0.475#arm_length_pixels * pixels_to_meters

            # Calcular el ángulo
            angle = calculate_angle(shoulder, elbow, wrist)

            # Actualizar ángulos mínimos y máximos
            if min_angle is None or angle < min_angle:
                min_angle = angle
            if max_angle is None or angle > max_angle:
                max_angle = angle

            # Calcular velocidad (en metros/segundo)
            current_time = time.time()
            if prev_angle is not None and prev_time is not None:
                delta_time = current_time - prev_time

                # Controlar el tiempo mínimo entre cuadros
                if delta_time > 0.01:  # Ignorar intervalos menores a 10ms
                    velocity = ((abs(angle - prev_angle) * np.pi / 270) * (arm_length_meters- (height * 0.175))) / delta_time

                    # Filtrar velocidades anómalas
                    if velocity < 300:  # Umbral razonable para la velocidad máxima
                        velocities.append(velocity)

            prev_angle = angle
            prev_time = current_time

            # Lógica de contador de repeticiones
            if angle > 150:
                stage = "down"
            if angle < 9 and stage == 'down':
                stage = "up"
                counter += 1

                # Calcular ROM como diferencia entre ángulo máximo y mínimo (normalizado)
                if max_angle is not None and min_angle is not None:
                    print(arm_length_meters)
                    raw_rom = (max_angle - min_angle) * np.pi / 270 * arm_length_meters
                    rom = raw_rom - arm_length_meters - (height * 0.175)#normalize_value(raw_rom, 0, 1)/2  # Normalizar ROM

                # Calcular y guardar los valores finales de VMED y VMAX (normalizados)
                if velocities:
                    raw_vmed = sum(velocities) / len(velocities)
                    raw_vmax = max(velocities) 
                    last_vmed = raw_vmed/1.5#normalize_value(raw_vmed, 0, 1)/2
                    last_vmax = normalize_value(raw_vmax, 0, 2)/10

                # Reiniciar ángulos mínimo y máximo para la próxima repetición
                min_angle = None
                max_angle = None

                # Reiniciar velocidades para la próxima repetición
                velocities = []

        except Exception as e:
            print(f"Error procesando landmarks: {e}")

        # Crear el texto a mostrar
        texts = [
            f"REPS: {counter}"
        ]
        # Ajustar el tamaño del recuadro con bordes redondeados
        top_left = (20, 40)  # Mantén el punto superior izquierdo
        bottom_right = (220, 180)  # Reduce el tamaño del rectángulo
        draw_rounded_rectangle(image_bgr, top_left, bottom_right, 20, (128, 0, 128), -1)
        # Dibujar texto formateado en la parte superior izquierda
        draw_formatted_text1(
            image=image_bgr,
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
        draw_rounded_rectangle(image_bgr, recuadro_top_left, recuadro_bottom_right, 20, (128, 0, 128), -1)

        # Crear el texto con las métricas solicitadas
        texts = [
            f"ROM: {rom:.4f} m",
            f"VMED: {last_vmed:.4f} m/s",
            f"VMAX: {last_vmax:.4f} m/s"
        ]

        # Posición inicial del texto dentro del recuadro
        text_position = (40, frame_height - 90)

        # Dibujar el texto formateado
        draw_formatted_text_row(image_bgr, texts, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, 10)

        # Mostrar el frame procesado
        cv2.imshow('Mediapipe Pose Detection', image_bgr)

        # Detener si presionas la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Liberar la cámara y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()