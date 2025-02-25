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

# Variable para almacenar el valor suavizado de progreso
smooth_progress = 0
smooth_factor = 0.1  # Controla la cantidad de suavizado (0 es sin suavizado, 1 es muy suave)

# Variables
min_angle_c = None
max_angle_c = None
start_time_c = None
elapsed_time_c = 0
smooth_progress_c = 0
smooth_factor_c = 0.1
stage_c = None  # Etapa del movimiento: "subida" o "bajada"

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

# Función para normalizar un valor en un rango dado
def normalize_value_bar(value, min_value, max_value):
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
    prev_angle = None
    prev_stage_c = None
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

            angle = calculate_angle(shoulder, elbow, wrist)

            # Actualizar el ángulo mínimo y máximo si es la primera vez que se detectan
            if min_angle is None or angle < min_angle:
                min_angle = angle
            if max_angle is None or angle > max_angle:
                max_angle = angle

            progress = 0
            if min_angle is not None and max_angle is not None:
                progress = normalize_value_bar(angle, min_angle, max_angle)
                progress = max(0, min(1, progress))  # Asegurarse de que el progreso esté entre 0 y 1

                # Aplicar suavizado al progreso
                smooth_progress = smooth_factor * progress + (1 - smooth_factor) * smooth_progress

                # Dibujar la barra de progreso
                # Dibujar la barra de progreso vertical
                frame_height, frame_width, _ = frame.shape
                draw_progress_bar(image_bgr, smooth_progress, (frame_width - 110, 50), 20, 300, (128, 0, 128), 10)
            
            # Calcular progreso invertido
            progress_c = 1 - ((angle - min_angle) / (max_angle - min_angle)) if max_angle > min_angle else 0
            smooth_progress_c = smooth_factor_c * progress_c + (1 - smooth_factor_c) * smooth_progress_c

            # Detectar la etapa del movimiento
            if angle > 160:  # Brazo completamente extendido
                stage_c = "bajada"
            elif angle < 160:  # Brazo completamente doblado
                stage_c = "subida"

            # Reiniciar el tiempo al cambiar de etapa
            if stage_c != prev_stage_c:
                if stage_c== "subida":  # Iniciar temporizador en "subida"
                    start_time_c = time.time()
                elif stage_c == "bajada" :  # Reiniciar temporizador en "bajada"
                    elapsed_time_c = 0
                    start_time_c = None

            # Calcular tiempo transcurrido en la etapa "subida"
            if stage_c == "subida" and start_time_c is not None:
                elapsed_time_c = time.time() - start_time_c

            prev_stage_c = stage_c
            
            # Dibujar rueda de progreso
            center = (frame.shape[1] - 100, frame.shape[0] - frame.shape[0] // 3 - 50)
            radius = 50
            thickness = 10
            color = (128, 0, 128) if stage_c == "subida" else (128, 0, 128)  # Verde para "subida", rojo para "bajada"
            draw_progress_wheel(image_bgr, smooth_progress_c, center, radius, color, thickness, elapsed_time_c)
            
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
                    
                    raw_rom = (max_angle - min_angle) * np.pi / 270 * arm_length_meters
                    rom = raw_rom - arm_length_meters - (height * 0.175)#normalize_value(raw_rom, 0, 1)/2  # Normalizar ROM

                # Calcular y guardar los valores finales de VMED y VMAX (normalizados)
                if velocities:
                    raw_vmed = sum(velocities) / len(velocities)
                    raw_vmax = max(velocities) 
                    last_vmed = raw_vmed#/1.5#normalize_value(raw_vmed, 0, 1)/2
                    last_vmax = normalize_value(raw_vmax, 0, 2)/10

                # Reiniciar ángulos mínimo y máximo para la próxima repetición
                min_angle = None
                max_angle = None

                # Reiniciar velocidades para la próxima repetición
                velocities = []
            # Normalizar el ángulo para obtener el progreso (valor entre 0 y 1)
            
        except Exception as e:
            pass
            #print(f"Error procesando landmarks: {e}")

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