import numpy as np
import cv2

class MotionAnalyzer:
    """Clase para analizar y visualizar movimiento en tiempo real con OpenCV."""

    @staticmethod
    def calculate_angle(a, b, c):
        """
        Calcula el ángulo formado por tres puntos en el plano 2D.

        Parámetros:
            a (tuple): Coordenadas (x, y) del primer punto
            b (tuple): Coordenadas (x, y) del punto vértice
            c (tuple): Coordenadas (x, y) del tercer punto

        Retorna:
            float: Ángulo en grados entre los segmentos ba y bc
        """
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians) * 180.0 / np.pi
        if angle > 180.0:
            angle = 360 - angle
        return angle

    @staticmethod
    def calculate_distance(a, b):
        """
        Calcula la distancia euclidiana entre dos puntos en 2D.

        Parámetros:
            a (tuple): Coordenadas (x, y) del primer punto
            b (tuple): Coordenadas (x, y) del segundo punto

        Retorna:
            float: Distancia entre los puntos
        """
        a = np.array(a)
        b = np.array(b)
        return np.linalg.norm(a - b)

    @staticmethod
    def normalize_value(value, min_value, max_value):
        """
        Normaliza un valor en el rango [0, 2].

        Parámetros:
            value (float): Valor a normalizar
            min_value (float): Valor mínimo del rango original
            max_value (float): Valor máximo del rango original

        Retorna:
            float: Valor normalizado
        """
        return (value - min_value) / (max_value - min_value) * 2

    @staticmethod
    def normalize_value_bar(value, min_value, max_value):
        """
        Normaliza un valor en el rango [0, 1] para uso en barras de progreso.

        Parámetros:
            value (float): Valor a normalizar
            min_value (float): Valor mínimo del rango original
            max_value (float): Valor máximo del rango original

        Retorna:
            float: Valor normalizado entre 0 y 1
        """
        if min_value is None or max_value is None:
            return 0
        if max_value == min_value:
            return 0
        progress = (value - min_value) / (max_value - min_value)
        if progress < 0:
            return 0
        else:
            return (value - min_value) / (max_value - min_value)

    @staticmethod
    def draw_rounded_rect(image, position, bar_width, bar_height, color, border_radius):
        """
        Dibuja un rectángulo con bordes redondeados en una imagen.

        Parámetros:
            image (numpy.ndarray): Imagen donde se dibujará
            position (tuple): Posición (x, y) de la esquina superior izquierda
            bar_width (int): Ancho del rectángulo
            bar_height (int): Alto del rectángulo
            color (tuple): Color en formato BGR
            border_radius (int): Radio de los bordes redondeados
        """
        x, y = position
        overlay = image.copy()
        
        cv2.ellipse(overlay, (x + border_radius, y + border_radius), 
                   (border_radius, border_radius), 180, 0, 90, color, -1)
        cv2.ellipse(overlay, (x + bar_width - border_radius, y + border_radius), 
                   (border_radius, border_radius), 270, 0, 90, color, -1)
        cv2.ellipse(overlay, (x + border_radius, y + bar_height - border_radius), 
                   (border_radius, border_radius), 90, 0, 90, color, -1)
        cv2.ellipse(overlay, (x + bar_width - border_radius, y + bar_height - border_radius), 
                   (border_radius, border_radius), 0, 0, 90, color, -1)
        
        cv2.rectangle(overlay, (x + border_radius, y), 
                     (x + bar_width - border_radius, y + bar_height), color, -1)
        cv2.rectangle(overlay, (x, y + border_radius), 
                     (x + bar_width, y + bar_height - border_radius), color, -1)
        
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    def draw_progress_bar(self, image, progress, position, bar_width, bar_height, color, border_radius):
        """
        Dibuja una barra de progreso vertical con porcentaje.

        Parámetros:
            image (numpy.ndarray): Imagen donde se dibujará.
            progress (float): Progreso actual (0-1).
            position (tuple): Posición (x, y) de la barra.
            bar_width (int): Ancho de la barra.
            bar_height (int): Alto total de la barra.
            color (tuple): Color de la barra en BGR.
            border_radius (int): Radio de los bordes redondeados.
        """
        x, y = position
        # Invertir el progreso para que 0 represente barra vacía y 1 barra llena.
        inv_progress = 1 - progress
        progress_height = int(bar_height * inv_progress)
        
        # Dibujar el contorno base de la barra (siempre se dibuja)
        self.draw_rounded_rect(image, (x, y), bar_width, bar_height, (128, 0, 128), border_radius)
        
        # Si progress_height es menor o igual a 0, forzamos un valor mínimo para evitar ejes de 0
        if progress_height <= 0:
            fill_y = y + bar_height - 1  # La parte inferior de la barra
            fill_height = 1
        else:
            fill_y = y + bar_height - progress_height
            fill_height = progress_height

        # Dibujar la porción de la barra correspondiente al progreso actual
        self.draw_rounded_rect(image, (x, fill_y), bar_width, fill_height, color, min(border_radius, fill_height // 2))
        
        # Calcular y dibujar el porcentaje
        percentage_text = f"{int(inv_progress * 100)}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color_text = (255, 255, 255)
        thickness = 2
        text_size = cv2.getTextSize(percentage_text, font, font_scale, thickness)[0]
        
        box_width = text_size[0] + 10
        box_height = text_size[1] + 10
        box_x = x + (bar_width - box_width) // 2
        box_y = int(y + bar_height - (inv_progress * bar_height) - box_height - 5)
        
        if (box_x + box_width <= image.shape[1]) and (box_y + box_height <= image.shape[0]):
            self.draw_rounded_rect_T(image, (box_x, box_y), box_width, box_height, (75, 0, 130), 5)
            cv2.putText(image, percentage_text, (box_x + 5, box_y + box_height - 5), 
                        font, font_scale, color_text, thickness)

    @staticmethod
    def draw_rounded_rect_T(image, top_left, width, height, color, border_radius):
        """
        Dibuja un rectángulo con bordes redondeados (versión alternativa).

        Parámetros:
            image (numpy.ndarray): Imagen donde se dibujará
            top_left (tuple): Coordenadas (x, y) de la esquina superior izquierda
            width (int): Ancho del rectángulo
            height (int): Alto del rectángulo
            color (tuple): Color en formato BGR
            border_radius (int): Radio de los bordes
        """
        x, y = top_left
        radius = border_radius
        cv2.rectangle(image, (x + radius, y), (x + width - radius, y + height), color, -1)
        cv2.rectangle(image, (x, y + radius), (x + width, y + height - radius), color, -1)
        cv2.circle(image, (x + radius, y + radius), radius, color, -1)
        cv2.circle(image, (x + width - radius, y + radius), radius, color, -1)
        cv2.circle(image, (x + radius, y + height - radius), radius, color, -1)
        cv2.circle(image, (x + width - radius, y + height - radius), radius, color, -1)

    def draw_progress_wheel(self, image, progress, center, radius, color, thickness, elapsed_time):
        """
        Dibuja una rueda de progreso circular con tiempo transcurrido.

        Parámetros:
            image (numpy.ndarray): Imagen donde se dibujará
            progress (float): Progreso actual (0-1)
            center (tuple): Centro del círculo (x, y)
            radius (int): Radio del círculo
            color (tuple): Color en formato BGR
            thickness (int): Grosor de la línea
            elapsed_time (float): Tiempo transcurrido a mostrar
        """
        angle_start = -90
        angle_end = int(angle_start + 360 * progress)
        overlay = image.copy()
        cv2.ellipse(overlay, center, (radius, radius), 0, 0, 360, color, thickness)
        cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
        cv2.ellipse(image, center, (radius, radius), 0, angle_start, angle_end, color, thickness)

        time_text = f"{elapsed_time:.1f}s"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        text_size = cv2.getTextSize(time_text, font, font_scale, 2)[0]
        text_x = center[0] - text_size[0] // 2
        text_y = center[1] + text_size[1] // 2
        cv2.putText(image, time_text, (text_x, text_y), font, font_scale, color, 2)

    @staticmethod
    def draw_rounded_rectangle_rep(image, top_left, bottom_right, radius, color, thickness):
        """
        Dibuja un rectángulo redondeado para representar repeticiones.

        Parámetros:
            image (numpy.ndarray): Imagen donde se dibujará
            top_left (tuple): Esquina superior izquierda (x, y)
            bottom_right (tuple): Esquina inferior derecha (x, y)
            radius (int): Radio de los bordes
            color (tuple): Color en formato BGR
            thickness (int): Grosor del borde
        """

        x1, y1 = top_left
        x2, y2 = bottom_right
        cv2.circle(image, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(image, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(image, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(image, (x2 - radius, y2 - radius), radius, color, -1)
        cv2.rectangle(image, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(image, (x1, y1 + radius), (x2, y2 - radius), color, -1)

    @staticmethod
    def draw_formatted_text(image, texts, position, font, font_scale, color, thickness, line_spacing):
        """
        Dibuja texto formateado con etiquetas en cursiva y valores en negrita.

        Parámetros:
            image (numpy.ndarray): Imagen donde se dibujará
            texts (list): Lista de textos en formato "etiqueta: valor"
            position (tuple): Posición inicial (x, y)
            font: Fuente de OpenCV
            font_scale (float): Escala de la fuente
            color (tuple): Color en formato BGR
            thickness (int): Grosor del texto
            line_spacing (int): Espaciado entre líneas
        """
        x, y = position
        for text in texts:
            before_colon, after_colon = text.split(":")
            before_colon = before_colon.strip()
            after_colon = after_colon.strip()

            italic_font = font | cv2.FONT_ITALIC
            cv2.putText(image, before_colon + ":", (x, y), italic_font, 
                       font_scale, color, thickness, cv2.LINE_AA)

            text_size, _ = cv2.getTextSize(before_colon + ":", italic_font, font_scale, thickness)
            text_width = text_size[0]

            bold_font = font
            cv2.putText(image, after_colon, (x + text_width + 5, y), bold_font, 
                       font_scale, color, thickness, cv2.LINE_AA)

            y += text_size[1] + line_spacing

    @staticmethod
    def draw_formatted_text_metrics(image, texts, position, font, font_scale, color, thickness, spacing):
        """
        Dibuja métricas distribuidas uniformemente en una fila.

        Parámetros:
            image (numpy.ndarray): Imagen donde se dibujará
            texts (list): Lista de textos en formato "métrica: valor"
            position (tuple): Posición inicial (x, y)
            font: Fuente de OpenCV
            font_scale (float): Escala de la fuente
            color (tuple): Color en formato BGR
            thickness (int): Grosor del texto
            spacing (int): Espacio horizontal entre métricas
        """
        x_start, y = position
        image_width = image.shape[1]
        total_text_width = 0
        text_sizes = []

        for text in texts:
            before_colon, after_colon = text.split(":")
            before_colon = before_colon.strip()
            after_colon = after_colon.strip()

            italic_font = font | cv2.FONT_ITALIC
            bold_font = font

            before_colon_size, _ = cv2.getTextSize(before_colon + ":", italic_font, font_scale, thickness)
            after_colon_size, _ = cv2.getTextSize(after_colon, bold_font, font_scale, thickness)

            total_width = before_colon_size[0] + after_colon_size[0] + spacing
            text_sizes.append((before_colon_size, after_colon_size, total_width))
            total_text_width += total_width

        remaining_space = image_width - total_text_width
        gap = remaining_space // (len(texts) + 1) if remaining_space > 0 else 0

        x = x_start + gap
        for i, text in enumerate(texts):
            before_colon, after_colon = text.split(":")
            before_colon = before_colon.strip()
            after_colon = after_colon.strip()

            italic_font = font | cv2.FONT_ITALIC
            cv2.putText(image, before_colon + ":", (x, y+50), italic_font, 
                       font_scale, color, thickness, cv2.LINE_AA)

            before_colon_width = text_sizes[i][0][0]
            bold_font = font
            cv2.putText(image, after_colon, (x + before_colon_width + 5, y+50), 
                       bold_font, font_scale, color, thickness, cv2.LINE_AA)

            x += text_sizes[i][2] + gap

    @staticmethod
    def draw_formatted_text_rep(image, texts, position, font, font_scale, color, thickness, line_spacing):
        """
        Dibuja texto de repeticiones con formato especial para números.

        Parámetros:
            image (numpy.ndarray): Imagen donde se dibujará
            texts (list): Lista de textos en formato "etiqueta: valor"
            position (tuple): Posición inicial (x, y)
            font: Fuente de OpenCV
            font_scale (float): Escala de la fuente
            color (tuple): Color en formato BGR
            thickness (int): Grosor del texto
            line_spacing (int): Espaciado entre líneas

        Nota:
            Ajusta la posición horizontal basado en el valor numérico
        """
        x, y = position
        for text in texts:
            before_colon, after_colon = text.split(":")
            before_colon = before_colon.strip()
            after_colon = after_colon.strip()

            italic_font = font | cv2.FONT_ITALIC
            text_size, _ = cv2.getTextSize(before_colon + ":", italic_font, font_scale, thickness)
            text_width = text_size[0]

            bold_font = font
            n = int(after_colon)
            
            if n < 10:
                cv2.putText(image, after_colon, (x +50, y+45), bold_font, 
                           font_scale, color, thickness, cv2.LINE_AA)
            else:
                cv2.putText(image, after_colon, (x +20, y+45), bold_font, 
                           font_scale, color, thickness, cv2.LINE_AA)

            y += text_size[1] + line_spacing

# Uso
#motion_analyzer = MotionAnalyzer()
