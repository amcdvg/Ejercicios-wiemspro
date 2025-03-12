import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

# --------------------------
# Configuración inicial
# --------------------------
model = YOLO('models/yolo11n-pose.pt')
n = 4
video_path = f'videos/swing golf {n}.mp4'

cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
W, H = int(cap.get(3)), int(cap.get(4))

# Configurar VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(f'result/output_video {n}.mp4', fourcc, fps, (W, H))

# Configuración de visualización
COLORS = {
    'angle': (0, 255, 0),
    'velocity': (255, 0, 255),  # Magenta#(255, 0, 0),
    'velocity2': (0, 255, 255),#
    'impact': (0, 0, 255),
    'rom': (255, 0, 255),#(255, 255, 0),
    'ball_speed': (0, 255, 255)
}

# --------------------------
# Funciones utilitarias
# --------------------------
def calculate_angle(a, b, c):
    try:
        if any(point is None for point in [a, b, c]):
            return 0.0  # Devolver 0 si algún punto es None
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return angle if angle <= 180 else 360 - angle
    except Exception as e:
        print(f"Error en cálculo de ángulo: {e}")
        return 0.0

def calculate_velocity(points, window=5):
    try:
        if len(points) < window:
            return 0.0
        displacements = np.diff(points[-window:], axis=0)
        return np.mean(np.sqrt(displacements[:,0]**2 + displacements[:,1]**2)) * fps
    except:
        return 0.0  # Valor predeterminado si hay un error

def safe_value(value, default=0.0):
    return value if isinstance(value, (int, float)) else default

segment_lengths = {
    'shoulder': 0.7,  # Longitud del brazo
    'elbow': 0.3,     # Longitud del antebrazo
    'wrist': 0.1,     # Longitud de la mano
    'hip': 1.0,       # Longitud de la pierna
    'knee': 0.5,      # Longitud de la pantorrilla
    'ankle': 0.2      # Longitud del pie
} 

def calculate_rom_distance(joint, angle_degrees):
    """
    Convierte el ROM de grados a metros usando la longitud del segmento.
    """
    angle_radians = np.radians(angle_degrees)
    segment_length = segment_lengths.get(joint, 1.0)  # Longitud del segmento en metros
    rom_distance = abs(segment_length * np.cos(angle_radians))
    return rom_distance
# --------------------------
# Estructuras de datos
# --------------------------
keypoints_buffer = {k: deque(maxlen=15) for k in [
    'right_shoulder', 'right_elbow', 'right_wrist',
    'right_hip', 'right_knee', 'right_ankle'
]}

metrics = {
    'impact_speed': 0,
    'ball_speed': 0,
    'max_impact_speed': 0,
    'average_speed': 0,
    'rom': {k: [0.0, 0.0] for k in ['shoulder', 'elbow', 'wrist', 'hip', 'knee', 'ankle']},
    'current_angles': {k: 0.0 for k in ['shoulder', 'elbow', 'wrist', 'hip', 'knee', 'ankle']},
    'velocities': {k: [] for k in ['right_shoulder', 'right_elbow', 'right_wrist', 'right_hip', 'right_knee', 'right_ankle']},
    'current_velocities': {k: 0.0 for k in ['right_shoulder', 'right_elbow', 'right_wrist', 'right_hip', 'right_knee', 'right_ankle']},
    'max_velocities': {k: 0.0 for k in ['right_shoulder', 'right_elbow', 'right_wrist', 'right_hip', 'right_knee', 'right_ankle']},
    'average_velocities': {k: 0.0 for k in ['right_shoulder', 'right_elbow', 'right_wrist', 'right_hip', 'right_knee', 'right_ankle']},
}

# --------------------------
# Procesamiento de video
# --------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    results = model(frame, conf=0.25, verbose=False)
    annotated_frame = results[0].plot()

    # Filtrar solo la persona más cercana
    if results[0].boxes.shape[0] > 0:  # Si hay detecciones
        boxes = results[0].boxes.xyxy.cpu().numpy()  # Coordenadas de las cajas
        class_ids = results[0].boxes.cls.cpu().numpy()  # IDs de clase

        # Filtrar solo las personas (clase 0 en COCO)
        person_indices = np.where(class_ids == 0)[0]
        if len(person_indices) > 0:
            # Calcular el área de cada cuadro delimitador
            areas = [(boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]) for i in person_indices]
            # Seleccionar la persona con el área más grande (más cercana)
            closest_person_idx = person_indices[np.argmax(areas)]
            keypoints = results[0].keypoints.xy.cpu().numpy()[closest_person_idx]

            # Extracción de puntos clave
            points = {
                'right_shoulder': keypoints[6] if len(keypoints) > 6 else None,
                'right_elbow': keypoints[8] if len(keypoints) > 8 else None,
                'right_wrist': keypoints[10] if len(keypoints) > 10 else None,
                'right_hip': keypoints[12] if len(keypoints) > 12 else None,
                'right_knee': keypoints[14] if len(keypoints) > 14 else None,
                'right_ankle': keypoints[16] if len(keypoints) > 16 else None,
            }

            # Cálculo de ángulos y ROM
            try:
                # Hombro (cadera-hombro-codo)
                if all(point is not None for point in [points['right_hip'], points['right_shoulder'], points['right_elbow']]):
                    shoulder_angle = calculate_angle(points['right_hip'], points['right_shoulder'], points['right_elbow'])
                    metrics['current_angles']['shoulder'] = shoulder_angle
                    metrics['rom']['shoulder'][0] = min(metrics['rom']['shoulder'][0], shoulder_angle)
                    metrics['rom']['shoulder'][1] = max(metrics['rom']['shoulder'][1], shoulder_angle)
                else:
                    metrics['current_angles']['shoulder'] = 0.0

                # Codo (hombro-codo-muñeca)
                if all(point is not None for point in [points['right_shoulder'], points['right_elbow'], points['right_wrist']]):
                    elbow_angle = calculate_angle(points['right_shoulder'], points['right_elbow'], points['right_wrist'])
                    metrics['current_angles']['elbow'] = elbow_angle
                    metrics['rom']['elbow'][0] = min(metrics['rom']['elbow'][0], elbow_angle)
                    metrics['rom']['elbow'][1] = max(metrics['rom']['elbow'][1], elbow_angle)
                else:
                    metrics['current_angles']['elbow'] = 0.0

                # Cadera (hombro-cadera-rodilla)
                if all(point is not None for point in [points['right_shoulder'], points['right_hip'], points['right_knee']]):
                    hip_angle = calculate_angle(points['right_shoulder'], points['right_hip'], points['right_knee'])
                    metrics['current_angles']['hip'] = hip_angle
                    metrics['rom']['hip'][0] = min(metrics['rom']['hip'][0], hip_angle)
                    metrics['rom']['hip'][1] = max(metrics['rom']['hip'][1], hip_angle)
                else:
                    metrics['current_angles']['hip'] = 0.0

                # Rodilla (cadera-rodilla-tobillo)
                if all(point is not None for point in [points['right_hip'], points['right_knee'], points['right_ankle']]):
                    knee_angle = calculate_angle(points['right_hip'], points['right_knee'], points['right_ankle'])
                    metrics['current_angles']['knee'] = knee_angle
                    metrics['rom']['knee'][0] = min(metrics['rom']['knee'][0], knee_angle)
                    metrics['rom']['knee'][1] = max(metrics['rom']['knee'][1], knee_angle)
                else:
                    metrics['current_angles']['knee'] = 0.0

                # Tobillo (rodilla-tobillo-punto imaginario hacia abajo)
                if all(point is not None for point in [points['right_knee'], points['right_ankle']]):
                    ankle_angle = calculate_angle(
                        points['right_knee'],
                        points['right_ankle'],
                        [points['right_ankle'][0], points['right_ankle'][1] + 1]
                    )
                    metrics['current_angles']['ankle'] = ankle_angle
                    metrics['rom']['ankle'][0] = min(metrics['rom']['ankle'][0], ankle_angle)
                    metrics['rom']['ankle'][1] = max(metrics['rom']['ankle'][1], ankle_angle)
                else:
                    metrics['current_angles']['ankle'] = 0.0

                # Muñeca (codo-muñeca-punto imaginario hacia adelante)
                if all(point is not None for point in [points['right_elbow'], points['right_wrist']]):
                    wrist_angle = calculate_angle(
                        points['right_elbow'],
                        points['right_wrist'],
                        [points['right_wrist'][0] + 1, points['right_wrist'][1]]
                    )
                    metrics['current_angles']['wrist'] = wrist_angle
                    metrics['rom']['wrist'][0] = min(metrics['rom']['wrist'][0], wrist_angle)
                    metrics['rom']['wrist'][1] = max(metrics['rom']['wrist'][1], wrist_angle)
                else:
                    metrics['current_angles']['wrist'] = 0.0
            except Exception as e:
                print(f"Error en cálculo de ángulos: {e}")

            # Cálculo de velocidades
            for joint in ['right_shoulder', 'right_elbow', 'right_wrist', 'right_hip', 'right_knee', 'right_ankle']:
                if points[joint] is not None:
                    keypoints_buffer[joint].append(points[joint])
                    velocity = calculate_velocity(np.array(keypoints_buffer[joint]))
                    metrics['velocities'][joint].append(velocity)
                    metrics['current_velocities'][joint] = velocity
                    metrics['max_velocities'][joint] = max(metrics['max_velocities'][joint], velocity)
                else:
                    if keypoints_buffer[joint]:
                        last_point = keypoints_buffer[joint][-1]
                        keypoints_buffer[joint].append(last_point)
                    metrics['current_velocities'][joint] = 0.0

            # Velocidad media
            for joint in metrics['velocities']:
                metrics['average_velocities'][joint] = np.mean(metrics['velocities'][joint]) if metrics['velocities'][joint] else 0.0

            # Actualización de velocidad de impacto y pelota
            wrist_velocity = metrics['current_velocities']['right_wrist']
            if wrist_velocity > metrics['max_impact_speed']:
                metrics['max_impact_speed'] = wrist_velocity
                metrics['ball_speed'] = wrist_velocity * 1.5

    # --------------------------
    # Visualización de métricas
    # --------------------------
    # Velocidades globales
    y_offset_left = 30
    for metric, color in [
        (f"Vel. Impacto: {safe_value(metrics['max_impact_speed'])* (1/100):.1f} m/s", COLORS['impact']),
        (f"Vel. Pelota: {safe_value(metrics['ball_speed'])* (1/100):.1f} m/s", COLORS['ball_speed']),
        (f"Vel. Media: {safe_value(np.mean(list(metrics['average_velocities'].values())))* (1/100):.1f} m/s", COLORS['velocity'])
    ]:
        cv2.putText(annotated_frame, metric, (10, y_offset_left),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y_offset_left += 30

    # Ángulos actuales
    y_offset_right = 30
    for joint in ['shoulder', 'elbow', 'wrist', 'hip', 'knee', 'ankle']:
        angle_text = f"{joint.capitalize()}: {safe_value(metrics['current_angles'][joint]):.1f} deg"
        cv2.putText(annotated_frame, angle_text, (W - 750, y_offset_right ),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['angle'], 2)
        y_offset_right += 30

    # Rango de movimiento (ROM)
    y_offset_rom = H - 120
    for joint in ['shoulder', 'elbow', 'wrist', 'hip', 'knee', 'ankle']:
        min_ang, max_ang = metrics['rom'][joint]
        rom_range = safe_value(max_ang - min_ang)
        dist = calculate_rom_distance(joint, rom_range)
        rom_text = f"{joint.capitalize()} ROM: {dist:.1f} m (Min: {safe_value(min_ang):.1f} deg, Max: {safe_value(max_ang):.1f} deg )"
        cv2.putText(annotated_frame, rom_text, (10, y_offset_rom - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['rom'], 2)
        y_offset_rom += 30

    # Velocidades medias y máximas por articulación
    y_offset_velocities = 30
    for joint in ['right_shoulder', 'right_elbow', 'right_wrist', 'right_hip', 'right_knee', 'right_ankle']:
        vmed = safe_value(metrics['average_velocities'][joint])
        vmax = safe_value(metrics['max_velocities'][joint])
        velocity_text = f"{joint.split('_')[1].capitalize()}: Vmed={vmed * (1/100):.1f} m/s, Vmax={vmax * (1/100):.1f} m/s"
        cv2.putText(annotated_frame, velocity_text, (W - 500, y_offset_velocities),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['velocity2'], 2)
        y_offset_velocities += 30

    out.write(annotated_frame)
    cv2.imshow('Swing Analysis', annotated_frame)
    if cv2.waitKey(1) == ord('q'): break

# --------------------------
# Finalización
# --------------------------
cap.release()
out.release()
cv2.destroyAllWindows()

print("\n--- REPORTE FINAL ---")
print(f"Velocidad de Impacto Máxima: {safe_value(metrics['max_impact_speed']):.1f} px/s")
print(f"Velocidad de la Pelota: {safe_value(metrics['ball_speed']):.1f} px/s")
print(f"Velocidad Media Global: {safe_value(np.mean(list(metrics['average_velocities'].values()))):.1f} px/s")

print("\nRango de Movimiento (ROM):")
for joint in ['shoulder', 'elbow', 'wrist', 'hip', 'knee', 'ankle']:
    min_ang, max_ang = metrics['rom'][joint]
    print(f"{joint.capitalize()}: {safe_value(max_ang - min_ang):.1f}° (Min: {safe_value(min_ang):.1f}°, Max: {safe_value(max_ang):.1f}°)")

print("\nVelocidades por Articulación:")
for joint in ['right_shoulder', 'right_elbow', 'right_wrist', 'right_hip', 'right_knee', 'right_ankle']:
    print(f"{joint.capitalize()}: Max={safe_value(metrics['max_velocities'][joint]):.1f} px/s, Avg={safe_value(metrics['average_velocities'][joint]):.1f} px/s")