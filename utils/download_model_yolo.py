import torch
from ultralytics import YOLO

try:
    model = YOLO('models/yolo8n-pose.pt')
    print("Modelo cargado correctamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")