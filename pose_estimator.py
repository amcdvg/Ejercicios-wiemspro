# pose_estimator.py
import numpy as np
from ultralytics import YOLO
from constants import Constants as K

class PoseEstimator:
    """_summary_
    """
    def __init__(self, model_path='yolo11n-pose.pt'):
        self.model = YOLO(model_path)
        
    def estimate(self, frame, relevant_indices=None):
        """
        Procesa el frame y retorna una tupla:
          - (keypoints, confs) de la persona seleccionada
          - frame anotado (con detecciones dibujadas)
          
        Si se especifica relevant_indices, se selecciona la detección con mayor promedio de confianza en esos keypoints.
        """
        results = self.model(frame, verbose=False)
        annotated_frame = results[0].plot()
        
        if results[0].keypoints is None or len(results[0].keypoints.xy) == 0:
            return None, annotated_frame
        
        if relevant_indices is None:
            kpts = results[0].keypoints.xy[0].cpu().numpy()
            confs = results[0].keypoints.conf[0].cpu().numpy()
            return (kpts, confs), annotated_frame
        
        best_index = 0
        best_conf = -1
        for i in range(len(results[0].keypoints.xy)):
            confs = results[0].keypoints.conf[i].cpu().numpy()
            mean_conf = np.mean(confs[relevant_indices])
            if mean_conf > best_conf:
                best_conf = mean_conf
                best_index = i
                
        kpts = results[0].keypoints.xy[best_index].cpu().numpy()
        confs = results[0].keypoints.conf[best_index].cpu().numpy()
        return (kpts, confs), annotated_frame
