# pose_estimator.py
import numpy as np
from ultralytics import YOLO
from constants import Constants as K

class PoseEstimator:
    """_summary_
    """
    def __init__(self, model_path='models\yolo11n-pose.pt'):
        self.model = YOLO(model_path)
        
    def estimate(self, frame, relevant_indices=None):
        """_summary_

        Args:
            frame (_type_): _description_
            relevant_indices (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        results = self.model(frame, verbose=False)
        annotated_frame = results[0].plot()
        
        # Verificar si se detectaron keypoints y si contienen datos
        if (results[0].keypoints is None or 
            not hasattr(results[0].keypoints, "xy") or 
            len(results[0].keypoints.xy) == 0 or 
            results[0].keypoints.conf is None or 
            len(results[0].keypoints.conf) == 0):
            return None, annotated_frame

        if relevant_indices is None:
            try:
                kpts = results[0].keypoints.xy[0].cpu().numpy()
                confs = results[0].keypoints.conf[0].cpu().numpy()
            except Exception as e:
                return None, annotated_frame
            return (kpts, confs), annotated_frame

        best_index = 0
        best_conf = -1
        # Iterar sobre cada detección y usar try/except para manejar posibles Nones
        for i in range(len(results[0].keypoints.xy)):
            try:
                current_confs = results[0].keypoints.conf[i].cpu().numpy()
            except Exception:
                continue
            if current_confs is None:
                continue
            mean_conf = np.mean(current_confs[relevant_indices])
            if mean_conf > best_conf:
                best_conf = mean_conf
                best_index = i

        try:
            kpts = results[0].keypoints.xy[best_index].cpu().numpy()
            confs = results[0].keypoints.conf[best_index].cpu().numpy()
        except Exception:
            return None, annotated_frame

        return (kpts, confs), annotated_frame