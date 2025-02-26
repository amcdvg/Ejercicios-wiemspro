# main.py
import cv2
from constants import Constants as K
from pose_estimator import PoseEstimator

from exercises.curl import CurlExercise
from exercises.squat import SquatExercise
from exercises.pushup import PushupExercise
from exercises.plank import PlankExercise

exercises = ['curl', 'squat', 'pushup', 'plank']
def main():
    """_summary_
    """
    exercise_type = exercises[0]
    if exercise_type == 'curl':
        exercise = CurlExercise()
        relevant_indices = [K.YOLO_POSE_KEYPOINTS['RIGHT_SHOULDER'],
                            K.YOLO_POSE_KEYPOINTS['RIGHT_ELBOW'],
                            K.YOLO_POSE_KEYPOINTS['RIGHT_WRIST']]
    elif exercise_type == 'squat':
        exercise = SquatExercise()
        relevant_indices = [K.YOLO_POSE_KEYPOINTS['RIGHT_HIP'],
                            K.YOLO_POSE_KEYPOINTS['RIGHT_KNEE'],
                            K.YOLO_POSE_KEYPOINTS['RIGHT_ANKLE']]
    elif exercise_type == 'pushup':
        exercise = PushupExercise()
        relevant_indices = [K.YOLO_POSE_KEYPOINTS['RIGHT_SHOULDER'],
                            K.YOLO_POSE_KEYPOINTS['RIGHT_ELBOW'],
                            K.YOLO_POSE_KEYPOINTS['RIGHT_WRIST']]
    elif exercise_type == 'plank':
        exercise = PlankExercise()
        relevant_indices = [K.YOLO_POSE_KEYPOINTS['RIGHT_SHOULDER'],
                            K.YOLO_POSE_KEYPOINTS['RIGHT_HIP'],
                            K.YOLO_POSE_KEYPOINTS['RIGHT_KNEE']]
    else:
        print("Ejercicio no reconocido")
        return
    
    pose_estimator = PoseEstimator()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error al abrir la cámara")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        data, annotated_frame = pose_estimator.estimate(frame, relevant_indices)
        if data is not None:
            keypoints, confs = data
            exercise.update(keypoints, confs)

        exercise.draw(annotated_frame)

        cv2.imshow('Virtual GYM', annotated_frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
