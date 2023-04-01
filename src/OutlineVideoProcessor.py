import cv2
import mediapipe as mp

from src.utils import angle_between_points

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose

bone_connections = [
    (mp_holistic.PoseLandmark.LEFT_KNEE.value, mp_holistic.PoseLandmark.LEFT_HIP.value),
    (mp_holistic.PoseLandmark.RIGHT_KNEE.value, mp_holistic.PoseLandmark.RIGHT_HIP.value),
    (mp_holistic.PoseLandmark.LEFT_HEEL.value, mp_holistic.PoseLandmark.LEFT_KNEE.value),
    (mp_holistic.PoseLandmark.RIGHT_HEEL.value, mp_holistic.PoseLandmark.RIGHT_KNEE.value),
    (mp_holistic.PoseLandmark.LEFT_HIP.value, mp_holistic.PoseLandmark.RIGHT_HIP.value),
    (mp_holistic.PoseLandmark.LEFT_HIP.value, mp_holistic.PoseLandmark.LEFT_SHOULDER.value),
    (mp_holistic.PoseLandmark.RIGHT_HIP.value, mp_holistic.PoseLandmark.RIGHT_SHOULDER.value),
    (mp_holistic.PoseLandmark.LEFT_SHOULDER.value, mp_holistic.PoseLandmark.RIGHT_SHOULDER.value),
    (mp_holistic.PoseLandmark.LEFT_SHOULDER.value, mp_holistic.PoseLandmark.LEFT_ELBOW.value),
    (mp_holistic.PoseLandmark.LEFT_SHOULDER.value, mp_holistic.PoseLandmark.LEFT_EAR.value),
    (mp_holistic.PoseLandmark.LEFT_ELBOW.value, mp_holistic.PoseLandmark.LEFT_WRIST.value),
    (mp_holistic.PoseLandmark.RIGHT_SHOULDER.value, mp_holistic.PoseLandmark.RIGHT_ELBOW.value),
    (mp_holistic.PoseLandmark.RIGHT_SHOULDER.value, mp_holistic.PoseLandmark.RIGHT_EAR.value),
    (mp_holistic.PoseLandmark.LEFT_EAR.value, mp_holistic.PoseLandmark.RIGHT_EAR.value),
    (mp_holistic.PoseLandmark.RIGHT_ELBOW.value, mp_holistic.PoseLandmark.RIGHT_WRIST.value),
]


class OutlineVideoProcessor:
    def __init__(self):
        # video outlining
        self.line_color = (0, 255, 0)
        self.line_thickness = 2

    def draw(self, frame, landmarks, pose_landmarks):
        # Draw lines
        for connection in bone_connections:
            start, end = connection
            cv2.line(frame, landmarks[start], landmarks[end], self.line_color, self.line_thickness)
