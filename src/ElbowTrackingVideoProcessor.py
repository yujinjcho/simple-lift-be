import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose


class ElbowTrackingVideoProcessor:

    def __init__(self):
        # self.left_elbow_history = []
        self.right_elbow_history = []
        self.line_color = (0, 200, 0)
        self.line_thickness = 2

    def draw(self, frame, landmarks, pose_landmarks):
        # left_elbow_idx = mp_holistic.PoseLandmark.LEFT_SHOULDER.value
        right_elbow_idx = mp_holistic.PoseLandmark.RIGHT_ELBOW.value

        # left_elbow = landmarks[left_elbow_idx]
        right_elbow = landmarks[right_elbow_idx]

        # self.left_elbow_history.append(left_elbow)
        self.right_elbow_history.append(right_elbow)

        for i in range(len(self.right_elbow_history) - 1):
            cv2.line(frame, self.right_elbow_history[i], self.right_elbow_history[i + 1], self.line_color,
                     self.line_thickness)
