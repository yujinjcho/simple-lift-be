import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose


class ShoulderTrackingVideoProcessor:

    def __init__(self):
        self.left_shoulder_history = []
        self.right_shoulder_history = []
        self.line_color = (255, 0, 0)
        self.line_thickness = 2

    def draw(self, frame, landmarks, pose_landmarks):
        left_shoulder_idx = mp_holistic.PoseLandmark.LEFT_SHOULDER.value
        right_shoulder_idx = mp_holistic.PoseLandmark.RIGHT_SHOULDER.value

        left_shoulder = landmarks[left_shoulder_idx]
        right_shoulder = landmarks[right_shoulder_idx]

        self.left_shoulder_history.append(left_shoulder)
        self.right_shoulder_history.append(right_shoulder)

        for i in range(len(self.left_shoulder_history) - 1):
            cv2.line(frame, self.left_shoulder_history[i], self.left_shoulder_history[i + 1], self.line_color,
                     self.line_thickness)

        for i in range(len(self.right_shoulder_history) - 1):
            cv2.line(frame, self.right_shoulder_history[i], self.right_shoulder_history[i + 1], self.line_color,
                     self.line_thickness)
