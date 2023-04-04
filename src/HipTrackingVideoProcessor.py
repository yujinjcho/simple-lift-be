import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose


class HipTrackingVideoProcessor:

    def __init__(self):
        # self.left_hip_history = []
        self.right_hip_history = []
        # BGR format
        self.line_color = (255, 255, 51)
        self.line_thickness = 2

    def draw(self, frame, landmarks, pose_landmarks):
        # left_hip_idx = mp_holistic.PoseLandmark.LEFT_SHOULDER.value
        right_hip_idx = mp_holistic.PoseLandmark.RIGHT_HIP.value

        # left_hip = landmarks[left_hip_idx]
        right_hip = landmarks[right_hip_idx]

        # self.left_hip_history.append(left_hip)
        self.right_hip_history.append(right_hip)

        for i in range(len(self.right_hip_history) - 1):
            cv2.line(frame, self.right_hip_history[i], self.right_hip_history[i + 1], self.line_color,
                     self.line_thickness)
