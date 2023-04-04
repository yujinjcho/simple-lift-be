import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose


class ShoulderTrackingVideoProcessor:

    def __init__(self):
        self.right_shoulder_history = []
        # BGR format
        self.line_color_up = (255, 128, 0)
        self.line_color_down = (153, 153, 255)
        self.line_thickness = 2

    def draw(self, frame, landmarks, pose_landmarks):
        right_shoulder_idx = mp_holistic.PoseLandmark.RIGHT_SHOULDER.value
        right_shoulder = landmarks[right_shoulder_idx]
        self.right_shoulder_history.append(right_shoulder)

        for i in range(len(self.right_shoulder_history) - 1):
            current_point = self.right_shoulder_history[i]
            next_point = self.right_shoulder_history[i + 1]

            # Choose color based on direction (up or down)
            if next_point[1] > current_point[1]:
                line_color = self.line_color_down
            else:
                line_color = self.line_color_up

            cv2.line(frame, current_point, next_point, line_color, self.line_thickness)
