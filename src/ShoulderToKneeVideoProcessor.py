import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose

class ShoulderToKneeVideoProcessor:
    def draw(self, frame, landmarks, pose_landmarks):
        line_color = (0, 255, 0)
        line_thickness = 2

        # Define the landmark indices for the shoulders and knees
        left_shoulder_idx = mp_holistic.PoseLandmark.LEFT_SHOULDER.value
        right_shoulder_idx = mp_holistic.PoseLandmark.RIGHT_SHOULDER.value
        left_knee_idx = mp_holistic.PoseLandmark.LEFT_KNEE.value
        right_knee_idx = mp_holistic.PoseLandmark.RIGHT_KNEE.value

        # Draw the lines
        cv2.line(frame, landmarks[left_shoulder_idx], landmarks[left_knee_idx], line_color, line_thickness)
        cv2.line(frame, landmarks[right_shoulder_idx], landmarks[right_knee_idx], line_color, line_thickness)
