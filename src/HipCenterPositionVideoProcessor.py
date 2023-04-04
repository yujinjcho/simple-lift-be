import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose


class HipCenterPositionVideoProcessor:

    def __init__(self):
        self.line_color = (255, 0, 0)  # BGR format
        self.line_thickness = 2

    def draw(self, frame, landmarks, pose_landmarks):
        right_shoulder_idx = mp_holistic.PoseLandmark.RIGHT_SHOULDER.value
        right_hip_idx = mp_holistic.PoseLandmark.RIGHT_HIP.value
        right_knee_idx = mp_holistic.PoseLandmark.RIGHT_KNEE.value

        right_shoulder = landmarks[right_shoulder_idx]
        right_hip = landmarks[right_hip_idx]
        right_knee = landmarks[right_knee_idx]

        # Draw horizontal lines from right shoulder, hip, and knee
        cv2.line(frame, (0, right_shoulder[1]), (frame.shape[1], right_shoulder[1]), self.line_color, self.line_thickness)
        cv2.line(frame, (0, right_hip[1]), (frame.shape[1], right_hip[1]), self.line_color, self.line_thickness)
        cv2.line(frame, (0, right_knee[1]), (frame.shape[1], right_knee[1]), self.line_color, self.line_thickness)
