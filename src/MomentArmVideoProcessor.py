import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose


class MomentArmVideoProcessor:

    def __init__(self):
        self.line_color = (255, 0, 0)
        self.line_thickness = 2

    def draw(self, frame, landmarks, pose_landmarks):
        right_shoulder_idx = mp_holistic.PoseLandmark.RIGHT_SHOULDER.value
        right_elbow_idx = mp_holistic.PoseLandmark.RIGHT_ELBOW.value
        right_hip_idx = mp_holistic.PoseLandmark.RIGHT_HIP.value

        right_shoulder = landmarks[right_shoulder_idx]
        right_elbow = landmarks[right_elbow_idx]
        right_hip = landmarks[right_hip_idx]

        # Draw a line between right shoulder and right hip
        cv2.line(frame, right_shoulder, right_hip, self.line_color, self.line_thickness)

        # Draw a vertical line that tracks the elbow
        vertical_line_color = (0, 0, 180)
        cv2.line(frame, (right_elbow[0], 0), (right_elbow[0], frame.shape[0]), vertical_line_color, self.line_thickness)

        # Draw a horizontal line from the right hip to the vertical line
        cv2.line(frame, right_hip, (right_elbow[0], right_hip[1]), self.line_color, self.line_thickness)
