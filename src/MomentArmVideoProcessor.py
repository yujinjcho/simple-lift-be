import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose


class MomentArmVideoProcessor:

    def __init__(self):
        # self.line_color = (255, 0, 0)
        self.line_color = (128, 179, 255)
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
        # vertical_line_color = (0, 0, 180)
        vertical_line_color = (179, 255, 191)
        cv2.line(frame, (right_elbow[0], 0), (right_elbow[0], frame.shape[0]), vertical_line_color, self.line_thickness)

        # Draw a horizontal line from the right hip to the vertical line
        cv2.line(frame, right_hip, (right_elbow[0], right_hip[1]), self.line_color, self.line_thickness)

        shoulder_hip = np.array(right_hip) - np.array(right_shoulder)
        horizontal_line = np.array([1, 0])

        epsilon = 1e-8
        angle_rad = np.arccos(np.dot(shoulder_hip, horizontal_line) / (np.linalg.norm(shoulder_hip) * np.linalg.norm(horizontal_line) + epsilon))
        angle_deg = np.degrees(angle_rad)

        # Adjust the angle to ensure that it shows 0 when the user's back is horizontal
        angle_deg = abs(angle_deg - 180)

        # Label
        label = f"Back Angle: {angle_deg:.1f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        font_thickness = 2
        text_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)

        # Shaded box background
        box_padding = 5
        box_tl = (0, int(frame.shape[0] * 0.1) - text_size[1] - 2 * box_padding)
        box_br = (text_size[0] + 2 * box_padding, int(frame.shape[0] * 0.1))
        cv2.rectangle(frame, box_tl, box_br, (50, 50, 50), -1)

        # White text
        text_origin = (box_padding, int(frame.shape[0] * 0.1) - box_padding)
        cv2.putText(frame, label, text_origin, font, font_scale, (255, 255, 255), font_thickness)
