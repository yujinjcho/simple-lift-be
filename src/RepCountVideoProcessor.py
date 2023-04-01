import cv2
import mediapipe as mp

from src.utils import angle_between_points

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose


class RepCountVideoProcessor:

    def __init__(self):
        self.rep_state = 'pending'
        self.rep_count = 0
        self.highest_confidence_ankles = None
        self.max_confidence = 0

    def draw(self, frame, landmarks, pose_landmarks):
        bone_connections = [
            (mp_holistic.PoseLandmark.LEFT_KNEE.value, mp_holistic.PoseLandmark.LEFT_HIP.value),
            (mp_holistic.PoseLandmark.RIGHT_KNEE.value, mp_holistic.PoseLandmark.RIGHT_HIP.value),
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

        # use estimated feet if none currently tracked
        if self.highest_confidence_ankles is None:
            bone_connections.append(
                (mp_holistic.PoseLandmark.LEFT_ANKLE.value, mp_holistic.PoseLandmark.LEFT_KNEE.value))
            bone_connections.append(
                (mp_holistic.PoseLandmark.RIGHT_ANKLE.value, mp_holistic.PoseLandmark.RIGHT_KNEE.value))

        # use stored feet position if tracked
        if self.highest_confidence_ankles is not None:
            left_ankle, right_ankle = self.highest_confidence_ankles
            landmarks[mp_holistic.PoseLandmark.LEFT_HEEL.value] = left_ankle
            landmarks[mp_holistic.PoseLandmark.RIGHT_HEEL.value] = right_ankle

            cv2.circle(frame, left_ankle, 5, (0, 0, 255), -1)
            cv2.circle(frame, right_ankle, 5, (0, 0, 255), -1)

        if self.rep_state == 'up':
            left_ankle_idx = mp_holistic.PoseLandmark.LEFT_HEEL.value
            right_ankle_idx = mp_holistic.PoseLandmark.RIGHT_HEEL.value
            left_ankle_confidence = pose_landmarks.landmark[left_ankle_idx].visibility
            right_ankle_confidence = pose_landmarks.landmark[right_ankle_idx].visibility
            average_confidence = (left_ankle_confidence + right_ankle_confidence) / 2

            if self.highest_confidence_ankles is None or average_confidence > self.max_confidence:
                left_ankle = landmarks[left_ankle_idx]
                right_ankle = landmarks[right_ankle_idx]
                self.highest_confidence_ankles = (left_ankle, right_ankle)
                self.max_confidence = average_confidence

        line_color = (0, 255, 0)
        line_thickness = 2

        for connection in bone_connections:
            start, end = connection
            cv2.line(frame, landmarks[start], landmarks[end], line_color, line_thickness)

        # Check if the first "up" state is reached
        first_up_state_reached = self.rep_count > 0

        if self.rep_state == "up" or first_up_state_reached:
            # Draw the connections between knee and ankle only after the first "up" state is reached
            left_ankle = landmarks[mp_holistic.PoseLandmark.LEFT_HEEL.value]
            left_knee = landmarks[mp_holistic.PoseLandmark.LEFT_KNEE.value]
            right_ankle = landmarks[mp_holistic.PoseLandmark.RIGHT_HEEL.value]
            right_knee = landmarks[mp_holistic.PoseLandmark.RIGHT_KNEE.value]

            cv2.line(frame, left_knee, left_ankle, line_color, line_thickness)
            cv2.line(frame, right_knee, right_ankle, line_color, line_thickness)

        left_shoulder = landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value]
        left_hip = landmarks[mp_holistic.PoseLandmark.LEFT_HIP.value]
        left_knee = landmarks[mp_holistic.PoseLandmark.LEFT_KNEE.value]

        right_shoulder = landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value]
        right_hip = landmarks[mp_holistic.PoseLandmark.RIGHT_HIP.value]
        right_knee = landmarks[mp_holistic.PoseLandmark.RIGHT_KNEE.value]

        left_angle = angle_between_points(left_shoulder, left_hip, left_knee)
        right_angle = angle_between_points(right_shoulder, right_hip, right_knee)

        angle_threshold = 100

        if self.rep_state == 'pending' and left_angle < angle_threshold and right_angle < angle_threshold:
            self.rep_state = 'start'
        elif self.rep_state == 'start' and left_angle > angle_threshold and right_angle > angle_threshold:
            self.rep_state = 'up'
        elif self.rep_state == 'up' and left_angle < angle_threshold and right_angle < angle_threshold:
            self.rep_state = 'start'
            self.rep_count += 1

        rep_text = f"Rep: {self.rep_count}"
        state_text = f"State: {self.rep_state}"
        left_angle = f"Left angle: {left_angle}"
        right_angle = f"Right angle: {right_angle}"

        text_color = (255, 255, 255)
        font_scale = 1.5
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_thickness = 2
        line_type = cv2.LINE_AA

        text_size_rep = cv2.getTextSize(rep_text, font, font_scale, text_thickness)[0]
        text_size_state = cv2.getTextSize(state_text, font, font_scale, text_thickness)[0]
        text_size_left_angle = cv2.getTextSize(left_angle, font, font_scale, text_thickness)[0]
        text_size_right_angle = cv2.getTextSize(right_angle, font, font_scale, text_thickness)[0]

        rep_text_y = int(frame.shape[0] * 0.05) + text_size_rep[1]
        state_text_y = rep_text_y + text_size_state[1] + 20
        left_angle_text_y = state_text_y + text_size_left_angle[1] + 20
        right_angle_text_y = left_angle_text_y + text_size_right_angle[1] + 20

        cv2.putText(frame, rep_text, (10, rep_text_y), font, font_scale, text_color, text_thickness, line_type)
        cv2.putText(frame, state_text, (10, state_text_y), font, font_scale, text_color, text_thickness, line_type)
        cv2.putText(frame, left_angle, (10, left_angle_text_y), font, font_scale, text_color, text_thickness, line_type)
        cv2.putText(frame, right_angle, (10, right_angle_text_y), font, font_scale, text_color, text_thickness, line_type)


