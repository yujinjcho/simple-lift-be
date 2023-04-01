import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose


class HeadPointsVideoProcessor:

    def draw(self, frame, landmarks, pose_landmarks):
        circle_radius = 5
        circle_color = (0, 255, 0)
        text_color = (255, 255, 255)
        font_scale = 0.5
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_offset = (0, -5)
        rectangle_color = (0, 0, 0)
        rectangle_opacity = 0.6

        neck_up_landmark_indices = [
            mp_holistic.PoseLandmark.NOSE.value,
            mp_holistic.PoseLandmark.LEFT_EYE_INNER.value,
            mp_holistic.PoseLandmark.LEFT_EYE.value,
            mp_holistic.PoseLandmark.LEFT_EYE_OUTER.value,
            mp_holistic.PoseLandmark.RIGHT_EYE_INNER.value,
            mp_holistic.PoseLandmark.RIGHT_EYE.value,
            mp_holistic.PoseLandmark.RIGHT_EYE_OUTER.value,
            mp_holistic.PoseLandmark.LEFT_EAR.value,
            mp_holistic.PoseLandmark.RIGHT_EAR.value,
            mp_holistic.PoseLandmark.MOUTH_LEFT.value,
            mp_holistic.PoseLandmark.MOUTH_RIGHT.value,
        ]

        for idx in neck_up_landmark_indices:
            x, y = landmarks[idx]
            cv2.circle(frame, (x, y), circle_radius, circle_color, thickness=-1)

            landmark_name = mp_holistic.PoseLandmark(idx).name
            confidence_text = f"{landmark_name}: {pose_landmarks.landmark[idx].visibility:.2f}"

            text_size, _ = cv2.getTextSize(confidence_text, font, font_scale, 1)
            text_width, text_height = text_size

            rectangle_top_left = (x + text_offset[0] - 2, y + text_offset[1] - text_height - 2)
            rectangle_bottom_right = (x + text_offset[0] + text_width + 2, y + text_offset[1] + 2)

            overlay = frame.copy()
            cv2.rectangle(overlay, rectangle_top_left, rectangle_bottom_right, rectangle_color, -1)
            cv2.addWeighted(overlay, rectangle_opacity, frame, 1 - rectangle_opacity, 0, frame)

            cv2.putText(frame, confidence_text, (x + text_offset[0], y + text_offset[1]), font, font_scale,
                        text_color, 1)
