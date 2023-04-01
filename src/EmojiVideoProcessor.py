import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose

class EmojiVideoProcessor:
    def __init__(self, emoji_path):
        self.emoji = cv2.imread(emoji_path, cv2.IMREAD_UNCHANGED)

    def draw(self, frame, landmarks, pose_landmarks):
        nose_x, nose_y = landmarks[mp_holistic.PoseLandmark.NOSE.value]
        left_ear_x, left_ear_y = landmarks[mp_holistic.PoseLandmark.LEFT_EAR.value]
        right_ear_x, right_ear_y = landmarks[mp_holistic.PoseLandmark.RIGHT_EAR.value]

        ear_distance = int(((right_ear_x - left_ear_x) ** 2 + (right_ear_y - left_ear_y) ** 2) ** 0.5)

        emoji_size = ear_distance * 3
        resized_emoji = cv2.resize(self.emoji, (emoji_size, emoji_size), interpolation=cv2.INTER_AREA)

        emoji_width, emoji_height, _ = resized_emoji.shape
        top_left_x = nose_x - emoji_width // 2
        top_left_y = nose_y - emoji_height // 2

        bottom_right_x = top_left_x + emoji_width
        bottom_right_y = top_left_y + emoji_height

        # Ensure the emoji is within the frame boundaries
        if top_left_x < 0 or top_left_y < 0 or bottom_right_x > frame.shape[1] or bottom_right_y > frame.shape[0]:
            return

        for i in range(emoji_height):
            for j in range(emoji_width):
                if resized_emoji[i, j, 3] > 128:  # Check the alpha channel to ensure transparency
                    frame[top_left_y + i, top_left_x + j] = resized_emoji[i, j, :3]
