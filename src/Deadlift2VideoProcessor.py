import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose


class Deadlift2VideoProcessor:
    def __init__(self, *processors):
        self.processors = processors

    def draw(self, frame, landmarks, pose_landmarks):
        for processor in self.processors:
            processor.draw(frame, landmarks, pose_landmarks)

