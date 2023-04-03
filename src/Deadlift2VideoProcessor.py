import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose


class Deadlift2VideoProcessor:
    def __init__(self, elbow_processor, moment_arm_processor):
        self.elbow_processor = elbow_processor
        self.moment_arm_processor = moment_arm_processor

    def draw(self, frame, landmarks, pose_landmarks):
        self.elbow_processor.draw(frame, landmarks, pose_landmarks)
        self.moment_arm_processor.draw(frame, landmarks, pose_landmarks)

