import cv2
import mediapipe as mp

from src.utils import create_writer_and_frame_generator, angle_between_points

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose

rep_state = "pending"  # Initial state of the rep
rep_count = 0  # Initial count of the reps

def draw_landmarks(image, landmarks, pose_landmarks):
    global rep_state, rep_count

    if not hasattr(draw_landmarks, 'highest_confidence_ankles'):
        draw_landmarks.highest_confidence_ankles = None
        draw_landmarks.max_confidence = 0

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

    if draw_landmarks.highest_confidence_ankles is None:
        bone_connections.append((mp_holistic.PoseLandmark.LEFT_ANKLE.value, mp_holistic.PoseLandmark.LEFT_KNEE.value))
        bone_connections.append((mp_holistic.PoseLandmark.RIGHT_ANKLE.value, mp_holistic.PoseLandmark.RIGHT_KNEE.value))

    if draw_landmarks.highest_confidence_ankles is not None:
        left_ankle, right_ankle = draw_landmarks.highest_confidence_ankles
        landmarks[mp_holistic.PoseLandmark.LEFT_HEEL.value] = left_ankle
        landmarks[mp_holistic.PoseLandmark.RIGHT_HEEL.value] = right_ankle

        cv2.circle(image, left_ankle, 5, (0, 0, 255), -1)
        cv2.circle(image, right_ankle, 5, (0, 0, 255), -1)

    if rep_state == 'up':
        left_ankle_idx = mp_holistic.PoseLandmark.LEFT_HEEL.value
        right_ankle_idx = mp_holistic.PoseLandmark.RIGHT_HEEL.value
        left_ankle_confidence = pose_landmarks.landmark[left_ankle_idx].visibility
        right_ankle_confidence = pose_landmarks.landmark[right_ankle_idx].visibility
        average_confidence = (left_ankle_confidence + right_ankle_confidence) / 2

        if draw_landmarks.highest_confidence_ankles is None or average_confidence > draw_landmarks.max_confidence:
            left_ankle = landmarks[left_ankle_idx]
            right_ankle = landmarks[right_ankle_idx]
            draw_landmarks.highest_confidence_ankles = (left_ankle, right_ankle)
            draw_landmarks.max_confidence = average_confidence

    # Draw the connections on the image
    line_color = (0, 255, 0)
    line_thickness = 2

    for connection in bone_connections:
        start, end = connection
        cv2.line(image, landmarks[start], landmarks[end], line_color, line_thickness)

    # Check if the first "up" state is reached
    first_up_state_reached = rep_count > 0

    if rep_state == "up" or first_up_state_reached:
        # Draw the connections between knee and ankle only after the first "up" state is reached
        left_ankle = landmarks[mp_holistic.PoseLandmark.LEFT_HEEL.value]
        left_knee = landmarks[mp_holistic.PoseLandmark.LEFT_KNEE.value]
        right_ankle = landmarks[mp_holistic.PoseLandmark.RIGHT_HEEL.value]
        right_knee = landmarks[mp_holistic.PoseLandmark.RIGHT_KNEE.value]

        cv2.line(image, left_knee, left_ankle, line_color, line_thickness)
        cv2.line(image, right_knee, right_ankle, line_color, line_thickness)


    left_shoulder = landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value]
    left_hip = landmarks[mp_holistic.PoseLandmark.LEFT_HIP.value]
    left_knee = landmarks[mp_holistic.PoseLandmark.LEFT_KNEE.value]

    right_shoulder = landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value]
    right_hip = landmarks[mp_holistic.PoseLandmark.RIGHT_HIP.value]
    right_knee = landmarks[mp_holistic.PoseLandmark.RIGHT_KNEE.value]

    left_angle = angle_between_points(left_shoulder, left_hip, left_knee)
    right_angle = angle_between_points(right_shoulder, right_hip, right_knee)

    angle_threshold = 100

    if rep_state == 'pending' and left_angle < angle_threshold and right_angle < angle_threshold:
        rep_state = 'start'
    elif rep_state == 'start' and left_angle > angle_threshold and right_angle > angle_threshold:
        rep_state = 'up'
    elif rep_state == 'up' and left_angle < angle_threshold and right_angle < angle_threshold:
        rep_state = 'start'
        rep_count += 1

    rep_text = f"Rep: {rep_count}"
    state_text = f"State: {rep_state}"
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

    rep_text_y = int(image.shape[0] * 0.05) + text_size_rep[1]
    state_text_y = rep_text_y + text_size_state[1] + 20
    left_angle_text_y = state_text_y + text_size_left_angle[1] + 20
    right_angle_text_y = left_angle_text_y + text_size_right_angle[1] + 20

    cv2.putText(image, rep_text, (10, rep_text_y), font, font_scale, text_color, text_thickness, line_type)
    cv2.putText(image, state_text, (10, state_text_y), font, font_scale, text_color, text_thickness, line_type)
    cv2.putText(image, left_angle, (10, left_angle_text_y), font, font_scale, text_color, text_thickness, line_type)
    cv2.putText(image, right_angle, (10, right_angle_text_y), font, font_scale, text_color, text_thickness, line_type)


def process_rep_count(input_file, output_file, reduce_resolution=False):
    writer, frame_generator = create_writer_and_frame_generator(input_file, output_file)

    with mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8) as pose:
        for frame in frame_generator:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            result = pose.process(image_rgb)
            if result.pose_landmarks:
                # Get the landmarks on the original frame
                landmarks = {}
                for idx, lm in enumerate(result.pose_landmarks.landmark):
                    x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                    landmarks[idx] = (x, y)

                draw_landmarks(frame, landmarks, result.pose_landmarks)

            writer.write(frame)

    writer.release()
    cv2.destroyAllWindows()