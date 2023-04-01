import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose

rep_state = "pending"  # Initial state of the rep
rep_count = 0  # Initial count of the reps

def angle_between_points(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)


def draw_landmarks(image, landmarks):
    global rep_state, rep_count

    bone_connections = [
        (mp_holistic.PoseLandmark.LEFT_ANKLE.value, mp_holistic.PoseLandmark.LEFT_KNEE.value),
        (mp_holistic.PoseLandmark.LEFT_KNEE.value, mp_holistic.PoseLandmark.LEFT_HIP.value),
        (mp_holistic.PoseLandmark.RIGHT_ANKLE.value, mp_holistic.PoseLandmark.RIGHT_KNEE.value),
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

    # Draw the connections on the image
    line_color = (0, 255, 0)
    line_thickness = 2

    for connection in bone_connections:
        start, end = connection
        cv2.line(image, landmarks[start], landmarks[end], line_color, line_thickness)

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


def process_mp(input_file, output_file, reduce_resolution=False):
    video_capture = cv2.VideoCapture(input_file)

    # Get video properties
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))

    if reduce_resolution:
        # Reduce resolution for mobile devices
        target_width = 480
        target_height = int(height * (target_width / width))
    else:
        target_width = width
        target_height = height

    # Reduce frame rate by a factor of 1.5
    frame_rate_reduction_factor = 1.5
    target_fps = int(fps / frame_rate_reduction_factor)

    # Initialize video writer for the output file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_file, fourcc, target_fps, (target_width, target_height))

    frame_count = 0
    frame_write_count = 0

    with mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8) as pose:
        while video_capture.isOpened():
            has_frame, frame = video_capture.read()
            if not has_frame:
                break

            # Process frames based on the frame rate reduction factor
            if frame_count % frame_rate_reduction_factor < 1:
                if reduce_resolution:
                    # Resize the frame
                    frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

                # Convert the frame to RGB
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process the image and get landmarks
                result = pose.process(image_rgb)

                if result.pose_landmarks:
                    # Get the landmarks on the original frame
                    landmarks = {}
                    for idx, lm in enumerate(result.pose_landmarks.landmark):
                        x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                        landmarks[idx] = (x, y)

                    # Draw landmarks on the frame
                    draw_landmarks(frame, landmarks)

                # Write the frame with overlay to the output file
                writer.write(frame)
                frame_write_count += 1

            frame_count += 1

    # Release the video capture, video writer, and close the window
    video_capture.release()
    writer.release()
    cv2.destroyAllWindows()