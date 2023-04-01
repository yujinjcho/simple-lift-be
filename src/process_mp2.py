import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose

def draw_landmarks(image, landmarks):
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


def draw_landmarks_with_circles(image, landmarks, result):
    circle_radius = 5
    circle_color = (0, 255, 0)
    text_color = (255, 255, 255)
    font_scale = 0.5
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_offset = (0, -5)
    rectangle_color = (0, 0, 0)
    rectangle_opacity = 0.6

    for idx, lm in enumerate(result.pose_landmarks.landmark):
        x, y = landmarks[idx]
        cv2.circle(image, (x, y), circle_radius, circle_color, thickness=-1)

        landmark_name = mp_holistic.PoseLandmark(idx).name
        confidence_text = f"{landmark_name}: {lm.visibility:.2f}"

        text_size, _ = cv2.getTextSize(confidence_text, font, font_scale, 1)
        text_width, text_height = text_size

        rectangle_top_left = (x + text_offset[0] - 2, y + text_offset[1] - text_height - 2)
        rectangle_bottom_right = (x + text_offset[0] + text_width + 2, y + text_offset[1] + 2)

        overlay = image.copy()
        cv2.rectangle(overlay, rectangle_top_left, rectangle_bottom_right, rectangle_color, -1)
        cv2.addWeighted(overlay, rectangle_opacity, image, 1 - rectangle_opacity, 0, image)

        cv2.putText(image, confidence_text, (x + text_offset[0], y + text_offset[1]), font, font_scale, text_color, 1)

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
                    draw_landmarks_with_circles(frame, landmarks, result)

                # Write the frame with overlay to the output file
                writer.write(frame)
                frame_write_count += 1

            frame_count += 1

    # Release the video capture, video writer, and close the window
    video_capture.release()
    writer.release()
    cv2.destroyAllWindows()