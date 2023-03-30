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

def process_mp(input_file, output_file):
    video_capture = cv2.VideoCapture(input_file)

    # Get video properties
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))

    # Initialize video writer for the output file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    tracking_confidence = .85
    # with mp_holistic.Holistic(min_detection_confidence=tracking_confidence, min_tracking_confidence=tracking_confidence) as holistic:
    with mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8) as pose:
        while video_capture.isOpened():
            has_frame, frame = video_capture.read()
            if not has_frame:
                break

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

    # Release the video capture, video writer, and close the window
    video_capture.release()
    writer.release()
    cv2.destroyAllWindows()
