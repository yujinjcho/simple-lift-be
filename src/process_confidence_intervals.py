import cv2
import mediapipe as mp

from src.utils import create_writer_and_frame_generator

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose

def process_confidence(input_file, output_file):
    writer, frame_generator = create_writer_and_frame_generator(input_file, output_file)

    with mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8) as pose:
        for frame in frame_generator:
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

    # Release the video capture, video writer, and close the window
    writer.release()
    cv2.destroyAllWindows()

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
