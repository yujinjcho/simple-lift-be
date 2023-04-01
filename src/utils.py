import cv2
import numpy as np

def create_writer_and_frame_generator(input_file, output_file):
    video_capture = cv2.VideoCapture(input_file)

    # Get video properties
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))

    # Reduce frame rate by a factor of 1.5
    frame_rate_reduction_factor = 1.5
    target_fps = int(fps / frame_rate_reduction_factor)

    # Initialize video writer for the output file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_file, fourcc, target_fps, (width, height))

    def frame_generator():
        nonlocal video_capture
        while video_capture.isOpened():
            has_frame, frame = video_capture.read()
            if not has_frame:
                break

            if int(video_capture.get(cv2.CAP_PROP_POS_FRAMES)) % frame_rate_reduction_factor < 1:
                yield frame

        video_capture.release()

    return writer, frame_generator()


def angle_between_points(a, b, c, round_by=5):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    degrees = np.degrees(angle)
    rounded_degrees = 5 * round(degrees / round_by)

    return rounded_degrees