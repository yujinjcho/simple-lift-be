import cv2
import mediapipe as mp

from src.utils import create_writer_and_frame_generator

mp_pose = mp.solutions.pose


def run_pipeline():
    # prior
    # user uploads to s3
    # on completion, create video_upload record
    # user_id, lift_id, status (pending)

    # fetch video_upload for lift on detail page
    # if complete, show button to download/play lift / also add option to download (if easy)

    # get pending video uploads

    # for each pending video
    # download video locally (data/user/lift.ext)
    # process video (data/processed/user/lift.ext)
    # upload to s3 'processed/user/lift_id.ext'
    # delete original upload
    # update video_upload (status:complete)

    # Steps 1:
    # create video_upload table
    # frontend: after upload, create a record
    # run pipeline: integrate with supabase
    # run pipeline: get pending video_upload
    return


def process_video(input_file, output_file, video_processor):
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

                video_processor.draw(frame, landmarks, result.pose_landmarks)

            writer.write(frame)

    writer.release()
