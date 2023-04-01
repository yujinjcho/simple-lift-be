import os
import argparse

from src.ConfidenceIntervalVideoProcessor import ConfidenceIntervalVideoProcessor
from src.RepCountVideoProcessor import RepCountVideoProcessor
from src.pipeline import process_video

confidence_interval_type = 'confidence'
rep_count_type = 'rep_count'

if __name__ == '__main__':
    print('starting main')

    parser = argparse.ArgumentParser(description='Process a video file for deadlift analysis.')
    parser.add_argument('-f', '--input_filename', type=str, required=True, help='Filename of the input video file (located in data/raw directory).')
    parser.add_argument('-t', '--processor_type', type=str, choices=[confidence_interval_type, rep_count_type], required=True, help='Type of video processor to use.')

    args = parser.parse_args()

    input_file = os.path.join('data', 'raw', args.input_filename)
    file_root, file_ext = os.path.splitext(args.input_filename)
    output_filename = f"{file_root}_{args.processor_type}{file_ext}"
    output_directory = os.path.join('data', 'processed')
    output_file = os.path.join(output_directory, output_filename)
    processor_type = args.processor_type

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    if processor_type == rep_count_type:
        video_processor = RepCountVideoProcessor()
    elif processor_type == confidence_interval_type:
        video_processor = ConfidenceIntervalVideoProcessor()
    else:
        raise ValueError(f'Unknown processor type: {processor_type}')

    process_video(input_file, output_file, video_processor)
