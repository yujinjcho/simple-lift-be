import os
import argparse
from enum import Enum

from src.ConfidenceIntervalVideoProcessor import ConfidenceIntervalVideoProcessor
from src.Deadlift2VideoProcessor import Deadlift2VideoProcessor
from src.DeadliftVideoProcessor import DeadliftVideoProcessor
from src.ElbowTrackingVideoProcessor import ElbowTrackingVideoProcessor
from src.EmojiVideoProcessor import EmojiVideoProcessor
from src.HeadPointsVideoProcessor import HeadPointsVideoProcessor
from src.DeadliftRepCountVideoProcessor import DeadliftRepCountVideoProcessor
from src.MomentArmVideoProcessor import MomentArmVideoProcessor
from src.OutlineVideoProcessor import OutlineVideoProcessor
from src.ShoulderToKneeVideoProcessor import ShoulderToKneeVideoProcessor
from src.ShoulderTrackingVideoProcessor import ShoulderTrackingVideoProcessor
from src.pipeline import process_video

class ProcessorType(Enum):
    confidence = 'confidence'
    dl_rep_count = 'dl_rep'
    dl = 'dl'
    dl2 = 'dl2'
    outline = 'outline'
    shoulder_knee = 'shoulder_knee'
    head = 'head'
    emoji = 'emoji'
    shoulder_track = 'shoulder_track'
    elbow_track = 'elbow_track'
    moment_arm = 'moment_arm'

processor_types = [processor_type.value for processor_type in ProcessorType.__members__.values()]

if __name__ == '__main__':
    print('starting main')

    parser = argparse.ArgumentParser(description='Process a video file for deadlift analysis.')
    parser.add_argument('-f', '--input_filename', type=str, required=True,
                        help='Filename of the input video file (located in data/raw directory).')
    parser.add_argument('-t', '--processor_type', type=str, choices=processor_types, required=True,
                        help='Type of video processor to use.')

    args = parser.parse_args()

    input_file = os.path.join('data', 'raw', args.input_filename)
    file_root, file_ext = os.path.splitext(args.input_filename)
    output_filename = f"{file_root}_{args.processor_type}{file_ext}"
    output_directory = os.path.join('data', 'processed')
    output_file = os.path.join(output_directory, output_filename)

    if args.processor_type not in processor_types:
        raise ValueError(f"Invalid processor type: {args.processor_type}")
    processor_type = ProcessorType[args.processor_type]

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    if processor_type == ProcessorType.dl_rep_count:
        video_processor = DeadliftRepCountVideoProcessor()
    elif processor_type == ProcessorType.dl:
        video_processor = DeadliftVideoProcessor(ShoulderTrackingVideoProcessor())
    elif processor_type == ProcessorType.dl2:
        video_processor = Deadlift2VideoProcessor(ElbowTrackingVideoProcessor(), MomentArmVideoProcessor())
    elif processor_type == ProcessorType.confidence:
        video_processor = ConfidenceIntervalVideoProcessor()
    elif processor_type == ProcessorType.shoulder_knee:
        video_processor = ShoulderToKneeVideoProcessor()
    elif processor_type == ProcessorType.head:
        video_processor = HeadPointsVideoProcessor()
    elif processor_type == ProcessorType.outline:
        video_processor = OutlineVideoProcessor()
    elif processor_type == ProcessorType.emoji:
        video_processor = EmojiVideoProcessor('emoji.png')
    elif processor_type == ProcessorType.shoulder_track:
        video_processor = ShoulderTrackingVideoProcessor()
    elif processor_type == ProcessorType.elbow_track:
        video_processor = ElbowTrackingVideoProcessor()
    elif processor_type == ProcessorType.moment_arm:
        video_processor = MomentArmVideoProcessor()
    else:
        raise ValueError(f'Unknown processor type: {processor_type}')

    process_video(input_file, output_file, video_processor)
