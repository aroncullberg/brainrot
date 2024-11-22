#!/usr/bin/env python3

import argparse
from pathlib import Path

def parse_arguments():
    """Parse command line arguments for the video processing script."""
    parser = argparse.ArgumentParser(
        description="Video Processing Tool with Whisper Integration",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Required Arguments
    required = parser.add_argument_group('Required Arguments')
    required.add_argument(
        '-v', '--video',
        required=True,
        type=str,
        help='Input video file path'
    )
    required.add_argument(
        '-a', '--audio',
        required=True,
        type=str,
        help='Input audio file path'
    )
    required.add_argument(
        '-o', '--output',
        required=True,
        type=str,
        help='Output video file path'
    )

    # Model Settings
    model_group = parser.add_argument_group('Model Settings')
    model_group.add_argument(
        '-m', '--model',
        type=str,
        default='base',
        choices=['tiny', 'base', 'small', 'medium', 'large'],
        help='Whisper model size [default: %(default)s]'
    )

    # Caption Styling
    styling_group = parser.add_argument_group('Caption Styling')
    styling_group.add_argument(
        '--font-size',
        type=int,
        default=24,
        help='Caption font size in pixels [default: %(default)s]'
    )

    # Processing Control
    processing_group = parser.add_argument_group('Processing Control')
    processing_group.add_argument(
        '--force',
        action='store_true',
        default=False,
        help='Overwrite output file if exists [default: %(default)s]'
    )
    processing_group.add_argument(
        '--preview',
        action='store_true',
        default=False,
        help='Process only first 30 seconds [default: %(default)s]'
    )
    processing_group.add_argument(
        '--threads',
        type=int,
        default=4,
        help='Number of CPU threads [default: %(default)s]\nUse 0 for auto-detection'
    )
    processing_group.add_argument(
        '--verbose',
        action='store_true',
        default=False,
        help='Show detailed processing information [default: %(default)s]'
    )

    # Audio Processing
    audio_group = parser.add_argument_group('Audio Processing')
    audio_group.add_argument(
        '--normalize-audio',
        action='store_true',
        default=False,
        help='Normalize audio levels [default: %(default)s]'
    )
    audio_group.add_argument(
        '--audio-offset',
        type=float,
        default=0.0,
        help='Offset audio timing in seconds [default: %(default)s]\n'
             '(Positive values delay audio, negative values advance audio)'
    )

    args = parser.parse_args()

    # Validate paths
    args.video = Path(args.video).resolve()
    args.audio = Path(args.audio).resolve()
    args.output = Path(args.output).resolve()

    return args

if __name__ == "__main__":
    args = parse_arguments()
    
    print("Parsed arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")