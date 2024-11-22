import argparse
from pathlib import Path
from dataclasses import dataclass

@dataclass
class VideoProcessingConfig:
    video: Path
    audio: Path 
    output: Path
    model: str = 'base'
    font_size: int = 24
    force: bool = False
    preview: bool = False
    threads: int = 4
    verbose: bool = False
    normalize_audio: bool = False
    audio_offset: float = 0.0

class ArgumentParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="Video Processing Tool with Whisper Integration",
            formatter_class=argparse.RawTextHelpFormatter
        )
        self._add_arguments()

    def _add_arguments(self):
        required = self.parser.add_argument_group('Required Arguments')
        required.add_argument('-v', '--video', required=True, type=str,
                            help='Input video file path')
        required.add_argument('-a', '--audio', required=True, type=str,
                            help='Input audio file path')
        required.add_argument('-o', '--output', required=True, type=str,
                            help='Output video file path')

        # Model Settings
        model_group = self.parser.add_argument_group('Model Settings')
        model_group.add_argument('-m', '--model', type=str, default='base',
                               choices=['tiny', 'base', 'small', 'medium', 'large'],
                               help='Whisper model size [default: %(default)s]')

        # Caption Styling
        styling_group = self.parser.add_argument_group('Caption Styling')
        styling_group.add_argument('--font-size', type=int, default=24,
                                 help='Caption font size in pixels [default: %(default)s]')

        # Processing Control
        processing_group = self.parser.add_argument_group('Processing Control')
        processing_group.add_argument('--force', action='store_true', default=False,
                                    help='Overwrite output file if exists [default: %(default)s]')
        processing_group.add_argument('--preview', action='store_true', default=False,
                                    help='Process only first 30 seconds [default: %(default)s]')
        processing_group.add_argument('--threads', type=int, default=4,
                                    help='Number of CPU threads [default: %(default)s]\nUse 0 for auto-detection')
        processing_group.add_argument('--verbose', action='store_true', default=False,
                                    help='Show detailed processing information [default: %(default)s]')

        # Audio Processing
        audio_group = self.parser.add_argument_group('Audio Processing')
        audio_group.add_argument('--normalize-audio', action='store_true', default=False,
                               help='Normalize audio levels [default: %(default)s]')
        audio_group.add_argument('--audio-offset', type=float, default=0.0,
                               help='Offset audio timing in seconds [default: %(default)s]\n'
                                    '(Positive values delay audio, negative values advance audio)')

    def parse(self) -> VideoProcessingConfig:
        args = self.parser.parse_args()
        return VideoProcessingConfig(
            video=Path(args.video).resolve(),
            audio=Path(args.audio).resolve(),
            output=Path(args.output).resolve(),
            model=args.model,
            font_size=args.font_size,
            force=args.force,
            preview=args.preview,
            threads=args.threads,
            verbose=args.verbose,
            normalize_audio=args.normalize_audio,
            audio_offset=args.audio_offset
        )