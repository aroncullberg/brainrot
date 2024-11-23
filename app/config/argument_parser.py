import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Union

@dataclass
class BrainrotConfig:
    video: Optional[ Path ]
    url: Optional[ str ]
    audio: Path 
    output: Path
    model: str = 'base'
    font_size: int = 24
    force: bool = False
    preview: bool = False
    threads: int = 4
    loglevel: str = 'INFO'
    normalize_audio: bool = False
    speakers: int = 2
    device: str = 'cpu'
    # audio_offset: float = 0.0

    @property
    def video_input(self) -> Union[Path, str]:
        return self.video if self.video else self.url

class ArgumentParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="Video Processing Tool with Whisper Integration",
            formatter_class=argparse.RawTextHelpFormatter
        )
        self._add_arguments()

    def _add_arguments(self):
        # Mutually exclusive but required arguments
        video_group = self.parser.add_mutually_exclusive_group(required=True)
        video_group.add_argument('-v', '--video', type=str,
                                help='Input video file path')
        video_group.add_argument('-u', '--url', type=str,
                                help='Input video URL')

        # Required Arguments
        required = self.parser.add_argument_group('Required Arguments')
        required.add_argument('-a', '--audio', required=True, type=str,
                            help='Input audio file path')
        required.add_argument('-o', '--output', required=True, type=str,
                            help='Output video file path')

        # Model Settings
        model_group = self.parser.add_argument_group('Model Settings')
        model_group.add_argument('-m', '--model', type=str, default='base',
                               choices=['tiny', 'base', 'small', 'medium', 'large', 'turbo'],
                               help='Whisper model size [default: %(default)s]')
        model_group.add_argument('--speakers', type=int, default=2,
                                 help='Number of speakers in the audio [default: %(default)s]')
        model_group.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], 
                                 help='Device to run inference [default: %(default)s]')

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
        processing_group.add_argument('--loglevel', type=str, default='INFO',
                                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                                      help='Set logging level [default: %(default)s]')
            

        # Audio Processing
        audio_group = self.parser.add_argument_group('Audio Processing')
        audio_group.add_argument('--normalize-audio', action='store_true', default=False,
                               help='Normalize audio levels [default: %(default)s]')
        audio_group.add_argument('--audio-offset', type=float, default=0.0,
                               help='Offset audio timing in seconds [default: %(default)s]\n'
                                    '(Positive values delay audio, negative values advance audio)')

    def parse(self) -> BrainrotConfig:
        args = self.parser.parse_args()
        return BrainrotConfig(
            video=Path(args.video).resolve() if args.video else None,
            url=args.url,
            audio=Path(args.audio).resolve(),
            output=Path(args.output).resolve(),
            model=args.model,
            font_size=args.font_size,
            force=args.force,
            preview=args.preview,
            threads=args.threads,
            loglevel=args.loglevel,
            normalize_audio=args.normalize_audio,
            speakers=args.speakers,
            device=args.device
            # audio_offset=args.audio_offset
        )