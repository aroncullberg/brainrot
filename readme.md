# Video Processing Tool with Speaker Diarization

A command-line tool that processes video files with advanced speaker diarization, transcription, and color-coded subtitle generation. This tool combines FFmpeg, OpenAI Whisper, and Pyannote.audio to create enhanced video content with speaker-identified subtitles.

## Features

- Speaker diarization using Pyannote.audio
- Speech-to-text transcription using OpenAI Whisper
- Color-coded subtitles based on speaker identification
- Customizable font styles and subtitle appearance
- Preview mode for quick testing
- Multi-threaded processing support
- Audio normalization and timing adjustment options

## System Architecture

### Processing Pipeline

The following flowchart illustrates the main processing pipeline:

```mermaid
---
title: Processing Pipeline
---
graph TD
    subgraph Input
        V[Video File] --> VV[Video Processing]
        A[Audio File] --> AP[Audio Processing]
    end

    subgraph Audio_Processing
        AP --> PY[Pyannote.audio]
        PY --> SD[Speaker Diarization]
        SD --> SS[Split Audio by Speaker]
        SS --> ST[Segments with Timestamps]
    end

    subgraph Transcription
        ST --> W[OpenAI Whisper]
        W --> TR[Generate Transcripts]
        TR --> CT[Colored Transcripts by Speaker]
    end

    subgraph Video_Processing
        VV --> EX[Extract Video Stream]
        CT --> FF[FFmpeg Processing]
        EX --> FF
        FF --> OV[Add Colored Subtitles]
    end

    subgraph Output
        OV --> FV[Final Video with Colored Transcripts]
    end
```

### Class Structure

The application is built using the following class structure:

```mermaid
---
title: Class Structure
---
classDiagram
    ArgumentParser --> ProcessingConfig
    VideoProcessor --> ProcessingConfig
    VideoProcessor --> Logger
    VideoProcessor --> AudioManager
    VideoProcessor --> TranscriptionManager
    VideoProcessor --> SubtitleManager
    VideoProcessor --> FFmpegManager

    class ArgumentParser {
        +parse_arguments()
        -validate_paths()
    }

    class VideoProcessor {
        -config: ProcessingConfig
        -logger: Logger
        +process()
        -validate_input_files()
        -create_output_directory()
    }

    class AudioManager {
        -audio_path: Path
        -whisper_model: str
        -pyannote_pipeline
        +extract_segments()
        +get_speaker_diarization()
        -normalize_audio()
        -adjust_timing()
    }

    class TranscriptionManager {
        -whisper_model
        -segments: List
        +generate_transcripts()
        -process_segment()
        -align_timestamps()
    }

    class SubtitleManager {
        -font_size: int
        -colors: Dict
        +generate_subtitles()
        -format_subtitle()
        -assign_colors()
    }

    class FFmpegManager {
        -input_path: Path
        -output_path: Path
        -threads: int
        +process_video()
        -add_subtitles()
        -handle_preview()
    }

    class ProcessingConfig {
        +video_path: Path
        +audio_path: Path
        +output_path: Path
        +model_size: str
        +font_size: int
        +force: bool
        +preview: bool
        +threads: int
        +verbose: bool
        +normalize_audio: bool
        +audio_offset: float
    }

    class Logger {
        -verbose: bool
        +info()
        +error()
        +debug()
        +progress()
    }
```

## Project Structure

```
project/
├── __init__.py
├── main.py
├── config/
│   ├── __init__.py
│   ├── argument_parser.py
│   └── processing_config.py
├── core/
│   ├── __init__.py
│   ├── video_processor.py
│   ├── audio_processor.py
│   └── ffmpeg_manager.py
├── utils/
│   ├── __init__.py
│   └── logger.py
└── tests/
    ├── __init__.py
    ├── test_audio_manager.py
    ├── test_transcription_manager.py
    └── ...
```

## Installation
```bash
# Clone the repository
git clone https://github.com/fetafisken00/brainrot.git
cd brainrot

# Create and activate conda environment from yml file
conda env create -f environment.yml
conda activate brainrot
```
You must have ffmpeg in your path.
Also you need to run it with administrative privileges, or create a pull request with a check for linux and use that.

Note: you need to be able to use admin with conda environment this might be an issue if its not in path(?), sudo is an option but idk what else you can use if you dont want to put in path.

Also create huggingface account and create key https://huggingface.co/pyannote/speaker-diarization-3.1
put it int .env, if no .env run program once.
## Usage

Basic usage:
```bash
python main.py -v input_video.mp4 -a input_audio.wav -o output_video.mp4
```

## Dependencies

- FFmpeg
- OpenAI Whisper
- Pyannote.audio
- Python 3.8+

## License
Haha, funny joke

## Contributing
?