#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
import subprocess
import tempfile

from dotenv import load_dotenv
from typing import List
from core import AudioProcessor, TranscribedSegment
from config.argument_parser import ArgumentParser

from typing import List
import subprocess
from pathlib import Path

import tempfile

import subprocess
import json
from typing import List
from pathlib import Path
import re
from datetime import timedelta

def get_video_metadata(video_path: str) -> dict:
    """Get video duration, dimensions and fps using ffprobe"""
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate,duration',
        '-show_entries', 'format=duration',
        '-of', 'json',
        video_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout)
    
    # Parse framerate (comes as fraction like '24000/1001')
    fps_num, fps_den = map(int, data['streams'][0]['r_frame_rate'].split('/'))
    fps = fps_num / fps_den
    
    return {
        'width': data['streams'][0]['width'],
        'height': data['streams'][0]['height'],
        'fps': fps,
        'duration': float(data['format']['duration'])
    }

def render_text_on_video(
    video_path: str,
    segments: List[TranscribedSegment],
    output_path: str,
    font_size: int = 24
) -> None:
    """Render transcript using FFmpeg with drawtext filter and progress tracking"""
    
    # Get video info
    metadata = get_video_metadata(video_path)
    final_duration = max(segment.end for segment in segments)
    
    print(f"\nVideo Information:")
    print(f"Resolution: {metadata['width']}x{metadata['height']}")
    print(f"Aspect Ratio: {metadata['width']}:{metadata['height']}")
    print(f"FPS: {metadata['fps']:.2f}")
    print(f"Original Duration: {timedelta(seconds=int(metadata['duration']))}")
    print(f"Output Duration: {timedelta(seconds=int(final_duration))}\n")

    # Create filter complex for text segments
    filter_complex = []
    for segment in segments:
        # Properly escape text for FFmpeg
        text = segment.text.strip()
        text = text.replace('\\', '\\\\')
        text = text.replace(':', '\\:')
        text = text.replace('\'', '\\\'')
        text = text.replace('[', '\\[').replace(']', '\\]')
        text = text.replace(',', '\\,')
        text = text.replace(';', '\\;')
        
        filter_str = (
            f"drawtext=text='{text}'"
            f":fontsize={font_size}"
            ":fontcolor=white"
            ":box=1"
            ":boxcolor=black@0.6"
            ":boxborderw=10"
            ":x=(w-text_w)/2"
            ":y=h-text_h-40"
            f":enable='between(t,{segment.start},{segment.end})'"
        )
        filter_complex.append(filter_str)

    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-vf', ','.join(filter_complex),
        '-t', str(final_duration),
        '-c:v', 'h264',
        '-c:a', 'aac',
        '-preset', 'ultrafast',
        '-threads', '0',
        '-progress', 'pipe:1',    # Output progress to stdout
        '-y',
        output_path
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )

    # Track progress
    duration_ms = final_duration * 1000
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        
        if output:
            # Parse progress info
            if 'out_time_ms=' in output:
                time = int(output.split('out_time_ms=')[1])
                progress = min((time / duration_ms) * 100, 100)
                print(f"\rProgress: {progress:.1f}%", end='', flush=True)

    if process.returncode != 0:
        error = process.stderr.read()
        raise RuntimeError(f"FFmpeg failed: {error}")
    
    print("\nRendering complete!")

def main(args):
    # parser = ArgumentParser()
    # config = parser.parse()

    load_dotenv()

    # Initialize processor
    processor = AudioProcessor(
        whisper_model_size="tiny",
        hf_token=os.getenv("HUGGINGFACE_TOKEN"),
        num_speakers=2
    )

    # Process audio file
    segments = processor.process_file("audio.wav")

    # Print results
    for seg in segments:
        print(seg)
        print("\n")
        # print(f"[{seg.speaker}] {seg.start:.1f}s - {seg.end:.1f}s: {seg.text}")
    
    render_text_on_video("video.mp4", segments, "output.mp4", font_size=24)

if __name__ == "__main__":
    args = None #parse_arguments()
    
    main(args)