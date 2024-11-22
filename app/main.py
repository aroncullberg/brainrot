#!/usr/bin/env python3
from collections import defaultdict
import os
from pathlib import Path
from typing import List
import cv2
import numpy as np
from audio.transcribed_segment import TranscribedSegment
from dotenv import load_dotenv
from config.env import loadAPIKeys
from config.argument_parser import ArgumentParser, BrainrotConfig
from audio.audio_manager import AudioManager
import subprocess
import logging
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import math

def get_video_properties(video_path: Path):
    """Get video properties using OpenCV."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    return width, height, fps, frame_count


def add_text_overlay(input_video: Path, input_audio: Path, output_video: Path, 
                    segments: List[TranscribedSegment], font_size: int = 90):
    """Add text overlay to video using OpenCV efficiently and stop at the last segment."""
    # Get video properties
    width, height, fps, frame_count = get_video_properties(input_video)
    
    # Setup color mapping for speakers
    colors = [(255, 255, 255), (0, 255, 255), (255, 255, 0), 
              (0, 255, 0), (255, 192, 203), (0, 165, 255)]  # BGR format
    speaker_colors = defaultdict(lambda: (255, 255, 255))
    
    unique_speakers = {seg.speaker for seg in segments if seg.speaker}
    for i, speaker in enumerate(unique_speakers):
        speaker_colors[speaker] = colors[i % len(colors)]
    
    # Temp output file for video without audio
    temp_output = str(output_video).replace('.mp4', '_temp.mp4')
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
    

    cap = cv2.VideoCapture(str(input_video))
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = font_size / 24
    bold_thickness = 10  # Thicker text for bold effect

    
    # Get the end time of the last segment
    last_segment_end = max(seg.end for seg in segments)
    frames_to_process = math.ceil(last_segment_end * fps)
    
    print(f"Processing frames up to {last_segment_end:.2f} seconds ({frames_to_process} frames)...")
    for frame_number in range(frames_to_process):
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = frame_number / fps
        
        # Stop processing if the current time exceeds the last segment's end time
        if current_time > last_segment_end:
            print("Reached the end of the last segment. Stopping processing.")
            break
        
        # Find active segments for the current time
        active_segments = [
            seg for seg in segments 
            if seg.start <= current_time <= seg.end
        ]
        
        # Add text for each active segment
        for segment in active_segments:
            text = segment.text.strip()
            color = speaker_colors[segment.speaker]
            

            # Wrap text to fit video width
            wrapped_text = wrap_text_to_fit(text, width, font, font_scale, bold_thickness)

            # print(f"Wrapped text for '{text}': {wrapped_text}")  # Debugging output

            
            # Calculate starting position for vertical alignment
            y_start = (height // 2) - (len(wrapped_text) * 50 // 2)  # Center vertically

            for i, line in enumerate(wrapped_text):
                # Get text size for alignment
                (text_width, text_height), baseline = cv2.getTextSize(
                    line, font, font_scale, bold_thickness)
                # print(f"Text width: {text_width}, height: {text_height}")  # Debugging output<<
                # print(f"Text baseline: {baseline}")  # Debugging output
                # print(f"Text: {text}")  # Debugging output
                
                # Center text horizontally
                x = max((width - text_width) // 2, 0)  # Ensure x-coordinate is within bounds
                y = min(max(y_start + i * (text_height + 10), 0), height - text_height)  # Keep y within bounds


                # print(f"Drawing text '{line}' at position ({x}, {y})")  # Debugging output
                # # print vidoe withd andheight
                # print(f"Video width: {width}, height: {height}")  # Debugging output
                
                # Draw black border (stroke)
                cv2.putText(
                    frame,
                    line,
                    (x, y),
                    font,
                    font_scale,
                    (0, 0, 0),  # Black color for the border
                    bold_thickness + 10,  # Border thickness (slightly larger)
                    cv2.LINE_AA
                )

                # Draw the main text
                cv2.putText(
                    frame,
                    line,
                    (x, y),
                    font,
                    font_scale,
                    color,  # Main text color
                    bold_thickness,
                    cv2.LINE_AA
                )
        # Write frame to output video
        out.write(frame)
        
        # Log progress relative to last_segment_end
        if frame_number % 100 == 0 or frame_number == frames_to_process - 1:
            progress = (frame_number + 1) / frames_to_process * 100
            print(f"Processed {frame_number + 1}/{frames_to_process} frames ({progress:.2f}% complete)")
    
    cap.release()
    out.release()
    
    print("Merging audio...")
    # Combine video and audio using FFmpeg
    cmd = [
        'ffmpeg', '-y',
        '-i', temp_output,
        '-i', str(input_audio),
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-strict', 'experimental',
        str(output_video)
    ]
    subprocess.run(cmd, check=True)
    
    # Remove temporary file
    os.remove(temp_output)
    print("Processing complete!")

    # Remove temporary file
def wrap_text_to_fit(text, max_width, font, font_scale, thickness):
    """
    Wrap text into lines that fit within the given max_width.
    Returns a list of lines.
    """
    words = text.split()
    lines = []
    current_line = words[0]

    for word in words[1:]:
        # Check the width of the current line plus the new word
        test_line = f"{current_line} {word}"
        (text_width, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
        
        if text_width > max_width:
            # Add current line to lines and start a new line
            lines.append(current_line)
            current_line = word
        else:
            # Add the word to the current line
            current_line = test_line

    # Add the last line

    lines.append(current_line)
    return lines

def main(config: BrainrotConfig):
    # Initialize logger
    logging.basicConfig(level=config.loglevel)
    LOG = logging.getLogger(__name__)

    # Load environment variables (API keys)
    loadAPIKeys(LOG)

    # Initialize processor
    audio = AudioManager(
        whisper_model_size=config.model,
        hf_token=os.environ.get("HF_TOKEN"),
        num_speakers=config.speakers,
        device=config.device,
        threads=config.threads,
        logger=LOG
    )

    segments = audio.process_file(config.audio)

    # Log results
    LOG.info("Speaker \t| Text")
    for seg in segments:
        LOG.info(f"{seg.speaker} \t| {seg.text}")
    
    add_text_overlay(config.video, config.audio, config.output, segments)


if __name__ == "__main__":
    parser = ArgumentParser()
    config = parser.parse()
    main(config)