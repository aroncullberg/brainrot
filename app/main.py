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
from PIL import Image, ImageDraw, ImageFont


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

def wrap_text_pil(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
    """Wrap text to fit within max_width using PIL's font metrics."""
    words = text.split()
    lines = []
    current_line = words[0]
    
    for word in words[1:]:
        test_line = f"{current_line} {word}"
        bbox = font.getbbox(test_line)
        if bbox[2] > max_width:  # bbox[2] is the width
            lines.append(current_line)
            current_line = word
        else:
            current_line = test_line
    
    lines.append(current_line)
    return lines

def cv2_to_pil(cv2_image):
    """Convert CV2 image to PIL Image."""
    cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv2_image)

def pil_to_cv2(pil_image):
    """Convert PIL Image to CV2 image."""
    cv2_image = np.array(pil_image)
    return cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR)

def add_text_overlay(input_video: Path, input_audio: Path, output_video: Path, 
                    segments: List[TranscribedSegment], font_size: int = 90):
    """Add text overlay to video using PIL for custom font rendering."""
    # Get video properties
    width, height, fps, frame_count = get_video_properties(input_video)
    
    # Setup color mapping for speakers
    colors = [(255, 255, 255), (255, 255, 0), (0, 255, 255),
             (0, 255, 0), (255, 192, 203), (255, 165, 0)]  # RGB format
    speaker_colors = defaultdict(lambda: (255, 255, 255))
    
    unique_speakers = {seg.speaker for seg in segments if seg.speaker}
    for i, speaker in enumerate(unique_speakers):
        speaker_colors[speaker] = colors[i % len(colors)]
    
    # Load custom font
    try:
        font = ImageFont.truetype("sfuidisplay_bold.ttf", font_size)
    except OSError:
        print("Warning: SF UI Display Bold font not found, using default font")
        font = ImageFont.load_default()
    
    # Temp output file for video without audio
    temp_output = str(output_video).replace('.mp4', '_temp.mp4')
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
    
    cap = cv2.VideoCapture(str(input_video))
    
    # Get the end time of the last segment
    last_segment_end = max(seg.end for seg in segments)
    frames_to_process = math.ceil(last_segment_end * fps)
    
    print(f"Processing frames up to {last_segment_end:.2f} seconds ({frames_to_process} frames)...")
    
    # Calculate stroke width based on font size
    stroke_width = max(3, font_size // 25)
    
    for frame_number in range(frames_to_process):
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = frame_number / fps
        
        if current_time > last_segment_end:
            print("Reached the end of the last segment. Stopping processing.")
            break
        
        # Convert frame to PIL Image for text rendering
        pil_frame = cv2_to_pil(frame)
        draw = ImageDraw.Draw(pil_frame)
        
        # Find active segments for the current time
        active_segments = [
            seg for seg in segments 
            if seg.start <= current_time <= seg.end
        ]
        
        # Add text for each active segment
        for segment in active_segments:
            text = segment.text.strip()
            color = speaker_colors[segment.speaker]
            
            # Wrap text to fit video width (leaving margins)
            margin = width * 0.1  # 10% margin on each side
            wrapped_text = wrap_text_pil(text, font, width - 2 * margin)
            
            # Calculate text block height
            line_spacing = font_size * 0.3  # 30% of font size
            total_height = len(wrapped_text) * (font_size + line_spacing)
            
            # Position text block in lower third of screen
            y_start = height - total_height - height * 0.2  # 20% from bottom
            
            # Draw each line
            for i, line in enumerate(wrapped_text):
                # Get text width for centering
                bbox = font.getbbox(line)
                text_width = bbox[2] - bbox[0]
                x = (width - text_width) // 2
                y = y_start + i * (font_size + line_spacing)
                
                # Draw text stroke (outline)
                for offset_x in range(-stroke_width, stroke_width + 1):
                    for offset_y in range(-stroke_width, stroke_width + 1):
                        draw.text(
                            (x + offset_x, y + offset_y), 
                            line, 
                            font=font,
                            fill=(0, 0, 0)  # Black outline
                        )
                
                # Draw main text
                draw.text(
                    (x, y),
                    line,
                    font=font,
                    fill=color
                )
        
        # Convert back to CV2 format and write
        cv2_frame = pil_to_cv2(pil_frame)
        out.write(cv2_frame)
        
        # Log progress
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