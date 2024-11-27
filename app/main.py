#!/usr/bin/env python3
from collections import defaultdict
import os
from pathlib import Path
from typing import List, Tuple
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
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from concurrent.futures import ThreadPoolExecutor
import threading


from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import threading
from typing import List, Dict, Tuple

class TextRenderer:
    """Cache text renderings to avoid redundant operations."""
    def __init__(self, font_path: str, font_size: int, width: int, height: int):
        try:
            self.font = ImageFont.truetype(font_path, font_size)
        except OSError:
            print("Warning: Custom font not found, using default font")
            self.font = ImageFont.load_default()
            
        self.font_size = font_size
        self.width = width
        self.height = height
        
        # Border styling options
        self.stroke_width = max(3, font_size // 25)
        self.border_color = (0, 0, 0, 255)  # Black border by default
        self.border_opacity = 255  # Full opacity
        self.blur_radius = 0  # No blur by default
        self.double_border = False  # Single border by default
        self.outer_border_color = (128, 128, 128, 255)  # Grey outer border
        self.outer_stroke_width = 2  # Width of outer border
        
        self.cache = {}
        self.cache_lock = threading.Lock()

    def wrap_text_pil(self, text: str, max_width: int) -> List[str]:
        """Wrap text to fit within max_width using PIL's font metrics."""
        words = text.split()
        if not words:
            return []
            
        lines = []
        current_line = words[0]
        
        for word in words[1:]:
            test_line = f"{current_line} {word}"
            bbox = self.font.getbbox(test_line)
            if bbox[2] > max_width:
                lines.append(current_line)
                current_line = word
            else:
                current_line = test_line
        
        lines.append(current_line)
        return lines

    def set_border_style(self, 
                        color: Tuple[int, int, int] = (0, 0, 0),
                        opacity: int = 255,
                        stroke_width: int = None,
                        blur_radius: int = 0,
                        double_border: bool = False,
                        outer_color: Tuple[int, int, int] = (128, 128, 128),
                        outer_width: int = 2):
        """Configure border styling options."""
        self.border_color = (*color, opacity)
        if stroke_width is not None:
            self.stroke_width = stroke_width
        self.blur_radius = blur_radius
        self.double_border = double_border
        self.outer_border_color = (*outer_color, opacity)
        self.outer_stroke_width = outer_width
        # Clear cache when style changes
        with self.cache_lock:
            self.cache.clear()

    def apply_blur(self, img: Image.Image, radius: int) -> Image.Image:
        """Apply gaussian blur to the image."""
        if radius > 0:
            return img.filter(ImageFilter.GaussianBlur(radius))
        return img

    def render_text_to_mask(self, text: str, color: Tuple[int, int, int]) -> np.ndarray:
        """Render text to a transparent RGBA mask with custom border styling."""
        # Check cache first
        cache_key = f"{text}_{color}"
        with self.cache_lock:
            if cache_key in self.cache:
                return self.cache[cache_key].copy()
        
        margin = self.width * 0.1
        wrapped_lines = self.wrap_text_pil(text, self.width - 2 * margin)
        
        line_spacing = self.font_size * 0.3
        total_height = len(wrapped_lines) * (self.font_size + line_spacing)
        
        # Create a transparent image for the text
        mask = Image.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(mask)
        
        y_start = self.height - total_height - self.height * 0.2
        
        # Draw text with custom border
        for i, line in enumerate(wrapped_lines):
            bbox = self.font.getbbox(line)
            text_width = bbox[2] - bbox[0]
            x = (self.width - text_width) // 2
            y = y_start + i * (self.font_size + line_spacing)
            
            if self.double_border:
                # Draw outer border first
                outer_width = self.stroke_width + self.outer_stroke_width
                for offset_x in range(-outer_width, outer_width + 1):
                    for offset_y in range(-outer_width, outer_width + 1):
                        draw.text(
                            (x + offset_x, y + offset_y),
                            line,
                            font=self.font,
                            fill=self.outer_border_color
                        )
            
            # Draw main border
            for offset_x in range(-self.stroke_width, self.stroke_width + 1):
                for offset_y in range(-self.stroke_width, self.stroke_width + 1):
                    draw.text(
                        (x + offset_x, y + offset_y),
                        line,
                        font=self.font,
                        fill=self.border_color
                    )
            
            # Draw main text
            draw.text(
                (x, y),
                line,
                font=self.font,
                fill=(*color, 255)
            )
        
        # Apply blur if specified
        if self.blur_radius > 0:
            mask = self.apply_blur(mask, self.blur_radius)
        
        # Convert to numpy array and cache
        result = np.array(mask)
        with self.cache_lock:
            self.cache[cache_key] = result.copy()
        
        return result

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

# old with cpu
# def process_frame(args):
#     """Process a single frame with text overlay."""
#     frame, texts_and_colors, renderer = args
    
#     # Convert frame to RGBA
#     frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    
#     # Blend all text masks
#     for text, color in texts_and_colors:
#         text_mask = renderer.render_text_to_mask(text, color)
#         # Blend text mask with frame
#         alpha = text_mask[:, :, 3:4] / 255.0
#         frame_rgba = frame_rgba * (1 - alpha) + text_mask[:, :, :4] * alpha
    
#     # Convert back to BGR
#     return cv2.cvtColor(frame_rgba.astype(np.uint8), cv2.COLOR_RGBA2BGR)

# new with gpu
def process_frame(args):
    """Process a single frame with text overlay using GPU acceleration."""
    frame, texts_and_colors, renderer = args
    
    # Create UMat object to process on GPU
    frame_gpu = cv2.UMat(frame)
    
    # Convert frame to RGBA on GPU
    frame_rgba = cv2.cvtColor(frame_gpu, cv2.COLOR_BGR2RGBA)
    
    # Download to CPU for text rendering (PIL doesn't support GPU)
    frame_rgba = frame_rgba.get()
    
    # Blend all text masks
    for text, color in texts_and_colors:
        text_mask = renderer.render_text_to_mask(text, color)
        # Blend text mask with frame
        alpha = text_mask[:, :, 3:4] / 255.0
        frame_rgba = frame_rgba * (1 - alpha) + text_mask[:, :, :4] * alpha
    
    # Convert back to BGR using GPU
    frame_gpu = cv2.UMat(frame_rgba.astype(np.uint8))
    return cv2.cvtColor(frame_gpu, cv2.COLOR_RGBA2BGR).get()


def add_text_overlay(input_video: Path, input_audio: Path, output_video: Path, 
                    segments: List[TranscribedSegment], font_size: int = 32):
    # Enable OpenCL
    if cv2.ocl.haveOpenCL():
        cv2.ocl.setUseOpenCL(True)
        print("OpenCL acceleration enabled")
    else:
        print("OpenCL not available, falling back to CPU")


    """Add text overlay to video using PIL for custom font rendering."""
    # Get video properties
    width, height, fps, frame_count = get_video_properties(input_video)
    

    MAX_HEIGHT = 720
    if height > MAX_HEIGHT:
        # Resize video to fit within 720p
        scale_factor = MAX_HEIGHT / height
        width = int(width * scale_factor)
        height = MAX_HEIGHT

    TARGET_FPS = 60 
    fps = min(fps, TARGET_FPS)

    # Setup color mapping for speakers
    colors = [(255, 255, 255), (255, 255, 0), (0, 255, 255),
             (0, 255, 0), (255, 192, 203), (255, 165, 0)]  # RGB format
    speaker_colors = defaultdict(lambda: (255, 255, 255))

    renderer = TextRenderer("sfuidisplay_bold.ttf", font_size, width, height)
    renderer.set_border_style(
        color=(0, 0, 0),
        stroke_width=5,
        opacity=255
    )

    unique_speakers = {seg.speaker for seg in segments if seg.speaker}
    for i, speaker in enumerate(unique_speakers):
        speaker_colors[speaker] = colors[i % len(colors)]
    
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
    
    with ThreadPoolExecutor(max_workers=os.cpu_count() - 2) as executor:
        frame_count = 0
        while frame_count < frames_to_process:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Resize frame if necessary
            if frame.shape[0] != height or frame.shape[1] != width:
                frame_gpu = cv2.UMat(frame)
                frame = cv2.resize(frame_gpu, (width, height), interpolation=cv2.INTER_AREA).get()


            
            # Skip frames to match target FPS if necessary
            if fps > TARGET_FPS and frame_count % int(fps / TARGET_FPS) != 0:
                frame_count += 1
                continue
                
            current_time = frame_count / fps
            
            if current_time > last_segment_end:
                break
            
            # Find active segments for the current time
            active_segments = [
                seg for seg in segments 
                if seg.start <= current_time <= seg.end
            ]

            # Prepare text and add colors for active segments
            text_and_colors = [
                (seg.text.strip(), speaker_colors[seg.speaker])
                for seg in active_segments
            ]

            # Process frame with text overlay
            frame_with_text = process_frame((frame, text_and_colors, renderer))
            out.write(frame_with_text)
            
            if frame_count % 100 == 0:
                progress = (frame_count + 1) / frames_to_process * 100
                print(f"Processed {frame_count + 1}/{frames_to_process} frames ({progress:.2f}% complete)")
            
            frame_count += 1


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
        # '-r', str(TARGET_FPS),
        '-strict', 'experimental',
        str(output_video)
    ]
    # cmd = [
    #     'ffmpeg', '-y',              # Yes to overwrite output
    #     '-i', temp_output,           # Input video
    #     '-i', str(input_audio),      # Input audio
    #     '-c:v', 'libx264',          # CPU H.264 encoder
    #     '-preset', 'medium',         # Balance between speed/compression (options: ultrafast to veryslow)
    #     '-crf', '23',               # Constant Rate Factor - quality (18-28, lower = better)
    #     '-maxrate', '4M',            # Maximum bitrate
    #     '-b:a', '128k',             # Audio bitrate - valid
    #     '-r', str(TARGET_FPS),      # Frame rate - valid if you need to specify
    #     '-movflags', '+faststart',   # Valid - helps web playback
    #     str(output_video)
    # ]


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